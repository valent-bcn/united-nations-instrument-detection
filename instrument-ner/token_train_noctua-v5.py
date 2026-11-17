"""
Weakly-supervised NER pipeline for detecting instrument/convention/protocol names in UN
General Assembly resolution text.

Training entry point. The data steps (1-3b) live in ner_data.py and the
metrics / mention-level report (5) in ner_eval.py.

Pipeline:
    1. Build a list of names (official titles + aliases) from several instrument
       reference tables (Wikipedia, OHCHR, UNO, UNESCO, conventions/protocols).
    2. Use a spaCy PhraseMatcher over the resolution corpus to weakly label
       spans that match a known name ("known_docs"); resolutions with no
       match are kept separately ("unknown_docs").
    3. Convert the character-level spans from step 2 into token-level BIO
       tags (B-INSTRUMENT / I-INSTRUMENT / O) for HF token classification.
       NEW: tokens inside an "instrument-looking" phrase that the lexicon did
       NOT match are tagged IGN instead of O -> label -100, i.e. they are
       neither rewarded nor punished. These are the likely false negatives of
       the weak labelling. The detector (CandidateFinder) is case-insensitive
       and expands both ways from a trigger word (convention, rules, ...), so
       it handles lowercase text and names like "Beijing Rules".
    4. Fine-tune a token classifier on the BIO data with the HF Trainer,
       logging seqeval precision/recall/F1 per epoch (stdout + CSV). Long
       documents are split into overlapping windows (sliding window).
       NEW (train split only):
         * mention replacement: with probability --aug-prob, each lexicon
           mention is replaced by another lexicon name (adopted before the
           eval year and never seen in the eval docs) -> the model cannot
           solve the task by memorising the ~300 names present in the text;
         * negative window sampling: windows without any labelled mention are
           kept with probability --neg-window-ratio.
    5. NEW: mention-level report on the eval split: recall on names seen /
       unseen in the training years, strict precision, and a CSV with every
       predicted span that is not in the weak gold (to inspect by hand and
       to grow the lexicon).

Baseline (previous behaviour):
    --mask-candidates none --neg-window-ratio 1 --aug-prob 0
"""

import argparse
import os
from functools import partial
from collections import Counter

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download, list_repo_files
from packaging import version
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ner_data import (
    ID2LABEL, IGNORE_TAG, LABEL2ID, LABEL_LIST,
    CandidateFinder, augment_mentions, build_augmentation_pool, build_golden_list,
    build_name_vocab, corpus_doc_freq, get_or_build_partition, load_instruments,
    load_resolutions, split_mentions, to_bio_examples, unmask,
)
from ner_eval import MetricsCSVLogger, compute_metrics, mention_level_report

# Sliding-window chunking for long documents.
# MAX_LENGTH: model window (BERT / RoBERTa / DeBERTa-v3 base & large are all 512).
# STRIDE: number of subword tokens shared by two consecutive windows. Any mention
#   of <= STRIDE subwords that is cut by one window edge is fully contained in the
#   next window. Spans: mean 51.9 + 2*33.4 = ~119 chars ~= 25-30 subwords.
# consecutive windows (Hugging Face tokenizer convention).
MAX_LENGTH = 512
STRIDE = 30


# ---------------------------------------------------------------------------
# 4. TOKEN CLASSIFIER FINE-TUNING (HF Trainer)
# ---------------------------------------------------------------------------

def _word_groups(tags):
    """
    Map every word index to the tuple of word indices of the unit it belongs to:
    the whole BIO span for entity words, or just itself for "O"/IGN words.
    """
    group_of = {}
    cur = None
    for i, t in enumerate(tags):
        if t.startswith("I-") and cur is not None:
            cur.append(i)
        else:
            cur = [i]
            if t == "O":
                group_of[i] = (i,)
                cur = None
                continue
        for w in cur:
            group_of[w] = cur  # list, finalised below
    return {w: tuple(g) for w, g in group_of.items()}


def align_labels(examples, indices, tokenizer, label2id=LABEL2ID,
                 max_length=MAX_LENGTH, stride=STRIDE):
    """
    Batched, model-agnostic alignment of word-level BIO labels to subword tokens,
    with sliding-window chunking instead of truncation. Must be mapped with
    batched=True, with_indices=True (one document -> several windows/rows).

    Each word is "owned" by exactly one window: the first window that contains
    all the subwords of its unit (the whole mention for entity words, the word
    itself otherwise). Only the owner window gives it a label (first subword);
    everywhere else -> -100. IGN words are owned but labelled -100.

    Extra columns (dropped automatically by the Trainer, used for the
    mention-level report):
      example_idx    index of the source document in the input list
      pred_word_ids  per subword position: the word id it owns, or -1
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
    )
    sample_map = tokenized.pop("overflow_to_sample_mapping")

    full = tokenizer(examples["tokens"], is_split_into_words=True, truncation=False)
    total_counts = [
        Counter(w for w in full.word_ids(j) if w is not None)
        for j in range(len(examples["tokens"]))
    ]
    groups = [_word_groups(tags) for tags in examples["ner_tags"]]
    done = [set() for _ in examples["tokens"]]

    all_labels, all_pred_words, example_idx = [], [], []
    for i, s in enumerate(sample_map):
        tags = examples["ner_tags"][s]
        total = total_counts[s]
        word_ids = tokenized.word_ids(batch_index=i)
        in_chunk = Counter(w for w in word_ids if w is not None)

        def complete(w):
            # words with 0 subwords (e.g. spaCy whitespace tokens) never block a span
            return total.get(w, 0) == 0 or in_chunk.get(w, 0) == total[w]

        aligned, pred_words = [], []
        prev_word = None
        done_here = set()
        for wid in word_ids:
            if wid is None or wid == prev_word:
                aligned.append(-100)
                pred_words.append(-1)
            elif wid in done[s] or not all(complete(x) for x in groups[s][wid]):
                aligned.append(-100)
                pred_words.append(-1)
            else:
                done_here.add(wid)
                pred_words.append(wid)
                aligned.append(-100 if tags[wid] == IGNORE_TAG else label2id[tags[wid]])
            prev_word = wid
        done[s].update(done_here)
        all_labels.append(aligned)
        all_pred_words.append(pred_words)
        example_idx.append(indices[s])

    tokenized["labels"] = all_labels
    tokenized["pred_word_ids"] = all_pred_words
    tokenized["example_idx"] = example_idx
    return tokenized


def sample_negative_windows(ds, neg_ratio, seed=42):
    """Keep every window with a labelled mention, and a `neg_ratio` fraction of the rest."""
    b_id = LABEL2ID["B-INSTRUMENT"]
    has_pos = np.array([b_id in labels for labels in ds["labels"]], dtype=bool)
    rng = np.random.default_rng(seed)
    keep = has_pos | (rng.random(len(ds)) < neg_ratio)
    print(f"train windows: {int(has_pos.sum())} with a labelled mention | "
          f"kept {int((keep & ~has_pos).sum())}/{int((~has_pos).sum())} without")
    return ds.select(np.flatnonzero(keep).tolist())


# Converted checkpoints are cached next to this script (not the launch directory),
# so each model is converted only once no matter where the script is started from.
CONVERTED_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "converted-models"
)


def ensure_safetensors(model_name, cache_root=CONVERTED_MODELS_DIR):
    """
    Return something from_pretrained can load safely on torch < 2.6.

    transformers refuses to torch.load a pytorch_model.bin on torch < 2.6
    (CVE-2025-32434) but has no such restriction for safetensors. Some hub repos
    (e.g. microsoft/deberta-v3-base) only ship the .bin file on their main branch.
    """
    if version.parse(torch.__version__) >= version.parse("2.6"):
        return model_name
    if os.path.isdir(model_name):
        return model_name

    out = os.path.join(cache_root, model_name.replace("/", "-"))
    if os.path.exists(os.path.join(out, "model.safetensors")):
        return out

    try:
        files = list_repo_files(model_name)
    except Exception:
        return model_name
    if any(f.endswith(".safetensors") for f in files):
        return model_name
    if "pytorch_model.bin" not in files:
        return model_name

    print(f"{model_name}: no safetensors on the hub and torch < 2.6, converting to {out}")
    p = hf_hub_download(model_name, "pytorch_model.bin")
    # weights_only=False unpickles arbitrary objects: only do this for trusted repos
    sd = torch.load(p, map_location="cpu", weights_only=False)
    sd = {k: v.clone().contiguous() for k, v in sd.items()}

    os.makedirs(out, exist_ok=True)
    AutoConfig.from_pretrained(model_name).save_pretrained(out)
    save_file(sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    return out


def train_token_classifier(bio_examples,
                           doc_years,
                           eval_from_year=2015,
                           model_name="bert-base-uncased",
                           output_dir="./checkpoints/",
                           mask_mode="both",
                           neg_window_ratio=0.5,
                           aug_pool=None,
                           aug_prob=0.0,
                           train_mentions=(),
                           lowercase_input=False,
                           max_length=MAX_LENGTH,
                           stride=STRIDE,
                           seed=42):
    """
    Fine-tune a token classification model to detect INSTRUMENT spans.

    mask_mode:
        "none"  IGN -> O everywhere (original weak labelling)
        "train" IGN ignored in train, counted as O in eval
        "both"  IGN ignored in train and in eval (eval F1 no longer penalises
                predictions on unlabelled instrument-looking phrases)
        Eval numbers are only comparable between runs with the same eval masking.
    neg_window_ratio:
        fraction of train windows without a labelled mention that are kept (1 = all).
    aug_pool / aug_prob:
        mention replacement on the train split (see augment_mentions).
    """
    if not 0 < stride < max_length // 2:
        raise ValueError(f"stride ({stride}) must be in (0, max_length/2)")

    model_type = AutoConfig.from_pretrained(model_name).model_type
    tok_kwargs = {"add_prefix_space": True} if model_type in {"roberta", "gpt2", "longformer", "bart"} else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)

    model_path = ensure_safetensors(model_name)
    
    train_ex = [ex for ex in bio_examples if doc_years[ex["doc_id"]] < eval_from_year]
    eval_ex_raw = [ex for ex in bio_examples if doc_years[ex["doc_id"]] >= eval_from_year]
    print(f"train: {len(train_ex)} | eval (>= {eval_from_year}): {len(eval_ex_raw)}")

    if mask_mode == "none":
        train_ex = unmask(train_ex)
    eval_ex = eval_ex_raw if mask_mode == "both" else unmask(eval_ex_raw)

    if aug_pool and aug_prob > 0:
        train_ex = augment_mentions(train_ex, aug_pool, aug_prob, seed=seed)

    if lowercase_input:
        # removes the train/eval casing shift for cased models (no-op for *-uncased)
        lower = lambda exs: [{**ex, "tokens": [t.lower() for t in ex["tokens"]]} for ex in exs]
        train_ex, eval_ex = lower(train_ex), lower(eval_ex)

    align = partial(align_labels, tokenizer=tokenizer, max_length=max_length, stride=stride)
    cols = ["tokens", "ner_tags", "doc_id"]
    train_ds = Dataset.from_list(train_ex).map(
        align, batched=True, with_indices=True, remove_columns=cols)
    eval_ds = Dataset.from_list(eval_ex).map(
        align, batched=True, with_indices=True, remove_columns=cols)
    print(f"windows ({max_length} tok, stride {stride}) -> "
          f"train: {len(train_ds)} | eval: {len(eval_ds)}")

    if neg_window_ratio < 1:
        train_ds = sample_negative_windows(train_ds, neg_window_ratio, seed=seed)

    # Force float32 weights (deberta-v3 stores fp16 weights; pure-fp16 AdamW -> NaN).
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        dtype=torch.float32,
    )
    print(f"model dtype: {next(model.parameters()).dtype}")
    collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=6,
        warmup_ratio=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[MetricsCSVLogger(output_dir)],
    )
    trainer.train()

    mention_level_report(trainer, eval_ds, eval_ex_raw, train_mentions, output_dir)

    return trainer, tokenizer, ID2LABEL


# ---------------------------------------------------------------------------
# DATA LOAD & RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a token classifier for instrument NER.")
    parser.add_argument("--model-name", default="bert-base-uncased",
                        help="HF model id, e.g. bert-base-cased, roberta-base, microsoft/deberta-v3-base")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to ./checkpoints-<model-name> (slashes replaced by '-')")
    parser.add_argument("--eval-from-year", type=int, default=2015)
    parser.add_argument("--mask-candidates", choices=["none", "train", "both"], default="both",
                        help="Ignore (-100) unlabelled instrument-looking phrases instead of O")
    parser.add_argument("--neg-window-ratio", type=float, default=0.5,
                        help="Fraction of train windows without a mention to keep (1 = all)")
    parser.add_argument("--aug-prob", type=float, default=0.3,
                        help="Probability of replacing a train mention with another lexicon name (0 = off)")
    parser.add_argument("--max-df", type=float, default=0.05,
                        help="A word is 'rare' (can anchor a candidate) if it occurs in "
                             "less than this fraction of documents")
    parser.add_argument("--lowercase-input", action=argparse.BooleanOptionalAction, default=True,
                        help="Lowercase all tokens before tokenisation (useful for cased models)")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolutions",
                        default="/home/user/branes/NER-data/ga_resolutions_1946_2025.csv")
    parser.add_argument("--instruments", default="/home/user/branes/data/df_instruments",
                        help="Lexicon CSV (title, year, alternative_name) or a directory of them")
    parser.add_argument("--partition-dir", default="/home/user/branes/NER-data/data/partition/")
    args_cli = parser.parse_args()

    safe_name = args_cli.model_name.replace("/", "-")
    output_dir = args_cli.output_dir or f"./checkpoints-{safe_name}-seed{args_cli.seed}"
    print(f"model: {args_cli.model_name} | output_dir: {output_dir}")

    df_res, doc_years = load_resolutions(args_cli.resolutions)
    df_instrument = load_instruments(args_cli.instruments)

    aliases = build_golden_list(df_instrument)
    known_docs, unknown_docs = get_or_build_partition(df_res, aliases, out_dir=args_cli.partition_dir)
    print(f"Docs with a known match: {len(known_docs)} | without match: {len(unknown_docs)}")

    train_mentions, eval_mentions = split_mentions(known_docs, doc_years, args_cli.eval_from_year)

    aug_pool = None
    if args_cli.aug_prob > 0:
        aug_pool = build_augmentation_pool(df_instrument, args_cli.eval_from_year, eval_mentions)

    finder = None
    if args_cli.mask_candidates != "none":
        name_vocab = build_name_vocab(aliases)
        doc_freq, n_docs = corpus_doc_freq(d["text"] for d in known_docs)
        finder = CandidateFinder(name_vocab, doc_freq, n_docs, max_df=args_cli.max_df)
        print(f"candidate finder: {len(name_vocab)} lexicon words, "
              f"rare = df < {finder.max_df_count}/{n_docs} docs")

    bio_examples = to_bio_examples(
        known_docs, finder=finder, doc_years=doc_years,
        eval_from_year=args_cli.eval_from_year,
        candidates_csv=os.path.join(output_dir, "candidates_ign.csv"),
    )
    trainer, tokenizer, id2label = train_token_classifier(
        bio_examples,
        doc_years,
        eval_from_year=args_cli.eval_from_year,
        model_name=args_cli.model_name,
        output_dir=output_dir,
        mask_mode=args_cli.mask_candidates,
        neg_window_ratio=args_cli.neg_window_ratio,
        aug_pool=aug_pool,
        aug_prob=args_cli.aug_prob,
        train_mentions=train_mentions,
        lowercase_input=args_cli.lowercase_input,
        max_length=args_cli.max_length,
        stride=args_cli.stride,
        seed=args_cli.seed,
    )
