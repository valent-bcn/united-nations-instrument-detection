#!/usr/bin/env python
"""
Evaluate zero-shot GLiNER + a fine-tuned BERT-style NER model COMBINED against the
manually annotated gold standard (single unlabelled class: INSTRUMENT).

Three systems are scored in one run (same documents, same gold, same metrics):
    gliner    GLiNER alone. It is asked for the generic tags in --tags and EVERY entity it
              returns counts as an INSTRUMENT prediction.
    bert      the fine-tuned model alone. Only the INSTRUMENT label is kept (PER / ORG / ...
              are switched off).
    combined  GLiNER + BERT merged according to --merge.

Pipeline
  1. load the gold file and the documents it refers to (same loader as eval_gliner.py /
     eval_ner_checkpoints.py: gold end is exclusive, text == content[start:end]).
     A document that contains NO instrument is declared with a gold row holding only the
     res_id2 (text, start, end empty), e.g.   A/HRC/RES/XX/1,,,
     It is evaluated like any other document, every prediction on it is a false positive and
     the number of such predictions is reported (neg_docs / neg_doc_fp),
  2. run GLiNER once over every document (overlapping windows, 384-word limit),
     keep predictions with score >= --gliner-threshold (default 0.85),
  3. for every selected BERT checkpoint run the fine-tuned model over every document
     (overlapping token windows). By default (--pretokenize spacy) the document is first
     split into spaCy words and fed with is_split_into_words=True, exactly like in
     token_train_noctua.py, and every word gets the prediction of its first sub-token.
     The score of a predicted span is the mean, over its words, of the probability of
     being inside an INSTRUMENT (1 - P(O)).
     Keep spans with score >= --bert-threshold (default 0.85),
  4. merge both prediction sets (--merge):
       bert-priority (default)  every BERT span is kept; a GLiNER span is only added when it
                                overlaps NO BERT span. The instruments the BERT found are
                                therefore always in the output and GLiNER only fills the gaps.
       score                    overlapping spans compete, the highest score wins
                                (the two models' scores are not calibrated against each other).
       union                    keep everything, overlapping spans allowed.
  5. compute precision / recall / F1 at two granularities
       * entity level (exact span match), and
       * token level (word units, partial credit),
  6. print everything and write one row per system (and per BERT checkpoint) to
     ./GLiNER_BERT_test_metrics-combined.csv

Example
-------
    python eval_gliner_BERT.py --bert-model checkpoints-roberta-large
    python eval_gliner_BERT.py --bert-model checkpoints-roberta-large \
        --bert-checkpoints checkpoint-8128 checkpoint-10160
    python eval_gliner_BERT.py --bert-model checkpoints-roberta-large/checkpoint-8128 \
        --merge score --save-predictions
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional
    def tqdm(iterable, **kwargs):
        return iterable

DEFAULT_DOC_DF = "/home/user/branes/NER-data/ga_resolutions_1946_2025.csv"
GOLD = "/home/user/branes/NER-data/NER_golden_truth_10.csv"

# --- GLiNER ---------------------------------------------------------------- #
GLINER_MODEL_NAME = "urchade/gliner_large-v2.1"
GLINER_SCORE_THRESHOLD = 0.85
TAGS = [
    "agenda",
    "agreement",
    "goals",
    "framework",
    "policy",
    "convention",
    "covenant",
    "charter",
    "pact",
    "treaty",
    "declaration",
    "programme of action",
]

# --- fine-tuned BERT ------------------------------------------------------- #
DEFAULT_ENTITY_TYPE = "INSTRUMENT"
BERT_SCORE_THRESHOLD = 0.0
CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

# GLiNER splits text into "words" with this pattern; max_len (384) counts those words.
GLINER_WORD_RE = re.compile(r"\w+(?:[-_]\w+)*|\S", re.UNICODE)
# Word units used for the token-level metric (model independent).
WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Every entity in this script is a tuple (start, end_exclusive, score, label, text, source),
# with source in {"gliner", "bert"}.


# --------------------------------------------------------------------------- #
# Data (same gold loader as eval_gliner.py / eval_ner_checkpoints.py)
# --------------------------------------------------------------------------- #
def resolve_span(content: str, text: str, start: int, end: int, mode: str):
    """Map a gold (start, end) pair onto `content`.

    Returns (start, end_exclusive, convention) when content[a:b] == text EXACTLY
    (no whitespace normalisation), otherwise None.
    mode: 'exclusive' -> text == content[start:end]      (Python slice)
          'inclusive' -> text == content[start:end + 1]
          'auto'      -> whichever of the two reproduces the gold text
    """
    options = {"exclusive": (start, end), "inclusive": (start, end + 1)}
    for conv in (["exclusive", "inclusive"] if mode == "auto" else [mode]):
        a, b = options[conv]
        if 0 <= a < b <= len(content) and content[a:b] == text:
            return a, b, conv
    return None


def load_data(docs_path: str, gold_path: str, gold_end: str = "auto"):
    """Returns (docs_text, gold_spans).

    docs_text  : {res_id2: content}  restricted to documents present in the gold file
    gold_spans : {res_id2: {(start, end_exclusive), ...}}

    A document WITHOUT any instrument is declared by a gold row that contains only the
    res_id2 (text, start and end left empty). It is kept in docs_text with an empty gold set,
    so every span predicted on it counts as a false positive.
    """
    docs = pd.read_csv(docs_path, dtype={"res_id2": str})
    docs["res_id2"] = docs["res_id2"].str.strip()
    docs["content"] = docs["content"].fillna("").astype(str)
    n_dup = int(docs["res_id2"].duplicated().sum())
    # every distinct text version of each document (normally exactly one)
    variants = {rid: list(dict.fromkeys(g))
                for rid, g in docs.groupby("res_id2", sort=False)["content"]}

    gold = pd.read_csv(gold_path, dtype={"res_id2": str})
    gold = gold.dropna(subset=["res_id2"])  # completely empty lines such as ",,,"
    gold["res_id2"] = gold["res_id2"].str.strip()
    gold["text"] = gold["text"].fillna("").astype(str)
    # empty start/end cells become NaN (otherwise int(r.start) below would raise a ValueError)
    gold["start"] = pd.to_numeric(gold["start"], errors="coerce")
    gold["end"] = pd.to_numeric(gold["end"], errors="coerce")

    # rows with only a res_id2 = "this document has no instruments"
    no_start_end = gold["start"].isna() | gold["end"].isna()
    blank = no_start_end & gold["start"].isna() & gold["end"].isna() & (gold["text"].str.strip() == "")
    partial = no_start_end & ~blank  # e.g. a text but no offsets: malformed, cannot be used
    if partial.any():
        print(f"[warn] {int(partial.sum())} gold rows have a missing start/end but are not completely "
              f"empty; they are ignored (e.g. {gold.loc[partial, 'res_id2'].iloc[0]}).")
    neg_ids = set(gold.loc[blank, "res_id2"])
    gold = gold[~no_start_end].drop_duplicates(["res_id2", "start", "end"])

    items_by_doc = {rid: [(r.text, int(r.start), int(r.end)) for r in rows.itertuples(index=False)]
                    for rid, rows in gold.groupby("res_id2", sort=True)}
    clash = sorted(neg_ids & set(items_by_doc))
    if clash:
        print(f"[warn] {len(clash)} documents are marked as empty (row with only res_id2) but also have "
              f"gold spans; the spans are used and the empty row is ignored (e.g. {clash[0]}).")
    for rid in neg_ids:
        items_by_doc.setdefault(rid, [])

    missing = sorted(rid for rid in items_by_doc if rid not in variants)
    if missing:
        n_lost = sum(len(items_by_doc[rid]) for rid in missing)
        print(f"[warn] {len(missing)} gold documents ({n_lost} spans) refer to res_id2 not found in docs "
              f"(e.g. {missing[0]}); they are dropped.")
        for rid in missing:
            del items_by_doc[rid]

    docs_text, gold_spans = {}, {}
    n_conv = {"exclusive": 0, "inclusive": 0, "unmatched": 0}
    n_conflict = 0
    bad = []
    for rid, items in sorted(items_by_doc.items()):
        versions = variants[rid]
        if len(versions) > 1:
            n_conflict += 1
        # duplicated res_id2 with different text: use the version in which most gold spans
        # reproduce the gold text exactly (ties -> first version)
        content = max(versions, key=lambda c: sum(
            resolve_span(c, t, s, e, gold_end) is not None for t, s, e in items))
        docs_text[rid] = content
        spans = set()
        for t, s, e in items:
            hit = resolve_span(content, t, s, e, gold_end)
            if hit is None:
                n_conv["unmatched"] += 1
                bad.append((rid, s, e, t, content[max(0, s - 30):e + 30]))
                spans.add((s, e + 1) if gold_end == "inclusive" else (s, e))
            else:
                n_conv[hit[2]] += 1
                spans.add((hit[0], hit[1]))
        gold_spans[rid] = spans

    n_gold = sum(len(v) for v in gold_spans.values())
    n_empty = sum(1 for v in gold_spans.values() if not v)
    print(f"Gold: {n_gold} unique spans in {len(docs_text)} documents "
          f"({n_empty} without any instrument).")
    print(f"Gold end convention (--gold-end {gold_end}): {n_conv['exclusive']} end-exclusive, "
          f"{n_conv['inclusive']} end-inclusive, {n_conv['unmatched']} NOT reproducing the gold text.")
    if n_dup:
        print(f"[info] {n_dup} duplicated rows by res_id2 in docs; {n_conflict} gold documents have "
              f"more than one different text version (the one matching the gold spans is used).")
    if bad:
        print(f"[warn] {len(bad)} gold spans do not reproduce the gold text exactly. First ones:")
        for rid, s, e, t, ctx in bad[:5]:
            print(f"       {rid} start={s} end={e} gold={t!r}  context={ctx!r}")
    return docs_text, gold_spans


# --------------------------------------------------------------------------- #
# GLiNER inference (same as eval_gliner.py)
# --------------------------------------------------------------------------- #
def load_gliner(model_name, device):
    from gliner import GLiNER  # imported here so the rest of the file works without gliner

    model = GLiNER.from_pretrained(model_name)
    try:
        model = model.to(device)
    except Exception as exc:  # keep going on whatever device GLiNER chose
        print(f"[warn] could not move GLiNER to {device}: {exc}")
    print(f"GLiNER model: {model_name}  |  device: {device}")
    return model


def gliner_predict_spans(gliner_model, text, labels, threshold, stats=None):
    """Returns [(start_char, end_char, score, label, text), ...] from GLiNER.

    end_char is exclusive (text[start:end] == entity text). Every span is checked against
    `text` and repaired if GLiNER's offsets are off by one; spans that cannot be located
    are dropped and counted in stats["dropped"].
    """
    stats = stats if stats is not None else {}
    out = []
    for e in gliner_model.predict_entities(text, labels, threshold=threshold):
        s, en, ent = int(e["start"]), int(e["end"]), e["text"]
        if text[s:en] != ent:
            stats["misaligned"] = stats.get("misaligned", 0) + 1
            if text[s:en + 1] == ent:
                en += 1
            else:
                j = text.find(ent, max(0, s - 10))
                if j < 0:
                    stats["dropped"] = stats.get("dropped", 0) + 1
                    continue
                s, en = j, j + len(ent)
        out.append((s, en, float(e["score"]), e["label"], ent))
    return out


def predict_long_text(gliner_model, text, labels, threshold, chunk_words=300, margin_words=50, stats=None):
    """GLiNER over a document of any length.

    The document is split into consecutive "cores" of (chunk_words - 2 * margin_words) words;
    each core is processed together with `margin_words` words of context on both sides
    (at most chunk_words words per call, below GLiNER's 384-word limit). An entity is taken
    from the window whose core contains its first character, so windows never produce the same
    entity twice and an entity crossing a core boundary is still seen whole.
    """
    core = chunk_words - 2 * margin_words
    if core < 1:
        raise ValueError("--chunk-words must be larger than 2 * --margin-words")
    words = [m.span() for m in GLINER_WORD_RE.finditer(text)]
    n = len(words)
    if n == 0:
        return []
    if n <= chunk_words:
        return gliner_predict_spans(gliner_model, text, labels, threshold, stats)

    out = []
    for core_start in range(0, n, core):
        core_end = min(core_start + core, n)
        lo, hi = max(0, core_start - margin_words), min(n, core_end + margin_words)
        a, b = words[lo][0], words[hi - 1][1]
        own_a = 0 if core_start == 0 else words[core_start][0]
        own_b = len(text) + 1 if core_end == n else words[core_end][0]
        for s, e, score, label, t in gliner_predict_spans(gliner_model, text[a:b], labels, threshold, stats):
            if own_a <= s + a < own_b:
                out.append((s + a, e + a, score, label, t))
    return out


def flatten(ents):
    """Keep non-overlapping entities, highest score first (GLiNER's flat NER, applied document-wide)."""
    kept = []
    for ent in sorted(ents, key=lambda x: (-x[2], -(x[1] - x[0]), x[0])):
        if all(ent[1] <= k[0] or ent[0] >= k[1] for k in kept):
            kept.append(ent)
    return sorted(kept, key=lambda x: x[0])


# --------------------------------------------------------------------------- #
# Fine-tuned BERT inference (as eval_ner_checkpoints.py, plus a span score)
# --------------------------------------------------------------------------- #
def parse_label(label: str):
    """'B-INSTRUMENT' -> ('B', 'INSTRUMENT'); 'O' -> ('O', None); 'INSTRUMENT' -> ('I', 'INSTRUMENT')."""
    if label == "O":
        return "O", None
    m = re.match(r"^([BI])-(.+)$", label)
    if m:
        return m.group(1), m.group(2)
    return "I", label  # plain label without BIO prefix


def trim_offsets(text: str, s: int, e: int):
    """Some tokenizers (sentencepiece / DeBERTa-v3) include the leading space in offsets."""
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


# Model types whose byte-level BPE tokenizer needs add_prefix_space=True for pre-split input
# (same set as token_train_noctua.py).
PREFIX_SPACE_TYPES = {"roberta", "gpt2", "longformer", "bart"}

_SPACY_NLP = None


def spacy_words(text: str):
    """Split `text` EXACTLY as token_train_noctua.py does (spacy.blank("en") tokenizer).

    Returns (words, offsets) with offsets[i] = (start, end_exclusive) of words[i] in `text`.
    """
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy  # imported here so --pretokenize none works without spaCy
        _SPACY_NLP = spacy.blank("en")
    doc = _SPACY_NLP.make_doc(text)
    return [t.text for t in doc], [(t.idx, t.idx + len(t.text)) for t in doc]


def decode_spans(units, label_info, o_ids, threshold, entity_type, text):
    """BIO decoding over ordered units.

    units: [(start, end_exclusive, avg_probs or None), ...] in text order.
      probs None -> the unit has no prediction and closes any open span;
      a whitespace-only unit is skipped (it neither opens nor closes a span).
    Span score = mean over its units of P(inside an INSTRUMENT) = 1 - P(O).
    """
    spans, cur = [], None  # cur = [start, end, [unit probs]]
    for s, e, avg in units:
        if avg is None:
            if not text[s:e].strip():
                continue
            if cur is not None:
                spans.append(cur)
                cur = None
            continue
        prefix, typ = label_info[int(np.argmax(avg))]
        if typ is None:  # "O"
            if cur is not None:
                spans.append(cur)
                cur = None
            continue
        p_ent = float(1.0 - avg[o_ids].sum()) if o_ids else 1.0
        if prefix == "B" or cur is None:
            if cur is not None:
                spans.append(cur)
            cur = [s, e, [p_ent]]
        else:  # "I" continuing the current entity
            cur[1] = e
            cur[2].append(p_ent)
    if cur is not None:
        spans.append(cur)

    out = []
    for s, e, ps in spans:
        s, e = trim_offsets(text, s, e)
        score = float(np.mean(ps))
        if e > s and score >= threshold:
            out.append((s, e, score, entity_type, text[s:e]))
    return out


class BertPredictor:
    def __init__(self, ckpt_dir, device, entity_type, max_length, stride, batch_size,
                 first_subword, threshold, pretokenize="spacy"):
        self.model = AutoModelForTokenClassification.from_pretrained(ckpt_dir).to(device).eval()
        tok_kwargs = ({"add_prefix_space": True}
                      if self.model.config.model_type in PREFIX_SPACE_TYPES else {})
        self.tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True, **tok_kwargs)
        if not self.tok.is_fast:
            raise RuntimeError(f"{ckpt_dir}: a fast tokenizer is required (character offsets).")
        self.device = device
        self.entity_type = entity_type
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.first_subword = first_subword
        self.threshold = threshold
        self.pretokenize = pretokenize

        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.label_info = {i: parse_label(l) for i, l in id2label.items()}
        self.labels = [id2label[i] for i in sorted(id2label)]
        allowed = [i for i, (_, t) in self.label_info.items()
                   if t is None or t.upper() == entity_type.upper()]
        if not any(t is not None for i, (_, t) in self.label_info.items() if i in allowed):
            raise ValueError(f"{ckpt_dir}: no '{entity_type}' label in {self.labels}")
        # Every label of another type (PER, ORG, LOC, ...) is switched off.
        mask = torch.full((len(id2label),), float("-inf"))
        mask[allowed] = 0.0
        self.logit_mask = mask.to(device)
        self.o_ids = [i for i, (_, t) in self.label_info.items() if t is None]
        self.input_names = list(self.tok.model_input_names)

    @torch.inference_mode()
    def _chunk_probs(self, enc):
        """Yields (chunk_index, probs[n_tokens, n_labels]) for every window in `enc`."""
        names = [k for k in self.input_names if k in enc]
        n_chunks = len(enc["input_ids"])
        for b in range(0, n_chunks, self.batch_size):
            top = min(b + self.batch_size, n_chunks)
            batch = {k: torch.tensor(enc[k][b:top], device=self.device) for k in names}
            logits = self.model(**batch).logits.float() + self.logit_mask
            probs = logits.softmax(-1).cpu().numpy()
            for j in range(top - b):
                yield b + j, probs[j]

    def predict(self, text: str):
        """Returns [(start, end_exclusive, score, label, text), ...] with score >= threshold.

        score = mean over the span's words (or tokens) of P(inside an INSTRUMENT) = 1 - P(O).
        """
        if not text.strip():
            return []
        if self.pretokenize == "spacy":
            return self._predict_words(text)
        return self._predict_raw(text)

    def _predict_words(self, text: str):
        """Inference with EXACTLY the tokenization used in training.

        token_train_noctua.py splits every document with spaCy and feeds the word list with
        is_split_into_words=True; only the first sub-token of each word carries a label and a
        word is labelled only in a window that contains all of its sub-tokens. This reproduces
        that: one prediction per spaCy word, read from its first sub-token, taken only from
        windows that contain the whole word (averaged when several windows do).
        Spans therefore always start and end on spaCy word boundaries.
        """
        words, offsets = spacy_words(text)
        enc = self.tok(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            padding=True,
        )
        # sub-tokens per word over the whole document, to detect words cut by a window edge
        total = self._subword_counts(words)

        acc: dict[int, list] = {}  # word index -> [sum_probs, count]
        fallback: dict[int, np.ndarray] = {}  # words never complete in any window
        for ci, p in self._chunk_probs(enc):
            wids = enc.word_ids(ci)
            in_chunk = Counter(w for w in wids if w is not None)
            seen = set()
            for ti, w in enumerate(wids):
                if w is None or w in seen:
                    continue
                seen.add(w)  # ti is the first sub-token of word w in this window
                if in_chunk[w] != total.get(w, 0):
                    fallback.setdefault(w, p[ti].copy())
                    continue
                slot = acc.get(w)
                if slot is None:
                    acc[w] = [p[ti].copy(), 1]
                else:  # word fully inside two overlapping windows -> average
                    slot[0] += p[ti]
                    slot[1] += 1

        units = []
        for w, (s, e) in enumerate(offsets):
            if w in acc:
                units.append((s, e, acc[w][0] / acc[w][1]))
            else:  # whitespace word with no sub-token, or a word longer than the window stride
                units.append((s, e, fallback.get(w)))
        return decode_spans(units, self.label_info, self.o_ids, self.threshold,
                            self.entity_type, text)

    def _subword_counts(self, words):
        """Number of sub-tokens of every word (no special tokens, no truncation)."""
        counts = Counter()
        enc = self.tok(words, is_split_into_words=True, add_special_tokens=False,
                       truncation=False, verbose=False)
        for w in enc.word_ids():
            if w is not None:
                counts[w] += 1
        return counts

    def _predict_raw(self, text: str):
        """Previous behaviour: the raw text goes straight to the tokenizer (--pretokenize none).

        NOT the tokenization used in training: around punctuation without a space, e.g.
        '(Sustainable' or '“Transforming', the sub-tokens differ from those seen in training.
        """
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            padding=True,
        )
        acc: dict[tuple, list] = {}  # (start, end) -> [sum_probs, count]
        for ci, p in self._chunk_probs(enc):
            if self.first_subword:  # label of the first sub-token is shared by the whole word
                src = np.arange(p.shape[0])
                first = {}
                for ti, w in enumerate(enc.word_ids(ci)):
                    if w is not None:
                        first.setdefault(w, ti)
                        src[ti] = first[w]
                p = p[src]
            offsets = enc["offset_mapping"][ci]
            special = enc["special_tokens_mask"][ci]
            att = enc["attention_mask"][ci]
            for ti, (s, e) in enumerate(offsets):
                if special[ti] or not att[ti]:
                    continue
                s, e = trim_offsets(text, s, e)
                if e <= s:
                    continue
                slot = acc.get((s, e))
                if slot is None:
                    acc[(s, e)] = [p[ti].copy(), 1]
                else:  # token seen in two overlapping windows -> average
                    slot[0] += p[ti]
                    slot[1] += 1

        units = [(s, e, acc[(s, e)][0] / acc[(s, e)][1]) for (s, e) in sorted(acc)]
        return decode_spans(units, self.label_info, self.o_ids, self.threshold,
                            self.entity_type, text)


# --------------------------------------------------------------------------- #
# Merging GLiNER + BERT
# --------------------------------------------------------------------------- #
def merge_predictions(gliner_ents, bert_ents, mode):
    """Merge the (already thresholded) entity lists of one document."""
    if mode == "bert-priority":
        kept = list(bert_ents)
        for g in gliner_ents:
            if all(g[1] <= b[0] or g[0] >= b[1] for b in bert_ents):
                kept.append(g)
        return sorted(kept, key=lambda x: x[0])
    if mode == "score":
        return flatten(list(bert_ents) + list(gliner_ents))
    if mode == "union":
        return sorted(list(bert_ents) + list(gliner_ents), key=lambda x: x[0])
    raise ValueError(f"unknown merge mode: {mode}")


def source_breakdown(merged, gold_spans):
    """Entity-level contribution of each source inside the combined output."""
    c = Counter()
    for rid, ents in merged.items():
        g = gold_spans.get(rid, set())
        src_of = {}
        for ent in ents:  # an identical span found by both models counts for BERT
            key = (ent[0], ent[1])
            if key not in src_of or ent[5] == "bert":
                src_of[key] = ent[5]
        for key, src in src_of.items():
            c[f"n_pred_from_{src}"] += 1
            if key in g:
                c[f"tp_from_{src}"] += 1
    return {k: c.get(k, 0) for k in
            ("n_pred_from_bert", "n_pred_from_gliner", "tp_from_bert", "tp_from_gliner")}


# --------------------------------------------------------------------------- #
# Metrics (same as eval_gliner.py / eval_ner_checkpoints.py)
# --------------------------------------------------------------------------- #
def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _covered(n_chars, spans, word_starts, word_ends):
    """Boolean array: does each word overlap any of the spans?"""
    mask = np.zeros(n_chars + 1, dtype=np.int32)
    for s, e in spans:
        mask[max(s, 0):e] = 1
    cum = np.concatenate(([0], np.cumsum(mask)))
    return (cum[word_ends] - cum[word_starts]) > 0


def evaluate(docs_text, gold_spans, pred_spans):
    e_tp = e_fp = e_fn = 0
    t_tp = t_fp = t_fn = 0
    neg_docs = neg_doc_fp = 0  # documents WITHOUT gold instruments and the spans predicted there
    for rid, text in docs_text.items():
        g = gold_spans.get(rid, set())
        p = pred_spans.get(rid, set())
        # entity level, exact match
        e_tp += len(g & p)
        e_fp += len(p - g)
        e_fn += len(g - p)
        if not g:
            neg_docs += 1
            neg_doc_fp += len(p)
        # token level (word units), partial credit
        words = [m.span() for m in WORD_RE.finditer(text)]
        if not words:
            continue
        ws = np.array([w[0] for w in words], dtype=np.int64)
        we = np.array([w[1] for w in words], dtype=np.int64)
        gc_ = _covered(len(text), g, ws, we)
        pc_ = _covered(len(text), p, ws, we)
        t_tp += int((gc_ & pc_).sum())
        t_fp += int((~gc_ & pc_).sum())
        t_fn += int((gc_ & ~pc_).sum())

    ep, er, ef = prf(e_tp, e_fp, e_fn)
    tp_, tr, tf = prf(t_tp, t_fp, t_fn)
    return {
        "entity_tp": e_tp, "entity_fp": e_fp, "entity_fn": e_fn,
        "entity_precision": ep, "entity_recall": er, "entity_f1": ef,
        "token_tp": t_tp, "token_fp": t_fp, "token_fn": t_fn,
        "token_precision": tp_, "token_recall": tr, "token_f1": tf,
        "neg_docs": neg_docs, "neg_doc_fp": neg_doc_fp,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def bert_short_name(model_dir: str) -> str:
    path = os.path.abspath(os.path.normpath(model_dir))
    base = os.path.basename(path)
    if CKPT_RE.match(base):  # --bert-model points directly at a checkpoint folder
        base = os.path.basename(os.path.dirname(path))
    return re.sub(r"^checkpoints?[-_]", "", base)


def resolve_bert_checkpoints(model_dir: str, wanted=None):
    """Returns [(step_or_None, checkpoint_name, path), ...]."""
    model_dir = os.path.normpath(model_dir)
    if not os.path.isdir(model_dir):
        sys.exit(f"--bert-model not found: {model_dir}")
    if os.path.isfile(os.path.join(model_dir, "config.json")):
        m = CKPT_RE.match(os.path.basename(os.path.abspath(model_dir)))
        return [(int(m.group(1)) if m else None, os.path.basename(os.path.abspath(model_dir)), model_dir)]

    found = []
    for name in os.listdir(model_dir):
        m = CKPT_RE.match(name)
        path = os.path.join(model_dir, name)
        if m and os.path.isdir(path):
            found.append((int(m.group(1)), name, path))
    found.sort()
    if not found:
        sys.exit(f"No config.json and no checkpoint-<step> folders found in {model_dir}")

    if wanted:
        if [w.lower() for w in wanted] == ["all"]:
            return found
        steps = set()
        for w in wanted:
            m = re.search(r"(\d+)$", w)  # "checkpoint-8128" or just "8128"
            if m:
                steps.add(int(m.group(1)))
        sel = [c for c in found if c[0] in steps]
        if not sel:
            sys.exit(f"None of {wanted} found in {model_dir} (available: {[c[1] for c in found]})")
        return sel

    print(f"[info] no --bert-checkpoints given: using the latest one, {found[-1][1]} "
          f"(available: {', '.join(c[1] for c in found)}; use '--bert-checkpoints all' for all of them)")
    return [found[-1]]


def read_epoch(ckpt_path: str):
    try:
        with open(os.path.join(ckpt_path, "trainer_state.json")) as f:
            return float(json.load(f)["epoch"])
    except Exception:
        return float("nan")


def to_span_sets(ents_by_doc):
    return {rid: {(e[0], e[1]) for e in ents} for rid, ents in ents_by_doc.items()}


def make_row(system, args, bert_name, ckpt, n_docs, n_gold, pred_spans, metrics, extra=None):
    uses_gliner = system in ("gliner", "combined")
    uses_bert = system in ("bert", "combined")
    row = {
        "system": system,
        "gliner_model": args.gliner_model if uses_gliner else None,
        "gliner_threshold": args.gliner_threshold if uses_gliner else None,
        "tags": ";".join(args.tags) if uses_gliner else None,
        "bert_model": bert_name if uses_bert else None,
        "bert_checkpoint": ckpt[1] if uses_bert else None,
        "step": ckpt[0] if uses_bert else None,
        "epoch": read_epoch(ckpt[2]) if uses_bert else None,
        "bert_threshold": args.bert_threshold if uses_bert else None,
        "bert_pretokenize": args.pretokenize if uses_bert else None,
        "merge": args.merge if system == "combined" else None,
        "n_docs": n_docs,
        "n_gold": n_gold,
        "n_pred": sum(len(v) for v in pred_spans.values()),
        **metrics,
        **(extra or {}),
    }
    return row


def print_system(label, row):
    print(f"  {label:<9} entity P/R/F1 = {row['entity_precision']:.4f} / {row['entity_recall']:.4f} / "
          f"{row['entity_f1']:.4f}   (TP {row['entity_tp']}, FP {row['entity_fp']}, FN {row['entity_fn']})")
    print(f"  {'':<9} token  P/R/F1 = {row['token_precision']:.4f} / {row['token_recall']:.4f} / "
          f"{row['token_f1']:.4f}   (TP {row['token_tp']}, FP {row['token_fp']}, FN {row['token_fn']})")
    if row.get("neg_docs"):
        print(f"  {'':<9} {row['neg_docs']} document(s) without gold instruments: "
              f"{row['neg_doc_fp']} span(s) predicted there (all false positives)")


def save_metrics(rows, output):
    df = pd.DataFrame(rows)
    keep_exact = {"gliner_threshold", "bert_threshold"}
    float_cols = [c for c in df.columns
                  if pd.api.types.is_float_dtype(df[c]) and c not in keep_exact]
    df[float_cols] = df[float_cols].round(6)
    for c in ("step", "n_pred_from_bert", "n_pred_from_gliner", "tp_from_bert", "tp_from_gliner",
              "neg_docs", "neg_doc_fp"):
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    ap.add_argument("--docs", default=DEFAULT_DOC_DF, help="CSV with res_id2, date_c, content")
    ap.add_argument("--gold", default=GOLD, help="CSV with res_id2, text, start, end")
    ap.add_argument("--gold-end", choices=["auto", "exclusive", "inclusive"], default="auto",
                    help="Meaning of the gold 'end' column: exclusive = text == content[start:end] "
                         "(Python slice), inclusive = content[start:end+1], auto = check each span "
                         "against the document text (default).")
    # GLiNER
    ap.add_argument("--gliner-model", "--model-name", dest="gliner_model", default=GLINER_MODEL_NAME,
                    help="GLiNER model on the Hugging Face hub / local path")
    ap.add_argument("--tags", nargs="+", default=TAGS, help="GLiNER labels; every one counts as INSTRUMENT")
    ap.add_argument("--gliner-threshold", type=float, default=GLINER_SCORE_THRESHOLD,
                    help="Keep GLiNER predictions with score >= threshold (default: %(default)s)")
    ap.add_argument("--chunk-words", type=int, default=300,
                    help="Max words per GLiNER call (GLiNER's limit is ~384). Default: %(default)s")
    ap.add_argument("--margin-words", type=int, default=50,
                    help="Context words added on each side of a window's core. Default: %(default)s")
    # fine-tuned BERT
    ap.add_argument("--bert-model", "--model-dir", dest="bert_model", required=True,
                    help="Fine-tuned model: either one checkpoint folder (contains config.json) or a folder "
                         "with checkpoint-<step> sub-folders, e.g. checkpoints-roberta-large")
    ap.add_argument("--bert-checkpoints", nargs="+", default=None,
                    help="Checkpoints to evaluate when --bert-model has checkpoint-<step> sub-folders "
                         "(names like checkpoint-8128, just 8128, or 'all'). Default: the latest one.")
    ap.add_argument("--bert-threshold", type=float, default=BERT_SCORE_THRESHOLD,
                    help="Keep BERT spans with score >= threshold (default: %(default)s)")
    ap.add_argument("--entity-type", default=DEFAULT_ENTITY_TYPE,
                    help="Entity type to keep from the BERT output (default: %(default)s)")
    ap.add_argument("--batch-size", type=int, default=16, help="BERT windows per forward pass")
    ap.add_argument("--max-length", type=int, default=512, help="BERT window size in tokens")
    ap.add_argument("--stride", type=int, default=128, help="Token overlap between BERT windows")
    ap.add_argument("--pretokenize", choices=["spacy", "none"], default="spacy",
                    help="spacy (default): split each document with spacy.blank('en') and tokenize the words "
                         "with is_split_into_words=True, one prediction per word from its first sub-token. "
                         "This is exactly the input format of token_train_noctua.py. "
                         "none: tokenize the raw text (previous behaviour, does NOT match training).")
    ap.add_argument("--first-subword", action="store_true",
                    help="Only used with --pretokenize none: use the first sub-token's label for the whole "
                         "word. --pretokenize spacy always works per word.")
    # combination / output
    ap.add_argument("--merge", choices=["bert-priority", "score", "union"], default="bert-priority",
                    help="How to combine the two prediction sets (see top of the file). Default: %(default)s")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--output", default=None, help="Metrics CSV. Default: %(default)s")
    ap.add_argument("--save-predictions", action="store_true",
                    help="Also write GLiNER_BERT_test_predictions-combined.csv next to --output "
                         "(bert_checkpoint, res_id2, start, end [exclusive], score, label, source, text, in_gold).")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    bert_name = bert_short_name(args.bert_model)
    bert_name = bert_name.replace("/", "-")
    
    if args.output is None:
        args.output = f"./GLiNER_{bert_name}_test_metrics-combined.csv"
    
    ckpts = resolve_bert_checkpoints(args.bert_model, args.bert_checkpoints)
    print(f"BERT model: {bert_name}  |  {len(ckpts)} checkpoint(s): " + ", ".join(c[1] for c in ckpts))
    print(f"Device: {device}  |  merge: {args.merge}  |  thresholds: GLiNER >= {args.gliner_threshold}, "
          f"BERT >= {args.bert_threshold}  |  BERT input: --pretokenize {args.pretokenize}")

    docs_text, gold_spans = load_data(args.docs, args.gold, args.gold_end)
    n_docs = len(docs_text)
    n_gold = sum(len(v) for v in gold_spans.values())

    # ---- 1. GLiNER, once ------------------------------------------------- #
    gliner_model = load_gliner(args.gliner_model, device)
    stats = {}
    gliner_ents = {}
    for rid, text in tqdm(docs_text.items(), desc="GLiNER", unit="doc"):
        ents = predict_long_text(gliner_model, text, args.tags, args.gliner_threshold,
                                 args.chunk_words, args.margin_words, stats)
        ents = [x for x in ents if x[2] >= args.gliner_threshold]
        gliner_ents[rid] = [(*x, "gliner") for x in flatten(ents)]
    if stats.get("misaligned"):
        print(f"[warn] {stats['misaligned']} GLiNER spans had offsets not matching their text "
              f"(repaired; {stats.get('dropped', 0)} dropped).")
    del gliner_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    by_label = Counter(x[3] for ents in gliner_ents.values() for x in ents)
    print(f"Predictions by GLiNER tag: {dict(by_label)}")

    g_spans = to_span_sets(gliner_ents)
    g_row = make_row("gliner", args, bert_name, None, n_docs, n_gold, g_spans,
                     evaluate(docs_text, gold_spans, g_spans))
    rows = [g_row]
    print("\nGLiNER alone")
    print_system("gliner", g_row)

    # ---- 2. fine-tuned BERT, per checkpoint, + merge ---------------------- #
    pred_rows = []
    for rank, ckpt in enumerate(ckpts, start=1):
        step, ckpt_name, ckpt_path = ckpt
        predictor = BertPredictor(ckpt_path, device, args.entity_type, args.max_length, args.stride,
                                  args.batch_size, args.first_subword, args.bert_threshold,
                                  args.pretokenize)
        if rank == 1:
            print(f"\n{bert_name} labels: {predictor.labels}  ->  keeping only '{args.entity_type}'")

        bert_ents = {}
        for rid, text in tqdm(docs_text.items(), desc=ckpt_name, unit="doc"):
            bert_ents[rid] = [(*x, "bert") for x in predictor.predict(text)]

        merged = {rid: merge_predictions(gliner_ents[rid], bert_ents[rid], args.merge)
                  for rid in docs_text}

        b_spans = to_span_sets(bert_ents)
        c_spans = to_span_sets(merged)
        b_row = make_row("bert", args, bert_name, ckpt, n_docs, n_gold, b_spans,
                         evaluate(docs_text, gold_spans, b_spans))
        c_row = make_row("combined", args, bert_name, ckpt, n_docs, n_gold, c_spans,
                         evaluate(docs_text, gold_spans, c_spans),
                         extra=source_breakdown(merged, gold_spans))
        rows += [b_row, c_row]

        print(f"\n{ckpt_name}")
        print_system(f"{bert_name}", b_row)
        print_system("combined", c_row)
        print(f"  combined = {c_row['n_pred_from_bert']} spans from {bert_name} (TP {c_row['tp_from_bert']}) "
              f"+ {c_row['n_pred_from_gliner']} added by GLiNER (TP {c_row['tp_from_gliner']})")

        if args.save_predictions:
            for rid, ents in merged.items():
                gold_doc = gold_spans.get(rid, set())
                for s, e, score, label, t, src in ents:
                    pred_rows.append({"bert_checkpoint": ckpt_name, "res_id2": rid, "start": s, "end": e,
                                      "score": round(score, 6), "label": label, "source": src,
                                      "text": t, "in_gold": (s, e) in gold_doc})
            out_dir = os.path.dirname(os.path.abspath(args.output))
            pred_path = os.path.join(out_dir, f"GLiNER_{bert_name}_test_predictions-combined.csv")
            pd.DataFrame(pred_rows, columns=["bert_checkpoint", "res_id2", "start", "end", "score",
                                             "label", "source", "text", "in_gold"]).to_csv(pred_path, index=False)

        # Save after every checkpoint so a crash does not lose finished work.
        df = save_metrics(rows, args.output)

        del predictor
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- 3. summary ------------------------------------------------------- #
    show = ["system", "bert_checkpoint", "entity_precision", "entity_recall", "entity_f1",
            "token_precision", "token_recall", "token_f1"]
    table = df[show].astype({"bert_checkpoint": object}).fillna("-")
    print("\n" + "=" * 100)
    print(f"GLiNER ({args.gliner_model}, >= {args.gliner_threshold})  +  "
          f"{bert_name} (>= {args.bert_threshold}), merge = {args.merge}")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    comb = df[df["system"] == "combined"]
    if len(comb) > 1:
        best = comb.loc[comb["entity_f1"].idxmax()]
        print(f"\nBest combined entity F1: {best['bert_checkpoint']} ({best['entity_f1']:.4f})")
        best = comb.loc[comb["token_f1"].idxmax()]
        print(f"Best combined token  F1: {best['bert_checkpoint']} ({best['token_f1']:.4f})")
    if args.save_predictions:
        print(f"Saved predictions: {os.path.join(os.path.dirname(os.path.abspath(args.output)), f'GLiNER_{bert_name}_test_predictions-combined.csv')}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
