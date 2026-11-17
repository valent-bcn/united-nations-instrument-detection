"""
Evaluation helpers for the instrument NER pipeline:
    * compute_metrics: token-classification seqeval P/R/F1 for the HF Trainer;
    * MetricsCSVLogger: per-epoch metrics to CSV;
    * mention_level_report: recall on seen / unseen names, strict precision and
      a CSV of predicted spans that are not in the (weak) gold.
Reusable for the hand-annotated test set.
"""

import csv
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import TrainerCallback

from ner_data import ID2LABEL, IGNORE_TAG, _norm


def compute_metrics(eval_pred, id2label=ID2LABEL):
    """Compute seqeval precision/recall/F1 from an HF EvalPrediction."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=2)

    true_labels, true_preds = [], []
    for pred_row, label_row in zip(preds, labels):
        seq_labels, seq_preds = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:  # padding / continuation subwords / IGN words
                continue
            seq_labels.append(id2label[l])
            seq_preds.append(id2label[p])
        true_labels.append(seq_labels)
        true_preds.append(seq_preds)

    print(classification_report(true_labels, true_preds))

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }


class MetricsCSVLogger(TrainerCallback):
    """
    Appends the eval metrics dict to <output_dir>/metrics_per_epoch.csv
    every time evaluation runs (end of each epoch).
    """
    def __init__(self, output_dir, filename="metrics_per_epoch.csv"):
        os.makedirs(output_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, filename)
        self._header_written = False
        self._last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._last_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        row = {"epoch": state.epoch, "train_loss": self._last_train_loss,
               **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}}
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)



# ---------------------------------------------------------------------------
# 5. MENTION-LEVEL REPORT (seen vs unseen names, unmatched predictions)
# ---------------------------------------------------------------------------

def bio_spans(tags):
    """[start, end) word spans from BIO tags; an orphan I- starts a new span."""
    spans, start = [], None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "B-INSTRUMENT" or (t == "I-INSTRUMENT" and start is None):
            if start is not None:
                spans.append((start, i))
            start = i
        elif t != "I-INSTRUMENT":
            if start is not None:
                spans.append((start, i))
            start = None
    return spans


def mention_level_report(trainer, eval_ds, eval_ex_raw, train_mentions, output_dir):
    """
    Exact-match, mention-level evaluation on the eval docs.

    * recall split by whether the gold name (normalised) occurs in the
      training years -> recall_unseen is the generalisation number;
    * strict precision against the weak gold;
    * every predicted span not in the weak gold is written to
      unmatched_predictions.csv with a category:
        boundary  - overlaps a gold mention (wrong boundaries)
        candidate - falls in an unlabelled instrument-looking phrase (IGN):
                    very likely a real instrument missing from the lexicon
        other     - anything else (probable false positive, check by hand)
    `eval_ex_raw` must contain the IGN tags (same tokens as eval_ds).
    """
    train_norm = {_norm(m) for m in train_mentions}
    out = trainer.predict(eval_ds)
    preds = np.argmax(out.predictions, axis=2)

    word_preds = [["O"] * len(ex["tokens"]) for ex in eval_ex_raw]
    owned = [set() for _ in eval_ex_raw]
    for row, (k, pw) in enumerate(zip(eval_ds["example_idx"], eval_ds["pred_word_ids"])):
        for pos, wid in enumerate(pw):
            if wid >= 0:
                word_preds[k][wid] = ID2LABEL[int(preds[row][pos])]
                owned[k].add(wid)

    stats = Counter()
    unmatched = []
    for k, ex in enumerate(eval_ex_raw):
        toks, gold = ex["tokens"], ex["ner_tags"]
        gold_sp = [sp for sp in bio_spans(gold) if all(w in owned[k] for w in range(*sp))]
        pred_sp = bio_spans(word_preds[k])
        gold_set, pred_set = set(gold_sp), set(pred_sp)
        gold_words = {w for a, b in gold_sp for w in range(a, b)}
        ign_words = {i for i, t in enumerate(gold) if t == IGNORE_TAG}

        for a, b in gold_sp:
            key = "seen" if _norm("".join(toks[a:b])) in train_norm else "unseen"
            stats[f"gold_{key}"] += 1
            stats[f"hit_{key}"] += (a, b) in pred_set

        for a, b in pred_sp:
            stats["pred"] += 1
            if (a, b) in gold_set:
                stats["pred_correct"] += 1
                continue
            words = set(range(a, b))
            if words & gold_words:
                cat = "boundary"
            elif words & ign_words:
                cat = "candidate"
            else:
                cat = "other"
            stats[f"pred_{cat}"] += 1
            unmatched.append({
                "doc_id": ex["doc_id"],
                "category": cat,
                "text": " ".join(t for t in toks[a:b] if t.strip()),
                "context": " ".join(t for t in toks[max(0, a - 12):b + 12] if t.strip()),
            })

    def ratio(x, y):
        return round(x / y, 4) if y else None

    summary = {
        "recall_seen": ratio(stats["hit_seen"], stats["gold_seen"]),
        "recall_unseen": ratio(stats["hit_unseen"], stats["gold_unseen"]),
        "precision_strict": ratio(stats["pred_correct"], stats["pred"]),
        **{k: int(v) for k, v in stats.items()},
    }
    print("mention-level eval:", json.dumps(summary, indent=2))
    with open(os.path.join(output_dir, "mention_eval.json"), "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(unmatched, columns=["doc_id", "category", "text", "context"]) \
        .to_csv(os.path.join(output_dir, "unmatched_predictions.csv"), index=False)
    top = Counter(r["text"] for r in unmatched if r["category"] == "candidate").most_common(25)
    if top:
        print("most frequent predicted names missing from the lexicon:")
        for text, n in top:
            print(f"  {n:4d}  {text}")
    return summary
