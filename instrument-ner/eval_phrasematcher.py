#!/usr/bin/env python
"""
Evaluate the weak-labelling PhraseMatcher "baseline" against the manually annotated gold
standard (single unlabelled class: INSTRUMENT), overall, per document, and with
document-level bootstrap confidence intervals.

This is NOT a trained model. It is the exact deterministic rule that produced the weak
("silver") BIO labels the fine-tuned token classifiers were trained on in train_ner.py:
a lowercase alias -> official-title lookup built from the instrument reference tables
(Wikipedia, OHCHR, UNO, UNESCO, conventions/protocols, ad hoc), matched with a spaCy
PhraseMatcher(attr="LOWER") over spacy.blank("en"). Scoring it against the gold data tells
you how good the weak supervision itself is, independently of anything the NER models then
learned.

Pipeline
  1. load the gold file and the documents it refers to (IDENTICAL gold loader to
     eval_gliner.py / eval_ner_checkpoints.py / eval_gliner_BERT.py),
  2. build the golden_list and a PhraseMatcher over it -- same construction, same spaCy
     pipeline, same overlap rule as weak_label_corpus() in train_ner.py,
  3. run that PhraseMatcher over every gold document,
  4. compute TP / FP / FN and precision / recall / F1 PER DOCUMENT at
       * entity level (exact span match), and
       * token level (word units, partial credit),
  5. aggregate them into corpus-level (micro) and document-average (macro) scores,
  6. bootstrap over documents (resample documents with replacement, n_boot times) to get
     percentile confidence intervals for the micro and macro P/R/F1,
  7. write everything to --output-dir (default: <NER_DATA_DIR>/phrasematcher_eval):
       phrasematcher_metrics.csv       one row, overall (micro) counts and scores
       phrasematcher_per_doc.csv       one row per document: counts + P/R/F1
       phrasematcher_bootstrap_ci.csv  point estimate, bootstrap mean/std and CI for every
                                       level x aggregation x metric
       phrasematcher_predictions.csv   every predicted span with an in_gold flag
                                       (disable with --no-save-predictions)

Micro vs macro
--------------
* micro: TP/FP/FN are summed over the (resampled) documents, then P/R/F1 are computed once.
  This is the same number the original script reported; documents with many mentions weigh
  more.
* macro: P/R/F1 are computed per document, then averaged. Each document weighs the same.
  Per-document metrics that are undefined are NaN and are left out of the average:
    precision is undefined if the matcher predicted nothing in that document,
    recall    is undefined if the document has no gold instrument,
    F1 = 2TP / (2TP + FP + FN) is undefined only if the document has neither gold nor
    predictions (a correct "nothing here" document).

Bootstrap
---------
Documents are the resampling unit (mentions within a document are not independent). Each
replicate draws n_docs documents with replacement, and the micro and macro metrics are
recomputed on that sample. The CI is the percentile interval [alpha/2, 1-alpha/2] of the
n_boot replicates (default 10 000 replicates, 95 % CI, seed 42).

Why precision at all for a deterministic rule?
------------------------------------------------
The matcher always returns the same spans for the same document, but that doesn't make every
span it returns correct: an alias can be ambiguous (e.g. a short acronym used for something
else) and a match can stop short of a longer mention the matcher never learned as a full
alias. Recall additionally doubles as an upper bound on how much of the gold INSTRUMENT
signal the weak BIO labels could have carried into training.

Example
-------
    python eval_phrasematcher.py
    python eval_phrasematcher.py --n-boot 2000 --alpha 0.1 --seed 1
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional
    def tqdm(iterable, **kwargs):
        return iterable

NER_DATA_DIR = "/home/user/branes/NER-data"
DEFAULT_DOC_DF = f"{NER_DATA_DIR}/ga_resolutions_1946_2025.csv"
GOLD = f"{NER_DATA_DIR}/NER_golden_truth_10.csv"
DEFAULT_OUTPUT_DIR = f"{NER_DATA_DIR}/phrasematcher_eval"

# Instrument reference tables concatenated into df_instrument, exactly as in train_ner.py's
# __main__.
DEFAULT_WIKI = f"{NER_DATA_DIR}/wiki-treaties_formatted.csv"
DEFAULT_OHCHR = f"{NER_DATA_DIR}/ohchr_instruments_detailed-instit.csv"
DEFAULT_UNO = f"{NER_DATA_DIR}/UNO-Treaties.csv"
DEFAULT_UNESCO = f"{NER_DATA_DIR}/UNESCO_legal_instruments_detail.csv"
DEFAULT_CONV_PROT_REC = f"{NER_DATA_DIR}/conventions-protocols-recommendations.csv"
DEFAULT_AD_HOC = f"{NER_DATA_DIR}/ad_hoc_instruments.csv"

# Number of texts per nlp.pipe() batch (same constant as train_ner.py).
BATCH_SIZE = 64

# Bootstrap defaults
N_BOOT = 10_000
ALPHA = 0.05
SEED = 42
BOOT_CHUNK = 500  # replicates processed at once (bounds memory: chunk x n_docs x 6 floats)

ENTITY_TYPE = "INSTRUMENT"
LEVELS = ("entity", "token")
METRICS = ("precision", "recall", "f1")
WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)  # word units for the token-level metric
ACRONYM_RE = re.compile(r"^[A-Z]{2,6}\.?$")  # crude "looks like a bare acronym" heuristic


# --------------------------------------------------------------------------- #
# Gold loading -- identical to eval_gliner.py / eval_ner_checkpoints.py / eval_gliner_BERT.py
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

    A res_id2 duplicated in `docs_path` (more than one text version) is resolved by keeping
    the version that reproduces the most gold spans exactly, ties broken by whichever version
    was seen first.
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
# Lexicon + PhraseMatcher -- identical construction to train_ner.py
# --------------------------------------------------------------------------- #
def build_golden_list(df_instrument: pd.DataFrame) -> list[str]:
    """
    Build the backbone of the weak labelling (copied verbatim from train_ner.py).

    Official titles and aliases receive the same treatment: either can be used in training,
    so both go into one flat list the PhraseMatcher looks for.
    """
    if df_instrument.empty:
        return []

    golden_list = []

    titles = [
        t.strip()
        for t in df_instrument["title"].dropna()
        if t.strip()
    ]

    aliases = []
    for value in df_instrument["alternative_name"].dropna():
        aliases.extend(
            p.strip()
            for p in value.split(";")
            if p.strip()
        )

    golden_list.extend(titles)
    golden_list.extend(aliases)

    return golden_list


def build_instrument_table(args) -> pd.DataFrame:
    """Reload + concatenate the six instrument reference tables, exactly as train_ner.py's
    __main__ does, so the golden_list here is the same one that produced the training labels.
    """
    cols = ["title", "year", "alternative_name"]

    df_wiki = pd.read_csv(args.wiki)
    df_wiki = df_wiki.rename(columns={"cleaned_title": "title", "cleaned_note": "alternative_name"})[cols]

    df_ohchr = pd.read_csv(args.ohchr)
    df_ohchr["year"] = pd.to_datetime(df_ohchr["adoption_date"], format="%d %B %Y").dt.year
    df_ohchr = df_ohchr[cols]

    df_uno = pd.read_csv(args.uno)
    df_uno["year"] = pd.to_datetime(df_uno["date"], format="%d %B %Y").dt.year
    df_uno["alternative_name"] = ""
    df_uno = df_uno[cols]

    df_unesco = pd.read_csv(args.unesco)
    df_unesco["year"] = pd.to_datetime(df_unesco["date"], format="%d %B %Y").dt.year
    df_unesco["alternative_name"] = ""
    df_unesco = df_unesco[cols]

    df_conv_prot_rec = pd.read_csv(args.conv_prot_rec)
    df_conv_prot_rec["alternative_name"] = ""
    df_conv_prot_rec = df_conv_prot_rec[cols]

    ad_hoc = pd.read_csv(args.ad_hoc)
    ad_hoc = ad_hoc[cols]

    return pd.concat(
        [df_wiki, df_ohchr, df_uno, df_unesco, df_conv_prot_rec, ad_hoc],
        ignore_index=True,
    )


def build_phrase_matcher(golden_list: list[str]):
    """spacy.blank("en") + PhraseMatcher(attr="LOWER"), same as weak_label_corpus()."""
    nlp = spacy.blank("en")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [
        nlp.make_doc(name)
        for name in golden_list
        if isinstance(name, str) and name.strip()
    ]
    matcher.add(ENTITY_TYPE, patterns)
    return nlp, matcher, len(patterns)


def predict_phrasematcher(nlp, matcher, docs_text: dict, batch_size: int = BATCH_SIZE):
    """Run the PhraseMatcher over every document in docs_text.

    Overlap handling is IDENTICAL to weak_label_corpus() in train_ner.py: on overlap the
    longest match wins (ties broken by scan order), so a short alias contained inside a longer
    one never fires separately. Returns {res_id2: [(start, end_exclusive, matched_text), ...]}.
    """
    preds = {}
    rids = list(docs_text)
    texts = (docs_text[rid] for rid in rids)
    for rid, doc in tqdm(zip(rids, nlp.pipe(texts, batch_size=batch_size)),
                         total=len(rids), desc="PhraseMatcher", unit="doc"):
        matches = matcher(doc)
        if not matches:
            preds[rid] = []
            continue

        matches_sorted = sorted(
            matches,
            key=lambda m: (doc[m[1]:m[2]].end_char - doc[m[1]:m[2]].start_char),
            reverse=True,
        )

        spans, seen = [], []
        for _, start, end in matches_sorted:
            span = doc[start:end]
            if any(span.start_char < seen_end and span.end_char > seen_start
                   for seen_start, seen_end in seen):
                continue
            seen.append((span.start_char, span.end_char))
            spans.append((span.start_char, span.end_char, span.text))

        preds[rid] = sorted(spans, key=lambda s: s[0])
    return preds


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def prf(tp: int, fp: int, fn: int):
    """Corpus-level P/R/F1 (undefined -> 0.0, same convention as the other eval scripts)."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _safe_div(num, den, fill):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, fill, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out


def prf_arrays(tp, fp, fn, fill):
    """Vectorised P/R/F1. `fill` is the value used when a metric is undefined
    (np.nan for per-document metrics, 0.0 for corpus-level/micro metrics).
    F1 = 2TP / (2TP + FP + FN), identical to 2PR/(P+R) whenever both are defined."""
    p = _safe_div(tp, tp + fp, fill)
    r = _safe_div(tp, tp + fn, fill)
    f = _safe_div(2 * tp, 2 * tp + fp + fn, fill)
    return p, r, f


def _covered(n_chars, spans, word_starts, word_ends):
    """Boolean array: does each word overlap any of the spans?"""
    mask = np.zeros(n_chars + 1, dtype=np.int32)
    for s, e in spans:
        mask[max(s, 0):e] = 1
    cum = np.concatenate(([0], np.cumsum(mask)))
    return (cum[word_ends] - cum[word_starts]) > 0


def evaluate_per_doc(docs_text, gold_spans, pred_spans, pred_by_doc=None) -> pd.DataFrame:
    """One row per document: entity- and token-level TP/FP/FN and P/R/F1 (NaN if undefined)."""
    rows = []
    for rid, text in docs_text.items():
        g = gold_spans.get(rid, set())
        p = pred_spans.get(rid, set())
        row = {
            "res_id2": rid,
            "n_chars": len(text),
            "n_gold": len(g),
            "n_pred": len(p),
            "negative_doc": not g,
            # entity level, exact match
            "entity_tp": len(g & p),
            "entity_fp": len(p - g),
            "entity_fn": len(g - p),
            "token_tp": 0, "token_fp": 0, "token_fn": 0,
        }
        # token level (word units), partial credit
        words = [m.span() for m in WORD_RE.finditer(text)]
        if words:
            ws = np.array([w[0] for w in words], dtype=np.int64)
            we = np.array([w[1] for w in words], dtype=np.int64)
            gc_ = _covered(len(text), g, ws, we)
            pc_ = _covered(len(text), p, ws, we)
            row["token_tp"] = int((gc_ & pc_).sum())
            row["token_fp"] = int((~gc_ & pc_).sum())
            row["token_fn"] = int((gc_ & ~pc_).sum())
        if pred_by_doc is not None:
            row["fp_acronym_like"] = sum(
                1 for s, e, t in pred_by_doc.get(rid, [])
                if (s, e) not in g and ACRONYM_RE.match(t.strip()))
        rows.append(row)

    df = pd.DataFrame(rows)
    for lvl in LEVELS:
        p, r, f = prf_arrays(df[f"{lvl}_tp"].to_numpy(), df[f"{lvl}_fp"].to_numpy(),
                             df[f"{lvl}_fn"].to_numpy(), fill=np.nan)
        df[f"{lvl}_precision"], df[f"{lvl}_recall"], df[f"{lvl}_f1"] = p, r, f
    return df


def aggregate(per_doc: pd.DataFrame) -> dict:
    """Corpus-level (micro) metrics from the per-document counts -- identical to the numbers
    the original single-pass evaluate() produced."""
    out = {}
    for lvl in LEVELS:
        tp, fp, fn = (int(per_doc[f"{lvl}_{k}"].sum()) for k in ("tp", "fp", "fn"))
        p, r, f = prf(tp, fp, fn)
        out.update({f"{lvl}_tp": tp, f"{lvl}_fp": fp, f"{lvl}_fn": fn,
                    f"{lvl}_precision": p, f"{lvl}_recall": r, f"{lvl}_f1": f})
    for lvl in LEVELS:  # document-averaged (macro) metrics, undefined documents left out
        for m in METRICS:
            col = per_doc[f"{lvl}_{m}"]
            out[f"{lvl}_macro_{m}"] = float(col.mean()) if col.notna().any() else float("nan")
    neg = per_doc["negative_doc"]
    out["neg_docs"] = int(neg.sum())
    out["neg_doc_fp"] = int(per_doc.loc[neg, "n_pred"].sum())
    return out


def print_metrics(row):
    for lvl in LEVELS:
        print(f"  {lvl:<6} micro P/R/F1 = {row[f'{lvl}_precision']:.4f} / {row[f'{lvl}_recall']:.4f} / "
              f"{row[f'{lvl}_f1']:.4f}   (TP {row[f'{lvl}_tp']}, FP {row[f'{lvl}_fp']}, "
              f"FN {row[f'{lvl}_fn']})")
        print(f"  {lvl:<6} macro P/R/F1 = {row[f'{lvl}_macro_precision']:.4f} / "
              f"{row[f'{lvl}_macro_recall']:.4f} / {row[f'{lvl}_macro_f1']:.4f}")
    if row.get("neg_docs"):
        print(f"  {row['neg_docs']} document(s) without gold instruments: "
              f"{row['neg_doc_fp']} span(s) predicted there (all false positives)")


# --------------------------------------------------------------------------- #
# Bootstrap over documents
# --------------------------------------------------------------------------- #
def bootstrap_ci(per_doc: pd.DataFrame, n_boot: int = N_BOOT, alpha: float = ALPHA,
                 seed: int = SEED, chunk: int = BOOT_CHUNK):
    """Resample documents with replacement and recompute micro and macro P/R/F1.

    Returns (report_df, samples_df):
      report_df  : one row per level x aggregation x metric with the point estimate (full
                   data), bootstrap mean / std, and the percentile CI.
      samples_df : the n_boot replicate values (one column per level/aggregation/metric).
    """
    n = len(per_doc)
    if n == 0:
        raise ValueError("No documents to bootstrap.")
    count_cols = [f"{lvl}_{k}" for lvl in LEVELS for k in ("tp", "fp", "fn")]
    metric_cols = [f"{lvl}_{m}" for lvl in LEVELS for m in METRICS]
    C = per_doc[count_cols].to_numpy(dtype=float)   # (n, 6) counts
    M = per_doc[metric_cols].to_numpy(dtype=float)  # (n, 6) per-doc metrics, NaN if undefined

    rng = np.random.default_rng(seed)
    micro = np.empty((n_boot, len(metric_cols)))
    macro = np.empty((n_boot, len(metric_cols)))
    for start in tqdm(range(0, n_boot, chunk), desc="Bootstrap", unit="chunk"):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        S = C[idx].sum(axis=1)  # (b, 6) summed counts per replicate
        for li in range(len(LEVELS)):
            tp, fp, fn = S[:, 3 * li], S[:, 3 * li + 1], S[:, 3 * li + 2]
            # undefined in a replicate (e.g. only gold-empty docs drawn) -> NaN, excluded from
            # the CI and reported via n_boot_valid, rather than counted as a 0
            p, r, f = prf_arrays(tp, fp, fn, fill=np.nan)
            micro[start:start + b, 3 * li:3 * li + 3] = np.column_stack([p, r, f])
        with warnings.catch_warnings():  # all-NaN replicate (e.g. only negative docs drawn)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            macro[start:start + b] = np.nanmean(M[idx], axis=1)

    full = aggregate(per_doc)
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rows, samples = [], {}
    for agg, arr in (("micro", micro), ("macro", macro)):
        for j, col in enumerate(metric_cols):
            lvl, m = col.split("_", 1)
            vals = arr[:, j]
            valid = vals[~np.isnan(vals)]
            point = full[f"{lvl}_{m}"] if agg == "micro" else full[f"{lvl}_macro_{m}"]
            n_defined = int(per_doc[col].notna().sum()) if agg == "macro" else n
            rows.append({
                "system": "phrasematcher",
                "level": lvl,
                "aggregation": agg,
                "metric": m,
                "point_estimate": point,
                "boot_mean": float(valid.mean()) if valid.size else np.nan,
                "boot_std": float(valid.std(ddof=1)) if valid.size > 1 else np.nan,
                "ci_low": float(np.percentile(valid, lo_q)) if valid.size else np.nan,
                "ci_high": float(np.percentile(valid, hi_q)) if valid.size else np.nan,
                "ci_level": 1 - alpha,
                "n_boot": n_boot,
                "n_boot_valid": int(valid.size),
                "n_docs": n,
                "n_docs_defined": n_defined,
                "seed": seed,
            })
            samples[f"{lvl}_{agg}_{m}"] = vals
    return pd.DataFrame(rows), pd.DataFrame(samples)


def print_bootstrap(report: pd.DataFrame):
    ci = int(round(report["ci_level"].iloc[0] * 100))
    print(f"\nBootstrap over documents ({report['n_boot'].iloc[0]} replicates, {ci}% percentile CI)")
    for (lvl, agg), grp in report.groupby(["level", "aggregation"], sort=False):
        parts = [f"{r.metric[0].upper()} {r.point_estimate:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]"
                 for r in grp.itertuples()]
        print(f"  {lvl:<6} {agg}:  " + "   ".join(parts))


# --------------------------------------------------------------------------- #
# False-positive diagnostics
# --------------------------------------------------------------------------- #
def false_positive_breakdown(docs_text, gold_spans, pred_by_doc, top_n=15):
    """Which matched strings fire most often where gold disagrees, and how many of those look
    like bare acronyms (a likely source of ambiguous short aliases)."""
    fp_counter = Counter()
    n_fp = 0
    n_fp_acronym = 0
    for rid, spans in pred_by_doc.items():
        g = gold_spans.get(rid, set())
        for s, e, t in spans:
            if (s, e) not in g:
                n_fp += 1
                fp_counter[t] += 1
                if ACRONYM_RE.match(t.strip()):
                    n_fp_acronym += 1
    print(f"\nFalse-positive diagnostics: {n_fp} false-positive span(s), "
          f"{n_fp_acronym} ({n_fp_acronym / n_fp:.1%} of them) look like a bare acronym "
          f"(2-6 upper-case letters, optional trailing '.')." if n_fp else
          "\nFalse-positive diagnostics: no false positives.")
    if fp_counter:
        print(f"  Most frequent false-positive strings (top {top_n}):")
        for text, count in fp_counter.most_common(top_n):
            flag = "  [acronym-like]" if ACRONYM_RE.match(text.strip()) else ""
            print(f"    {count:>4}x  {text!r}{flag}")
    return n_fp, n_fp_acronym, fp_counter


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    ap.add_argument("--docs", default=DEFAULT_DOC_DF, help="CSV with res_id2, date_c, content")
    ap.add_argument("--gold", default=GOLD, help="CSV with res_id2, text, start, end")
    ap.add_argument("--gold-end", choices=["auto", "exclusive", "inclusive"], default="auto",
                    help="Meaning of the gold 'end' column: exclusive = text == content[start:end] "
                         "(Python slice), inclusive = content[start:end+1], auto = check each span "
                         "against the document text (default).")
    # instrument reference tables (same six sources + columns as train_ner.py)
    ap.add_argument("--wiki", default=DEFAULT_WIKI)
    ap.add_argument("--ohchr", default=DEFAULT_OHCHR)
    ap.add_argument("--uno", default=DEFAULT_UNO)
    ap.add_argument("--unesco", default=DEFAULT_UNESCO)
    ap.add_argument("--conv-prot-rec", default=DEFAULT_CONV_PROT_REC)
    ap.add_argument("--ad-hoc", default=DEFAULT_AD_HOC)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="nlp.pipe() batch size")
    # bootstrap
    ap.add_argument("--n-boot", type=int, default=N_BOOT,
                    help="Number of bootstrap replicates (documents resampled with replacement). "
                         "Default: %(default)s")
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help="1 - confidence level of the percentile CI. Default: %(default)s (95%% CI)")
    ap.add_argument("--seed", type=int, default=SEED, help="Bootstrap RNG seed. Default: %(default)s")
    # output
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help="Directory all outputs are written to. Created if missing. "
                         "Default: %(default)s")
    ap.add_argument("--no-save-predictions", action="store_true",
                    help="Do NOT write phrasematcher_predictions.csv (written by default).")
    ap.add_argument("--save-bootstrap-samples", action="store_true",
                    help="Also write every bootstrap replicate to phrasematcher_bootstrap_samples.csv.")
    ap.add_argument("--fp-top-n", type=int, default=15,
                    help="How many of the most frequent false-positive strings to print/save. "
                         "Default: %(default)s")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    docs_text, gold_spans = load_data(args.docs, args.gold, args.gold_end)
    n_docs = len(docs_text)
    n_gold = sum(len(v) for v in gold_spans.values())

    df_instrument = build_instrument_table(args)
    golden_list = build_golden_list(df_instrument)
    nlp, matcher, n_patterns = build_phrase_matcher(golden_list)
    print(f"Lexicon: {len(golden_list)} title(s)/alias(es) from {len(df_instrument)} instrument "
          f"row(s)  ->  {n_patterns} PhraseMatcher pattern(s).")

    pred_by_doc = predict_phrasematcher(nlp, matcher, docs_text, args.batch_size)
    pred_spans = {rid: {(s, e) for s, e, _ in spans} for rid, spans in pred_by_doc.items()}

    # ---- per-document metrics ------------------------------------------------
    per_doc = evaluate_per_doc(docs_text, gold_spans, pred_spans, pred_by_doc)
    metrics = aggregate(per_doc)
    print("\nPhraseMatcher baseline")
    print_metrics(metrics)
    n_fp, n_fp_acronym, fp_counter = false_positive_breakdown(
        docs_text, gold_spans, pred_by_doc, args.fp_top_n)

    # ---- bootstrap -----------------------------------------------------------
    boot_report, boot_samples = bootstrap_ci(per_doc, args.n_boot, args.alpha, args.seed)
    print_bootstrap(boot_report)

    # ---- save ----------------------------------------------------------------
    def _round(df):
        float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
        df[float_cols] = df[float_cols].round(6)
        return df

    saved = []

    row = {
        "system": "phrasematcher",
        "n_patterns": n_patterns,
        "n_docs": n_docs,
        "n_gold": n_gold,
        "n_pred": sum(len(v) for v in pred_spans.values()),
        **metrics,
        "fp_acronym_like": n_fp_acronym,
    }
    # add the CI bounds to the one-row summary as well
    for r in boot_report.itertuples():
        key = f"{r.level}_{r.aggregation}_{r.metric}"
        row[f"{key}_ci_low"] = r.ci_low
        row[f"{key}_ci_high"] = r.ci_high
    row.update({"n_boot": args.n_boot, "ci_level": 1 - args.alpha, "seed": args.seed})
    path = os.path.join(args.output_dir, "phrasematcher_metrics.csv")
    _round(pd.DataFrame([row])).to_csv(path, index=False)
    saved.append(path)

    path = os.path.join(args.output_dir, "phrasematcher_per_doc.csv")
    _round(per_doc.copy()).to_csv(path, index=False)
    saved.append(path)

    path = os.path.join(args.output_dir, "phrasematcher_bootstrap_ci.csv")
    _round(boot_report.copy()).to_csv(path, index=False)
    saved.append(path)

    if args.save_bootstrap_samples:
        path = os.path.join(args.output_dir, "phrasematcher_bootstrap_samples.csv")
        _round(boot_samples.copy()).to_csv(path, index_label="replicate")
        saved.append(path)

    if fp_counter:
        path = os.path.join(args.output_dir, "phrasematcher_fp_strings.csv")
        pd.DataFrame(
            [{"text": t, "count": c, "acronym_like": bool(ACRONYM_RE.match(t.strip()))}
             for t, c in fp_counter.most_common()]
        ).to_csv(path, index=False)
        saved.append(path)

    if not args.no_save_predictions:
        rows = []
        for rid, spans in pred_by_doc.items():
            g = gold_spans.get(rid, set())
            for s, e, t in spans:
                rows.append({"res_id2": rid, "start": s, "end": e, "label": ENTITY_TYPE,
                             "text": t, "in_gold": (s, e) in g,
                             "acronym_like": bool(ACRONYM_RE.match(t.strip()))})
        path = os.path.join(args.output_dir, "phrasematcher_predictions.csv")
        pd.DataFrame(rows, columns=["res_id2", "start", "end", "label", "text", "in_gold",
                                    "acronym_like"]).to_csv(path, index=False)
        saved.append(path)

    print("\nSaved:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
