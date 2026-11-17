"""
Data side of the instrument NER pipeline: everything that runs before the model.

    0. Loading the resolutions and the instrument lexicon.
    1. Name list (official titles + aliases).
    2. Weak labelling with a spaCy PhraseMatcher (+ cached partition on disk).
    2b. CandidateFinder: instrument-looking phrases the lexicon missed (-> IGN).
    3. Word-level BIO tags.
    3b. Mention replacement (anti-memorisation augmentation).

Imported by the training script and by any evaluation script, so it must not
import torch / transformers.
"""

import json
import os
import random
import re
from collections import Counter

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.blank("en")

# Number of texts per nlp.pipe() batch.
BATCH_SIZE = 64

LABEL_LIST = ["O", "B-INSTRUMENT", "I-INSTRUMENT"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

# Word-level tag for "unknown": an instrument-looking phrase the lexicon did not
# match. Aligned to -100, so it is ignored by the loss and by seqeval.
IGNORE_TAG = "IGN"


# ---------------------------------------------------------------------------
# 0. LOADING
# ---------------------------------------------------------------------------

def load_resolutions(path):
    """Resolutions CSV -> (df with columns id, content, year; {doc_id: year})."""
    df_res = pd.read_csv(path).rename(columns={"res_id2": "id"})
    df_res["year"] = pd.to_datetime(df_res["date_c"], format="%d %B %Y", errors="coerce").dt.year
    return df_res, dict(zip(df_res["id"], df_res["year"]))


def load_instruments(path):
    """
    Instrument lexicon with columns title, year, alternative_name
    (aliases separated by ';'). `path` is one CSV, or a directory whose CSVs
    all have those columns and are concatenated.
    """
    cols = ["title", "year", "alternative_name"]
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv"))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        df = pd.read_csv(path)
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df = df[cols].copy()
    df["alternative_name"] = df["alternative_name"].fillna("")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    print(f"instrument lexicon: {len(df)} rows from {path}")
    return df


def split_mentions(known_docs, doc_years, eval_from_year):
    """Weak-label mention strings of the train years and of everything else."""
    def is_train(doc_id):
        y = doc_years.get(doc_id)
        return y is not None and not pd.isna(y) and y < eval_from_year

    train = [s[2] for d in known_docs if is_train(d["doc_id"]) for s in d["spans"]]
    # everything that is not train counts as "eval side" for leak prevention
    rest = [s[2] for d in known_docs if not is_train(d["doc_id"]) for s in d["spans"]]
    return train, rest



# ---------------------------------------------------------------------------
# 1. LIST GENERATION ACROSS ALL ENTITIES
# ---------------------------------------------------------------------------
def _row_names(title, alternative_name) -> list[str]:
    """Official title + ';'-separated aliases of one lexicon row."""
    names = []
    if isinstance(title, str) and title.strip():
        names.append(title.strip())
    if isinstance(alternative_name, str):
        names.extend(p.strip() for p in alternative_name.split(";") if p.strip())
    return names


def build_golden_list(df_instrument: pd.DataFrame) -> list[str]:
    """
    Build the backbone of the weak labelling.

    Official titles and aliases receive the same treatment. Either the
    official name or an alias can be used in training.

    Parameters
    ----------
    df_instrument:
        DataFrame containing at least ["title", "alternative_name", ...].

    Returns
    -------
    list[str]
        Names (official titles and alternative names) that the
        PhraseMatcher should detect.
    """
    if df_instrument.empty:
        return []

    titles = [t.strip() for t in df_instrument["title"].dropna() if t.strip()]
    aliases = []
    for value in df_instrument["alternative_name"].dropna():
        aliases.extend(p.strip() for p in value.split(";") if p.strip())

    return titles + aliases


# ---------------------------------------------------------------------------
# 2. WEAK LABELING WITH PhraseMatcher
# ---------------------------------------------------------------------------

def weak_label_corpus(df_res: pd.DataFrame, golden_list: list[str], batch_size: int = BATCH_SIZE):
    """
    Weakly label documents using a list of instrument names.

    Returns
    -------
    known_docs:
        [{"doc_id": ..., "text": ..., "spans": [(start_char, end_char, matched_text)]}]
    unknown_docs:
        [{"doc_id": ..., "text": ...}]
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    patterns = [
        nlp.make_doc(name)
        for name in golden_list
        if isinstance(name, str) and name.strip()
    ]
    matcher.add("INSTRUMENT", patterns)

    known_docs = []
    unknown_docs = []

    rows = list(df_res[["id", "content"]].itertuples(index=False, name=None))
    texts = (text for _, text in rows)

    for (doc_id, text), doc in zip(rows, nlp.pipe(texts, batch_size=batch_size)):
        matches = matcher(doc)

        if not matches:
            unknown_docs.append({"doc_id": doc_id, "text": text})
            continue

        # Longest matches first so that, in case of overlap, the longer match is kept.
        matches_sorted = sorted(
            matches,
            key=lambda match: (
                doc[match[1]:match[2]].end_char - doc[match[1]:match[2]].start_char
            ),
            reverse=True,
        )

        spans = []
        seen = []
        for _, start, end in matches_sorted:
            span = doc[start:end]
            if any(
                span.start_char < seen_end and span.end_char > seen_start
                for seen_start, seen_end in seen
            ):
                continue
            seen.append((span.start_char, span.end_char))
            spans.append((span.start_char, span.end_char, span.text))

        known_docs.append({"doc_id": doc_id, "text": text, "spans": spans})

    return known_docs, unknown_docs


def save_partition(known_docs, unknown_docs, out_dir="./data/partition"):
    os.makedirs(out_dir, exist_ok=True)

    known_rows = [
        {"doc_id": d["doc_id"], "text": d["text"], "spans": json.dumps(d["spans"])}
        for d in known_docs
    ]
    known_df = pd.DataFrame(known_rows, columns=["doc_id", "text", "spans"])
    known_path = os.path.join(out_dir, "partition_known.csv")
    known_df.to_csv(known_path, index=False)

    unknown_rows = [{"doc_id": d["doc_id"], "text": d["text"]} for d in unknown_docs]
    unknown_df = pd.DataFrame(unknown_rows, columns=["doc_id", "text"])
    unknown_path = os.path.join(out_dir, "partition_unknown.csv")
    unknown_df.to_csv(unknown_path, index=False)

    return known_path, unknown_path


def partition_exists(out_dir="./data/partition"):
    known_path = os.path.join(out_dir, "partition_known.csv")
    unknown_path = os.path.join(out_dir, "partition_unknown.csv")
    return os.path.exists(known_path) and os.path.exists(unknown_path)


def load_partition(out_dir="./data/partition"):
    known_path = os.path.join(out_dir, "partition_known.csv")
    unknown_path = os.path.join(out_dir, "partition_unknown.csv")

    known_df = pd.read_csv(known_path)
    unknown_df = pd.read_csv(unknown_path)

    known_docs = [
        {
            "doc_id": row["doc_id"],
            "text": row["text"],
            "spans": [tuple(span) for span in json.loads(row["spans"])],
        }
        for _, row in known_df.iterrows()
    ]
    unknown_docs = [
        {"doc_id": row["doc_id"], "text": row["text"]}
        for _, row in unknown_df.iterrows()
    ]
    return known_docs, unknown_docs


def get_or_build_partition(
    df_res,
    aliases,
    out_dir="/home/user/branes/NER-data/data/partition/"):
    if partition_exists(out_dir):
        print(f"Found existing partition in {out_dir}, loading from disk...")
        known_docs, unknown_docs = load_partition(out_dir)
    else:
        print("No existing partition found, running weak_label_corpus...")
        known_docs, unknown_docs = weak_label_corpus(df_res, aliases)
        save_partition(known_docs, unknown_docs, out_dir)
    return known_docs, unknown_docs


# ---------------------------------------------------------------------------
# 2b. INSTRUMENT-LOOKING PHRASES (likely false negatives of the lexicon)
# ---------------------------------------------------------------------------
# Case-insensitive, trigger-anchored detector. It works on spaCy tokens and
# does NOT require capitalisation (the pre-2015 text is ~80% lowercase), and
# the trigger can be anywhere in the phrase: we expand to the LEFT (modifiers:
# "Beijing Rules", "Optional Protocol", "International Covenant") and to the
# RIGHT ("... on the Rights of the Child", "... against Torture and Other ...").
#
# A token is "name-like" if any of these holds:
#   * it occurs in a lexicon name (all ~1000 rows, not only the 300 matched);
#   * it is capitalised and not sentence-initial (only a bonus signal: the
#     detector never depends on it);
#   * it is a year / number / roman numeral ("1949", "Protocol II");
#   * fallback for unknown names: alphabetic word, not a stopword, not an
#     -ing/-ed/-ly form, and RARE in the corpus (doc frequency < max_df),
#     e.g. "bangkok", "munitions". Common words ("relevant", "new") never pass.
#
# Acceptance:
#   * strong triggers (convention, treaty, ...): at least one name-like token
#     besides the trigger(s);
#   * weak triggers (rules, principles, standards, ...): at least one ANCHOR,
#     i.e. a name-like token that is rare in the corpus, a year/number, or
#     capitalised mid-sentence. "beijing rules" passes, "the new rules" does not.
# Over-inclusion is cheap (those tokens are only ignored), but every IGN token
# is a lost "O" example, so check the IGN rate and candidates_ign.csv.

STRONG_TRIGGERS = ("convention", "covenant", "treaty", "protocol", "charter", "statute", "pact")
WEAK_TRIGGERS = ("agreement", "declaration", "accord", "principles", "rules",
                 "guidelines", "standards")


def _trigger_forms(words):
    out = set()
    for w in words:
        out.add(w)
        out.add(w[:-1] if w.endswith("s") else w + "s")   # rules<->rule, pact<->pacts
        if w.endswith("y"):
            out.add(w[:-1] + "ies")                          # treaty -> treaties
    return out


STRONG_FORMS = _trigger_forms(STRONG_TRIGGERS)
WEAK_FORMS = _trigger_forms(WEAK_TRIGGERS)
TRIGGER_FORMS = STRONG_FORMS | WEAK_FORMS

# Words allowed INSIDE a name, between name-like tokens (right expansion only).
_CONNECTORS = frozenset((
    "of", "the", "and", "on", "for", "in", "to", "against", "relating",
    "concerning", "between", "with", "at", "by", "from", "or", "a", "an",
))
# Words that always end an expansion, even if they occur in a lexicon name.
_STOP = frozenset((
    "this", "that", "these", "those", "present", "said", "such", "which", "who",
    "whose", "whom", "it", "its", "their", "they", "we", "our", "his", "her",
    "shall", "should", "may", "must", "will", "would", "can", "could",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "not", "no", "all", "any", "each", "also", "further", "as", "so", "than",
    "there", "here", "whereby", "including", "inter", "alia", "relevant",
    "calls", "urges", "requests", "invites", "decides", "notes", "welcomes",
    "recalls", "reaffirms", "recognizes", "recognises", "encourages",
    "emphasizes", "stresses", "takes", "adopts", "approves", "expresses",
    "adopted", "entered", "signed", "ratified", "acceded", "contained", "annexed",
    "under", "pursuant", "accordance", "within", "through", "since", "until",
    "upon", "regarding", "per", "via", "thereto", "thereof", "therein", "herein",
    "hereto", "respectively",
))
_HYPHENS = frozenset(("-", "\u2010", "\u2011", "\u2013"))
_ARTICLES = frozenset(("the", "a", "an"))
_ROMAN = frozenset(("ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"))
_SENT_END = frozenset((".", ";", ":", "!", "?"))

# Normalised phrases (lowercase, no spaces) that must never be ignored.
# Grow it after reading candidates_ign.csv.
CANDIDATE_BLOCKLIST = {
    "rulesofprocedure", "rulesofprocedureofthegeneralassembly",
    "financialregulationsandrules", "treatyseries", "treatybodies",
}

MAX_LEFT = 5          # modifiers before the trigger
MAX_RIGHT = 25        # tokens after the trigger
MAX_FALLBACK_RUN = 2  # consecutive unknown-but-rare words accepted


def build_name_vocab(names) -> set[str]:
    """Lowercase content words of every lexicon name (titles + aliases)."""
    vocab = set()
    for n in names:
        if not isinstance(n, str):
            continue
        for t in nlp.make_doc(n.lower()):
            w = t.text
            if t.is_alpha and len(w) > 1 and w not in _CONNECTORS and w not in _STOP:
                vocab.add(w)
    return vocab


def corpus_doc_freq(texts) -> tuple[Counter, int]:
    """Lowercase word -> number of documents containing it."""
    df, n = Counter(), 0
    for text in texts:
        if isinstance(text, str):
            df.update(set(re.findall(r"[a-z]+", text.lower())))
            n += 1
    return df, n


class CandidateFinder:
    def __init__(self, name_vocab, doc_freq, n_docs, max_df=0.05):
        self.vocab = name_vocab
        self.max_df_count = max(1, int(max_df * max(n_docs, 1)))
        self.df = doc_freq

    def _rare(self, w):
        return self.df.get(w, 0) < self.max_df_count

    @staticmethod
    def _is_break_space(tok):
        return tok.is_space and tok.text.count("\n") >= 2

    @staticmethod
    def _sent_initial(doc, i):
        j = i - 1
        while j >= 0 and doc[j].is_space and "\n" not in doc[j].text:
            j -= 1
        return j < 0 or doc[j].text in _SENT_END or (doc[j].is_space and "\n" in doc[j].text)

    def classify(self, doc, i):
        """
        None if the token cannot be part of a name, else a tuple
        (kind, is_anchor) with kind in {"trigger", "vocab", "num", "cap", "fallback"}.
        """
        tok = doc[i]
        w = tok.lower_
        if w in TRIGGER_FORMS:
            return ("trigger", False)
        if w in _STOP or w in _CONNECTORS or tok.is_punct or tok.is_space:
            return None
        if tok.like_num or w in _ROMAN or (tok.text == "I" and i > 0 and doc[i - 1].lower_ in TRIGGER_FORMS):
            return ("num", True)
        cap_mid = tok.text[:1].isupper() and not self._sent_initial(doc, i) and not w.endswith("ing")
        if w in self.vocab:
            return ("vocab", self._rare(w) or cap_mid)
        if cap_mid and tok.is_alpha:
            return ("cap", True)
        if (tok.is_alpha and len(w) > 2 and self._rare(w)
                and not w.endswith(("ing", "ed", "ly"))):
            return ("fallback", True)
        return None

    def _skip_soft_space(self, doc, i, step):
        while 0 <= i < len(doc) and doc[i].is_space and not self._is_break_space(doc[i]):
            i += step
        return i

    def _list_ahead(self, doc, i, cls):
        """After a comma at i-1: accept 'x, y and z' enumerations inside a name."""
        k, seen = i, 0
        while k < len(doc) and seen < 8:
            k = self._skip_soft_space(doc, k, 1)
            if k >= len(doc):
                return False
            if doc[k].lower_ in ("and", "or"):
                return seen > 0
            if doc[k].text == "," or cls(k) is not None:
                seen += 1
                k += 1
                continue
            return False
        return False

    def spans(self, doc) -> list[tuple[int, int]]:
        cache = {}

        def cls(i):
            if i not in cache:
                cache[i] = self.classify(doc, i)
            return cache[i]

        raw = []
        for t in range(len(doc)):
            if doc[t].lower_ not in TRIGGER_FORMS:
                continue
            # left: only adjacent modifiers, never connectors
            start, i, n_left = t, t - 1, 0
            while n_left < MAX_LEFT:
                i = self._skip_soft_space(doc, i, -1)
                if i < 0:
                    break
                c = cls(i)
                if c is None:
                    break
                start, i, n_left = i, i - 1, n_left + 1
            # right: (connector* name-like)+ with enumerations "x, y and z"
            end, i, fb_run = t + 1, t + 1, 0
            while i < len(doc) and i - t <= MAX_RIGHT:
                i = self._skip_soft_space(doc, i, 1)
                if i >= len(doc):
                    break
                j, prev_conn = i, None
                coordination = False
                while j < len(doc) and (doc[j].lower_ in _CONNECTORS or doc[j].text in _HYPHENS or
                                        (doc[j].is_space and not self._is_break_space(doc[j])) or
                                        (doc[j].text == "," and self._list_ahead(doc, j + 1, cls))):
                    w = doc[j].lower_
                    if not doc[j].is_space:
                        # "X and the Y" / "X, the Y" -> two names, not one
                        if w in _ARTICLES and prev_conn in ("and", "or", ","):
                            coordination = True
                            break
                        prev_conn = w
                    j += 1
                if coordination or j >= len(doc):
                    break
                c = cls(j)
                if c is None:
                    break
                if prev_conn in _CONNECTORS:
                    fb_run = 0            # a connector word restarts the unknown-word run
                fb_run = fb_run + 1 if c[0] == "fallback" else 0
                if fb_run > MAX_FALLBACK_RUN:
                    break
                end, i = j + 1, j + 1
            raw.append((start, end))

        # merge overlapping / touching spans ("optional protocol" + "to the convention ...")
        merged = []
        for s, e in sorted(raw):
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        out = []
        for s, e in merged:
            if not self.accepts(doc, range(s, e), cls):
                continue
            if _norm(doc[s:e].text) in CANDIDATE_BLOCKLIST:
                continue
            out.append((s, e))
        return out

    def accepts(self, doc, idxs, cls=None):
        """
        Acceptance rule applied to an arbitrary set of token indices:
          * with a strong trigger: at least one other name-like token;
          * with weak triggers only: at least one anchor.
        Used on the whole candidate in spans(), and again in to_bio_examples()
        on the tokens left uncovered by the lexicon, so that a weak trigger
        glued to a known name ("principles of the <Charter of the United
        Nations>") is not turned into IGN.
        """
        cls = cls or (lambda i: self.classify(doc, i))
        idxs = [i for i in idxs if not doc[i].is_space]
        kinds = [k for k in (cls(i) for i in idxs) if k is not None]
        has_strong = any(doc[i].lower_ in STRONG_FORMS for i in idxs)
        others = [k for k in kinds if k[0] != "trigger"]
        if has_strong:
            return bool(others)
        return any(anchor for _, anchor in others)


# ---------------------------------------------------------------------------
# 3. SETTING BIO TAGS (formatting HF token classification)
# ---------------------------------------------------------------------------

def to_bio_examples(known_docs, finder=None, doc_years=None, eval_from_year=None,
                    candidates_csv=None, batch_size: int = BATCH_SIZE):
    """
    Convert each doc's character-level spans into tokens + BIO labels,
    using spaCy's tokenizer for the split (realigned to the model's own
    tokenizer later, in align_labels).

    Tokens inside an instrument-looking phrase (CandidateFinder) that the
    lexicon did not label are tagged IGNORE_TAG instead of "O". Whether IGN is
    actually used for a split is decided later (see unmask()).

    If `candidates_csv` is given, every ignored phrase is written there with
    its frequency (train / eval docs): read it to tune the triggers / blocklist
    and to grow the lexicon.
    """
    examples = []
    n_ign_docs = 0
    split_tok = Counter()       # (split, "ign"|"all") -> count
    phrase_stats = {}           # norm -> [example text, n_mentions, train_docs, eval_docs]

    def split_of(doc_id):
        if doc_years is None or eval_from_year is None:
            return "all"
        y = doc_years.get(doc_id)
        if y is None or pd.isna(y):
            return "unknown"
        return "train" if y < eval_from_year else "eval"

    docs = nlp.pipe((d["text"] for d in known_docs), batch_size=batch_size)

    for d, doc in zip(known_docs, docs):
        labels = ["O"] * len(doc)

        for start, end, _matched_text in d["spans"]:
            span = doc.char_span(start, end, alignment_mode="expand")
            if span is None:
                continue
            labels[span.start] = "B-INSTRUMENT"
            for i in range(span.start + 1, span.end):
                labels[i] = "I-INSTRUMENT"

        split = split_of(d["doc_id"])
        masked_here = 0
        seen_in_doc = set()
        for s, e in (finder.spans(doc) if finder is not None else []):
            free = [i for i in range(s, e) if labels[i] == "O"]
            if not any(not doc[i].is_space for i in free):
                continue                     # fully covered by a lexicon match
            covered = len(free) < e - s
            # If part of the candidate is already a lexicon match, the leftover
            # tokens must qualify as a name on their own; otherwise they stay O
            # (e.g. "principles of the" + <Charter of the United Nations>).
            if covered and not finder.accepts(doc, free):
                continue
            for i in free:
                labels[i] = IGNORE_TAG
                masked_here += not doc[i].is_space
            key = _norm(doc[s:e].text)
            free_text = " ".join(" ".join(doc[i].text for i in free if not doc[i].is_space).split())
            st = phrase_stats.setdefault(
                key, [" ".join(doc[s:e].text.split()), 0, 0, 0, free_text if covered else ""])
            st[1] += 1
            if key not in seen_in_doc:
                seen_in_doc.add(key)
                st[2 if split == "train" else 3] += 1
        n_ign_docs += masked_here > 0
        split_tok[(split, "ign")] += masked_here
        split_tok[(split, "all")] += sum(not t.is_space for t in doc)

        examples.append({
            "doc_id": d["doc_id"],
            "tokens": [t.text for t in doc],
            "ner_tags": labels,
        })

    print(f"IGN (unlabelled instrument-looking) tokens in {n_ign_docs} docs")
    for split in sorted({k[0] for k in split_tok}):
        ign, total = split_tok[(split, "ign")], split_tok[(split, "all")]
        print(f"  {split:>7}: {ign} IGN / {total} tokens ({100 * ign / max(total, 1):.2f}%)")

    if candidates_csv and phrase_stats:
        os.makedirs(os.path.dirname(os.path.abspath(candidates_csv)), exist_ok=True)
        # ignored_part: only the tokens actually tagged IGN when the candidate
        # partly overlaps a lexicon match (empty = the whole phrase is IGN)
        rows = [{"phrase": v[0], "ignored_part": v[4], "n_mentions": v[1],
                 "train_docs": v[2], "eval_docs": v[3]}
                for v in phrase_stats.values()]
        pd.DataFrame(rows).sort_values(["n_mentions", "phrase"], ascending=[False, True]) \
            .to_csv(candidates_csv, index=False)
        print(f"{len(rows)} distinct ignored phrases written to {candidates_csv}")
    return examples


def unmask(examples):
    """Turn IGN back into O (i.e. the original weak labelling)."""
    return [
        {**ex, "ner_tags": ["O" if t == IGNORE_TAG else t for t in ex["ner_tags"]]}
        for ex in examples
    ]


# ---------------------------------------------------------------------------
# 3b. MENTION REPLACEMENT (anti-memorisation augmentation, train split only)
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Lowercase, whitespace removed: makes spaCy-token joins comparable to raw text."""
    return re.sub(r"\s+", "", str(s).lower())


def build_augmentation_pool(df_instrument: pd.DataFrame, max_year: int,
                            exclude_mentions) -> list[str]:
    """
    Lexicon names that can be pasted into training sentences without leaking
    the eval set:
      * the instrument was adopted before `max_year` (eval start year);
      * no name of the instrument occurs in the eval docs. A row hit by an eval
        mention also excludes every other name of that row, and long names are
        compared by substring to catch near-duplicates from other tables.
    Rows without a parseable year are dropped (conservative).
    """
    years = pd.to_numeric(df_instrument["year"], errors="coerce")
    rows = [
        (_row_names(t, a), y)
        for (t, a), y in zip(
            df_instrument[["title", "alternative_name"]].itertuples(index=False, name=None),
            years,
        )
    ]
    excl_base = {_norm(m) for m in exclude_mentions} - {""}
    excl = set(excl_base)
    for names, _ in rows:
        normed = {_norm(n) for n in names}
        if normed & excl_base:
            excl |= normed

    def leaks(n):
        return any(
            n == e or (min(len(n), len(e)) >= 10 and (n in e or e in n))
            for e in excl
        )

    pool, n_excluded = set(), 0
    for names, year in rows:
        if not names or pd.isna(year) or year >= max_year:
            continue
        if any(leaks(_norm(n)) for n in names):
            n_excluded += 1
            continue
        pool.update(names)
    print(f"augmentation pool: {len(pool)} names "
          f"({n_excluded} lexicon rows excluded because they occur in eval docs)")
    return sorted(pool)


def _match_case(rep, orig):
    """
    Give the replacement the casing style of the mention it replaces, so that
    augmentation does not inject Title Case names into the (mostly lowercase)
    pre-2015 text: that would teach a cased model "capitalised = instrument".
    """
    letters = "".join(t for t in orig if any(c.isalpha() for c in t))
    if letters and letters == letters.lower():
        return [t.lower() for t in rep]
    if letters and letters == letters.upper():
        return [t.upper() for t in rep]
    return list(rep)


def augment_mentions(examples, pool, prob, seed=42):
    """
    Replace each B/I mention with a random pool name with probability `prob`.
    The surrounding context is untouched, so the model has to rely on context
    and on the shape of the name, not on its identity.
    """
    rng = random.Random(seed)
    pool_tokens = [[t.text for t in nlp.make_doc(n)] for n in pool]
    pool_tokens = [p for p in pool_tokens if p]
    if not pool_tokens:
        return examples

    out, n_replaced, n_total = [], 0, 0
    for ex in examples:
        toks, tags = ex["tokens"], ex["ner_tags"]
        new_toks, new_tags = [], []
        i = 0
        while i < len(toks):
            if tags[i] == "B-INSTRUMENT":
                j = i + 1
                while j < len(toks) and tags[j] == "I-INSTRUMENT":
                    j += 1
                n_total += 1
                # a mention nested in a longer unlabelled name ("optional protocol to
                # the <mention>") is left alone: replacing it builds a hybrid name
                nested = (i > 0 and tags[i - 1] == IGNORE_TAG) or (j < len(tags) and tags[j] == IGNORE_TAG)
                if not nested and rng.random() < prob:
                    rep = _match_case(rng.choice(pool_tokens), toks[i:j])
                    new_toks.extend(rep)
                    new_tags.extend(["B-INSTRUMENT"] + ["I-INSTRUMENT"] * (len(rep) - 1))
                    n_replaced += 1
                else:
                    new_toks.extend(toks[i:j])
                    new_tags.extend(tags[i:j])
                i = j
            else:
                new_toks.append(toks[i])
                new_tags.append(tags[i])
                i += 1
        out.append({**ex, "tokens": new_toks, "ner_tags": new_tags})
    print(f"mention replacement: {n_replaced}/{n_total} train mentions replaced")
    return out
