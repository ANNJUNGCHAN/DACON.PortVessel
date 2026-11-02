"""Latin-row generator and fleet-rename features + LLM mask on rename-mode rows (C3 of reports/diag/action_plan.md).

Port of experiments/diag/latin_residuals/{lr_lib.py, genops.py, wgen.py} (feature groups 'gen' and 'rename',
and run_cv.mask_llm). Every value depends only on one query row (its name and its candidate pairs) and on
catalog-only statistics (first-word counts, word IDF), so a row gets the same output alone, in any subset or
in any order. No labels and no cross-row statistics. Hangul rows get NaN.

Feature groups (Latin rows only)
  gen    : gen_d (0/1/2 delete/double/swap ops of an interior letter from the candidate's normalized name to
           the record's, after optional M/V stripping; 9 = farther), osa_dist, row_min_gen, n_row_gen_min,
           is_gen_min
  rename : first_eq, rest_gen, q_first_freq, c_first_freq, fleet_swap (first word differs, the rest is within
           2 ops, the catalog first word starts >= 5 catalog names), idf_overlap, row_has_fleet_swap,
           row_n_gt_exact

LLM mask (rename mode)
  On Latin rows where no candidate is within 2 word-interior ops (wgen_dist, after M/V stripping) of the
  record, ft_score / ft_rank are set to NaN. In train these are exactly the 13 renames; the LLM never scored
  the true vessel on such rows (10 of 10), and its features pushed the ranker to a wrong fleet sibling. The
  ranker then treats them like the synthetic renames, which never carry LLM scores. The gate is evaluated
  only on Latin rows the LLM scores (name_exact sum != 1, rank.finetune.select_pairs); on other rows the
  LLM columns are NaN already, so the result equals the prototype's all-row gate. The gate also skips the
  variant expansion when the joined names are more than 2 Damerau-Levenshtein edits apart (exact, see
  wgen_dist_fast) and stops scanning a row at its first near candidate.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from rapidfuzz.distance import OSA, DamerauLevenshtein

from common import is_hangul_name, normalize_key

GEN_COLS = ["gen_d", "osa_dist", "row_min_gen", "n_row_gen_min", "is_gen_min"]
RENAME_COLS = ["first_eq", "rest_gen", "q_first_freq", "c_first_freq", "fleet_swap", "idf_overlap",
               "row_has_fleet_swap", "row_n_gt_exact"]
LATIN_COLS = GEN_COLS + RENAME_COLS
LLM_COLS = ("ft_score", "ft_rank")
_MV = re.compile(r"^(M\s*[./]\s*[VT]\.?|MV|MT)\s+")


# ---------------------------------------------------------------------------
# name helpers
# ---------------------------------------------------------------------------
def words(name) -> list[str]:
    if not isinstance(name, str):
        return []
    name = unicodedata.normalize("NFKC", name).upper()
    return [w for w in re.split(r"[^0-9A-Z]+", name) if w]


def wkey(name) -> tuple[str, ...]:
    return tuple(words(name))


def strip_mv(name) -> str:
    if not isinstance(name, str):
        return ""
    return _MV.sub("", unicodedata.normalize("NFKC", name).upper().strip())


# ---------------------------------------------------------------------------
# generator distance on normalized keys (genops.py)
# ---------------------------------------------------------------------------
def _interior_letters(s: str) -> list[int]:
    letters = [i for i, ch in enumerate(s) if ch.isalpha()]
    return letters[1:-1]


def one_op_variants(s: str) -> set[str]:
    """Strings one generator op (delete / double / swap with the next letter) of an interior letter away."""
    out = set()
    for i in _interior_letters(s):
        out.add(s[:i] + s[i + 1:])                       # delete
        out.add(s[:i] + s[i] + s[i:])                    # double
        if i + 1 < len(s) and s[i + 1].isalpha():
            out.add(s[:i] + s[i + 1] + s[i] + s[i + 2:])  # transpose
    out.discard(s)
    return out


@lru_cache(maxsize=400_000)
def _variants1(s: str) -> frozenset:
    return frozenset(one_op_variants(s))


def gen_dist(record: str, cand: str, max_ops: int = 2) -> int:
    """0/1/2 generator ops from catalog key ``cand`` to record key ``record``; 9 if farther."""
    if record == cand:
        return 0
    if not record or not cand or abs(len(record) - len(cand)) > max_ops:
        return 9
    v1 = _variants1(cand)
    if record in v1:
        return 1
    if max_ops >= 2:
        for v in v1:  # ops are not symmetric: expand the candidate side twice
            if abs(len(record) - len(v)) <= 1 and record in _variants1(v):
                return 2
    return 9


# ---------------------------------------------------------------------------
# word-interior generator distance on word tuples (wgen.py)
# ---------------------------------------------------------------------------
def _word_variants1(w: str) -> set[str]:
    out = set()
    n = len(w)
    for i in range(1, n - 1):  # interior letters only
        out.add(w[:i] + w[i + 1:])            # delete
        out.add(w[:i] + w[i] + w[i:])         # double
        if i + 1 < n - 1:                     # transpose two interior letters
            out.add(w[:i] + w[i + 1] + w[i] + w[i + 2:])
    out.discard(w)
    return out


@lru_cache(maxsize=500_000)
def wvariants1(key: tuple[str, ...]) -> frozenset:
    out = set()
    for j, w in enumerate(key):
        for v in _word_variants1(w):
            out.add(key[:j] + (v,) + key[j + 1:])
    return frozenset(out)


def wgen_dist(rec: tuple, cand: tuple, max_ops: int = 2) -> int:
    """0/1/2 word-interior ops from catalog word tuple ``cand`` to record tuple ``rec``; 9 if farther."""
    if rec == cand:
        return 0
    if len(rec) != len(cand) or not rec:
        return 9
    diff = [j for j in range(len(rec)) if rec[j] != cand[j]]
    if len(diff) > max_ops or sum(abs(len(rec[j]) - len(cand[j])) for j in diff) > max_ops:
        return 9
    v1 = wvariants1(cand)
    if rec in v1:
        return 1
    if max_ops >= 2:
        for v in v1:
            if rec in wvariants1(v):
                return 2
    return 9


def wgen_dist_fast(rec: tuple, cand: tuple, rec_s: str, cand_s: str) -> int:
    """wgen_dist with an exact pre-filter (``*_s`` = the tuples joined by spaces).

    Each word-interior op is one single-character deletion, insertion (double) or adjacent transposition of
    the joined string, and unrestricted Damerau-Levenshtein is a metric with unit cost for each of them, so
    wgen_dist <= 2 implies DL(joined) <= 2. DL > 2 therefore means 9 without the variant expansion.
    """
    if rec == cand:
        return 0
    if len(rec) != len(cand) or not rec:
        return 9
    if DamerauLevenshtein.distance(rec_s, cand_s, score_cutoff=2) > 2:
        return 9
    return wgen_dist(rec, cand)


# ---------------------------------------------------------------------------
# catalog-only tables
# ---------------------------------------------------------------------------
class CatalogAux:
    """Normalized keys, first word / rest, first-word counts and word IDF of the catalog names."""

    def __init__(self, catalog: pd.DataFrame, fleet_min: int = 5):
        names = catalog["vessel_name"].fillna("").astype(str).tolist()
        self.names_raw = names
        self.norm = [normalize_key(n) for n in names]
        ws = [words(n) for n in names]
        self.first = [w[0] if len(w) >= 2 else "" for w in ws]
        self.rest = ["".join(w[1:]) if len(w) >= 2 else "" for w in ws]
        self.first_count = Counter(f for f in self.first if f)
        df = Counter(w for wl in ws for w in set(wl))
        n = len(names)
        self.idf = {w: math.log(n / (1 + c)) for w, c in df.items()}
        self.idf_default = math.log(n)
        self.words = [set(w) for w in ws]
        self.wkeys = [tuple(w) for w in ws]
        self.wjoined = [" ".join(w) for w in ws]
        self.fleet_min = fleet_min

    def first_freq(self, w: str) -> float:
        return math.log1p(self.first_count.get(w, 0)) if w else np.nan


_AUX: list = []  # [(catalog, CatalogAux)]: one entry, keyed by object identity (the catalog is kept alive)


def catalog_aux(catalog: pd.DataFrame) -> CatalogAux:
    if not (_AUX and _AUX[0][0] is catalog):
        _AUX[:] = [(catalog, CatalogAux(catalog))]
    return _AUX[0][1]


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def _hangul_rows(queries: pd.DataFrame) -> np.ndarray:
    return np.array([is_hangul_name(n) for n in queries["vessel_name"].tolist()], dtype=bool)


def latin_features(pairs: pd.DataFrame, queries: pd.DataFrame, aux: CatalogAux) -> pd.DataFrame:
    """gen + rename columns (LATIN_COLS) aligned to pairs.index; NaN on Hangul rows."""
    q = pairs["q"].to_numpy()
    c = pairs["c"].to_numpy()
    qnames = queries["vessel_name"].tolist()
    hangul = _hangul_rows(queries)
    qnorm = [normalize_key(nm) for nm in qnames]
    qstrip = [normalize_key(strip_mv(nm)) for nm in qnames]
    qwords = [words(nm) for nm in qnames]
    lat = ~hangul[q]
    lat_idx = np.flatnonzero(lat)
    n = len(pairs)
    out = {}

    osa = np.full(n, np.nan)
    gen = np.full(n, np.nan)
    for i in lat_idx:
        a, b = qnorm[q[i]], aux.norm[c[i]]
        d = OSA.distance(a, b, score_cutoff=9)
        osa[i] = min(d, 9)
        g = gen_dist(a, b) if d <= 2 else 9
        s = qstrip[q[i]]
        if s != a and g > 0:
            d2 = OSA.distance(s, b, score_cutoff=3)
            if d2 <= 2:
                g = min(g, gen_dist(s, b))
        gen[i] = g
    row_min = pd.DataFrame({"q": q, "gen": gen}).groupby("q")["gen"].transform("min").to_numpy()
    is_min = (gen == row_min) & lat
    n_min = pd.Series(is_min.astype(float)).groupby(q).transform("sum").to_numpy()
    out["gen_d"] = gen
    out["osa_dist"] = osa
    out["row_min_gen"] = np.where(lat, row_min, np.nan)
    out["n_row_gen_min"] = np.where(lat, n_min, np.nan)
    out["is_gen_min"] = np.where(lat, is_min.astype(float), np.nan)

    first_eq = np.full(n, np.nan)
    rest_gen = np.full(n, np.nan)
    qff = np.full(n, np.nan)
    cff = np.full(n, np.nan)
    swap = np.full(n, np.nan)
    idf_ov = np.full(n, np.nan)
    idf = aux.idf.get
    for i in lat_idx:
        qw = qwords[q[i]]
        cc = c[i]
        qf = qw[0] if len(qw) >= 2 else ""
        qr = "".join(qw[1:]) if len(qw) >= 2 else ""
        cf, cr = aux.first[cc], aux.rest[cc]
        first_eq[i] = float(bool(qf) and qf == cf)
        if qr and cr:
            d = OSA.distance(qr, cr, score_cutoff=3)
            rest_gen[i] = gen_dist(qr, cr) if d <= 2 else 9
        else:
            rest_gen[i] = 9
        qff[i] = aux.first_freq(qf)
        cff[i] = aux.first_freq(cf)
        swap[i] = float(qf != "" and cf != "" and qf != cf and rest_gen[i] <= 2
                        and aux.first_count.get(cf, 0) >= aux.fleet_min)
        qs, cs = set(qw), aux.words[cc]
        uni = qs | cs
        if uni:
            idf_ov[i] = (sum(idf(x, aux.idf_default) for x in qs & cs)
                         / sum(idf(x, aux.idf_default) for x in uni))
    out.update(first_eq=first_eq, rest_gen=rest_gen, q_first_freq=qff, c_first_freq=cff,
               fleet_swap=swap, idf_overlap=idf_ov)
    row_swap = pd.Series(np.nan_to_num(swap)).groupby(q).transform("max").to_numpy()
    out["row_has_fleet_swap"] = np.where(lat, row_swap, np.nan)
    gt_eq = pairs["gt_eq"].to_numpy(dtype=float)
    row_gt = pd.Series(gt_eq).groupby(q).transform("sum").to_numpy()
    out["row_n_gt_exact"] = np.where(lat, row_gt, np.nan)
    return pd.DataFrame(out, index=pairs.index)


# ---------------------------------------------------------------------------
# LLM mask on rename-mode rows
# ---------------------------------------------------------------------------
def gate_rows(pairs: pd.DataFrame, queries: pd.DataFrame) -> np.ndarray:
    """Per pair: True when its row is a Latin row the LLM scores (name_exact sum != 1)."""
    q = pairs["q"].to_numpy()
    keep = ~_hangul_rows(queries)[q]
    if "name_exact" in pairs:
        exact = pd.Series(pairs["name_exact"].to_numpy(dtype=float)).groupby(q).transform("sum").to_numpy()
        keep &= exact != 1
    return keep


def rename_mode_rows(pairs: pd.DataFrame, queries: pd.DataFrame, aux: CatalogAux,
                     restrict: bool = True) -> set[int]:
    """Row indices q (Latin rows; with ``restrict`` only those the LLM scores) whose every candidate is more
    than 2 word-interior ops from the record, after M/V stripping."""
    q = pairs["q"].to_numpy()
    c = pairs["c"].to_numpy()
    sel = gate_rows(pairs, queries) if restrict else ~_hangul_rows(queries)[q]
    qnames = queries["vessel_name"].tolist()
    best: dict[int, int] = {}
    keys: dict[int, tuple] = {}
    for i in np.flatnonzero(sel):
        qi = int(q[i])
        if best.get(qi, 9) <= 2:  # the row already has a near candidate; its min cannot exceed 2
            continue
        if qi not in keys:
            a, s = wkey(qnames[qi]), wkey(strip_mv(qnames[qi]))
            keys[qi] = (a, s, " ".join(a), " ".join(s))
        a, s, a_s, s_s = keys[qi]
        b, b_s = aux.wkeys[c[i]], aux.wjoined[c[i]]
        d = wgen_dist_fast(a, b, a_s, b_s)
        if d > 0 and s != a:
            d = min(d, wgen_dist_fast(s, b, s_s, b_s))
        best[qi] = min(best.get(qi, 9), d)
    return {qi for qi, d in best.items() if d > 2}


def mask_llm(X: pd.DataFrame, pairs: pd.DataFrame, queries: pd.DataFrame, aux: CatalogAux) -> pd.DataFrame:
    """X with ft_score / ft_rank set to NaN on rename-mode rows (X is aligned row-for-row with pairs)."""
    cols = [col for col in LLM_COLS if col in X.columns]
    if not cols:
        return X
    rows = rename_mode_rows(pairs, queries, aux)
    if not rows:
        return X
    sel = np.isin(pairs["q"].to_numpy(), np.fromiter(rows, dtype=np.int64, count=len(rows)))
    X = X.copy()
    for col in cols:
        X[col] = np.where(sel, np.nan, X[col].to_numpy(dtype=np.float64))
    return X
