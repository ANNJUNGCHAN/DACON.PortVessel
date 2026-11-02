"""Candidate generation + fold-independent pair features.

Every feature here depends only on one query row and the catalog, so the
output for a row never changes with the other rows in the batch.

Sources (unioned per row):
  window: tonnage-nearest catalog rows (HangulMatcher window), best k_window
          by a heuristic name-similarity minus tonnage-gap score
  base:   the official baseline retrieval (exact name, trigram, tonnage)
  key:    exact phonetic-key hits anywhere in the catalog (no tonnage needed)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from baseline_lib import CatalogIndex
from common import is_hangul_name, normalize_key
from hangul_match import HangulMatcher, _positive, _sim_matrix
from en2ko import name_variants
from translit import (hangul_core, name_digit_groups, name_digit_groups_forms, name_digits, name_digits_ext,
                      name_digits_v2, phonetic_keys, strip_leading_zeros, strip_number_words, to_jamo)

try:  # typo-tolerant search over the whole catalog (non-Hangul rows)
    from rapidfuzz import process as _rf_process
    from rapidfuzz.distance import OSA as _OSA
except ImportError:  # pragma: no cover - server without rapidfuzz: source disabled
    _rf_process = _OSA = None


def _name_words(name: str) -> list[str]:
    import re
    import unicodedata
    words = re.split(r"[^0-9A-Z]+", unicodedata.normalize("NFKC", name).upper())
    return [w for w in words if len(w) >= 2 and w != "NO"]


def _trigrams(key: str) -> set[str]:
    """Char 3-grams of a normalize_key string, padded so short keys still have one (latin_char_tfidf_k)."""
    s = f" {key} " if key else ""
    return {s[i:i + 3] for i in range(len(s) - 2)}

BASE_FEATURES = (
    "name_exact", "bigram_jac", "trigram_jac", "prefix_ratio", "len_ratio",
    "gross_gap", "intl_gap", "best_gap", "q_gt_missing", "c_gt_missing",
)


def tr_rank(cands: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """Row-local rank by transliteration similarity (ties -> lower catalog position)."""
    return np.argsort(np.lexsort((cands, -np.nan_to_num(tr, nan=-1.0))))


def _prefix_ratio(a: str, b: str) -> float:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(len(a), len(b), 1)


def topk_desc(sims: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values, ties -> lower index, in that order (experiments/diag/recall)."""
    if k >= len(sims):
        part = np.arange(len(sims))
    else:
        thr = np.partition(sims, len(sims) - k)[len(sims) - k]
        part = np.flatnonzero(sims >= thr)
    order = np.lexsort((part, -sims[part]))
    return part[order][:k]


class WordContainIndex:
    """Word-containment retrieval (S9 err5 fix B; port of experiments/s9_robust/err5/wcpath.WCIndex, thr 0.8).

    For a record: the catalog vessels whose name contains a match for EVERY record word (order-free), ranked by the
    gap to the nearest record tonnage (missing -> last), then fewer catalog words, then catalog position; top k.
    Latin words: _name_words after a leading M/V, M/T, MV or MT; a record word matches a catalog word when equal, or
    (both >= 4 chars) OSA normalized similarity >= thr. Hangul words: jamo of hangul_core per space-separated word
    (>= 3 jamo; a leading 엠브이 / 엠티 / 엠/브이 dropped); one matches a catalog word when one of that word's readings
    (aux["wread"], en2ko.word_variants) has normalized Levenshtein similarity >= thr. Row-local.
    ``aux``: rank.pair_feats4.build_aux table (words, wread, log_gt are used)."""

    def __init__(self, aux: dict, thr: float = 0.8) -> None:
        self.thr = thr
        self.log_gt = aux["log_gt"]
        self.nw = np.array([len(w) for w in aux["words"]])
        post: dict[str, list[int]] = {}
        for pos, ws in enumerate(aux["words"]):
            for w in set(ws):
                post.setdefault(w, []).append(pos)
        self.post = {w: np.array(v) for w, v in post.items()}
        self.vocab4 = [w for w in self.post if len(w) >= 4]
        rd: dict[str, set[str]] = {}
        for w, rs in aux["wread"].items():
            for r in rs:
                if r:
                    rd.setdefault(r, set()).add(w)
        self.rvocab = list(rd)
        self.r2w = [sorted(rd[r]) for r in self.rvocab]

    @staticmethod
    def _ko_words(name) -> list[str]:
        import unicodedata
        from rank.pair_feats4 import JAMO_HI, JAMO_LO
        if not isinstance(name, str):
            return []
        ws = unicodedata.normalize("NFKC", name).split()
        out = []
        for i, w in enumerate(ws):
            if i == 0 and len(ws) > 1 and w in ("엠브이", "엠티", "엠/브이"):
                continue
            j = "".join(ch for ch in to_jamo(hangul_core(w)) if JAMO_LO <= ch <= JAMO_HI)
            if len(j) >= 3:
                out.append(j)
        return out

    @staticmethod
    def _la_words(name) -> list[str]:
        import re
        import unicodedata
        if not isinstance(name, str):
            return []
        ws = _name_words(re.sub(r"^\s*M\s*/\s*[VT]\s+", "", unicodedata.normalize("NFKC", name).upper()))
        if len(ws) > 1 and ws[0] in ("MV", "MT"):
            ws = ws[1:]
        return ws

    def _match_la(self, w: str) -> list[str]:
        from rapidfuzz import process
        from rapidfuzz.distance import OSA
        out = [w] if w in self.post else []
        if len(w) >= 4:
            out += [m for m, _, _ in process.extract(w, self.vocab4, scorer=OSA.normalized_similarity,
                                                     score_cutoff=self.thr, limit=None) if m != w]
        return out

    def _match_ko(self, j: str) -> list[str]:
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein
        out: set[str] = set()
        for _, _, i in process.extract(j, self.rvocab, scorer=Levenshtein.normalized_similarity,
                                       score_cutoff=self.thr, limit=None):
            out.update(self.r2w[i])
        return sorted(out)

    def vessels(self, row: dict, hangul: bool) -> np.ndarray:
        """Catalog positions whose name contains a match for every record word (sorted)."""
        name = row.get("vessel_name")
        words = self._ko_words(name) if hangul else self._la_words(name)
        if not words:
            return np.array([], dtype=int)
        cur = None
        for w in dict.fromkeys(words):
            ms = self._match_ko(w) if hangul else self._match_la(w)
            if not ms:
                return np.array([], dtype=int)
            s = np.unique(np.concatenate([self.post[m] for m in ms]))
            cur = s if cur is None else np.intersect1d(cur, s, assume_unique=True)
            if not len(cur):
                return cur
        return cur

    def topk(self, row: dict, hangul: bool, k: int) -> np.ndarray:
        c = self.vessels(row, hangul)
        if not len(c):
            return c
        tv = []
        for key in ("gross_tonnage", "international_gross_tonnage"):
            try:
                v = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and v > 0:
                tv.append(np.log(v))
        lg = self.log_gt[c]
        gap = np.min([np.abs(lg - t) for t in tv], axis=0) if tv else np.full(len(c), 9.0)
        gap = np.nan_to_num(gap, nan=9.0)
        return c[np.lexsort((c, self.nw[c], gap))[:k]]

    def add_glued_pairs(self, names) -> None:
        """S9 next8 "hw" (port of experiments/s9_robust/next8/hw8.make_hw_index): every pair of adjacent catalog
        tokens A B (re.split on [^0-9A-Z]+ of the NFKC upper name, digit-only tokens dropped) is also indexed as
        the word "A+B", read as the concatenated jamo of the first 3 en2ko.word_variants of A and of B (spaces
        removed; a non-alphabetic token reads as itself). A Hangul record word written glued (흥아 -> Heung-A) can
        then match it. ``names``: the catalog vessel names in catalog order. No file is read."""
        import re
        import unicodedata
        from en2ko import word_variants
        post = {w: list(v) for w, v in self.post.items()}
        rd = {r: set(ws) for r, ws in zip(self.rvocab, self.r2w)}
        vcache: dict[str, list[str]] = {}

        def readings(w: str) -> list[str]:
            if w not in vcache:
                vcache[w] = ([to_jamo(v.replace(" ", "")) for v in list(dict.fromkeys(word_variants(w)))[:3]]
                             if w.isalpha() else [w])
            return vcache[w]
        for pos, nm in enumerate(names):
            toks = [t for t in re.split(r"[^0-9A-Z]+", unicodedata.normalize("NFKC", nm).upper())
                    if t and not t.isdigit()]
            for a, b in zip(toks, toks[1:]):
                key = a + "+" + b
                post.setdefault(key, []).append(pos)
                for ra in readings(a):
                    for rb in readings(b):
                        if ra and rb:
                            rd.setdefault(ra + rb, set()).add(key)
        self.post = {w: np.unique(np.asarray(v)) for w, v in post.items()}
        self.rvocab = list(rd)
        self.r2w = [sorted(rd[r]) for r in self.rvocab]


class CandidateGenerator:
    """S4 options (reports/s4_impl.md; all default off, and off reproduces the S3 generator exactly):

    mv_digits      'M/V X' does not read V as 5 (translit.name_digits_ext), for queries and catalog alike.
    num_words      C5 restricted Hangul number-word reader in digit_state (translit.name_digits_ext).
    notonnage_net  C4.4 safety net, ONLY for Hangul rows without any positive tonnage: key hits are
                   re-sorted by tr_sim before the k_key cut, and the k_global catalog vessels with the best
                   transliteration similarity over the whole catalog are added (source bit 128, no column).
                   Rows with a positive tonnage and Latin rows are untouched.
    global_band    S5b "G4" banded global transliteration retrieval, for EVERY Hangul row (subsumes notonnage_net):
                   key hits are re-sorted by tr_sim before the k_key cut, and the k_global catalog vessels with the
                   best transliteration similarity are added (source bit 128, no new column). With a positive
                   record tonnage the search only considers vessels whose catalog GT is within a factor ``band`` of
                   a record tonnage (vessels without GT are always allowed), and a hit is only added when its
                   similarity is >= (best tr_sim inside the tonnage window) - ``band_gate``. Without any tonnage
                   the whole catalog is searched and every hit is added (= notonnage_net). Latin rows untouched.
                   Port of experiments/diag/recall/global_index.py RecallCandidateGenerator(k_gtr=30,
                   key_by_tr=True, gtr_gate_delta=0.1, gtr_band=log 3, extra_cols=False).
    num_forms      S5c number-form reader for HANGUL record names in digit_state (translit.name_digit_groups_forms:
                   제N / N호 / 하나 투 / 하나투(호) / 에이스일레븐 / 하나II / 금양삼호 / 하나 이호), and for Hangul
                   records digit groups are compared without leading zeros (셀바라 2 == Selvara 02). Latin records
                   and catalog digit strings are untouched (experiments/s5_robust/s5c/REPORT.md).
    The digit options change only the digit_state column; the candidate sets stay the same.

    S9 retrieval options (experiments/s9_robust/PLAN.md item 2), all default off (0 / False = today's frame
    bit for bit). They only ADD candidates: every candidate and every column of the option-off frame is kept
    with the same value (row-relative columns of the existing candidates are computed over the existing set,
    as before). Candidates found only by these sources ("extras") are appended with src_extra = 1; their
    row-relative columns are their position in the union, measured against the existing set (n_cands and
    *_gap_best use the existing set), and finetune.select_pairs never sends them to the LLM. New columns
    (only when an option is on): src_extra, plus src_<option> per enabled option and the option's similarity.
    hangul_global_k     Hangul rows: top-k catalog vessels by transliteration similarity over the WHOLE catalog,
                        no tonnage (src_hglobal).
    latin_word_index_k  Latin rows: top-k vessels by IDF-weighted name-word cosine (_name_words; words shared
                        with the record, rare words weigh most) (src_lword, word_cos).
    latin_char_tfidf_k  Latin rows: top-k vessels by char 3-gram TF-IDF cosine over normalize_key names
                        (src_ltfidf, tfidf_sim).
    cs_fuzzy            every row: vessels whose train call sign is within 1 edit of the record's call sign
                        (src_cs). Needs ``cs_map`` {normalize_key(call sign): [catalog positions]} built from the
                        (fold-)train rows; such a generator cannot go through pipeline.candidate_generator.
    hangul_num_v2       Hangul records (port of the experiments/diag/hangul_fragility digits + numcore prototype):
                        Hangul number words are stripped from the name before every name similarity / key / reading
                        (translit.strip_number_words), and digit_state reads the record with translit.name_digits_v2
                        (with num_forms: only when the num_forms reader finds no digits) against the catalog read by
                        name_digits_v2. Latin rows untouched.

    S9 item 4b (experiments/s9_robust/item4/RESULT_4.md), default off (off = today's frame bit for bit):
    pair_feats4         append the 22 rank.pair_feats4.ALL columns (float32) as the LAST step of row_pairs (after
                        the S9 extras); the row-relative ones are computed over the FULL union frame. ``feats4_aux``:
                        the catalog side table (pair_feats4.build_aux; shipped as art["feats4_aux"]); None = build it
                        here from the catalog and the complete tr_cache (about 100 s, no file read).

    S9 err5 fix B (experiments/s9_robust/err5/RESULT_5.md), default 0 (off = today's frame bit for bit):
    word_contain_k      every row: the top-k catalog vessels whose name contains a match for EVERY record word
                        (WordContainIndex; order-free) join the S9 extras (src_extra = 1, no new column). Uses the
                        feats4_aux table (words, wread, log_gt), built here as for pair_feats4 when not given.

    S9 next8 "hw" (experiments/s9_robust/next8/RESULT_8.md P3), default 0 (off = today's frame bit for bit):
    hangul_wc2_k        Hangul rows only: the top-k of a second WordContainIndex (thr 0.75, plus glued adjacent
                        catalog tokens, WordContainIndex.add_glued_pairs) join the S9 extras like word_contain_k
                        (src_extra = 1, no new column; independent of word_contain_k). Built from the feats4_aux
                        table and the catalog names on first use (_wc2_index; no file read).

    S9 charft10 dual encoder (experiments/s9_robust/charft10/RESULT_10.md, v2), default 0 (off = today's frame bit for
    bit):
    hangul_de_k         Hangul rows only: the top-k catalog vessels by cosine of the char dual encoder
                        (``dual_encoder``, rank.dual_encoder.DualEncoder: the record name's embedding vs the shipped
                        catalog embeddings; ties -> lower catalog position) join the S9 extras like word_contain_k
                        (src_extra = 1, no new source column, never LLM-scored), and three columns are appended LAST
                        (after pair_feats4), row-relative over the full union frame, NaN on Latin rows:
                        de_cos (cosine), de_grank (rank in the row's DE top-k, k if outside), de_gap (row max de_cos
                        - de_cos). Row-local: each record name is encoded alone (rank.dual_encoder).

    S9 kode15 "v2-tok" union arm (experiments/s9_robust/kode15/RESULT_15.md), default 0 (off = today's frame bit for bit):
    hangul_de2_k        Hangul rows only: the top-k of a second dual encoder (``dual_encoder2``, the v2-tok model whose
                        query carries [TON][FLAG][TYPE] side tokens) join the S9 extras exactly like the hangul_de_k
                        top-k (src_extra = 1, never LLM-scored), and de2_cos, de2_grank, de2_gap (same definitions,
                        second encoder) are appended last, after the de_* columns. NaN on Latin rows.

    S9 hs22 third Hangul dual encoder (experiments/s9_robust/hs22), default 0 (off = the same frame bit for bit):
    hangul_de3_k        Hangul rows only: the top-k of a third dual encoder (``dual_encoder3``, the charft10 architecture
                        trained with human-spelling pseudo queries) join the S9 extras like the hangul_de_k top-k
                        (src_extra = 1, never LLM-scored); de3_cos, de3_grank, de3_gap are appended after de2_* (before
                        ld_*). NaN on Latin rows.

    S9 latde13 Latin dual encoder (experiments/s9_robust/latde13/RESULT_13.md, l10 arm), default 0 (off = today's frame
    bit for bit):
    latin_de_k          non-Hangul rows only: the top-k of the Latin char dual encoder (``latin_dual_encoder``, payload
                        kind "latin") join the S9 extras like word_contain_k (src_extra = 1, never LLM-scored), and
                        ld_cos, ld_grank, ld_gap (the de_* definitions) are appended last, after de2_*. NaN on Hangul rows.
    latin_de_win_k      (with latin_de_k; default 0 = off) non-Hangul rows with a tonnage: also the top-k by Latin DE
                        cosine among the tonnage-window vessels (HangulMatcher.candidates, the window every row already
                        uses) join the extras, and ld_wrank (rank in that window top-k, k if outside, NaN without a
                        window) is appended after ld_gap. Targets one-word fragments shared by many vessels (TORM, STAR)
                        whose global DE top-k misses the right one.
    """

    def __init__(self, catalog: pd.DataFrame, n_gt: int = 3000, k_window: int = 40,
                 k_key: int = 20, gap_weight: float = 0.5, k_tr: int = 20, k_gt_exact: int = 20,
                 tr_cache: dict[str, list[str]] | None = None, k_typo: int = 20,
                 mv_digits: bool = False, num_words: bool = False, notonnage_net: bool = False,
                 k_global: int = 30, global_band: bool = False, band: float = 3.0,
                 band_gate: float = 0.1, num_forms: bool = False, hangul_global_k: int = 0,
                 latin_word_index_k: int = 0, latin_char_tfidf_k: int = 0, cs_fuzzy: bool = False,
                 cs_map: dict[str, list[int]] | None = None, hangul_num_v2: bool = False,
                 pair_feats4: bool = False, feats4_aux: dict | None = None, word_contain_k: int = 0,
                 hangul_wc2_k: int = 0, hangul_de_k: int = 0, dual_encoder=None,
                 hangul_de2_k: int = 0, dual_encoder2=None, latin_de_k: int = 0,
                 latin_dual_encoder=None, hangul_de3_k: int = 0, dual_encoder3=None,
                 latin_de_win_k: int = 0, latin_de2_k: int = 0, latin_dual_encoder2=None) -> None:
        self.catalog = catalog.reset_index(drop=True)
        self.base = CatalogIndex(self.catalog)
        self.matcher = HangulMatcher(self.catalog, n_gt=n_gt, gap_weight=gap_weight)
        self.norm = np.array(self.base.names, dtype=object)
        self.mv_digits, self.num_words = bool(mv_digits), bool(num_words)
        self.notonnage_net, self.k_global = bool(notonnage_net), int(k_global)
        self.global_band, self.band, self.band_gate = bool(global_band), float(band), float(band_gate)
        self._tr_flat_list = None  # readings as a list for the whole-catalog search (notonnage_net)
        self.num_forms = bool(num_forms)
        self.digits = np.array([self._name_digits(n) for n in self.catalog["vessel_name"]], dtype=object)
        if self.num_forms:  # catalog digit groups without leading zeros, for Hangul records (S5c)
            self.digits_nz = np.array([strip_leading_zeros(name_digit_groups(n, self.mv_digits, self.num_words))
                                       for n in self.catalog["vessel_name"]], dtype=object)
        self.k_window, self.k_key, self.gap_weight = k_window, k_key, gap_weight
        self.names = self.catalog["vessel_name"].fillna("").astype(str).to_numpy()
        # vessel name -> jamo of its rule-based Hangul readings (shippable with a submission)
        self.tr_cache: dict[str, list[str]] = tr_cache if tr_cache is not None else {}
        self.k_tr, self.k_gt_exact, self.k_typo = k_tr, k_gt_exact, k_typo
        self.norm_list = self.norm.tolist()
        self._tr_flat = None  # flattened readings of the whole catalog, built on first Hangul row
        self.key_index: dict[str, list[int]] = defaultdict(list)
        for pos, (k0, k1) in enumerate(zip(self.matcher.key0, self.matcher.key1)):
            for k in {k0, k1}:
                if len(k) >= 2:
                    self.key_index[k].append(pos)
        self.hangul_global_k, self.latin_word_index_k = int(hangul_global_k), int(latin_word_index_k)
        self.latin_char_tfidf_k, self.cs_fuzzy = int(latin_char_tfidf_k), bool(cs_fuzzy)
        if self.cs_fuzzy and cs_map is None:
            raise ValueError("cs_fuzzy needs cs_map")
        self.cs_map = dict(cs_map or {})
        self._cs_keys = list(self.cs_map)
        self.word_contain_k = int(word_contain_k)
        self._wc_ix = None  # WordContainIndex, built on first use
        self.hangul_wc2_k = int(hangul_wc2_k)
        self._wc2_ix = None  # thr-0.75 WordContainIndex + glued pairs (hangul_wc2_k), built on first use
        self.hangul_de_k = int(hangul_de_k)
        self.dual_encoder = dual_encoder if self.hangul_de_k else None
        if self.hangul_de_k:
            if dual_encoder is None:
                raise ValueError("hangul_de_k needs dual_encoder")
            dual_encoder.check_catalog(self.catalog)
        self.hangul_de2_k = int(hangul_de2_k)
        self.dual_encoder2 = dual_encoder2 if self.hangul_de2_k else None
        if self.hangul_de2_k:
            if dual_encoder2 is None:
                raise ValueError("hangul_de2_k needs dual_encoder2")
            dual_encoder2.check_catalog(self.catalog)
        self.hangul_de3_k = int(hangul_de3_k)
        self.dual_encoder3 = dual_encoder3 if self.hangul_de3_k else None
        if self.hangul_de3_k:
            if dual_encoder3 is None:
                raise ValueError("hangul_de3_k needs dual_encoder3")
            dual_encoder3.check_catalog(self.catalog)
        self.latin_de_win_k = int(latin_de_win_k)
        self.latin_de2_k = int(latin_de2_k)
        self.latin_dual_encoder2 = latin_dual_encoder2 if self.latin_de2_k else None
        if self.latin_de2_k:
            if latin_dual_encoder2 is None or not latin_dual_encoder2.latin:
                raise ValueError("latin_de2_k needs a Latin latin_dual_encoder2")
            latin_dual_encoder2.check_catalog(self.catalog)
        self.latin_de_k = int(latin_de_k)
        self.latin_dual_encoder = latin_dual_encoder if self.latin_de_k else None
        if self.latin_de_k:
            if latin_dual_encoder is None or not latin_dual_encoder.latin:
                raise ValueError("latin_de_k needs a Latin latin_dual_encoder")
            latin_dual_encoder.check_catalog(self.catalog)
        self._s9 = bool(self.hangul_global_k or self.latin_word_index_k or self.latin_char_tfidf_k or self.cs_fuzzy
                        or self.word_contain_k or self.hangul_wc2_k or self.hangul_de_k or self.hangul_de2_k
                        or self.latin_de_k or self.hangul_de3_k or self.latin_de2_k)
        self._word_ix = self._char_ix = None  # sparse indexes, built on first use
        self.hangul_num_v2 = bool(hangul_num_v2)
        if self.hangul_num_v2:
            self.digits_v2 = np.array([name_digits_v2(n) for n in self.catalog["vessel_name"]], dtype=object)
        self.pair_feats4 = bool(pair_feats4)
        self.feats4_aux = feats4_aux
        if (self.pair_feats4 or self.word_contain_k or self.hangul_wc2_k) and self.feats4_aux is None:
            from rank.pair_feats4 import build_aux
            self._catalog_readings()  # completes tr_cache for every catalog name
            self.feats4_aux = build_aux(self.catalog, self.tr_cache)

    def _name_digits(self, name) -> str:
        if self.mv_digits or self.num_words:
            return name_digits_ext(name, mv_fix=self.mv_digits, num_words=self.num_words)
        return name_digits(name)

    def digit_state(self, name, cands: np.ndarray, tr: np.ndarray | None = None) -> np.ndarray:
        """0 no digits on either side, 1 same digits, 2 candidate has none, 3 record has none, 4 differ.

        num_forms, Hangul record: digits read by translit.name_digit_groups_forms, and when the record's digit
        groups carry no leading zero, a candidate whose digits equal them only without leading zeros (셀바라 2 vs
        Selvara 02) gets state 1 if its tr_sim (``tr``, row-local) is >= the best tr_sim among the row's exact-digit
        candidates (always when there is none; never when an exact candidate exists and ``tr`` is not given)."""
        if (self.num_forms or self.hangul_num_v2) and is_hangul_name(name):
            groups = name_digit_groups_forms(name) if self.num_forms else []
            if self.hangul_num_v2 and not groups:
                groups = [g for g in [name_digits_v2(name)] if g]
            qd = "".join(groups)
            ds = (self.digits_v2 if self.hangul_num_v2 else self.digits)[cands]
            st = np.array([0 if not qd and not d else 1 if qd == d else 2 if not d else 3 if not qd else 4
                           for d in ds], dtype=float)
            if self.num_forms and qd and not any(len(g) > 1 and g.startswith("0") for g in groups):
                zero_eq = (self.digits_nz[cands] == strip_leading_zeros(groups)) & (ds != qd)
                exact = ds == qd
                if zero_eq.any() and exact.any():
                    if tr is None:
                        zero_eq[:] = False
                    else:
                        tr = np.asarray(tr, dtype=float)
                        best = np.nanmax(np.where(exact, tr, np.nan)) if np.isfinite(tr[exact]).any() else -np.inf
                        zero_eq &= np.nan_to_num(tr, nan=-np.inf) >= best
                st[zero_eq] = 1.0
            return st
        qd = self._name_digits(name)
        return np.array([0 if not qd and not d else 1 if qd == d else 2 if not d else 3 if not qd else 4
                         for d in self.digits[cands]], dtype=float)

    def recompute_digit_state(self, pairs: pd.DataFrame, queries: pd.DataFrame) -> np.ndarray:
        """digit_state of cached pairs under this generator's digit options (pairs['q'] indexes queries;
        pairs['tr_sim'] feeds the num_forms leading-zero rule)."""
        names = queries["vessel_name"].tolist()
        out = np.empty(len(pairs))
        q = pairs["q"].to_numpy()
        c = pairs["c"].to_numpy()
        tr = pairs["tr_sim"].to_numpy(float) if "tr_sim" in pairs else None
        starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
        for a, b in zip(starts, np.r_[starts[1:], len(q)]):
            out[a:b] = self.digit_state(names[q[a]], c[a:b], None if tr is None else tr[a:b])
        return out

    def _global_tr(self, query: str, tvals: list[float] | None = None, floor: float | None = None,
                   k: int | None = None) -> np.ndarray:
        """Top-k_global catalog positions by transliteration similarity over the WHOLE catalog
        (port of experiments/diag/recall/global_index.py GlobalReadingIndex.tr_sims + topk_desc).

        ``tvals`` (global_band): positive record tonnages; only vessels with min |log(GT / t)| <= log(band), or
        without a catalog GT, are eligible. ``floor`` (global_band): hits below this similarity are dropped.
        Both None (notonnage_net): the plain whole-catalog top-k. ``k``: None = k_global (hangul_global_k)."""
        flat, start, _ = self._catalog_readings()
        if self._tr_flat_list is None:
            self._tr_flat_list = flat.tolist()
        if _rf_process is not None:
            from rapidfuzz.distance import Levenshtein
            sims = _rf_process.cdist([query], self._tr_flat_list, scorer=Levenshtein.normalized_similarity,
                                     dtype=np.float32, workers=1)[0]
        else:  # pragma: no cover - server without rapidfuzz
            sims = _sim_matrix([query], self._tr_flat_list)[0].astype(np.float32)
        sims = np.maximum.reduceat(sims, start)
        if not tvals and floor is None:
            return topk_desc(sims, self.k_global if k is None else k)
        if tvals:
            lg = self.matcher.log_gt
            gap = np.min([np.abs(lg - np.log(v)) for v in tvals], axis=0)
            sims = np.where(np.isnan(lg) | (gap <= np.log(self.band)), sims, np.float32(-1.0))
        top = topk_desc(sims, self.k_global)
        top = top[sims[top] >= 0]
        if floor is not None:
            top = top[sims[top] >= floor]
        return top

    def _transliterations(self, pos: int) -> list[str]:
        name = self.names[pos]
        hit = self.tr_cache.get(name)
        if hit is None:
            hit = [to_jamo(v.replace(" ", "")) for v in name_variants(name, max_variants=8)] or [""]
            self.tr_cache[name] = hit
        return hit

    def _catalog_readings(self):
        """Readings of every catalog vessel, flattened for vectorized window scoring."""
        if self._tr_flat is None:
            per_pos = [self._transliterations(p) for p in range(len(self.names))]
            lens = np.array([len(v) for v in per_pos])
            self._tr_flat = (np.array([x for v in per_pos for x in v], dtype=object),
                             np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
        return self._tr_flat

    def _tr_window(self, query: str, cands: np.ndarray) -> np.ndarray:
        """Max reading similarity per candidate, for a whole tonnage window at once."""
        flat, start, lens = self._catalog_readings()
        n = lens[cands]
        idx = np.repeat(start[cands], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        sims = _sim_matrix([query], flat[idx].tolist())[0]
        return np.maximum.reduceat(sims, np.cumsum(n) - n)

    def _tr_similarity(self, row: dict, cands: np.ndarray) -> np.ndarray:
        """Jamo similarity of the Hangul record name to rule-based readings of each candidate."""
        query = to_jamo(hangul_core(row.get("vessel_name")))
        if not query:
            return np.full(len(cands), np.nan)
        flat, owner = [], []
        for j, c in enumerate(cands):
            for v in self._transliterations(int(c)):
                flat.append(v)
                owner.append(j)
        sims = _sim_matrix([query], flat)[0]
        out = np.zeros(len(cands))
        np.maximum.at(out, np.array(owner), sims)
        return out

    # ---- per-row ---------------------------------------------------------
    def _similarity(self, row: dict, cands: np.ndarray, hangul: bool, qnorm: str):
        ph = self.matcher.name_similarity(row, cands)
        if hangul or not qnorm:
            return ph, np.full(len(cands), np.nan)
        nm = _sim_matrix([qnorm], self.norm[cands].tolist())[0]
        return ph, nm

    def _heuristic(self, ph, nm, gap):
        sim = np.fmax(ph, np.nan_to_num(nm, nan=0.0))
        return sim - self.gap_weight * np.minimum(np.nan_to_num(gap, nan=1.0), 1.0)

    def row_pairs(self, row: dict) -> pd.DataFrame:
        de = None
        if self.hangul_de_k and is_hangul_name(row.get("vessel_name")):
            q, cat = self.dual_encoder.query(row)
            sims = cat @ q
            de = (topk_desc(sims, self.hangul_de_k), sims)
        de2 = None
        if self.hangul_de2_k and is_hangul_name(row.get("vessel_name")):
            q, cat = self.dual_encoder2.query(row)
            sims = cat @ q
            de2 = (topk_desc(sims, self.hangul_de2_k), sims)
        de3 = None
        if self.hangul_de3_k and is_hangul_name(row.get("vessel_name")):
            q, cat = self.dual_encoder3.query(row)
            sims = cat @ q
            de3 = (topk_desc(sims, self.hangul_de3_k), sims)
        ld = None
        if self.latin_de_k and not is_hangul_name(row.get("vessel_name")):
            q, cat = self.latin_dual_encoder.query(row)
            sims = cat @ q
            ld = (topk_desc(sims, self.latin_de_k), sims)
        ld2 = None
        if self.latin_de2_k and not is_hangul_name(row.get("vessel_name")):
            q, cat = self.latin_dual_encoder2.query(row)
            sims = cat @ q
            ld2 = (topk_desc(sims, self.latin_de2_k), sims)
        ld_win = None
        if ld is not None and self.latin_de_win_k:
            win, _ = self.matcher.candidates(row)
            if len(win):
                ld_win = win[np.lexsort((win, -ld[1][win]))[: self.latin_de_win_k]]
        tops = [d[0] for d in (de, de2, de3) if d is not None]
        lts = [x for x in (None if ld is None else ld[0], ld_win, None if ld2 is None else ld2[0]) if x is not None]
        ld_top = np.concatenate(lts) if lts else None
        frame = self._row_pairs(row, np.concatenate(tops) if tops else None, ld_top)
        if self.pair_feats4:
            from rank.pair_feats4 import frame_feats
            F = frame_feats(row, frame, self.feats4_aux)
            for col in F.columns:
                frame[col] = F[col].to_numpy()
        if self.hangul_de_k:
            self._de_columns(frame, de)
        if self.hangul_de2_k:
            self._de_columns(frame, de2, "de2")
        if self.hangul_de3_k:
            self._de_columns(frame, de3, "de3")
        if self.latin_de_k:
            self._de_columns(frame, ld, "ld")
            if self.latin_de_win_k:
                if ld_win is None:
                    frame["ld_wrank"] = np.full(len(frame), np.nan, dtype=np.float32)
                else:
                    wr = {int(x): i for i, x in enumerate(ld_win)}
                    frame["ld_wrank"] = np.array([wr.get(int(x), len(ld_win)) for x in frame["c"].to_numpy()],
                                                 dtype=np.float32)
        if self.latin_de2_k:
            self._de_columns(frame, ld2, "ld2")
        return frame

    def _de_columns(self, frame: pd.DataFrame, de, prefix: str = "de") -> None:
        """hangul_de_k columns (last): de_cos, de_grank, de_gap over the row's full frame; NaN on Latin rows.
        ``prefix`` "de2": the hangul_de2_k columns."""
        n = len(frame)
        if de is None:
            for col in ("cos", "grank", "gap"):
                frame[f"{prefix}_{col}"] = np.full(n, np.nan, dtype=np.float32)
            return
        top, sims = de
        c = frame["c"].to_numpy()
        cos = sims[c].astype(np.float32)
        rank = {int(p): i for i, p in enumerate(top)}
        frame[f"{prefix}_cos"] = cos
        frame[f"{prefix}_grank"] = np.array([rank.get(int(x), len(top)) for x in c], dtype=np.float32)
        frame[f"{prefix}_gap"] = (cos.max() - cos).astype(np.float32)

    def _row_pairs(self, row: dict, de_top: np.ndarray | None = None,
                   ld_top: np.ndarray | None = None) -> pd.DataFrame:
        name = row.get("vessel_name")
        hangul = is_hangul_name(name)
        digit_name = name
        if self.hangul_num_v2 and hangul:  # name part only for every similarity; digits read from the full name
            stripped = strip_number_words(name)
            if stripped != name:
                name = stripped
                row = {**row, "vessel_name": stripped}
        qnorm = normalize_key(name)
        qkeys = phonetic_keys(name)

        sources: dict[int, int] = defaultdict(int)  # catalog pos -> bit flags
        win, gap = self.matcher.candidates(row)
        win_best = -1.0  # global_band gate: best transliteration similarity inside the tonnage window
        if len(win):
            ph, nm = self._similarity(row, win, hangul, qnorm)
            heur = self._heuristic(ph, nm, gap)
            for c in win[np.lexsort((win, -heur))[: self.k_window]]:
                sources[int(c)] |= 1
            query = to_jamo(hangul_core(name)) if hangul else ""
            if query:
                # Best rule-based Hangul readings in the window, and exact-tonnage vessels:
                # the old phonetic heuristic alone missed e.g. 신간 -> Singan (exact tonnage).
                trw = self._tr_window(query, win)
                if self.global_band:
                    win_best = float(trw.max())
                for c in win[np.lexsort((win, -trw))[: self.k_tr]]:
                    sources[int(c)] |= 16
                exact = gap < 1e-6
                if exact.any():
                    ex, ex_tr = win[exact], trw[exact]
                    for c in ex[np.lexsort((ex, -ex_tr))[: self.k_gt_exact]]:
                        sources[int(c)] |= 32
        if not hangul:
            for c in self.base.candidates(row):
                sources[int(c)] |= 2
            if qnorm and _rf_process is not None and self.k_typo:
                # Misspelled names with no tonnage (KETSRAVEN -> Kestraven) never reach the
                # n-gram/tonnage sources; OSA distance counts a transposition as one edit.
                for _, _, c in _rf_process.extract(qnorm, self.norm_list, scorer=_OSA.normalized_similarity,
                                                   limit=self.k_typo, score_cutoff=0.6):
                    sources[int(c)] |= 64
        hits = sorted({p for k in qkeys for p in self.key_index.get(k, ())})
        net_query = ""
        if (self.notonnage_net and hangul and _positive(row.get("gross_tonnage")) is None
                and _positive(row.get("international_gross_tonnage")) is None) or (self.global_band and hangul):
            # C4.4: a Hangul row without any positive tonnage. Rows with a tonnage never get here.
            # S5b global_band: every Hangul row (key re-sort + banded / gated global search below).
            net_query = to_jamo(hangul_core(name))
            if net_query and len(hits) > self.k_key:
                h = np.array(hits)
                hits = h[np.lexsort((h, -self._tr_similarity(row, h)))].tolist()
        hits = hits[: self.k_key]
        for c in hits:
            sources[int(c)] |= 4
        if net_query:
            if self.global_band:
                tvals = [v for v in (_positive(row.get("gross_tonnage")),
                                     _positive(row.get("international_gross_tonnage"))) if v]
                glob = self._global_tr(net_query, tvals, max(0.0, win_best - self.band_gate))
            else:
                glob = self._global_tr(net_query)
            for c in glob:
                sources[int(c)] |= 128
        if not sources:
            sources[0] |= 8
        frame = self._pair_frame(row, name, hangul, qnorm, qkeys, np.array(sorted(sources), dtype=int), sources,
                                 digit_name)
        if not self._s9:
            return frame
        return self._add_s9(row, name, hangul, qnorm, qkeys, frame, sources, digit_name, de_top, ld_top)

    def _pair_frame(self, row: dict, name, hangul: bool, qnorm: str, qkeys, cands: np.ndarray,
                    sources: dict[int, int], digit_name) -> pd.DataFrame:
        """Feature frame of ``cands`` (sorted catalog positions); row-relative columns over ``cands``."""
        ph, nm = self._similarity(row, cands, hangul, qnorm)
        values = [v for v in (_positive(row.get("gross_tonnage")),
                              _positive(row.get("international_gross_tonnage"))) if v]
        log_c = self.matcher.log_gt[cands]
        lg_gross = np.abs(log_c - np.log(values[0])) if values else np.full(len(cands), np.nan)
        lg_min = (np.min([np.abs(log_c - np.log(v)) for v in values], axis=0)
                  if values else np.full(len(cands), np.nan))
        cgt = np.exp(log_c)
        gt_eq = np.zeros(len(cands))
        for v in values:
            gt_eq = np.maximum(gt_eq, np.isclose(cgt, v, rtol=0, atol=0.5))
        heur = self._heuristic(ph, nm, lg_min)

        q0 = qkeys[0]
        k0 = self.matcher.key0[cands]
        base = np.array([self.base.pair_features(row, int(c)) for c in cands], dtype=float)

        frame = pd.DataFrame(base, columns=BASE_FEATURES)
        frame.insert(0, "c", cands)
        flags = np.array([sources[int(c)] for c in cands])
        frame["src_window"] = (flags & 1) > 0
        frame["src_base"] = (flags & 2) > 0
        frame["src_key"] = (flags & 4) > 0
        frame["src_tr"] = (flags & 16) > 0
        frame["src_gt_exact"] = (flags & 32) > 0
        frame["src_typo"] = (flags & 64) > 0
        if hangul or not qnorm or _OSA is None:
            frame["osa_sim"] = np.nan
            frame["word_overlap"] = np.nan
            frame["last_word_eq"] = np.nan
        else:
            frame["osa_sim"] = [_OSA.normalized_similarity(qnorm, self.norm[c]) for c in cands]
            qw = _name_words(name)
            cw = [_name_words(self.names[c]) for c in cands]
            frame["word_overlap"] = [len(set(qw) & set(w)) / max(len(set(qw)), len(set(w)), 1) for w in cw]
            frame["last_word_eq"] = [float(bool(qw) and bool(w) and qw[-1] == w[-1]) for w in cw]
        frame["is_hangul"] = hangul
        frame["ph_sim"] = ph
        frame["ph_exact"] = ph >= 1.0
        frame["ph_prefix"] = [_prefix_ratio(q0, k) for k in k0]
        frame["q_key_len"] = len(q0)
        frame["c_key_len"] = [len(k) for k in k0]
        frame["nm_sim"] = nm
        frame["tr_sim"] = self._tr_similarity(row, cands) if hangul else np.nan
        frame["tr_rank"] = (tr_rank(cands, frame["tr_sim"].to_numpy()) if hangul
                            else np.full(len(cands), 999))
        frame["lg_gross"] = lg_gross
        frame["lg_min"] = lg_min
        frame["gt_eq"] = gt_eq
        frame["digit_state"] = self.digit_state(digit_name, cands,
                                                frame["tr_sim"].to_numpy() if self.num_forms else None)
        frame["heur"] = heur
        order = np.argsort(np.lexsort((cands, -heur)))
        frame["heur_rank"] = order
        frame["heur_gap_best"] = heur.max() - heur
        frame["n_cands"] = len(cands)
        # Row-relative views: how far each candidate is from the row's best on each signal.
        for col in ("tr_sim", "ph_sim", "nm_sim"):
            v = frame[col].to_numpy(dtype=float)
            frame[f"{col}_gap_best"] = (np.nanmax(v) - v) if np.isfinite(v).any() else np.nan
        gap = np.nan_to_num(lg_min, nan=9.0)
        frame["gap_rank"] = np.argsort(np.lexsort((cands, gap)))
        frame["tr_top_gt_eq"] = ((frame["tr_rank"] == 0) & (frame["gt_eq"] > 0)).astype(np.int8)
        return frame

    # ---- S9 retrieval options (all default off) ----------------------------
    def _sparse_index(self, kind: str):
        """(vocab, idf, L2-normalised binary-tf x idf matrix as CSC) over the catalog names; built once."""
        attr = "_word_ix" if kind == "word" else "_char_ix"
        if getattr(self, attr) is None:
            from scipy import sparse
            docs = ([set(_name_words(n)) for n in self.names] if kind == "word"
                    else [_trigrams(k) for k in self.norm_list])
            vocab: dict[str, int] = {}
            rows, cols = [], []
            for pos, terms in enumerate(docs):
                for t in terms:
                    cols.append(vocab.setdefault(t, len(vocab)))
                    rows.append(pos)
            n = len(docs)
            m = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(vocab)))
            df = np.asarray(m.sum(axis=0)).ravel()
            idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
            m = m.multiply(idf[None, :]).tocsr()
            norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
            m = sparse.diags(1.0 / np.where(norm > 0, norm, 1.0)) @ m
            setattr(self, attr, (vocab, idf, np.log(1.0 + n) + 1.0, m.tocsc()))
        return getattr(self, attr)

    def _sparse_scores(self, kind: str, terms: set[str]) -> np.ndarray | None:
        """Cosine of the record's terms with every catalog name (None: the record has no terms)."""
        if not terms:
            return None
        vocab, idf, idf_oov, m = self._sparse_index(kind)
        known = sorted(vocab[t] for t in terms if t in vocab)
        qnorm = np.sqrt(np.sum(idf[known] ** 2) + (len(terms) - len(known)) * idf_oov ** 2)
        if not known:
            return np.zeros(m.shape[0])
        return np.asarray(m[:, known] @ idf[known]).ravel() / qnorm

    def _wc2_index(self) -> WordContainIndex:
        """hangul_wc2_k index (thr 0.75 + glued pairs), built once per process on first use (about 50 s, mostly
        en2ko.word_variants of the catalog tokens; the parallel build's workers each build their own, overlapping
        the LLM load)."""
        if self._wc2_ix is None:
            self._wc2_ix = WordContainIndex(self.feats4_aux, thr=0.75)
            self._wc2_ix.add_glued_pairs(self.names)
        return self._wc2_ix

    def _add_s9(self, row: dict, name, hangul: bool, qnorm: str, qkeys, frame: pd.DataFrame,
                sources: dict[int, int], digit_name, de_top: np.ndarray | None = None,
                ld_top: np.ndarray | None = None) -> pd.DataFrame:
        """Append the S9 sources' extra candidates to the option-off ``frame`` (kept as is) + the new columns."""
        hits: dict[int, int] = defaultdict(int)
        sims: dict[str, np.ndarray | None] = {}
        if hangul and self.hangul_global_k:
            query = to_jamo(hangul_core(name))
            if query:
                for c in self._global_tr(query, k=self.hangul_global_k):
                    hits[int(c)] |= 256
        latin = not hangul and bool(qnorm)
        for col, k, kind, bit, terms in (("word_cos", self.latin_word_index_k, "word", 512,
                                          lambda: set(_name_words(name))),
                                         ("tfidf_sim", self.latin_char_tfidf_k, "char", 1024,
                                          lambda: _trigrams(qnorm))):
            if not k:
                continue
            s = self._sparse_scores(kind, terms()) if latin else None
            sims[col] = s
            if s is not None:
                top = topk_desc(s, k)
                for c in top[s[top] > 0]:
                    hits[int(c)] |= bit
        if self.word_contain_k:
            if self._wc_ix is None:
                self._wc_ix = WordContainIndex(self.feats4_aux)
            for c in self._wc_ix.topk(row, hangul, self.word_contain_k):
                hits[int(c)] |= 4096
        if self.hangul_wc2_k and hangul:
            for c in self._wc2_index().topk(row, True, self.hangul_wc2_k):
                hits[int(c)] |= 4096
        if de_top is not None and hangul:  # hangul_de_k (+ hangul_de2_k): DE top-k, like word_contain_k
            for c in de_top:
                hits[int(c)] |= 4096
        if ld_top is not None and not hangul:  # latin_de_k: Latin DE top-k, like word_contain_k
            for c in ld_top:
                hits[int(c)] |= 4096
        if self.cs_fuzzy:
            cs = normalize_key(row.get("call_sign"))
            if cs and self._cs_keys:
                if _rf_process is not None:
                    from rapidfuzz.distance import Levenshtein
                    keys = [k for k, _, _ in _rf_process.extract(cs, self._cs_keys, scorer=Levenshtein.distance,
                                                                score_cutoff=1, limit=None)]
                else:  # pragma: no cover - server without rapidfuzz: exact call sign only
                    keys = [cs] if cs in self.cs_map else []
                for key in keys:
                    for c in self.cs_map[key]:
                        hits[int(c)] |= 2048
        ext = np.array(sorted(p for p in hits if p not in sources), dtype=int)
        out = frame
        if len(ext):
            fe = self._pair_frame(row, name, hangul, qnorm, qkeys, ext, hits, digit_name)
            self._relative_to_base(fe, frame, digit_name, hangul)
            out = pd.concat([frame, fe], ignore_index=True)
            out = out.iloc[np.argsort(out["c"].to_numpy(), kind="stable")].reset_index(drop=True)
        c = out["c"].to_numpy()
        bits = np.array([hits.get(int(x), 0) for x in c])
        out["src_extra"] = np.isin(c, ext)
        for col, on, bit in (("src_hglobal", self.hangul_global_k, 256), ("src_lword", self.latin_word_index_k, 512),
                             ("src_ltfidf", self.latin_char_tfidf_k, 1024), ("src_cs", self.cs_fuzzy, 2048)):
            if on:
                out[col] = (bits & bit) > 0
        for col in ("word_cos", "tfidf_sim"):
            if col in sims:
                out[col] = np.nan if sims[col] is None else sims[col][c]
        return out

    def _relative_to_base(self, fe: pd.DataFrame, base: pd.DataFrame, name, hangul: bool) -> None:
        """Row-relative columns of the extra candidates ``fe``: position in the union, gaps to the base set's best,
        n_cands of the base set (the base rows keep their option-off values)."""
        nb = len(base)
        call = np.r_[base["c"].to_numpy(), fe["c"].to_numpy()]
        heur = np.r_[base["heur"].to_numpy(float), fe["heur"].to_numpy(float)]
        fe["heur_rank"] = np.argsort(np.lexsort((call, -heur)))[nb:]
        tr_all = np.r_[base["tr_sim"].to_numpy(float), fe["tr_sim"].to_numpy(float)]
        if hangul:
            fe["tr_rank"] = tr_rank(call, tr_all)[nb:]
        if (self.num_forms or self.hangul_num_v2) and is_hangul_name(name):  # leading-zero rule reads row tr_sim
            fe["digit_state"] = self.digit_state(name, call, tr_all)[nb:]
        fe["heur_gap_best"] = base["heur"].max() - fe["heur"]
        fe["n_cands"] = nb
        for col in ("tr_sim", "ph_sim", "nm_sim"):
            v = base[col].to_numpy(dtype=float)
            fe[f"{col}_gap_best"] = (np.nanmax(v) - fe[col].to_numpy(dtype=float)) if np.isfinite(v).any() else np.nan
        gap = np.nan_to_num(np.r_[base["lg_min"].to_numpy(float), fe["lg_min"].to_numpy(float)], nan=9.0)
        fe["gap_rank"] = np.argsort(np.lexsort((call, gap)))[nb:]
        fe["tr_top_gt_eq"] = ((fe["tr_rank"] == 0) & (fe["gt_eq"] > 0)).astype(np.int8)

    def build(self, queries: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for q, row in enumerate(queries.to_dict("records")):
            f = self.row_pairs(row)
            f.insert(0, "q", q)
            frames.append(f)
        return concat_pairs(frames)


def concat_pairs(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Per-row frames (with their "q" column, in row order) -> the pairs frame CandidateGenerator.build returns."""
    out = pd.concat(frames, ignore_index=True)
    bool_cols = out.select_dtypes(bool).columns
    out[bool_cols] = out[bool_cols].astype(np.int8)
    return out


# ---------------------------------------------------------------------------
# Row-parallel build (S8 speed path, models/ranked_llm.py art["fast"]). row_pairs depends only on
# (row, catalog, options, tr_cache), so splitting the rows over worker processes and concatenating the
# frames in row order gives exactly CandidateGenerator.build's output. 'fork' (Linux): workers inherit the
# generator built in the parent; 'spawn' (Windows): each worker rebuilds it from (catalog, tr_cache, options).
# ---------------------------------------------------------------------------
_WORKER_GEN: CandidateGenerator | None = None


def _worker_init(gen: CandidateGenerator | None, spawn_args: tuple | None) -> None:
    global _WORKER_GEN
    if gen is not None:
        _WORKER_GEN = gen
    else:
        catalog, tr_cache, opts = spawn_args
        _WORKER_GEN = CandidateGenerator(catalog, tr_cache=tr_cache, **opts)


def _worker_rows(task: tuple) -> list[pd.DataFrame]:
    """(start, rows): rows start, start+1, ... ; (row indices, rows, Latin query embeddings) for the split build
    (the parent's latin_dual_encoder embeddings of these rows, added to the worker's encoder cache first)."""
    if len(task) == 3:
        idx, rows, emb = task
        for attr, e in (emb or {}).items():  # {encoder attribute: {cache key: embedding}}
            getattr(_WORKER_GEN, attr)._q.update(e)
    else:
        start, rows = task
        idx = range(start, start + len(rows))
    frames = []
    for i, row in zip(idx, rows):
        f = _WORKER_GEN.row_pairs(row)
        f.insert(0, "q", i)
        frames.append(f)
    return frames


def start_parallel_build(gen: CandidateGenerator, queries: pd.DataFrame, n_jobs: int, spawn_opts: dict,
                         chunk: int = 16):
    """Start gen.build(queries) on ``n_jobs`` worker processes without waiting; returns (pool, async result).

    ``spawn_opts``: the CandidateGenerator keyword options of ``gen`` (used only with 'spawn'; with pair_feats4 the
    generator's feats4_aux is passed along automatically). Call this before
    CUDA is initialised in the parent (fork). Collect with finish_parallel_build.
    hangul_de_k: every Hangul record is encoded HERE (gen.dual_encoder.prepare, one row at a time, in the parent)
    before the pool starts; fork workers inherit the embeddings and spawn workers receive them with the encoder, so
    no worker runs torch (no fork-after-OpenMP hazard).
    """
    rows = queries.to_dict("records")
    tasks = [(s, rows[s: s + chunk]) for s in range(0, len(rows), chunk)]
    if getattr(gen, "hangul_de_k", 0):  # encode every Hangul record here, before the fork: workers never run torch
        gen.dual_encoder.prepare(rows)
    if getattr(gen, "hangul_de2_k", 0):
        gen.dual_encoder2.prepare(rows)
    if getattr(gen, "hangul_de3_k", 0):
        gen.dual_encoder3.prepare(rows)
    if getattr(gen, "latin_de_k", 0):
        gen.latin_dual_encoder.prepare(rows)
    if getattr(gen, "latin_de2_k", 0):
        gen.latin_dual_encoder2.prepare(rows)
    pool = _make_pool(gen, n_jobs, spawn_opts)
    try:
        return pool, pool.map_async(_worker_rows, tasks)
    except BaseException:
        pool.terminate()
        raise


def _make_pool(gen: CandidateGenerator, n_jobs: int, spawn_opts: dict):
    import multiprocessing as mp
    if "fork" in mp.get_all_start_methods():
        pool = mp.get_context("fork").Pool(n_jobs, initializer=_worker_init, initargs=(gen, None))
    else:
        if getattr(gen, "pair_feats4", False):  # spawn workers get the parent's aux table instead of rebuilding it
            spawn_opts = {**spawn_opts, "pair_feats4": True, "feats4_aux": gen.feats4_aux}
        elif getattr(gen, "word_contain_k", 0) or getattr(gen, "hangul_wc2_k", 0):  # same table, word-containment
            spawn_opts = {**spawn_opts, "feats4_aux": gen.feats4_aux}
        if getattr(gen, "hangul_de_k", 0):  # the encoder with the parent's prepared query embeddings
            spawn_opts = {**spawn_opts, "dual_encoder": gen.dual_encoder}
        if getattr(gen, "hangul_de2_k", 0):
            spawn_opts = {**spawn_opts, "dual_encoder2": gen.dual_encoder2}
        if getattr(gen, "hangul_de3_k", 0):
            spawn_opts = {**spawn_opts, "dual_encoder3": gen.dual_encoder3}
        if getattr(gen, "latin_de_k", 0):
            spawn_opts = {**spawn_opts, "latin_dual_encoder": gen.latin_dual_encoder}
        if getattr(gen, "latin_de2_k", 0):
            spawn_opts = {**spawn_opts, "latin_dual_encoder2": gen.latin_dual_encoder2}
        pool = mp.get_context("spawn").Pool(n_jobs, initializer=_worker_init,
                                            initargs=(None, (gen.catalog, dict(gen.tr_cache), spawn_opts)))
    return pool


def start_split_build(gen: CandidateGenerator, queries: pd.DataFrame, n_jobs: int, spawn_opts: dict,
                      chunk: int = 16) -> dict:
    """S9 speed path (art["fast"]["split"]): start_parallel_build with the Hangul rows queued FIRST, so their pairs
    (the only ones the LLM scores) are ready while the workers still build the Latin rows. The Hangul dual encoders
    run before the pool starts (as in start_parallel_build); the Latin encoder runs here AFTER the pool starts, one
    chunk at a time, and each Latin task carries its rows' embeddings (computed exactly as prepare() does), so no
    worker runs torch and the Latin encoding overlaps the Hangul candidates. Every row is still built alone by
    row_pairs: finish_split_build returns exactly gen.build(queries)."""
    from common import is_hangul_name
    rows = queries.to_dict("records")
    ko = [i for i, r in enumerate(rows) if is_hangul_name(r.get("vessel_name"))]
    la = [i for i, r in enumerate(rows) if not is_hangul_name(r.get("vessel_name"))]
    if getattr(gen, "hangul_de_k", 0):
        gen.dual_encoder.prepare([rows[i] for i in ko])
    if getattr(gen, "hangul_de2_k", 0):
        gen.dual_encoder2.prepare([rows[i] for i in ko])
    if getattr(gen, "hangul_de3_k", 0):
        gen.dual_encoder3.prepare([rows[i] for i in ko])
    pool = _make_pool(gen, n_jobs, spawn_opts)
    try:
        h = pool.map_async(_worker_rows, [(ko[s: s + chunk], [rows[i] for i in ko[s: s + chunk]], None)
                                          for s in range(0, len(ko), chunk)])
        encs = [a for a, k in (("latin_dual_encoder", "latin_de_k"), ("latin_dual_encoder2", "latin_de2_k"))
                if getattr(gen, k, 0)]
        lat = []
        for s in range(0, len(la), chunk):
            idx = la[s: s + chunk]
            part = [rows[i] for i in idx]
            emb = {a: getattr(gen, a).embeddings(part) for a in encs} or None
            lat.append(pool.apply_async(_worker_rows, ((idx, part, emb),)))
    except BaseException:
        pool.terminate()
        raise
    return {"pool": pool, "hangul": h, "latin": lat}


def split_hangul_pairs(handle: dict, timeout: float | None = None) -> pd.DataFrame:
    """The Hangul rows' pairs (q = row index in the full query frame), as soon as their tasks are done."""
    try:
        frames = [f for part in handle["hangul"].get(timeout) for f in part]
    except BaseException:
        handle["pool"].terminate()
        raise
    handle["hangul_frames"] = frames
    return concat_pairs(frames) if frames else pd.DataFrame()


def finish_split_build(handle: dict, timeout: float | None = None) -> pd.DataFrame:
    """Every row's frame, in row order: the same frame as gen.build(queries)."""
    import time as _time
    pool = handle["pool"]
    try:
        t_end = None if timeout is None else _time.monotonic() + timeout
        left = lambda: None if t_end is None else max(1.0, t_end - _time.monotonic())  # noqa: E731
        frames = handle.get("hangul_frames")
        if frames is None:
            frames = [f for part in handle["hangul"].get(left()) for f in part]
        frames = list(frames) + [f for r in handle["latin"] for f in r.get(left())]
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        raise
    frames.sort(key=lambda f: int(f["q"].iat[0]))
    return concat_pairs(frames)


def finish_parallel_build(handle, timeout: float | None = None) -> pd.DataFrame:
    """Wait for start_parallel_build's workers; same frame as gen.build(queries). Raises on any worker error."""
    pool, res = handle
    try:
        frames = [f for part in res.get(timeout) for f in part]
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        raise
    return concat_pairs(frames)
