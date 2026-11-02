"""Ranker-side synthetic Hangul rows for S4 (reports/diag/action_plan.md C4.1 and C4.2). Fit time only.

Nothing here runs at inference: the sets are extra training rows for RankPipeline.fit(aug=...). Every
statistic is learned from the ``train`` frame passed in, so in CV pass the fold-train rows (fold-safe).

C4.1 synthfix(queries, pairs, train, catalog, gen)
    Port of experiments/diag/hangul_fragility/stress.py::fixed_synth_sets for the v5 synthetic Hangul set:
    (1) drop rows whose TARGET pair has digit_state 3/4 (record digits != catalog digits; 0 of the 12,222
        real train targets look like that - en2ko spells II as 아이아이);
    (2) give a share of the remaining rows a real-looking tonnage gap: the share and the |log GT| gaps are
        those of the Hangul rows of ``train`` with a nonzero target gap (28.5% / mostly 3-5% on the full
        train), random sign, both tonnages scaled; the candidates of the changed rows are rebuilt.

C4.2 hangul_sets(train, catalog, gen)
    Port of experiments/diag/augmentation/generator.py (HangulNameModelV4, hangul_set_v4,
    hangul_set_v1fix) and the alignment helpers of align.py they use. Data-driven Hangul records for
    catalog vessels that are NOT targets of ``train``:
      v4     lexicon of real readings + learned syllable edits, numbers rendered like real records,
             target (tr_sim, ph_sim) matched to a real profile (best of 8 sampled names);
      v1fix  the noisier v1 name model (no lexicon, no profile matching) with the v4 number rendering.
    Tonnage ratios / missing flag from a real Hangul donor row (DonorModel), Korean flag/type from the
    empirical P(record value | catalog value) (AttributeModel). No generative-model output is used.

Parity with the prototypes is checked by experiments/s4_impl/parity_synth.py (logs next to it).
"""

from __future__ import annotations

import copy
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

from common import is_hangul_name, normalize_key
from hangul_match import _positive

# Defaults of the measured configuration (augmentation.md sec. 3: v5 + v4 4000 + v1fix 4000, weight 0.5;
# eval_cv.py seeds 700 + k / 800 + k, k = fold -> 700 / 800 for the full-train fit).
N_V4, N_V1FIX, WEIGHT, SEED_V4, SEED_V1FIX = 4000, 4000, 0.5, 700, 800
SYNTHFIX_SEED = 7


# ---------------------------------------------------------------------------
# C4.1 synthfix
# ---------------------------------------------------------------------------
def hangul_target_gaps(train: pd.DataFrame, catalog: pd.DataFrame, log_gt: np.ndarray) -> np.ndarray:
    """|log GT gap| of each Hangul row of ``train`` to its target (NaN if no positive tonnage / target GT).

    The gap is that of the record tonnage (gross or international) whose rank among all catalog vessels
    by |log gap| is smallest - the definition of experiments/diag/hangul_fragility/tonnage_window.py
    (it equals the smaller of the two gaps on every train row; experiments/s4_impl/parity_synth.log).
    """
    pos_of = {v: i for i, v in enumerate(catalog["vessel_id"].astype(str))}
    valid = np.sort(log_gt[np.isfinite(log_gt)])
    out = []
    for r in train[train["vessel_name"].map(is_hangul_name).to_numpy()].to_dict("records"):
        vals = [v for v in (_positive(r["gross_tonnage"]), _positive(r["international_gross_tonnage"])) if v]
        t = pos_of[str(r["target_vessel_id"])]
        if not vals or not np.isfinite(log_gt[t]):
            out.append(np.nan)
            continue
        best = None
        for v in vals:
            lv = math.log(v)
            gap_t = abs(log_gt[t] - lv)
            rank = int((np.abs(valid - lv) < gap_t).sum())
            if best is None or rank < best[1]:
                best = (gap_t, rank)
        out.append(best[0])
    return np.array(out, dtype=float)


def _subset_pairs(pairs: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Pairs of query rows ``rows`` (sorted old indices), re-indexed 0..len(rows)-1."""
    remap = pd.Series(np.arange(len(rows)), index=rows)
    sub = pairs[pairs["q"].isin(rows)].copy()
    sub["q"] = remap.loc[sub["q"].to_numpy()].to_numpy()
    return sub


def synthfix(queries: pd.DataFrame, pairs: pd.DataFrame, train: pd.DataFrame, catalog: pd.DataFrame, gen,
             seed: int = SYNTHFIX_SEED, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """C4.1 on a synthetic Hangul set (queries with target_vessel_id, pairs from ``gen``); see module doc."""
    from rank.pipeline import labels

    queries = queries.reset_index(drop=True)
    lab = labels(pairs, queries, catalog).astype(bool)
    bad = set(pairs.loc[lab & (pairs["digit_state"].to_numpy() >= 3), "q"].tolist())
    keep = np.array([q for q in range(len(queries)) if q not in bad], dtype=int)
    sq = queries.iloc[keep].reset_index(drop=True)
    gaps_all = hangul_target_gaps(train, catalog, gen.matcher.log_gt)
    gaps = gaps_all[gaps_all > 1e-9]
    share = float((gaps_all > 1e-9).mean()) if len(gaps_all) else 0.0
    rng = np.random.default_rng(seed)
    noisy = rng.random(len(sq)) < share
    if len(gaps):
        f = np.exp(rng.choice(gaps, len(sq)) * rng.choice([-1, 1], len(sq)))
        f = np.where(noisy, f, 1.0)
    else:
        noisy[:] = False
        f = np.ones(len(sq))
    sq["gross_tonnage"] = sq["gross_tonnage"] * f
    sq["international_gross_tonnage"] = sq["international_gross_tonnage"] * f
    # Candidates are row-local: unchanged rows keep their pairs, only the re-tonnaged rows are rebuilt.
    sp = _subset_pairs(pairs, keep)
    sp = sp[~noisy[sp["q"].to_numpy()]]
    jit = np.flatnonzero(noisy)
    if len(jit):
        rebuilt = gen.build(sq.iloc[jit].reset_index(drop=True))
        rebuilt["q"] = jit[rebuilt["q"].to_numpy()]
        sp = pd.concat([sp, rebuilt[sp.columns]], ignore_index=True)
    sp = sp.sort_values(["q", "c"], kind="stable").reset_index(drop=True)
    log(f"synthfix: dropped {len(bad)} digit-mismatch rows, re-tonnaged {len(jit)}/{len(sq)} "
        f"(share {share:.3f} from {len(gaps_all)} train Hangul rows)")
    return sq, sp


# ---------------------------------------------------------------------------
# Alignment helpers (port of experiments/diag/augmentation/align.py)
# ---------------------------------------------------------------------------
def is_syl(ch: str) -> bool:
    return "가" <= ch <= "힣"


def decompose(ch: str) -> tuple[int, int, int]:
    code = ord(ch) - 0xAC00
    return code // 588, (code % 588) // 28, code % 28


def compose(cho: int, jung: int, jong: int) -> str:
    return chr(0xAC00 + cho * 588 + jung * 28 + jong)


def catalog_words(name: str) -> list[str]:
    """Words as en2ko.name_variants sees them (Latin words and digit runs, NO dropped)."""
    return [w for w in re.findall(r"[A-Za-z]+|\d+", unicodedata.normalize("NFKC", name)) if w.upper() != "NO"]


@lru_cache(maxsize=200_000)
def readings(word: str) -> tuple[str, ...]:
    from en2ko import word_variants
    if word.isdigit():
        return (word,)
    return tuple(v.replace(" ", "") for v in word_variants(word))


def record_core(name: str) -> tuple[str, dict]:
    """Record name without spaces/punctuation; flags for 제N prefix, 호 suffix, spacing."""
    s = unicodedata.normalize("NFKC", name).strip()
    info = {"n_words": len(s.split()), "je_prefix": bool(re.match(r"^제\s*\d", s)),
            "ho_suffix": bool(re.search(r"호$", s)) and len(s.replace(" ", "")) >= 3,
            "has_digit": bool(re.search(r"\d", s)), "has_latin": bool(re.search(r"[A-Za-z]", s))}
    core = re.sub(r"[^0-9A-Za-z가-힣]", "", s)
    if info["je_prefix"]:
        core = re.sub(r"^제", "", core)
    if info["ho_suffix"]:
        core = core[:-1]
    return core, info


def _jlen(s: str) -> int:
    from translit import to_jamo
    return len(to_jamo(s))


def word_align(record: str, catalog_name: str, max_chunk: int = 12):
    """Monotone DP: record core chunks <-> catalog words (cost = jamo Levenshtein to the closest reading)."""
    from rapidfuzz.distance import Levenshtein

    from translit import to_jamo

    core, info = record_core(record)
    words = catalog_words(catalog_name)
    n, m = len(core), len(words)
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    rd = [readings(w) for w in words]
    rj = [[to_jamo(v) for v in r] for r in rd]
    for i in range(n + 1):
        for j in range(m + 1):
            c0 = cost[i][j]
            if c0 == INF:
                continue
            if i < n:  # unmatched record char
                c = c0 + _jlen(core[i])
                if c < cost[i + 1][j]:
                    cost[i + 1][j], back[i + 1][j] = c, ("I", i, j, None, None)
            if j < m:
                c = c0 + len(rj[j][0])  # drop the catalog word
                if c < cost[i][j + 1]:
                    cost[i][j + 1], back[i][j + 1] = c, ("D", i, j, None, None)
                for L in range(1, min(max_chunk, n - i) + 1):
                    chunk = to_jamo(core[i:i + L])
                    best = min(range(len(rj[j])), key=lambda k: (Levenshtein.distance(chunk, rj[j][k]), k))
                    d = Levenshtein.distance(chunk, rj[j][best])
                    c = c0 + d
                    if c < cost[i + L][j + 1]:
                        cost[i + L][j + 1], back[i + L][j + 1] = c, ("M", i, j, L, best)
    i, j = n, m
    ops = []
    while (i, j) != (0, 0):
        op = back[i][j]
        ops.append(op)
        if op[0] == "I":
            i -= 1
        elif op[0] == "D":
            j -= 1
        else:
            i -= op[3]
            j -= 1
    ops.reverse()
    out, inserted = [], []
    for op in ops:
        if op[0] == "M":
            _, i0, j0, L, k = op
            chunk = core[i0:i0 + L]
            r = rd[j0][k]
            sim = Levenshtein.normalized_similarity(to_jamo(chunk), to_jamo(r))
            out.append({"word": words[j0], "chunk": chunk, "reading": r, "variant": k, "sim": sim,
                        "start": i0, "end": i0 + L})
        elif op[0] == "D":
            out.append({"word": words[op[2]], "chunk": "", "reading": rd[op[2]][0], "variant": -1, "sim": 0.0,
                        "start": op[1], "end": op[1]})
        else:
            inserted.append(core[op[1]])
    return {"core": core, "info": info, "words": out, "inserted": "".join(inserted),
            "total_cost": cost[n][m], "core_jamo_len": _jlen(core)}


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if is_syl(a) and is_syl(b):
        da, db = decompose(a), decompose(b)
        return sum(x != y for x, y in zip(da, db)) / 3.0 * 1.5  # 1 slot 0.5, 2 slots 1.0, 3 slots 1.5
    return 1.5


def syllable_align(src: str, dst: str):
    """Align reading (src) to record chunk (dst); ins/del cost 1, substitution slot-aware."""
    n, m = len(src), len(dst)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = float(i)
    for j in range(1, m + 1):
        D[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + _sub_cost(src[i - 1], dst[j - 1]))
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(D[i][j] - (D[i - 1][j - 1] + _sub_cost(src[i - 1], dst[j - 1]))) < 1e-9:
            ops.append(("M", src[i - 1], dst[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and abs(D[i][j] - (D[i - 1][j] + 1)) < 1e-9:
            ops.append(("D", src[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, dst[j - 1]))
            j -= 1
    ops.reverse()
    return ops


# ---------------------------------------------------------------------------
# Generator (port of experiments/diag/augmentation/generator.py, Hangul parts only)
# ---------------------------------------------------------------------------
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
# Fallback confusions (same as rank.finetune.jamo_noise) for jamo never seen substituted.
_CONF = [set("ㅓㅗ"), set("ㅐㅔ"), set("ㅕㅖ"), set("ㅅㅆ"), set("ㅈㅊㅉ"), set("ㄱㅋ"), set("ㄷㅌ"), set("ㅂㅍ"), set("ㅜㅠ")]


def _draw(counter: Counter, rng: random.Random, exclude=None):
    items = [(k, v) for k, v in counter.items() if v > 0 and k != exclude]
    if not items:
        return None
    tot = sum(v for _, v in items)
    x = rng.random() * tot
    for k, v in items:
        x -= v
        if x <= 0:
            return k
    return items[-1][0]


def _bucket(n_syl: int) -> int:
    return 0 if n_syl <= 2 else 1 if n_syl <= 4 else 2


class HangulNameModel:
    """v1: per-word en2ko variant index, learned syllable edits, spacing / 호 / 제N / digit attachment."""

    def fit(self, records, catalog_names) -> "HangulNameModel":
        self.variant = Counter()
        self.n_edits = [Counter() for _ in range(3)]
        self.etype = Counter()
        self.sub = {"cho": defaultdict(Counter), "jung": defaultdict(Counter), "jong": defaultdict(Counter)}
        self.occ = {"cho": Counter(), "jung": Counter(), "jong": Counter()}
        self.dele, self.dele_jung, self.ins = Counter(), Counter(), Counter()
        n_multi = n_joined = 0
        keep_num = keep_den = 0
        n_ho = n_rows = 0
        n_no = n_je = 0
        dig_sp = dig_tot = 0
        for rec, cname in dict.fromkeys(zip(records, catalog_names)):
            a = word_align(rec, cname)
            n_rows += 1
            n_ho += a["info"]["ho_suffix"]
            lw = [w for w in a["words"] if not w["word"].isdigit()]
            if len(lw) >= 2:
                n_multi += 1
                nw = a["info"]["n_words"]
                if nw == 1:
                    n_joined += 1
                else:
                    keep_num += min(nw - 1, len(lw) - 1)
                    keep_den += len(lw) - 1
            if re.search(r"\bNo\b\.?\s*\d", cname, flags=re.I):
                n_no += 1
                n_je += a["info"]["je_prefix"]
            if re.search(r"\d", rec):
                dig_tot += 1
                dig_sp += bool(re.search(r"\s\d", rec))
            for w in lw:
                if w["variant"] < 0:
                    continue
                self.variant[min(w["variant"], 3)] += 1
                r, chunk = w["reading"], w["chunk"]
                for s in r:
                    if is_syl(s):
                        d = decompose(s)
                        self.occ["cho"][d[0]] += 1
                        self.occ["jung"][d[1]] += 1
                        self.occ["jong"][d[2]] += 1
                k = 0
                for op, x, y in syllable_align(r, chunk):
                    if op == "M" and x != y and is_syl(x) and is_syl(y):
                        dx, dy = decompose(x), decompose(y)
                        for slot, i in (("cho", 0), ("jung", 1), ("jong", 2)):
                            if dx[i] != dy[i]:
                                self.sub[slot][dx[i]][dy[i]] += 1
                                self.etype[slot] += 1
                                k += 1
                    elif op == "D" and is_syl(x):
                        self.dele[x] += 1
                        self.dele_jung[decompose(x)[1]] += 1
                        self.etype["del"] += 1
                        k += 1
                    elif op == "I" and is_syl(y):
                        self.ins[y] += 1
                        self.etype["ins"] += 1
                        k += 1
                self.n_edits[_bucket(len(r))][min(k, 8)] += 1
        self.p_join = n_joined / max(n_multi, 1)
        self.p_keep = keep_num / max(keep_den, 1)
        self.p_ho = n_ho / max(n_rows, 1)
        self.p_je = n_je / max(n_no, 1)
        self.p_digit_space = dig_sp / max(dig_tot, 1)
        return self

    def _sub_target(self, slot: str, a: int, rng: random.Random):
        b = _draw(self.sub[slot].get(a, Counter()), rng)
        if b is not None:
            return b
        if slot in ("cho", "jung"):
            alph = CHO if slot == "cho" else JUNG
            for g in _CONF:
                if alph[a] in g:
                    return alph.index(rng.choice(sorted(g - {alph[a]})))
        return None

    def _rate(self, slot: str, a: int) -> float:
        n_sub = sum(self.sub[slot].get(a, Counter()).values())
        return (n_sub + 0.1) / (self.occ[slot][a] + 1.0)

    def _edit(self, syl: list[str], rng: random.Random) -> list[str]:
        kind = _draw(self.etype, rng)
        idx = [i for i, s in enumerate(syl) if is_syl(s)]
        if not idx:
            return syl
        if kind in ("cho", "jung", "jong"):
            slot_i = {"cho": 0, "jung": 1, "jong": 2}[kind]
            w = [self._rate(kind, decompose(syl[i])[slot_i]) for i in idx]
            i = rng.choices(idx, weights=w)[0]
            d = list(decompose(syl[i]))
            b = self._sub_target(kind, d[slot_i], rng)
            if b is None:
                return syl
            d[slot_i] = b
            syl = syl.copy()
            syl[i] = compose(*d)
        elif kind == "del":
            if len(idx) < 2:
                return syl
            w = [(self.dele[syl[i]] + 0.2 * self.dele_jung[decompose(syl[i])[1]] / 10 + 0.05) for i in idx]
            i = rng.choices(idx, weights=w)[0]
            syl = syl[:i] + syl[i + 1:]
        elif kind == "ins":
            s = _draw(self.ins, rng)
            if s is None:
                return syl
            i = rng.randint(1, len(syl))
            syl = syl[:i] + [s] + syl[i:]
        return syl

    def word_reading(self, word: str, rng: random.Random) -> str:
        if word.isdigit():
            return word
        vs = readings(word)
        probs = [self.variant.get(min(k, 3), 0) + 0.01 for k in range(len(vs))]
        k = rng.choices(range(len(vs)), weights=probs)[0]
        syl = list(vs[k])
        n = _draw(self.n_edits[_bucket(len(syl))], rng) or 0
        for _ in range(n):
            syl = self._edit(syl, rng)
        return "".join(syl) or vs[k]


class DonorModel:
    """Tonnage ratios + flag-missing copied from a real row with a similar target."""

    def fit(self, rows: pd.DataFrame, catalog: pd.DataFrame) -> "DonorModel":
        cat = catalog.set_index("vessel_id")
        cgt = rows["target_vessel_id"].map(cat["gross_tonnage"]).astype(float).to_numpy()
        kr = (rows["target_vessel_id"].map(cat["flag_name"]) == "South Korea").to_numpy()
        ok = np.isfinite(cgt) & (cgt > 0)
        self.log_cgt = np.log(cgt[ok])
        self.kr = kr[ok]
        self.r_gt = (rows["gross_tonnage"].astype(float).to_numpy() / np.where(ok, cgt, 1))[ok]
        self.r_intl = (rows["international_gross_tonnage"].astype(float).to_numpy() / np.where(ok, cgt, 1))[ok]
        self.flag_na = rows["flag_name"].isna().to_numpy()[ok]
        return self

    def sample(self, cgt: float, korean: bool, rng: random.Random, k: int = 40):
        same = np.flatnonzero(self.kr == korean)
        pool = same if len(same) >= k else np.arange(len(self.kr))
        d = np.abs(self.log_cgt[pool] - math.log(max(cgt, 1.0)))
        near = pool[np.argsort(d, kind="stable")[:k]]
        j = int(near[rng.randrange(len(near))])

        def apply(r):
            if not np.isfinite(r):
                return np.nan
            return float(cgt) if abs(r - 1.0) < 1e-9 else float(round(cgt * r))
        return apply(self.r_gt[j]), apply(self.r_intl[j]), bool(self.flag_na[j])


class AttributeModel:
    def fit(self, rows: pd.DataFrame, catalog: pd.DataFrame) -> "AttributeModel":
        cat = catalog.set_index("vessel_id")
        cf = rows["target_vessel_id"].map(cat["flag_name"])
        ct = rows["target_vessel_id"].map(cat["vessel_type"])
        self.flag = defaultdict(Counter)
        for r, c in zip(rows["flag_name"], cf):
            if isinstance(r, str):
                self.flag[c][r] += 1
        self.type = defaultdict(Counter)
        for r, c in zip(rows["vessel_type"], ct):
            if isinstance(r, str):
                self.type[c][r] += 1
        self.type_any = Counter(rows["vessel_type"].dropna())
        return self

    def sample(self, cflag, ctype, rng: random.Random):
        f = _draw(self.flag.get(cflag, Counter()), rng)
        t = _draw(self.type.get(ctype, Counter()), rng) or _draw(self.type_any, rng)
        return f, t


class TargetSampler:
    """Catalog targets: mix of uniform and the train target profile (flag x type x log-GT bin).
    Duplicated names, missing DWT, GT < min_gt and every target of the train frame are excluded."""

    def __init__(self, catalog: pd.DataFrame, exclude_ids, min_gt: float = 100.0):
        gt = pd.to_numeric(catalog["gross_tonnage"], errors="coerce")
        names = catalog["vessel_name"].fillna("").astype(str)
        norm = names.map(normalize_key)
        dup = norm.map(norm.value_counts()) > 1
        ok = (gt >= min_gt) & catalog["deadweight_tonnage"].notna() & names.str.contains("[A-Za-z]{3}") & ~dup
        ok &= ~catalog["vessel_id"].astype(str).isin(set(map(str, exclude_ids)))
        self.pos = np.flatnonzero(ok.to_numpy())
        self.catalog = catalog
        fb = catalog["flag_name"].fillna("?").to_numpy()
        tb = catalog["vessel_type"].fillna("?").to_numpy()
        gb = np.floor(np.log(np.where(gt > 0, gt, 1)) / 0.5).astype(int)
        self.keys = [(fb[p], tb[p], gb[p]) for p in range(len(catalog))]

    def weights(self, profile_ids, real_share: float = 0.5) -> np.ndarray:
        pos_of = {v: i for i, v in enumerate(self.catalog["vessel_id"].astype(str))}
        prof = Counter(self.keys[pos_of[str(v)]] for v in profile_ids)
        n_prof = sum(prof.values())
        keys_ok = Counter(self.keys[p] for p in self.pos)
        w_real = np.array([prof.get(self.keys[p], 0) / n_prof / keys_ok[self.keys[p]] for p in self.pos])
        w = (1 - real_share) / len(self.pos) + real_share * w_real / max(w_real.sum(), 1e-12)
        return w / w.sum()

    def draw(self, n: int, w: np.ndarray, seed: int) -> np.ndarray:
        rs = np.random.default_rng(seed)
        return np.sort(rs.choice(self.pos, size=min(n, len(self.pos)), replace=False, p=w))


def _row(i, tag, name, gt, intl, flag, vtype, vid):
    return {"sample_id": f"{tag}_{i:05d}", "vessel_name": name, "call_sign": "", "gross_tonnage": gt,
            "international_gross_tonnage": intl, "flag_name": flag, "vessel_type": vtype,
            "arrival_year": 2024, "target_vessel_id": vid}


_NUM_WORD_KO = {"1": ["원"], "2": ["투"], "3": ["쓰리", "스리"], "4": ["포"], "5": ["파이브"], "6": ["식스"],
                "7": ["세븐"], "8": ["에잇", "에이트"], "9": ["나인"], "10": ["텐"]}


def _number_value(word: str, i: int) -> str | None:
    """Digit value of a standalone Roman numeral / English number word (never the first word),
    exactly the words translit.name_digits reads as digits."""
    from translit import _NUMBER_WORDS
    if i == 0 or word.isdigit():
        return None
    v = _NUMBER_WORDS.get(word.upper())
    return v if v and word.isascii() else None


class HangulNameModelV4(HangulNameModel):
    """HangulNameModel + a real-reading lexicon (catalog word -> record chunks, used with the learned p_lex)
    and numbers rendered as Arabic digits or Hangul English number words, never letter by letter."""

    def fit(self, records, catalog_names) -> "HangulNameModelV4":
        super().fit(records, catalog_names)
        self.lex = defaultdict(Counter)
        by_word_vessel = defaultdict(lambda: defaultdict(Counter))
        n_num = n_num_word = 0
        for rec, cname in dict.fromkeys(zip(records, catalog_names)):
            a = word_align(rec, cname)
            for w in a["words"]:
                if w["variant"] >= 0 and w["chunk"] and not w["word"].isdigit():
                    self.lex[w["word"].upper()][w["chunk"]] += 1
                    by_word_vessel[w["word"].upper()][cname][w["chunk"]] += 1
            for i, w in enumerate(catalog_words(cname)):
                if _number_value(w, i):
                    n_num += 1
                    n_num_word += not bool(re.search(r"\d", rec))
        # p_lex: for words seen on >= 2 vessels, how often does a vessel's chunk also occur on another vessel?
        hit = tot = 0
        for word, per_vessel in by_word_vessel.items():
            if len(per_vessel) < 2:
                continue
            for v, chunks in per_vessel.items():
                others = Counter()
                for v2, c2 in per_vessel.items():
                    if v2 != v:
                        others.update(c2)
                for ch, n in chunks.items():
                    tot += n
                    hit += n * (ch in others)
        self.p_lex = hit / max(tot, 1)
        self.p_num_word = (n_num_word + 1) / (n_num + 2)
        self.n_num = n_num
        # 'X Y No. N' with a 제 record: 제N in front (제2 해마루) or 제N호 at the end (청솔마루제2호).
        n_front = n_end = 0
        for rec, cname in dict.fromkeys(zip(records, catalog_names)):
            if re.search(r"\bNo\b\.?\s*\d+\s*$", cname, flags=re.I) and "제" in rec:
                n_front += bool(re.match(r"^제\s*\d", rec))
                n_end += bool(re.search(r"제\s*\d+호?$", rec))
        self.p_je_end = (n_end + 1) / (n_front + n_end + 2)
        return self

    def word_reading(self, word: str, rng: random.Random) -> str:
        lex = self.lex.get(word.upper())
        if lex and rng.random() < self.p_lex:
            return _draw(lex, rng)
        return super().word_reading(word, rng)

    def sample(self, cname: str, rng: random.Random) -> str:
        words = catalog_words(cname)
        # hyphenated words are written together (Heung-A -> 흥아), not as two words
        nf = unicodedata.normalize("NFKC", cname)
        toks = [m for m in re.finditer(r"[A-Za-z]+|\d+", nf) if m.group().upper() != "NO"]
        glue = [i > 0 and nf[toks[i - 1].end():t.start()] == "-" for i, t in enumerate(toks)]
        if len(glue) != len(words):
            glue = [False] * len(words)
        has_no = bool(re.search(r"\bNo\b\.?\s*\d", cname, flags=re.I))
        no_end = bool(re.search(r"\bNo\b\.?\s*\d+\s*$", cname, flags=re.I))
        parts = []  # (is_digit, text, glued_by_hyphen, is_number_word)
        for i, w in enumerate(words):
            v = _number_value(w, i)
            if v is not None and rng.random() < self.p_num_word:
                parts.append((False, rng.choice(_NUM_WORD_KO[v]), False, True))
            elif v is not None:
                parts.append((True, v, False, False))
            else:
                parts.append((w.isdigit(), self.word_reading(w, rng), glue[i], False))
        je = has_no and rng.random() < self.p_je
        prefix = suffix = ""
        if je:  # move only the number that follows 'No' (KTX-30 No. 15 keeps its 30)
            num = re.search(r"\bNo\b\.?\s*(\d+)", cname, flags=re.I).group(1)
            k = max((j for j, x in enumerate(parts) if x[0] and x[1] == num), default=None)
            if k is not None:
                parts = parts[:k] + parts[k + 1:]
            if no_end and rng.random() < self.p_je_end:
                suffix = (" " if rng.random() < 0.5 else "") + "제" + num + "호"
            else:
                prefix = "제" + num
        n_latin = sum(1 for x in parts if not x[0])
        joined = n_latin >= 2 and rng.random() < self.p_join
        out = ""
        for i, (is_d, p, g, nw) in enumerate(parts):
            if i == 0:
                out = p
            elif nw:  # the record parser only reads a standalone number word (에이비 마린 투)
                out += " " + p
            elif is_d:
                out += (" " if rng.random() < self.p_digit_space else "") + p
            elif joined or g:
                out += p
            else:
                out += (" " if rng.random() < self.p_keep else "") + p
        out = prefix + out + suffix
        if not suffix and not (parts and parts[-1][3]) and rng.random() < self.p_ho:
            out += "호"
        return out


class ProfileMatcher:
    """Pick, among several generated names, the one whose target (tr_sim, ph_sim) is closest to a
    profile drawn from real train rows - matches the joint distribution of the two key features."""

    def __init__(self, gen, profiles: np.ndarray):
        self.gen = gen
        self.profiles = profiles[np.isfinite(profiles).all(axis=1)]

    def sims(self, name: str, pos: int) -> tuple[float, float]:
        from hangul_match import _sim_matrix
        from translit import hangul_core, to_jamo
        q = to_jamo(hangul_core(name))
        tr = float(_sim_matrix([q], self.gen._transliterations(int(pos)))[0].max()) if q else 0.0
        ph = float(self.gen.matcher.name_similarity({"vessel_name": name}, np.array([int(pos)]))[0])
        return tr, ph

    def choose(self, names: list[str], pos: int, rng: random.Random) -> str:
        p = self.profiles[rng.randrange(len(self.profiles))]
        names = list(dict.fromkeys(names))
        d = [(a - p[0]) ** 2 + (b - p[1]) ** 2 for a, b in (self.sims(n, pos) for n in names)]
        return names[int(np.argmin(d))]


def fit_hangul_models(train: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    """Every table from ``train`` only (fold-train in CV)."""
    cat = catalog.set_index("vessel_id")
    ko = train[train["vessel_name"].map(is_hangul_name).to_numpy()]
    return {
        "name4": HangulNameModelV4().fit(ko["vessel_name"].tolist(),
                                         ko["target_vessel_id"].map(cat["vessel_name"]).tolist()),
        "donor_ko": DonorModel().fit(ko, catalog),
        "attr": AttributeModel().fit(train, catalog),
        "ko_ids": ko["target_vessel_id"].unique(),
        "train_ids": train["target_vessel_id"].unique(),
    }


def target_profiles(train: pd.DataFrame, catalog: pd.DataFrame, gen) -> np.ndarray:
    """(tr_sim, ph_sim) of the target pair for each Hangul row of ``train`` (= the pair features the
    generator computes; prototype: generator.real_profiles on the cached train pairs)."""
    pos_of = {v: i for i, v in enumerate(catalog["vessel_id"].astype(str))}
    pm = ProfileMatcher(gen, np.zeros((0, 2)))
    ko = train[train["vessel_name"].map(is_hangul_name).to_numpy()]
    out = [pm.sims(n, pos_of[str(t)]) for n, t in zip(ko["vessel_name"], ko["target_vessel_id"])]
    return np.array(out, dtype=float).reshape(-1, 2)


def _hangul_rows(models: dict, name_model, catalog: pd.DataFrame, n: int, seed: int, tag: str,
                 real_share: float = 0.5, pm: ProfileMatcher | None = None, n_try: int = 8) -> pd.DataFrame:
    rng = random.Random(seed)
    ts = TargetSampler(catalog, models["train_ids"], min_gt=100.0)
    picks = ts.draw(n, ts.weights(models["ko_ids"], real_share), seed)
    rows = []
    for i, p in enumerate(picks):
        c = catalog.iloc[p]
        if pm is None:  # generator.hangul_set
            name = name_model.sample(c["vessel_name"], rng)
            if not is_hangul_name(name):
                continue
        else:  # generator.hangul_set_matched
            names = [x for x in (name_model.sample(c["vessel_name"], rng) for _ in range(n_try)) if is_hangul_name(x)]
            if not names:
                continue
            name = pm.choose(names, int(p), rng)
        gt, intl, flag_na = models["donor_ko"].sample(float(c["gross_tonnage"]), c["flag_name"] == "South Korea", rng)
        f, t = models["attr"].sample(c["flag_name"], c["vessel_type"], rng)
        rows.append(_row(i, tag, name, gt, intl, None if flag_na else f, t, c["vessel_id"]))
    return pd.DataFrame(rows)


def hangul_set_v4(models: dict, catalog: pd.DataFrame, n: int, gen, profiles: np.ndarray, seed: int = SEED_V4,
                  real_share: float = 0.5, n_try: int = 8) -> pd.DataFrame:
    pm = ProfileMatcher(gen, profiles)
    return _hangul_rows(models, models["name4"], catalog, n, seed, "SYNK4", real_share, pm, n_try)


def hangul_set_v1fix(models: dict, catalog: pd.DataFrame, n: int, seed: int = SEED_V1FIX,
                     real_share: float = 0.5) -> pd.DataFrame:
    name = copy.copy(models["name4"])
    name.p_lex = 0.0
    return _hangul_rows(models, name, catalog, n, seed, "SYNK1F", real_share)


def hangul_sets(train: pd.DataFrame, catalog: pd.DataFrame, gen, n_v4: int = N_V4, n_v1fix: int = N_V1FIX,
                weight: float = WEIGHT, seed_v4: int = SEED_V4, seed_v1fix: int = SEED_V1FIX,
                log=print) -> list[tuple[pd.DataFrame, pd.DataFrame, float]]:
    """C4.2: [(v4 queries, pairs, weight), (v1fix queries, pairs, weight)] for RankPipeline.fit(aug=...).

    Targets are catalog vessels outside ``train``'s targets; candidates come from ``gen`` (pass the
    generator with the pipeline's cand_opts). Flag/type are already Korean (no localize_attributes)."""
    import time
    t0 = time.perf_counter()
    models = fit_hangul_models(train, catalog)
    out = []
    if n_v4:
        q = hangul_set_v4(models, catalog, n_v4, gen, target_profiles(train, catalog, gen), seed=seed_v4)
        out.append((q.reset_index(drop=True), None, weight))
    if n_v1fix:
        q = hangul_set_v1fix(models, catalog, n_v1fix, seed=seed_v1fix)
        out.append((q.reset_index(drop=True), None, weight))
    out = [(q, gen.build(q), w) for q, _, w in out]
    log(f"synth_v6: {' + '.join(str(len(q)) for q, _, _ in out)} Hangul rows "
        f"(models from {len(train)} train rows) in {time.perf_counter() - t0:.0f}s")
    return out
