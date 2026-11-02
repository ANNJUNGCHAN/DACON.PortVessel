"""S9 item 4b: row-local pair features (one query row + the catalog only; no other rows, no train statistics).

Port of experiments/s9_robust/item4/feats4.py (RESULT_4.md "Porting spec" 1). Used by CandidateGenerator(
pair_feats4=True) as the last step of row_pairs; the catalog side table comes from build_aux(catalog, tr_cache)
(no file I/O), is built once at fit time and shipped as art["feats4_aux"].

Per pair (record q, catalog vessel c); same column names for both scripts (is_hangul tells the tree which):
  f_q_in_c     share of record words found in the candidate name. Latin: equal word, or OSA sim >= 0.8 for words
               of >= 4 chars. Hangul: a record word (jamo of hangul_core) is found when its best partial_ratio
               against the full rule-based readings of the candidate (tr_cache) is >= 0.8.
  f_c_in_q     share of candidate words found in the record. Hangul: a candidate Latin word is found when one of
               its word readings (en2ko.word_variants -> jamo) has partial_ratio >= 0.8 inside the record jamo.
  f_q_part     mean best match score of the record words (soft f_q_in_c).
  f_tset       order-free token_set_ratio / 100. Hangul: record word jamos vs first reading of each candidate word.
  f_tsort      token_sort_ratio / 100 (Latin only).
  f_idf_share_q / f_idf_share_c   IDF-weighted share of matched words on each side (catalog word IDF
               log((1+N)/(1+df))+1 over set(_name_words); OOV record words get the OOV idf. Hangul record
               words have no catalog idf, so f_idf_share_q = f_q_in_c there).
  f_idf_rare   largest IDF among matched candidate words (rarest shared word); 0 if none.
  f_idf_miss_c largest IDF among UNmatched candidate words (a rare candidate word that the record lacks).
  f_num_state  number groups after normalisation (no leading zeros; ONE..TWENTY and Roman I..XX as non-first
               word; M/V MV MT prefixes dropped; Hangul: num_forms reader, else name_digits_v2): 0 none, 1 equal,
               2 record subset of candidate, 3 candidate subset of record, 4 overlap, 5 disjoint,
               6 candidate only, 7 record only.   f_num_q / f_num_c: group counts.
  f_ton_lr     signed log(candidate GT / nearest record tonnage); NaN if missing.
  f_ton_bkt    bucket of |f_ton_lr|: 0 <0.01, 1 <0.05, 2 <0.15, 3 <0.3, 4 <0.75, 5 >=0.75, -1 missing.
  f_n_src      number of retrieval sources that proposed the candidate.
Row-relative (over the row's FULL candidate set, S9 extras included; still row-local):
  f_nstr               name strength: Hangul tr_sim; Latin max(nm_sim, osa_sim, f_tset)
  f_nstr_gap/_rank     gap to the row best / rank in the row (0 = best; ties -> lower catalog position)
  f_qin_gap, f_idfq_gap  gaps of f_q_in_c / f_idf_share_q to the row best
  f_row_bestname_bkt   f_ton_bkt of the best-name candidate, broadcast to the row
  f_row_bestton_ngap   f_nstr_gap of the tonnage-closest candidate (gap_rank 0), broadcast to the row
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

COLS = ["f_q_in_c", "f_c_in_q", "f_q_part", "f_tset", "f_tsort", "f_idf_share_q", "f_idf_share_c", "f_idf_rare",
        "f_idf_miss_c", "f_num_state", "f_num_q", "f_num_c", "f_ton_lr", "f_ton_bkt"]
CI = {c: i for i, c in enumerate(COLS)}
REL = ["f_n_src", "f_nstr", "f_nstr_gap", "f_nstr_rank", "f_qin_gap", "f_idfq_gap", "f_row_bestname_bkt",
       "f_row_bestton_ngap"]
ALL = COLS + REL
SRC = ["src_window", "src_base", "src_key", "src_tr", "src_gt_exact", "src_typo", "src_hglobal", "src_lword"]
_PREFIX = re.compile(r"^\s*(M\s*/\s*V|M\s*\.\s*V\s*\.?|M\s*/\s*T|MV|MT)\s+", re.I)
_ENG = {w: str(i) for i, w in enumerate(["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
                                          "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN",
                                          "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN", "TWENTY"]) if i}
_ROMAN = {w: str(i) for i, w in enumerate(["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI",
                                            "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]) if w}
JAMO_LO, JAMO_HI = chr(0x1100), chr(0x11FF)


def latin_num_groups(name) -> tuple:
    if not isinstance(name, str):
        return ()
    s = _PREFIX.sub("", unicodedata.normalize("NFKC", name).upper())
    toks = re.findall(r"[A-Z]+|\d+", s)
    out = set()
    for i, t in enumerate(toks):
        if t.isdigit():
            out.add(t.lstrip("0") or "0")
        elif i > 0 and t in _ENG:
            out.add(_ENG[t])
        elif i > 0 and t in _ROMAN:
            out.add(_ROMAN[t])
    return tuple(sorted(out))


def hangul_num_groups(name) -> tuple:
    from translit import name_digit_groups_forms, name_digits_v2
    g = name_digit_groups_forms(name)
    if not g:
        v = name_digits_v2(name)
        g = [v] if v else []
    return tuple(sorted({x.lstrip("0") or "0" for x in g}))


def _hangul_words(name) -> list:
    from translit import hangul_core, to_jamo
    if not isinstance(name, str):
        return []
    out = []
    for w in unicodedata.normalize("NFKC", name).split():
        j = "".join(ch for ch in to_jamo(hangul_core(w)) if JAMO_LO <= ch <= JAMO_HI)
        if len(j) >= 3:
            out.append(j)
    return out


def build_aux(catalog: pd.DataFrame, tr_cache: dict) -> dict:
    """Catalog side, once: words, word idf, per-word jamo readings, full readings, number groups, log GT.
    ``tr_cache``: vessel name -> jamo readings (the generator's / art["tr_cache"]); no file is read."""
    from en2ko import word_variants
    from hangul_match import HangulMatcher
    from rank.candidates import _name_words
    from translit import to_jamo
    names = catalog["vessel_name"].fillna("").astype(str).tolist()
    words = [_name_words(n) for n in names]
    df = {}
    for ws in words:
        for w in set(ws):
            df[w] = df.get(w, 0) + 1
    n = len(names)
    idf = {w: float(np.log((1 + n) / (1 + d)) + 1) for w, d in df.items()}
    rd = {w: list(dict.fromkeys(to_jamo(v.replace(" ", "")) for v in word_variants(w)))[:8]
          for w in df if w.isalpha()}
    full = [[r for r in (tr_cache.get(nm) or []) if r] for nm in names]
    return {"words": words, "idf": idf, "idf_oov": float(np.log(1 + n) + 1), "wread": rd, "full": full,
            "nums": [latin_num_groups(nm) for nm in names], "log_gt": HangulMatcher(catalog).log_gt}


def _num_state(a: tuple, b: tuple) -> int:
    if not a and not b:
        return 0
    if a and not b:
        return 7
    if b and not a:
        return 6
    sa, sb = set(a), set(b)
    if sa == sb:
        return 1
    if sa < sb:
        return 2
    if sb < sa:
        return 3
    return 4 if sa & sb else 5


def row_feats(row: dict, cands: np.ndarray, aux: dict) -> np.ndarray:
    """COLS (float64) for one record and its candidate catalog positions."""
    from rapidfuzz import fuzz
    from rapidfuzz.distance import OSA
    from common import is_hangul_name
    from rank.candidates import _name_words
    from translit import hangul_core, to_jamo
    A = aux
    idf, oov = A["idf"], A["idf_oov"]
    name = row.get("vessel_name")
    hangul = is_hangul_name(name)
    out = np.full((len(cands), len(COLS)), np.nan)
    tv = []
    for key in ("gross_tonnage", "international_gross_tonnage"):
        try:
            v = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and v > 0:
            tv.append(v)
    lg = A["log_gt"][cands]
    if tv:
        d = np.stack([lg - np.log(v) for v in tv])
        k = np.argmin(np.where(np.isnan(d), np.inf, np.abs(d)), axis=0)
        lr = d[k, np.arange(len(cands))]
    else:
        lr = np.full(len(cands), np.nan)
    a = np.abs(lr)
    out[:, CI["f_ton_lr"]] = lr
    out[:, CI["f_ton_bkt"]] = np.select([np.isnan(a), a < 0.01, a < 0.05, a < 0.15, a < 0.3, a < 0.75],
                                        [-1, 0, 1, 2, 3, 4], 5)
    if hangul:
        qn = hangul_num_groups(name)
        qws = _hangul_words(name)
        Q = to_jamo(hangul_core(name))
    else:
        qn = latin_num_groups(name)
        qws = _name_words(name) if isinstance(name, str) else []
        qset = sorted(set(qws))
        qi = np.array([idf.get(w, oov) for w in qset])
    for j, c in enumerate(cands):
        c = int(c)
        cws = A["words"][c]
        cn = A["nums"][c]
        r = out[j]
        r[CI["f_num_state"]], r[CI["f_num_q"]], r[CI["f_num_c"]] = _num_state(qn, cn), len(qn), len(cn)
        if hangul:
            if not qws or not Q:
                continue
            reads = A["full"][c]
            best = [max((fuzz.partial_ratio(qw, x) for x in reads), default=0.0) / 100 for qw in qws]
            calpha = [w for w in dict.fromkeys(cws) if A["wread"].get(w)]
            cm = np.array([max(fuzz.partial_ratio(x, Q) for x in A["wread"][w]) / 100 for w in calpha])
            qm = np.array(best) >= 0.8
            r[CI["f_q_in_c"]] = qm.mean()
            r[CI["f_q_part"]] = float(np.mean(best))
            r[CI["f_idf_share_q"]] = qm.mean()
            r[CI["f_tset"]] = fuzz.token_set_ratio(" ".join(qws), " ".join(A["wread"][w][0] for w in calpha)) / 100
            if len(calpha):
                cmb = cm >= 0.8
                ci = np.array([idf.get(w, oov) for w in calpha])
                r[CI["f_c_in_q"]] = cmb.mean()
                r[CI["f_idf_share_c"]] = float((ci * cmb).sum() / ci.sum())
                r[CI["f_idf_rare"]] = float(ci[cmb].max()) if cmb.any() else 0.0
                r[CI["f_idf_miss_c"]] = float(ci[~cmb].max()) if (~cmb).any() else 0.0
        else:
            if not qset:
                continue
            cset = sorted(set(cws))
            qbest = []
            for w in qset:
                if w in cset:
                    qbest.append(1.0)
                elif len(w) >= 4:
                    qbest.append(max((OSA.normalized_similarity(w, x) for x in cset if len(x) >= 4), default=0.0))
                else:
                    qbest.append(0.0)
            qm = np.array(qbest) >= 0.8
            r[CI["f_q_in_c"]] = qm.mean()
            r[CI["f_q_part"]] = float(np.mean(qbest))
            r[CI["f_idf_share_q"]] = float((qi * qm).sum() / qi.sum())
            r[CI["f_tset"]] = fuzz.token_set_ratio(" ".join(qset), " ".join(cset)) / 100
            r[CI["f_tsort"]] = fuzz.token_sort_ratio(" ".join(qws), " ".join(cws)) / 100
            if cset:
                cmb = np.array([x in qset or (len(x) >= 4 and max((OSA.normalized_similarity(x, w) for w in qset
                                                                    if len(w) >= 4), default=0.0) >= 0.8)
                                for x in cset], dtype=bool)
                ci = np.array([idf.get(w, oov) for w in cset])
                r[CI["f_c_in_q"]] = cmb.mean()
                r[CI["f_idf_share_c"]] = float((ci * cmb).sum() / ci.sum())
                r[CI["f_idf_rare"]] = float(ci[cmb].max()) if cmb.any() else 0.0
                r[CI["f_idf_miss_c"]] = float(ci[~cmb].max()) if (~cmb).any() else 0.0
    return out


def relative(pairs: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """Row-relative REL columns added to F (vectorised; pairs sorted by q, F aligned with pairs)."""
    q = pairs["q"].to_numpy()
    c = pairs["c"].to_numpy()
    hang = pairs["is_hangul"].to_numpy() > 0
    F["f_n_src"] = sum(pairs[s].to_numpy(dtype=float) for s in SRC if s in pairs)
    lat = np.fmax(np.fmax(pairs["nm_sim"].to_numpy(float), pairs["osa_sim"].to_numpy(float)),
                  F["f_tset"].to_numpy(float))
    F["f_nstr"] = np.where(hang, pairs["tr_sim"].to_numpy(float), lat)
    g = pd.DataFrame({"q": q, "c": c, "n": np.nan_to_num(F["f_nstr"].to_numpy(float), nan=-1.0),
                      "qin": F["f_q_in_c"].to_numpy(float), "idfq": F["f_idf_share_q"].to_numpy(float),
                      "bkt": F["f_ton_bkt"].to_numpy(float), "gr": pairs["gap_rank"].to_numpy()})
    grp = g.groupby("q", sort=False)
    F["f_nstr_gap"] = grp["n"].transform("max").to_numpy() - g["n"].to_numpy()
    order = np.lexsort((c, -g["n"].to_numpy(), q))
    qs = q[order]
    starts = np.r_[0, np.flatnonzero(qs[1:] != qs[:-1]) + 1]
    pos = np.arange(len(q)) - np.repeat(starts, np.diff(np.r_[starts, len(q)]))
    rank = np.empty(len(q), dtype=float)
    rank[order] = pos
    F["f_nstr_rank"] = rank
    F["f_qin_gap"] = grp["qin"].transform("max").to_numpy() - g["qin"].to_numpy()
    F["f_idfq_gap"] = grp["idfq"].transform("max").to_numpy() - g["idfq"].to_numpy()
    best = pd.Series(g["bkt"].to_numpy()[rank == 0], index=q[rank == 0])
    F["f_row_bestname_bkt"] = best.reindex(q).to_numpy()
    bt = g["gr"].to_numpy() == 0
    bng = pd.Series(F["f_nstr_gap"].to_numpy()[bt], index=q[bt])
    bng = bng[~bng.index.duplicated()]
    F["f_row_bestton_ngap"] = bng.reindex(q).to_numpy()
    return F


def frame_feats(row: dict, frame: pd.DataFrame, aux: dict) -> pd.DataFrame:
    """ALL columns (float32) for one record's candidate frame (CandidateGenerator.row_pairs output, before "q").
    Same dtype path as the prototype: COLS float64 -> float32, REL over them, everything float32."""
    F = pd.DataFrame(row_feats(row, frame["c"].to_numpy(), aux).astype(np.float32), columns=COLS)
    F = relative(frame.assign(q=0).reset_index(drop=True), F)
    return F[ALL].astype(np.float32)
