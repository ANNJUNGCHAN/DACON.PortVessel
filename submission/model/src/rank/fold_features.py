"""Pair features learned from the training fold only.

- flag_p / type_p: P(catalog flag | record flag), P(catalog type | record type),
  estimated from train rows joined to their target catalog vessel.
- mem_cs / mem_nm: how many train rows with the same call sign / normalized
  name point at this candidate; *_tot is the row-level total for the key.

With ``exclude_self=True`` (scoring the training rows themselves) each row's
own contribution is removed, i.e. the counts are leave-one-out, so the model
never sees a feature that trivially contains its own label.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from common import normalize_key

FOLD_FEATURES = (
    "flag_p", "flag_q_missing", "type_p",
    "mem_cs", "mem_cs_tot", "mem_nm", "mem_nm_tot",
)


def _clean(series: pd.Series) -> np.ndarray:
    return series.fillna("").astype(str).str.strip().to_numpy(dtype=object)


class FoldFeatures:
    def __init__(self, train: pd.DataFrame, catalog: pd.DataFrame) -> None:
        self.cflag = _clean(catalog["flag_name"])
        self.ctype = _clean(catalog["vessel_type"])
        pos_of = {v: i for i, v in enumerate(catalog["vessel_id"].astype(str))}
        tpos = np.array([pos_of[v] for v in train["target_vessel_id"].astype(str)])

        qf, qt = _clean(train["flag_name"]), _clean(train["vessel_type"])
        self.flag_pair = Counter(zip(qf, self.cflag[tpos]))
        self.flag_tot = Counter(qf)
        self.type_pair = Counter(zip(qt, self.ctype[tpos]))
        self.type_tot = Counter(qt)

        cs = [normalize_key(v) for v in train["call_sign"]]
        nm = [normalize_key(v) for v in train["vessel_name"]]
        self.cs_pair = Counter((k, p) for k, p in zip(cs, tpos) if k)
        self.cs_tot = Counter(k for k in cs if k)
        self.nm_pair = Counter((k, p) for k, p in zip(nm, tpos) if k)
        self.nm_tot = Counter(k for k in nm if k)
        self.pos_of = pos_of

    @staticmethod
    def _ratio(pair_n: np.ndarray, tot_n: np.ndarray) -> np.ndarray:
        # Laplace-smoothed share; NaN when the record value was never seen.
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot_n > 0, pair_n / (tot_n + 1.0), np.nan)

    def transform(self, pairs: pd.DataFrame, queries: pd.DataFrame, exclude_self: bool) -> pd.DataFrame:
        q = pairs["q"].to_numpy()
        c = pairs["c"].to_numpy()
        qf = _clean(queries["flag_name"])[q]
        qt = _clean(queries["vessel_type"])[q]
        cs = np.array([normalize_key(v) for v in queries["call_sign"]], dtype=object)[q]
        nm = np.array([normalize_key(v) for v in queries["vessel_name"]], dtype=object)[q]
        cf, ct = self.cflag[c], self.ctype[c]

        if exclude_self:
            own = np.array([self.pos_of[v] for v in queries["target_vessel_id"].astype(str)])[q]
            is_own = (c == own).astype(float)
            # flag_pair / type_pair count (record value, catalog value of the TARGET): the row itself contributes to
            # every candidate whose catalog flag / type equals its target's, not only to the target (S9r fix; before,
            # only the target lost the row's count, so at fit the target looked rarer than its same-flag rivals).
            own_f = (cf == self.cflag[own]).astype(float)
            own_t = (ct == self.ctype[own]).astype(float)
            if not getattr(self, "lo_fix", True):  # pre-S9r behaviour (pickled pipelines without the attribute: fixed)
                own_f = own_t = is_own
        else:
            is_own = own_f = own_t = np.zeros(len(c))
        selfn = 1.0 if exclude_self else 0.0

        def lookup(counter, keys):
            return np.fromiter((counter.get(k, 0) for k in keys), dtype=float, count=len(c))

        flag_pair = lookup(self.flag_pair, zip(qf, cf)) - own_f * (qf != "")
        flag_tot = lookup(self.flag_tot, qf) - selfn * (qf != "")
        type_pair = lookup(self.type_pair, zip(qt, ct)) - own_t * (qt != "")
        type_tot = lookup(self.type_tot, qt) - selfn * (qt != "")
        cs_pair = lookup(self.cs_pair, zip(cs, c)) - is_own * (cs != "")
        cs_tot = lookup(self.cs_tot, cs) - selfn * (cs != "")
        nm_pair = lookup(self.nm_pair, zip(nm, c)) - is_own * (nm != "")
        nm_tot = lookup(self.nm_tot, nm) - selfn * (nm != "")

        flag_p = self._ratio(flag_pair, flag_tot)
        flag_p[qf == ""] = np.nan
        type_p = self._ratio(type_pair, type_tot)
        type_p[qt == ""] = np.nan
        return pd.DataFrame({
            "flag_p": flag_p,
            "flag_q_missing": (qf == "").astype(np.int8),
            "type_p": type_p,
            "mem_cs": cs_pair, "mem_cs_tot": cs_tot,
            "mem_nm": nm_pair, "mem_nm_tot": nm_tot,
        }, index=pairs.index)

    def transform_lso(self, pairs: pd.DataFrame, queries: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
        """Leave-SOURCE-out features for noise-augmented copies of train rows (S9 item 4a; port of
        experiments/s9_robust/item4/exp4.fold_lso). ``src`` row i = the train row copy i was made from. A copy may
        carry a different call sign / name / flag / type than its source, so exclude_self=True would be wrong:
        the source row's contribution is removed wherever the copy's key equals the source's key (mem_cs/_tot,
        mem_nm/_tot, and flag_p / type_p recomputed from the adjusted counts). With src = queries this equals
        transform(exclude_self=True)."""
        X = self.transform(pairs, queries, exclude_self=False)
        q = pairs["q"].to_numpy()
        c = pairs["c"].to_numpy()
        own = np.array([self.pos_of[v] for v in queries["target_vessel_id"].astype(str)])[q]
        is_own = (c == own).astype(float)

        def same(fn, col):
            a = np.array([fn(v) for v in queries[col]], dtype=object)
            b = np.array([fn(v) for v in src[col]], dtype=object)
            return ((a == b) & (b != "")).astype(float)[q]

        def cl(s):
            return _clean(pd.Series([s]))[0]

        s_cs, s_nm = same(normalize_key, "call_sign"), same(normalize_key, "vessel_name")
        s_fl, s_ty = same(cl, "flag_name"), same(cl, "vessel_type")
        X["mem_cs"] -= is_own * s_cs
        X["mem_cs_tot"] -= s_cs
        X["mem_nm"] -= is_own * s_nm
        X["mem_nm_tot"] -= s_nm
        qf = _clean(queries["flag_name"])[q]
        qt = _clean(queries["vessel_type"])[q]
        cf, ct = self.cflag[c], self.ctype[c]
        fixed = getattr(self, "lo_fix", True)
        for col, pair_c, tot_c, qv, cv, s, ccat in (("flag_p", self.flag_pair, self.flag_tot, qf, cf, s_fl, self.cflag),
                                                    ("type_p", self.type_pair, self.type_tot, qt, ct, s_ty, self.ctype)):
            same_cat = (cv == ccat[own]).astype(float) if fixed else is_own  # S9r fix, as in transform
            pn = np.fromiter((pair_c.get(kv, 0) for kv in zip(qv, cv)), float, len(c)) - same_cat * s
            tn = np.fromiter((tot_c.get(kv, 0) for kv in qv), float, len(c)) - s
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(tn > 0, pn / (tn + 1.0), np.nan)
            r[qv == ""] = np.nan
            X[col] = r
        return X
