"""Train-memory post-hoc rules (C1 of reports/diag/action_plan.md): row-local, fit once on train.csv.

Both rules store plain dicts / sets of strings built at fit time, keyed by catalog ``vessel_id``,
so train.csv is NOT needed at inference and the catalog row order does not matter. At inference
each rule looks only at the row's own fields (call sign, vessel name) and the catalog, so a row
gets the same output when scored alone, in any subset or in any order. No cross-row statistics.

TrainTargetRule (port of experiments/diag/catalog_prior/tt_rule.py)
    penalty = 1 for a (row, candidate) pair when the candidate is a train target AND the row's
    normalized call sign is not that vessel's train call sign AND the row's normalized name is
    not among its train record names; 0 otherwise. Used as ``score -= w * penalty`` before the
    argmax. If every candidate of a row is penalized, the uniform shift leaves the argmax as is.

    Rationale (LB-free, reports/diag/catalog_prior_audit.md sec. 2.5): port calls per train vessel
    are {1: 3173, 2: 628, 3: 407, 4: 1643} rows. If the 4-capped pool had been split into train and
    test row by row (80/20), vessels with 3 train rows would be about as frequent as vessels with
    4 (n3/n4 >= 0.98). The observed n3/n4 is 407/1643 = 0.25, which matches a vessel-level split:
    train vessels are rarely test answers. A train vessel can still be the answer when the row
    proves it is that vessel, which is what the two gates check: in train every vessel has exactly
    one hashed call sign (and vice versa), and a seen vessel's later rows almost always repeat one
    of its train record names. The rule also explains why it helps without memorisation: dataset
    vessels cluster (sister ships, fleets), so train targets are frequent close competitors.
    Inputs: train.csv (targets, call signs, names), the catalog and the row itself only.

CallSignMemory
    normalized call sign -> train target ``vessel_id``, kept only for call signs that map to
    exactly one train target. A row whose call sign is in the map is predicted as that vessel
    (applied after the argmax). Inputs: train.csv and the row's own call sign only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import normalize_key


def _norm(values) -> np.ndarray:
    return pd.Series(values).map(normalize_key).to_numpy(dtype=object)


class TrainTargetRule:
    def __init__(self, train: pd.DataFrame, catalog: pd.DataFrame) -> None:
        known = set(catalog["vessel_id"].astype(str))
        tid = train["target_vessel_id"].astype(str).to_numpy()
        if not set(tid) <= known:
            raise ValueError("train targets missing from the catalog")
        self.call: dict[str, set[str]] = {}
        self.names: dict[str, set[str]] = {}
        for t, cs, nm in zip(tid, _norm(train["call_sign"]), _norm(train["vessel_name"])):
            self.call.setdefault(t, set())
            self.names.setdefault(t, set())
            if cs:
                self.call[t].add(cs)
            if nm:
                self.names[t].add(nm)
        self.targets: frozenset[str] = frozenset(self.call)

    def penalty(self, pairs: pd.DataFrame, queries: pd.DataFrame, catalog: pd.DataFrame) -> np.ndarray:
        """int8 per pair; ``pairs`` holds row index ``q`` (into ``queries``) and catalog position ``c``."""
        ids = catalog["vessel_id"].astype(str).to_numpy()
        q = pairs["q"].to_numpy()
        c = pairs["c"].to_numpy()
        qcs = _norm(queries["call_sign"])
        qnm = _norm(queries["vessel_name"])
        is_tt = np.fromiter((v in self.targets for v in ids), bool, len(ids))
        out = np.zeros(len(c), np.int8)
        for i in np.flatnonzero(is_tt[c]):
            t, qi = ids[c[i]], q[i]
            seen = (qcs[qi] and qcs[qi] in self.call[t]) or (qnm[qi] and qnm[qi] in self.names[t])
            out[i] = 0 if seen else 1
        return out


class CallSignMemory:
    def __init__(self, train: pd.DataFrame, keep_names: bool = False) -> None:
        targets: dict[str, set[str]] = {}
        for cs, t in zip(_norm(train["call_sign"]), train["target_vessel_id"].astype(str)):
            if cs:
                targets.setdefault(cs, set()).add(t)
        self.map: dict[str, str] = {cs: next(iter(ts)) for cs, ts in targets.items() if len(ts) == 1}
        if not keep_names:
            return
        # train record names per target (S9 cs_guard, cs_name_agreement); default off = the old object exactly
        self.names: dict[str, set[str]] = {}
        for nm, t in zip(train["vessel_name"], train["target_vessel_id"].astype(str)):
            if isinstance(nm, str) and nm.strip():
                self.names.setdefault(t, set()).add(nm)

    def lookup(self, queries: pd.DataFrame) -> dict[int, str]:
        """Row index -> remembered ``vessel_id`` for rows whose call sign is in the map."""
        out = {}
        for qi, cs in enumerate(_norm(queries["call_sign"])):
            if cs and cs in self.map:
                out[qi] = self.map[cs]
        return out


def cs_name_agreement(row: dict, vessel_pos: int, gen, record_names=()) -> float:
    """How well the record's name agrees with a call-sign vessel (S9 cs_guard), in [0, 1].

    Hangul record: best jamo similarity to the vessel's rule-based catalog readings (gen._tr_similarity) or to
    its Hangul train record names. Latin record: best OSA similarity of normalize_key names (catalog name, train
    record names). A record without a name returns 1.0 (nothing contradicts the call sign). Row-local."""
    from common import is_hangul_name
    from hangul_match import _sim_matrix
    from translit import hangul_core, to_jamo

    name = row.get("vessel_name")
    if not isinstance(name, str) or not normalize_key(name):
        return 1.0
    if is_hangul_name(name):
        best = float(np.nan_to_num(gen._tr_similarity(row, np.array([vessel_pos]))[0], nan=0.0))
        q = to_jamo(hangul_core(name))
        own = [to_jamo(hangul_core(n)) for n in record_names if is_hangul_name(n)]
        if q and own:
            best = max(best, float(_sim_matrix([q], own)[0].max()))
        return best
    from rapidfuzz.distance import OSA
    q = normalize_key(name)
    refs = [gen.norm[vessel_pos]] + [normalize_key(n) for n in record_names if not is_hangul_name(n)]
    return max(OSA.normalized_similarity(q, r) for r in refs if r) if any(refs) else 0.0


def tt_adjust(score: np.ndarray, penalty: np.ndarray, weight: float) -> np.ndarray:
    return np.asarray(score, dtype=np.float64) - float(weight) * penalty
