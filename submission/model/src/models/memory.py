"""Train-history memory resolver with baseline fallback.

Lookup tables are built only from the train rows passed to ``fit``:
normalized call sign -> vessel ids, normalized name -> vessel ids.
Each query row is resolved alone; no statistic spans query rows.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from common import normalize_key
from models import baseline


def _positive(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


class Memory:
    def __init__(self, train: pd.DataFrame, catalog: pd.DataFrame) -> None:
        self.by_call: dict[str, Counter] = defaultdict(Counter)
        self.by_name: dict[str, Counter] = defaultdict(Counter)
        for cs, name, vid in zip(train["call_sign"], train["vessel_name"], train["target_vessel_id"]):
            if (k := normalize_key(cs)):
                self.by_call[k][vid] += 1
            if (k := normalize_key(name)):
                self.by_name[k][vid] += 1
        self.catalog_gt = dict(zip(catalog["vessel_id"], pd.to_numeric(catalog["gross_tonnage"], errors="coerce")))

    def resolve(self, row: dict) -> str | None:
        call = self.by_call.get(normalize_key(row.get("call_sign")), Counter())
        name = self.by_name.get(normalize_key(row.get("vessel_name")), Counter())
        if not call and not name:
            return None
        votes = Counter()
        for vid, n in call.items():
            votes[vid] += 2 * n
        for vid, n in name.items():
            votes[vid] += n
        gross = _positive(row.get("gross_tonnage")) or _positive(row.get("international_gross_tonnage"))

        def gap(vid: str) -> float:
            cat = _positive(self.catalog_gt.get(vid))
            if gross is None or cat is None:
                return 1.0
            return min(abs(gross - cat) / max(gross, cat), 1.0)

        # Agreement of both keys dominates, then tonnage closeness, then counts.
        return max(votes, key=lambda v: (v in call and v in name, -round(gap(v), 2), votes[v], v))


def make_predictor(memory: Memory, rf, catalog: pd.DataFrame):
    index = baseline.catalog_index(catalog)

    def fallback(queries: pd.DataFrame) -> list[str]:
        return baseline.predict_ids(rf, queries, index)

    def predict(queries: pd.DataFrame) -> list[str]:
        rows = queries.to_dict("records")
        hits = [memory.resolve(r) for r in rows]
        miss = [i for i, h in enumerate(hits) if h is None]
        if miss:
            for i, pid in zip(miss, fallback(queries.iloc[miss])):
                hits[i] = pid
        return hits

    return predict


def fit_artifacts(train: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    rf = baseline.fit_model(train, baseline.catalog_index(catalog))
    return {"memory": Memory(train, catalog), "rf": rf}


def fit(train: pd.DataFrame, catalog: pd.DataFrame):
    art = fit_artifacts(train, catalog)
    return make_predictor(art["memory"], art["rf"], catalog)


def from_artifacts(art: dict, catalog: pd.DataFrame):
    return make_predictor(art["memory"], art["rf"], catalog)
