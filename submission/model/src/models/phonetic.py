"""Hangul rows -> phonetic matcher; other rows -> memory + baseline RF (as memory_v1).

Set MEMORY_FIRST to let a train-history hit override the phonetic matcher on
Hangul rows.
"""

from __future__ import annotations

import pandas as pd

from common import is_hangul_name
from hangul_match import HangulMatcher
from models import memory

MEMORY_FIRST = True

_MATCHER_CACHE: dict[int, HangulMatcher] = {}


def hangul_matcher(catalog: pd.DataFrame) -> HangulMatcher:
    key = id(catalog)
    if key not in _MATCHER_CACHE:
        _MATCHER_CACHE.clear()
        _MATCHER_CACHE[key] = HangulMatcher(catalog)
    return _MATCHER_CACHE[key]


def make_predictor(mem: memory.Memory, rf, catalog: pd.DataFrame):
    base = memory.make_predictor(mem, rf, catalog)
    matcher = hangul_matcher(catalog)

    def predict(queries: pd.DataFrame) -> list[str]:
        preds = list(base(queries))
        for i, row in enumerate(queries.to_dict("records")):
            if not is_hangul_name(row.get("vessel_name")):
                continue
            if MEMORY_FIRST and mem.resolve(row) is not None:
                continue
            hit = matcher.predict_row(row)
            if hit is not None:
                preds[i] = hit
        return preds

    return predict


def fit(train: pd.DataFrame, catalog: pd.DataFrame):
    art = memory.fit_artifacts(train, catalog)
    return make_predictor(art["memory"], art["rf"], catalog)


fit_artifacts = memory.fit_artifacts


def from_artifacts(art: dict, catalog: pd.DataFrame):
    return make_predictor(art["memory"], art["rf"], catalog)
