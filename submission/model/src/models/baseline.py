"""Official baseline features + RandomForest, retrained from source.

The distributed rf.pkl is not unpickled; the same feature code is retrained
on each fold so CV numbers are leak-free.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from baseline_lib import CatalogIndex, make_training_pairs, predict_ids  # noqa: F401 (re-exported)

_INDEX_CACHE: dict[int, CatalogIndex] = {}


def catalog_index(catalog: pd.DataFrame) -> CatalogIndex:
    key = id(catalog)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = CatalogIndex(catalog)
    return _INDEX_CACHE[key]


def fit_model(train: pd.DataFrame, index: CatalogIndex) -> RandomForestClassifier:
    x, y, recall = make_training_pairs(train, index)
    print(f"  baseline candidate recall on fit rows: {recall:.4f}", flush=True)
    model = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=0
    )
    model.fit(x, y)
    return model


def fit(train: pd.DataFrame, catalog: pd.DataFrame):
    index = catalog_index(catalog)
    model = fit_model(train, index)
    return lambda queries: predict_ids(model, queries, index)
