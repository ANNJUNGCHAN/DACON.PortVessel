"""hbag (2026-10-31): Hangul ranker as an average of the five fold-trained rankers (same recipe as the shipped S9o
Hangul ranker, each fitted on 4/5 of the train vessels). Packaged as rank/bagrank.py; LangSplit.ko is replaced by a
BagRanker so Latin rows keep E08's ranker byte-for-byte."""
from __future__ import annotations

import numpy as np


class BagRanker:
    name = "bag"

    def __init__(self, members, extra=None):
        self.members = list(members)
        self.extra = extra  # optional callable(X) -> X with extra feature columns (e5ko v4 features)

    def fit(self, X, y, qid, weight):  # pragma: no cover - assembled from fitted rankers only
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.members], axis=0)


class WeightedRanker:
    """Weighted sum of fitted rankers' scores (each selects its own columns from X): e.g. the shipped S9o plus the
    e5 refit minus its no-e5 twin (transfers the e5 effect onto S9o), or an average."""
    name = "weighted"

    def __init__(self, members, weights):
        assert len(members) == len(weights)
        self.members, self.weights = list(members), [float(w) for w in weights]

    def fit(self, X, y, qid, weight):  # pragma: no cover - assembled from fitted rankers only
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        return sum(w * m.predict(X) for m, w in zip(self.members, self.weights))
