"""Hangul-name rows: tonnage-nearest catalog candidates scored by phonetic keys.

Everything here is built from the catalog alone; each query row is scored
independently of every other query row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from translit import phonetic_keys

try:
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    def _sim_matrix(query: list[str], choices: list[str]) -> np.ndarray:
        return process.cdist(query, choices, scorer=Levenshtein.normalized_similarity)
except ImportError:  # server image without rapidfuzz: same metric, slower
    def _levenshtein(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    def _sim_matrix(query: list[str], choices: list[str]) -> np.ndarray:
        out = np.empty((len(query), len(choices)))
        for i, q in enumerate(query):
            for j, c in enumerate(choices):
                longest = max(len(q), len(c))
                out[i, j] = 1.0 - _levenshtein(q, c) / longest if longest else 1.0
        return out


def _positive(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 0 else None


class HangulMatcher:
    def __init__(self, catalog: pd.DataFrame, n_gt: int = 3000, gap_weight: float = 0.5) -> None:
        self.ids = catalog["vessel_id"].astype(str).to_numpy()
        gt = pd.to_numeric(catalog["gross_tonnage"], errors="coerce").to_numpy(dtype=float)
        self.log_gt = np.where(np.isfinite(gt) & (gt > 0), np.log(np.where(gt > 0, gt, 1)), np.nan)
        valid = np.flatnonzero(np.isfinite(self.log_gt))
        self.order = valid[np.argsort(self.log_gt[valid], kind="stable")]
        self.log_sorted = self.log_gt[self.order]
        keys = [phonetic_keys(n) for n in catalog["vessel_name"]]
        # Two key variants per catalog name (second = letter-spelled acronym form if any).
        self.key0 = np.array([k[0] for k in keys], dtype=object)
        self.key1 = np.array([k[1] if len(k) > 1 else k[0] for k in keys], dtype=object)
        self.n_gt = n_gt
        self.gap_weight = gap_weight

    def _nearest(self, value: float | None) -> np.ndarray:
        if value is None:
            return np.empty(0, dtype=int)
        p = int(np.searchsorted(self.log_sorted, np.log(value)))
        lo, hi = max(0, p - self.n_gt), min(len(self.order), p + self.n_gt)
        window = self.order[lo:hi]
        gaps = np.abs(self.log_sorted[lo:hi] - np.log(value))
        return window[np.argsort(gaps, kind="stable")[: self.n_gt]]

    def candidates(self, row: dict) -> tuple[np.ndarray, np.ndarray]:
        """Candidate catalog positions and their best log-tonnage gap."""
        values = [v for v in (_positive(row.get("gross_tonnage")),
                              _positive(row.get("international_gross_tonnage"))) if v]
        if not values:
            return np.empty(0, dtype=int), np.empty(0)
        cands = np.unique(np.concatenate([self._nearest(v) for v in values]))
        gap = np.min([np.abs(self.log_gt[cands] - np.log(v)) for v in values], axis=0)
        return cands, gap

    def name_similarity(self, row: dict, cands: np.ndarray) -> np.ndarray:
        query = phonetic_keys(row.get("vessel_name"))
        sims = [_sim_matrix(query, self.key0[cands].tolist()),
                _sim_matrix(query, self.key1[cands].tolist())]
        return np.max([s.max(axis=0) for s in sims], axis=0)

    def predict_row(self, row: dict) -> str | None:
        cands, gap = self.candidates(row)
        if len(cands) == 0:
            return None
        score = self.name_similarity(row, cands) - self.gap_weight * np.minimum(gap, 1.0)
        # Stable tie-break on catalog position keeps the result order-independent.
        best = np.lexsort((cands, -score))[0]
        return self.ids[cands[best]]
