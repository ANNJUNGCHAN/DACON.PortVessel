"""Shared data loading, normalization and the official weighted metric."""

from __future__ import annotations

from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "open"


def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")


def load_catalog() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "vessel_catalog.csv", encoding="utf-8-sig")


def normalize_key(value: object) -> str:
    """NFKC, upper case, alphanumerics only. Empty string for missing values."""
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).upper()
    return "".join(ch for ch in value if ch.isalnum())


def is_hangul_name(value: object) -> bool:
    return isinstance(value, str) and any("가" <= ch <= "힣" for ch in value)


def weighted_accuracy(truth: pd.DataFrame, predicted_ids) -> dict[str, float]:
    """0.7 x Hangul-name accuracy + 0.3 x non-Hangul-name accuracy."""
    predicted = np.asarray(predicted_ids, dtype=object)
    if len(truth) != len(predicted):
        raise ValueError("Prediction count differs from validation data")
    hangul = truth["vessel_name"].map(is_hangul_name).to_numpy(dtype=bool)
    correct = truth["target_vessel_id"].to_numpy(dtype=object) == predicted
    ko = float(correct[hangul].mean()) if hangul.any() else float("nan")
    en = float(correct[~hangul].mean()) if (~hangul).any() else float("nan")
    return {
        "score": 0.7 * ko + 0.3 * en,
        "hangul": ko,
        "non_hangul": en,
        "n_hangul": int(hangul.sum()),
        "n_non_hangul": int((~hangul).sum()),
    }
