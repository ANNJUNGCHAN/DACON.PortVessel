"""S9w: LLM name-variant copies of train rows for ranker training (fit time only; never used at inference).

DACON talk 418501: an open pretrained LLM may rewrite train.csv vessel names into other notations (train data only).
The variants (experiments/s9_robust/llmgen30: Qwen3-32B-AWQ over the 5,851 train target vessels, filtered by
parse30.py) are read from a parquet file (vessel_id, lang, variant, op). Same construction as llmgen30/llmparts30.py:
  llm    row name -> a variant of its target vessel in the row's own language
  llmko  row name -> a Hangul variant (Latin rows become Hangul rows; attributes unchanged)
  llmabbr Latin rows only -> an abbreviation variant (op abbr*) of the target vessel
pick = crc32("<part>#<row position>") over the vessel's variants sorted by text (호-suffixed ones with probability
HO_SHARE). Rows without a variant are skipped.
Returns (copies, sources) like rank.noise_aug.noise_copies: sources = the copied train rows (leave-source-out)."""
from __future__ import annotations

import zlib

import pandas as pd

KEY_COLS = ["vessel_name", "gross_tonnage", "international_gross_tonnage", "flag_name", "vessel_type", "call_sign"]


HO_SHARE = 0.1  # real train Hangul rows ending in 호: 8.1%; the LLM gives ~1 호 variant per vessel


def _pick(c: list[str], key: str) -> int:
    """with probability HO_SHARE a 호-suffixed variant (if any), else crc32 over the other variants."""
    h = zlib.crc32(key.encode())
    ho = [j for j, x in enumerate(c) if x.endswith("호")]
    rest = [j for j in range(len(c)) if j not in ho]
    if ho and (not rest or zlib.crc32((key + "#ho").encode()) / 2**32 < HO_SHARE):
        return ho[h % len(ho)]
    return rest[h % len(rest)]


def llm_copies(train: pd.DataFrame, variants: pd.DataFrame, part: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from common import is_hangul_name
    if part not in ("llm", "llmko", "llmabbr"):
        raise ValueError(part)
    if part == "llmabbr":  # Latin rows only, abbreviation variants only (llmgen30/abbr31.py)
        variants = variants[(variants["lang"] == "la") & variants["op"].astype(str).str.startswith("abbr")]
    by = {k: g.sort_values("variant")["variant"].tolist() for k, g in variants.groupby(["vessel_id", "lang"])}
    rows, src = [], []
    base = train[KEY_COLS + ["target_vessel_id"]].reset_index(drop=True)
    for i, r in enumerate(base.to_dict("records")):
        lang = "ko" if (part == "llmko" or is_hangul_name(r["vessel_name"])) else "la"
        if part == "llmabbr" and lang == "ko":
            continue
        c = by.get((str(r["target_vessel_id"]), lang))
        if not c:
            continue
        r = dict(r)
        r["vessel_name"] = c[_pick(c, f"{part}#{i}")]
        rows.append(r)
        src.append(i)
    copies = pd.DataFrame(rows, columns=KEY_COLS + ["target_vessel_id"])
    for col in ("flag_name", "vessel_type", "call_sign"):
        copies[col] = copies[col].astype(object).where(copies[col].notna(), None)
    return copies, train.reset_index(drop=True).iloc[src].reset_index(drop=True)


def rule_abbr_copies(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S9w control (Codex R6): Latin train rows -> stress_v2 la_abbr_word or la_abbr_first (crc32 order, the other as
    fallback; experiments/s9_robust/llmgen30/abbr31.py part "rabbr"). Hangul rows and rows where both fail are
    skipped. Returns (copies, sources)."""
    import random

    from common import is_hangul_name
    from rank.noise_aug import _families
    families, _ = _families()
    rows, src = [], []
    base = train[KEY_COLS + ["target_vessel_id"]].reset_index(drop=True)
    for i, r in enumerate(base.to_dict("records")):
        if is_hangul_name(r["vessel_name"]):
            continue
        order = ["la_abbr_word", "la_abbr_first"]
        if zlib.crc32(f"rabbr#{i}".encode()) % 2:
            order.reverse()
        new = None
        for f in order:
            new = families[f][2](dict(r), random.Random(zlib.crc32(f"rabbr#{f}#{i}".encode())), {})
            if new is not None:
                break
        if new is None:
            continue
        r = dict(r)
        r["vessel_name"] = new["vessel_name"]
        rows.append(r)
        src.append(i)
    copies = pd.DataFrame(rows, columns=KEY_COLS + ["target_vessel_id"])
    for col in ("flag_name", "vessel_type", "call_sign"):
        copies[col] = copies[col].astype(object).where(copies[col].notna(), None)
    return copies, train.reset_index(drop=True).iloc[src].reset_index(drop=True)
