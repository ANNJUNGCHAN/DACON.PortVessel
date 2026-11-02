"""S9 item 4a: noise-augmented copies of train rows for ranker training (fit time only; never used at inference).

Port of experiments/s9_robust/item4/aug.py make_rows (RESULT_4.md "Porting spec" 4). One perturbed copy per
selected train row, drawn from the S9 PLAN mixture WITHOUT its clean share (tonnage 15 : call sign 10 : name 25 :
two-family mixture 20, renormalised), with the stress_v2 perturbation families and the same in-group fallback.
The family code is imported from experiments/diag/stress_v2/families.py (+ hangul_fragility/stress.py) at call
time; the record context (call-sign pool for cs_other, Hangul first words, flags, types) comes from the given
train rows only, so copies of fold-train rows never see validation vessels.

Seeds (``seed`` = 4242 in the prototype; the shift CV uses 20261020), for train row gq (the frame index label):
  group / family pick  random.Random(seed*1_000_003 + 7919*k + 104729*rep + gq), rep = 0
  family rng           crc32("aug<seed>#<family>#<tag>#0#<gq>")
  jamo-noise families  read row["_gq"] = gq + 1_000_000*(1 + 2k + rep)
  selection            crc32("sub#<tag>#<gq>") / 2**32 < frac
with (k, tag) = (fold, str(fold)) for a CV fold (reproduces the prototype's rep-0 copies of that fold) and
(5, "full") for the full-train fit (fold=None).
"""
from __future__ import annotations

import random
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLS = ["vessel_name", "gross_tonnage", "international_gross_tonnage", "flag_name", "vessel_type", "call_sign"]
REPS = 2  # copies per row and fold in the prototype; only rep 0 is drawn here (the +50% setting)
FULL_K = 5  # fold index used in the integer seeds for the full-train fit (CV folds are 0..4)
TON = ["ton_x0.5", "ton_x0.67", "ton_x0.8", "ton_x1.25", "ton_x1.5", "ton_x2", "ton_gt_only_x1.5", "ton_dwt",
       "ton_miss_gt", "ton_miss_igt", "ton_miss_both"]
CS = ["cs_miss", "cs_other"]
NAME = {
    "la": ["la_word_drop_first", "la_word_drop_last", "la_word_swap", "la_trunc", "la_num_drop", "la_num_change",
           "la_num_words", "la_num_roman", "la_prefix_M/V", "la_prefix_MV", "la_prefix_MT", "la_space_del",
           "la_space_add", "la_hyphen", "la_typo1", "la_typo2", "la_ocr", "la_sub1"],
    "ko": ["ko_word_drop_first", "ko_word_drop_last", "ko_word_swap", "ko_trunc", "ko_syl_drop", "ko_num_drop",
           "ko_num_change", "ko_num_sino", "ko_num_eng", "ko_num_eng_glued", "ko_num_native", "ko_num_roman",
           "ko_prefix_embeui", "ko_prefix_M/V", "ko_space_del", "ko_space_add", "ko_jamo0.15", "ko_jamo0.30"],
}


def _families():
    """(FAMILIES, contexts) from experiments/diag/stress_v2 (training only)."""
    import importlib.util
    from common import ROOT
    d = ROOT / "experiments" / "diag" / "stress_v2"
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    from families import FAMILIES
    spec = importlib.util.spec_from_file_location("stress_v2_run", d / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    return FAMILIES, run.contexts


def _changed(new: dict, row: dict) -> bool:
    return any(not (pd.isna(new[c]) and pd.isna(row[c])) and new[c] != row[c] for c in KEY_COLS)


def _apply(families, fam_list, first, row, ctx, tag, pick, seed):
    rest = [f for f in fam_list if f != first]
    pick.shuffle(rest)
    for fam in [first] + rest:
        rng = random.Random(zlib.crc32(f"aug{seed}#{fam}#{tag}".encode()))
        new = families[fam][2](dict(row), rng, ctx)
        if new is not None and _changed(new, row):
            return fam, new
    return None, None


def noise_copies(train: pd.DataFrame, catalog: pd.DataFrame, frac: float = 0.5, seed: int = 4242,
                 fold: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perturbed copies of about ``frac`` of ``train``'s rows (frac <= 1; one copy per selected row).

    ``train``: the rows to copy (the full train, or fold-train rows with their original index labels = gq).
    Returns (copies: KEY_COLS + target_vessel_id, one row per copy; sources: the train row of each copy, same
    order). Copies keep the source's target vessel. A selected row whose families all fail is skipped."""
    from common import is_hangul_name
    if not 0 < frac <= 1:
        raise ValueError(f"frac {frac!r}: expected (0, 1]")
    families, contexts = _families()
    k, tag = (FULL_K, "full") if fold is None else (int(fold), str(int(fold)))
    rep = 0
    ctx = contexts(train.reset_index(drop=True), catalog, np.arange(len(train)))
    crec = catalog.set_index("vessel_id").to_dict("index")
    out, src = [], []
    for gq, row in zip(train.index, train.to_dict("records")):
        gq = int(gq)
        if zlib.crc32(f"sub#{tag}#{gq}".encode()) / 2**32 >= frac:
            continue
        row0 = dict(row)
        ctx["crow"] = crec[row0["target_vessel_id"]]
        lang = "ko" if is_hangul_name(row0["vessel_name"]) else "la"
        row0["_gq"] = gq + 1_000_000 * (1 + k * REPS + rep)
        pick = random.Random(seed * 1_000_003 + 7919 * k + 104729 * rep + gq)
        u = pick.random() * 0.70
        if u < 0.15:
            comps = ["tonnage"]
        elif u < 0.25:
            comps = ["callsign"]
        elif u < 0.50:
            comps = ["name"]
        else:
            comps = sorted(pick.sample(["tonnage", "callsign", "name"], 2))
        new, fams = dict(row0), []
        for g in comps:
            lst = CS if g == "callsign" else (TON if g == "tonnage" else NAME[lang])
            first = pick.choice(lst)
            fam, nxt = _apply(families, [first] if g == "callsign" else lst, first, new, ctx, f"{tag}#{rep}#{gq}",
                              pick, seed)
            if fam is None:
                continue
            fams.append(fam)
            new = nxt
        if not fams:
            continue
        out.append({c: new[c] for c in KEY_COLS + ["target_vessel_id"]})
        src.append(gq)
    copies = pd.DataFrame(out, columns=KEY_COLS + ["target_vessel_id"])
    for c in ("flag_name", "vessel_type", "call_sign"):
        copies[c] = copies[c].astype(object).where(copies[c].notna(), None)
    return copies, train.loc[src].reset_index(drop=True)
