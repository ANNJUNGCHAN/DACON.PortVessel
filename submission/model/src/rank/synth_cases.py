"""Ranker-side synthetic rows for the case types the new evaluation data names (S4, reports/s4_ablation.md). Fit time
only: these rows are extra training rows for RankPipeline.fit(aug=...); nothing here runs at inference.

Port of experiments/s4/case_synth.py (Agent B), generators (a) and (b), line for line:
  same   same_name_rows       target = a member of a duplicated-name catalog group (normalize_key duplicates), so the
                              other members are the decoys. One Hangul row (en2ko reading, jamo noise 0.08 on half) and
                              one Latin row (catalog name upper-cased, one word-interior typo with p 0.2) per target and
                              pass; GT exact or with a 3-5 % gap (p 0.35); 10 % of the Latin rows without tonnage;
                              flag/type mapped to the train frame's record vocabulary; 10 % of the flags dropped.
  hton   hangul_tonnage_rows  Hangul rows (en2ko reading + jamo noise 0.12, 10 % with 호) of catalog vessels with
                              GT >= 1000 and a Latin name, each with one tonnage defect: both missing 40 %, one missing
                              10 %, x1.5 15 %, x0.67 15 %, DWT recorded as GT 20 %.
Every table comes from the ``train`` frame passed in (fold-train in CV); targets exclude every target of that frame.
Each row draws from its own random.Random(crc(kind, seed, vessel, pass)), so rows are deterministic and independent
of n. Parity with the prototype: tests/test_synth_cases.py.

Spec string (env RANKER_CASE in models/ranked.py): "hton:2000:0.5,same:600:0.5" = [(kind, n rows, weight)].
"""
from __future__ import annotations

import hashlib
import math
import random
import unicodedata
import zlib
from collections import Counter

import numpy as np
import pandas as pd

from common import is_hangul_name, normalize_key

QUERY_COLS = ["sample_id", "vessel_name", "call_sign", "gross_tonnage", "international_gross_tonnage",
              "flag_name", "vessel_type", "arrival_year", "target_vessel_id"]
HTON_MIX = {"miss_both": 0.40, "miss_one": 0.10, "x1.5": 0.15, "x0.67": 0.15, "dwt": 0.20}


def crc(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8"))


def fresh_call_sign(token: str) -> str:
    """A hashed call sign in the train format (CS_ + 20 hex) that no train row uses."""
    return "CS_" + hashlib.sha1(f"s4-unseen|{token}".encode("utf-8")).hexdigest()[:20].upper()


def faithful_typo(name: str, rng, n_ops: int):
    """n_ops delete / double / transpose edits of word-interior characters (upper-cased); the train typo generator."""
    s = list(unicodedata.normalize("NFKC", name).upper())
    for _ in range(n_ops):
        interior = [i for i in range(1, len(s) - 1) if s[i].isalnum() and s[i - 1].isalnum() and s[i + 1].isalnum()]
        if not interior:
            return None
        i = rng.choice(interior)
        op = rng.random()
        if op < 0.305:
            if s[i + 1].isalnum() and i + 2 < len(s) and s[i + 2].isalnum() and s[i + 1] != s[i]:
                s[i], s[i + 1] = s[i + 1], s[i]
            else:
                del s[i]
        elif op < 0.665:
            del s[i]
        else:
            s.insert(i, s[i])
    return "".join(s)


def dup_groups(catalog: pd.DataFrame) -> pd.DataFrame:
    """Catalog rows whose normalize_key name is shared with another catalog row: pos, vessel_id, nkey, size."""
    k = catalog["vessel_name"].map(normalize_key)
    k = k.where(k != "", None)
    cnt = k.map(k.value_counts())
    m = (cnt >= 2).to_numpy()
    out = pd.DataFrame({"pos": np.flatnonzero(m), "vessel_id": catalog["vessel_id"].astype(str).to_numpy()[m],
                        "nkey": k.to_numpy()[m], "size": cnt.to_numpy()[m].astype(int)})
    return out.reset_index(drop=True)


def localization_maps(train: pd.DataFrame, catalog: pd.DataFrame) -> tuple[dict, dict]:
    """English catalog flag / type -> the most common record value among the frame's rows with that target value."""
    cat = catalog.set_index("vessel_id")
    joined = train.join(cat[["flag_name", "vessel_type"]], on="target_vessel_id", rsuffix="_cat")
    flag_map = {k: Counter(g.dropna()).most_common(1)[0][0]
                for k, g in joined.groupby("flag_name_cat")["flag_name"] if g.notna().any()}
    type_map = {k: Counter(g.dropna()).most_common(1)[0][0]
                for k, g in joined.groupby("vessel_type_cat")["vessel_type"] if g.notna().any()}
    return flag_map, type_map


def _order(ids, kind: str, seed: int) -> list:
    return sorted(ids, key=lambda v: (crc("s4case", kind, seed, v), v))


def _gt(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) and v > 0 else float("nan")


def render_hangul(name: str, rng: random.Random, noise: float = 0.0, p_noise: float = 1.0, p_alt: float = 0.4,
                  p_nospace: float = 0.3, p_ho: float = 0.0) -> str | None:
    """A Hangul record for a Latin catalog name (en2ko readings). None when the reading has no Hangul or its digits
    differ from the catalog name's (the 'II read as 아이아이' artifact)."""
    from en2ko import name_variants
    from rank.finetune import jamo_noise
    from translit import name_digits
    variants = name_variants(name)
    if not variants:
        return None
    ko = variants[0] if (rng.random() >= p_alt or len(variants) == 1) else rng.choice(variants[1:])
    if rng.random() < p_nospace:
        ko = ko.replace(" ", "")
    if noise and rng.random() < p_noise:
        ko = jamo_noise(ko, rng, noise)
    if p_ho and rng.random() < p_ho:
        ko += "호"
    if not is_hangul_name(ko) or name_digits(ko) != name_digits(name):
        return None
    return ko


def _typo1(name: str, rng: random.Random) -> str:
    t = faithful_typo(name, rng, 1)
    return t if t else name


def _frame(rows: list[dict]) -> pd.DataFrame:
    cols = QUERY_COLS + ["s4_kind", "s4_mode", "s4_group"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in ("vessel_name", "call_sign", "flag_name", "vessel_type", "sample_id", "target_vessel_id"):
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    for c in ("gross_tonnage", "international_gross_tonnage"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df[cols]


# --------------------------------------------------------------------------------------------- (a) same name
def same_name_row(crow: dict, lang: str, rng: random.Random, flag_map: dict, type_map: dict, token: str,
                  p_gap: float, p_nogt: float, p_flag_drop: float, p_typo: float, ko_noise: float) -> dict | None:
    name = str(crow["vessel_name"])
    if lang == "ko":
        rec = render_hangul(name, rng, noise=ko_noise, p_noise=0.5)
        if rec is None:
            return None
    else:
        rec = name.upper()
        if p_typo and rng.random() < p_typo:
            rec = _typo1(rec, rng)
    gt = _gt(crow["gross_tonnage"])
    mode = "exact"
    if np.isfinite(gt) and rng.random() < p_gap:
        gt = float(max(1.0, np.round(gt * (1 + rng.choice([-1, 1]) * rng.uniform(0.03, 0.05)))))
        mode = "gap"
    if lang == "lat" and p_nogt and rng.random() < p_nogt:
        gt, mode = float("nan"), "nogt"
    flag = flag_map.get(crow["flag_name"])
    if p_flag_drop and rng.random() < p_flag_drop:
        flag = None
    return {"vessel_name": rec, "call_sign": fresh_call_sign(token), "gross_tonnage": gt,
            "international_gross_tonnage": gt, "flag_name": flag, "vessel_type": type_map.get(crow["vessel_type"]),
            "arrival_year": 2024, "target_vessel_id": str(crow["vessel_id"]), "s4_mode": f"{lang}_{mode}"}


def same_name_rows(train: pd.DataFrame, catalog: pd.DataFrame, n: int, seed: int = 0, langs=("ko", "lat"),
                   p_gap: float = 0.35, p_nogt_lat: float = 0.1, p_flag_drop: float = 0.1, p_typo_lat: float = 0.2,
                   ko_noise: float = 0.08, exclude_ids=(), keys=None, tag: str = "SN") -> pd.DataFrame:
    """(a) n rows whose targets are duplicated-name group members. ``keys``: restrict to these group keys."""
    g = dup_groups(catalog)
    if keys is not None:
        g = g[g["nkey"].isin(set(keys))]
    banned = set(train["target_vessel_id"].astype(str)) | {str(v) for v in exclude_ids}
    g = g[~g["vessel_id"].isin(banned)]
    if not len(g) or n <= 0:
        return _frame([])
    flag_map, type_map = localization_maps(train, catalog)
    crec = catalog.to_dict("records")
    pos_of = dict(zip(g["vessel_id"], g["pos"]))
    key_of = dict(zip(g["vessel_id"], g["nkey"]))
    order = _order(g["vessel_id"].tolist(), "same", seed)
    rows, j = [], 0
    while len(rows) < n and j < 50:
        added = 0
        for vid in order:
            for lang in langs:
                rng = random.Random(crc("s4same", seed, vid, j, lang))
                r = same_name_row(crec[pos_of[vid]], lang, rng, flag_map, type_map, f"{tag}|{seed}|{vid}|{j}|{lang}",
                                  p_gap, p_nogt_lat, p_flag_drop, p_typo_lat, ko_noise)
                if r is None:
                    continue
                r.update(sample_id=f"S4{tag}_{len(rows):05d}", s4_kind="same_name", s4_group=key_of[vid])
                rows.append(r)
                added += 1
                if len(rows) >= n:
                    break
            if len(rows) >= n:
                break
        if not added:
            break
        j += 1
    return _frame(rows)


# --------------------------------------------------------------------------------------------- (b) Hangul tonnage
def hangul_tonnage_rows(train: pd.DataFrame, catalog: pd.DataFrame, n: int, seed: int = 0, mix: dict | None = None,
                        exclude_ids=(), noise: float = 0.12, p_flag_drop: float = 0.1, tag: str = "HT") -> pd.DataFrame:
    """(b) n Hangul rows of catalog vessels outside the train frame, each with a tonnage defect drawn from ``mix``."""
    mix = dict(mix or HTON_MIX)
    modes, probs = list(mix), np.cumsum([mix[m] for m in mix]) / sum(mix.values())
    gt = pd.to_numeric(catalog["gross_tonnage"], errors="coerce")
    names = catalog["vessel_name"].fillna("").astype(str)
    ok = (gt >= 1000).to_numpy() & names.str.contains("[A-Za-z]{3}").to_numpy()
    ids = catalog["vessel_id"].astype(str).to_numpy()
    banned = set(train["target_vessel_id"].astype(str)) | {str(v) for v in exclude_ids}
    pool = [v for v, o in zip(ids, ok) if o and v not in banned]
    if not pool or n <= 0:
        return _frame([])
    pos_of = {v: i for i, v in enumerate(ids)}
    flag_map, type_map = localization_maps(train, catalog)
    crec = catalog.to_dict("records")
    rows, j = [], 0
    order = _order(pool, "hton", seed)
    while len(rows) < n and j < 5:
        for vid in order:
            crow = crec[pos_of[vid]]
            rng = random.Random(crc("s4hton", seed, vid, j))
            rec = render_hangul(str(crow["vessel_name"]), rng, noise=noise, p_noise=1.0, p_ho=0.1)
            if rec is None:
                continue
            mode = modes[int(np.searchsorted(probs, rng.random(), side="right"))]
            g = _gt(crow["gross_tonnage"])
            gg = ig = g
            if mode == "miss_both":
                gg = ig = float("nan")
            elif mode == "miss_one":
                if rng.random() < 0.5:
                    gg = float("nan")
                else:
                    ig = float("nan")
            elif mode in ("x1.5", "x0.67"):
                gg = ig = float(max(1.0, np.round(g * (1.5 if mode == "x1.5" else 1 / 1.5))))
            else:  # dwt
                d = _gt(crow.get("deadweight_tonnage"))
                v = d if (np.isfinite(d) and abs(math.log(d / g)) > 0.1) else g * 1.6
                gg = ig = float(max(1.0, np.round(v)))
            flag = flag_map.get(crow["flag_name"])
            if rng.random() < p_flag_drop:
                flag = None
            rows.append({"sample_id": f"S4{tag}_{len(rows):05d}", "vessel_name": rec,
                         "call_sign": fresh_call_sign(f"{tag}|{seed}|{vid}|{j}"), "gross_tonnage": gg,
                         "international_gross_tonnage": ig, "flag_name": flag,
                         "vessel_type": type_map.get(crow["vessel_type"]), "arrival_year": 2024,
                         "target_vessel_id": vid, "s4_kind": "hangul_tonnage", "s4_mode": mode, "s4_group": ""})
            if len(rows) >= n:
                break
        j += 1
    return _frame(rows)


# --------------------------------------------------------------------------------------------- specs
KINDS = ("same", "hton")


def parse_spec(spec: str) -> list[tuple[str, int, float]]:
    """'hton:2000:0.5,same:600:0.5' -> [(kind, n, weight)]; weight defaults to 0.5."""
    out = []
    for part in filter(None, (s.strip() for s in (spec or "").split(","))):
        bits = part.split(":")
        if bits[0] not in KINDS:
            raise ValueError(f"unknown case kind {bits[0]!r} (have {KINDS})")
        out.append((bits[0], int(bits[1]) if len(bits) > 1 else 1000, float(bits[2]) if len(bits) > 2 else 0.5))
    return out


def case_sets(train: pd.DataFrame, catalog: pd.DataFrame, spec, seed: int = 0, same_keys=None,
              log=print) -> list[tuple[str, pd.DataFrame, float]]:
    """[(name, queries, weight)] for a spec. ``same_keys``: restrict (a) to these duplicated-name groups (None = all;
    the CV restricted it to the half of the groups the S4 suite does not evaluate on). Flag/type are already in the
    record vocabulary: do not localize_attributes these rows."""
    spec = parse_spec(spec) if isinstance(spec, str) else spec
    out = []
    for kind, n, w in spec:
        if kind == "same":
            q = same_name_rows(train, catalog, n, seed=seed, keys=same_keys)
        else:
            q = hangul_tonnage_rows(train, catalog, n, seed=seed)
        out.append((f"case_{kind}{n}", q, w))
    log(f"synth_cases: {', '.join(f'{name} {len(q)} rows (w {w:g})' for name, q, w in out)}")
    return out
