"""Candidate generation + fold features + LightGBM lambdarank.

v3: en2ko transliteration similarity, row-relative features, and training-time
augmentation with noisy synthetic Hangul rows rendered from the catalog.
"""

from __future__ import annotations

import os

import pandas as pd

from rank.pipeline import RankPipeline, candidate_generator, set_tr_cache

RANKER = "lgb_lambdarank"
SYNTH_ROWS, SYNTH_NOISE, SYNTH_WEIGHT, SYNTH_SEED = 4000, 0.12, 0.5, 0
SYNTH_EN_ROWS, SYNTH_EN_WEIGHT = 4000, 0.5  # misspelled Latin rows (v5)
SYNTH_RENAME_ROWS, SYNTH_RENAME_WEIGHT = 2000, 0.5  # fleet renames: OCEAN FINCH -> BLUE Finch
# S4 fit-time flags (reports/s4_impl.md), default off = the S3 training data:
#   RANKER_SYNTHFIX=1  C4.1: v5 synthetic Hangul set without digit_state 3/4 targets + real-looking GT gaps
#   RANKER_SYNTH_V6=1  C4.2: + data-driven Hangul sets (v4 4000 + v1fix 4000, weight 0.5), rank/synth_v6.py
SYNTHFIX = os.environ.get("RANKER_SYNTHFIX", "") == "1"
SYNTH_V6 = os.environ.get("RANKER_SYNTH_V6", "") == "1"
# S9 item 4a (experiments/s9_robust/item4/RESULT_4.md), default off: RANKER_NOISE_AUG=<frac> adds noise-augmented
# copies of about frac of the train rows (rank/noise_aug.py, seed NOISE_AUG_SEED) with weight RANKER_NOISE_AUG_W
# (default 0.5) x a real row of the same language, fold features leave-source-out.
NOISE_AUG = float(os.environ.get("RANKER_NOISE_AUG") or 0)
NOISE_AUG_W = float(os.environ.get("RANKER_NOISE_AUG_W") or 0.5)
NOISE_AUG_SEED = 4242
# S9w (experiments/s9_robust/llmgen30), default off: RANKER_LLM_AUG=llm,llmko adds LLM name-variant copies of the train
# rows (rank/llm_aug.py; variants read from RANKER_LLM_AUG_FILE) with weight RANKER_LLM_AUG_W (default 0.5).
LLM_AUG = [p for p in (os.environ.get("RANKER_LLM_AUG") or "").split(",") if p]
LLM_AUG_W = float(os.environ.get("RANKER_LLM_AUG_W") or 0.5)
LLM_AUG_FILE = os.environ.get("RANKER_LLM_AUG_FILE", "")
# RANKER_RULE_ABBR_W=<w> (default off): rule abbreviation copies of the Latin train rows (rank/llm_aug.rule_abbr_copies).
RULE_ABBR_W = float(os.environ.get("RANKER_RULE_ABBR_W") or 0)


def synthetic_augmentation(train: pd.DataFrame, catalog: pd.DataFrame, cand_opts: dict | None = None,
                           synthfix: bool | None = None, synth_v6: bool | None = None,
                           noise_aug: float | None = None, noise_w: float | None = None):
    """Synthetic training rows [(queries, pairs, weight), ...]. Every train-derived statistic comes from
    ``train`` (the fold-train in CV). ``cand_opts``: the pipeline's CandidateGenerator options, so the
    synthetic pairs match the real ones. ``synthfix`` / ``synth_v6``: None -> the env flags above.
    ``noise_aug`` / ``noise_w`` (None -> NOISE_AUG / NOISE_AUG_W): > 0 appends the item
    (copies, pairs, noise_w, source rows) of rank.noise_aug.noise_copies(train, frac=noise_aug)."""
    from rank.finetune import localize_attributes, synthetic_queries

    from rank.finetune import synthetic_latin_queries, synthetic_rename_queries

    synthfix = SYNTHFIX if synthfix is None else synthfix
    synth_v6 = SYNTH_V6 if synth_v6 is None else synth_v6
    gen = candidate_generator(catalog, **(cand_opts or {}))
    synth = synthetic_queries(catalog, SYNTH_ROWS, SYNTH_SEED, noise=SYNTH_NOISE)
    synth_pairs = gen.build(synth)
    if synthfix:
        from rank.synth_v6 import synthfix as fix
        synth, synth_pairs = fix(synth, synth_pairs, train, catalog, gen)
    augs = [(localize_attributes(synth, train, catalog, seed=1), synth_pairs, SYNTH_WEIGHT)]
    if SYNTH_EN_ROWS:
        latin = synthetic_latin_queries(catalog, SYNTH_EN_ROWS, SYNTH_SEED)
        augs.append((localize_attributes(latin, train, catalog, seed=3), gen.build(latin), SYNTH_EN_WEIGHT))
    if SYNTH_RENAME_ROWS:
        renames = synthetic_rename_queries(catalog, SYNTH_RENAME_ROWS, SYNTH_SEED)
        augs.append((localize_attributes(renames, train, catalog, seed=5), gen.build(renames), SYNTH_RENAME_WEIGHT))
    if synth_v6:
        from rank.synth_v6 import hangul_sets
        augs += hangul_sets(train, catalog, gen)
    noise_aug = NOISE_AUG if noise_aug is None else noise_aug
    if noise_aug:
        from rank.noise_aug import noise_copies
        copies, sources = noise_copies(train, catalog, frac=noise_aug, seed=NOISE_AUG_SEED)
        augs.append((copies, gen.build(copies), NOISE_AUG_W if noise_w is None else noise_w, sources))
        print(f"noise aug: {len(copies)} copies of {len(train)} train rows, {len(augs[-1][1])} pairs")
    if LLM_AUG:
        from rank.llm_aug import llm_copies
        variants = pd.read_parquet(LLM_AUG_FILE)
        for part in LLM_AUG:
            copies, sources = llm_copies(train, variants, part)
            augs.append((copies, gen.build(copies), LLM_AUG_W, sources))
            print(f"llm aug {part}: {len(copies)} copies, {len(augs[-1][1])} pairs, w {LLM_AUG_W}")
    if RULE_ABBR_W:
        from rank.llm_aug import rule_abbr_copies
        copies, sources = rule_abbr_copies(train)
        augs.append((copies, gen.build(copies), RULE_ABBR_W, sources))
        print(f"rule abbr aug: {len(copies)} copies, {len(augs[-1][1])} pairs, w {RULE_ABBR_W}")
    return augs


def fit_artifacts(train: pd.DataFrame, catalog: pd.DataFrame, pairs: pd.DataFrame | None = None) -> dict:
    aug = synthetic_augmentation(train, catalog) if SYNTH_ROWS else None
    pipe = RankPipeline(RANKER).fit(train, catalog, pairs=pairs, aug=aug)
    # Ship the catalog transliterations so inference skips the ~5 min precompute.
    return {"pipeline": pipe, "tr_cache": dict(candidate_generator(catalog).tr_cache)}


def from_artifacts(art: dict, catalog: pd.DataFrame):
    if art.get("tr_cache"):
        set_tr_cache(art["tr_cache"])
    pipe = art["pipeline"]
    return lambda queries: pipe.predict(queries, catalog)


def fit(train: pd.DataFrame, catalog: pd.DataFrame):
    return from_artifacts(fit_artifacts(train, catalog), catalog)
