"""v3 ranker + fine-tuned Qwen3.5 pair scores (ft_score / ft_rank) as features.

Training: the ranker sees out-of-fold LLM scores for the train rows (from
experiments/finetune_oof.py), so it never learns from in-sample LLM scores.
Inference: the LLM fine-tuned on all train rows scores every candidate of each
Hangul test row (one batch per row, fixed padding -> row independent).

Submission layout: model/llm/ holds the merged LLM weights + tokenizer.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from models.ranked import NOISE_AUG, NOISE_AUG_SEED, NOISE_AUG_W, synthetic_augmentation
from rank.pipeline import (RankPipeline, candidate_generator, set_dual_encoder, set_dual_encoder2, set_dual_encoder3,
                           set_feats4_aux, set_latin_dual_encoder, set_latin_dual_encoder2, set_tr_cache)

RANKER = "lgb_lambdarank"
BACKBONE = os.environ.get("LLM_BACKBONE", "qwen35_0.8b")
OOF_FILE = os.environ.get("LLM_OOF_FILE", "ft_oof_q35_0.8b_v4.parquet")  # v4, all Hangul candidates scored
USE_SYNTH_AUG = True
SCORE_K = int(os.environ["LLM_SCORE_K"]) if os.environ.get("LLM_SCORE_K") else None  # None: all candidates
LATIN_K = int(os.environ["LLM_LATIN_K"]) if os.environ.get("LLM_LATIN_K") else None  # None: Hangul rows only
# S9s (Codex round 2): the OOF file carries 4B scores for hard Latin train rows, but with LATIN_K None the shipped
# scorer leaves every Latin row NaN. RANKER_LATIN_LLM_FIT_OFF=1 drops the Latin OOF pairs at fit (train = inference).
LATIN_LLM_FIT_OFF = os.environ.get("RANKER_LATIN_LLM_FIT_OFF", "") == "1"
LLM_DIR = Path(__file__).resolve().parents[2] / "llm"  # model/src/models -> model/llm
# C3 (reports/diag/action_plan.md): Latin gen/rename features + LLM mask on rename-mode Latin rows. The flags
# are stored on the fitted RankPipeline, so inference needs no environment variable.
LATIN_C3 = os.environ.get("RANKER_LATIN_C3", "") == "1"
# S4 (reports/s4_impl.md), all default off. Inference-time options live on the pickled RankPipeline
# (ranker object, cand_opts), so the server needs no environment variable either:
#   RANKER_NAME=lgb_lambdarank_bag3   C4.5 3-seed LightGBM bag (z-average, constants fixed at fit)
#   RANKER_MV_DIGITS=1                C4.3 'M/V X' does not read V as 5 (queries and catalog)
#   RANKER_NUM_WORDS=1                C5 restricted Hangul number-word reader (digit_state only)
#   RANKER_NOTONNAGE_NET=1            C4.4 safety net for Hangul rows without any tonnage
#   RANKER_SYNTHFIX=1, RANKER_SYNTH_V6=1   fit-time synthetic rows (models/ranked.py)
RANKER = os.environ.get("RANKER_NAME") or RANKER
CAND_OPTS = {k: True for k, env in (("mv_digits", "RANKER_MV_DIGITS"), ("num_words", "RANKER_NUM_WORDS"),
                                    ("notonnage_net", "RANKER_NOTONNAGE_NET")) if os.environ.get(env, "") == "1"}
# S9 (experiments/s9_robust), all default off: RANKER_NUM_FORMS=1 (S5c number-form reader, S7),
# RANKER_HANGUL_GLOBAL_K=k / RANKER_LATIN_WORD_INDEX_K=k (additive retrieval; extras are never LLM-scored).
if os.environ.get("RANKER_NUM_FORMS", "") == "1":
    CAND_OPTS["num_forms"] = True
for _k, _env in (("hangul_global_k", "RANKER_HANGUL_GLOBAL_K"), ("latin_word_index_k", "RANKER_LATIN_WORD_INDEX_K"),
                 ("latin_char_tfidf_k", "RANKER_LATIN_CHAR_TFIDF_K")):  # S9y: char 3-gram TF-IDF extras (probe_outcand)
    if int(os.environ.get(_env) or 0) > 0:
        CAND_OPTS[_k] = int(os.environ[_env])
# S9 item 4 (experiments/s9_robust/item4/RESULT_4.md), default off: RANKER_PAIR_FEATS4=1 (22 row-local pair
# features, rank/pair_feats4.py; catalog table shipped as art["feats4_aux"]); RANKER_NOISE_AUG=<frac> /
# RANKER_NOISE_AUG_W=<w> noise-augmented train copies (models/ranked.py; training only).
if os.environ.get("RANKER_PAIR_FEATS4", "") == "1":
    CAND_OPTS["pair_feats4"] = True
# S9 err5 fix B (experiments/s9_robust/err5/RESULT_5.md), default off: RANKER_WORD_CONTAIN_K=k word-containment
# retrieval extras (never LLM-scored); uses the same shipped art["feats4_aux"] table.
if int(os.environ.get("RANKER_WORD_CONTAIN_K") or 0) > 0:
    CAND_OPTS["word_contain_k"] = int(os.environ["RANKER_WORD_CONTAIN_K"])
# S9 next8 "hw" (experiments/s9_robust/next8/RESULT_8.md P3), default off: RANKER_HANGUL_WC2_K=k second Hangul
# word-containment index (thr 0.75 + glued catalog token pairs) extras (never LLM-scored); same art["feats4_aux"].
if int(os.environ.get("RANKER_HANGUL_WC2_K") or 0) > 0:
    CAND_OPTS["hangul_wc2_k"] = int(os.environ["RANKER_HANGUL_WC2_K"])
# S9 charft10 (experiments/s9_robust/charft10/RESULT_10.md, v2), default off: RANKER_HANGUL_DE_K=k char dual-encoder
# top-k extras for Hangul rows (never LLM-scored) + de_cos / de_grank / de_gap. Fit: the ranker-training rows get the
# group-fold encoders of DE_DIR (strictly out-of-fold); the full model ships as art["dual_encoder"] (weights + catalog
# embeddings inside the artifacts; nothing is read outside the package at inference).
if int(os.environ.get("RANKER_HANGUL_DE_K") or 0) > 0:
    CAND_OPTS["hangul_de_k"] = int(os.environ["RANKER_HANGUL_DE_K"])
DE_DIR = os.environ.get("RANKER_DE_DIR")  # None: <repo>/experiments/s9_robust/charft10 (fit time only)
DE_SUFFIX = os.environ.get("RANKER_DE_SUFFIX", "_v2")
# S9 kode15 (experiments/s9_robust/kode15/RESULT_15.md, union arm), default off: RANKER_HANGUL_DE2_K=k second char dual
# encoder "v2-tok" ([TON][FLAG][TYPE] side tokens before the jamo query): its top-k join the extras (never LLM-scored)
# + de2_cos / de2_grank / de2_gap. Fit: group-fold models of DE2_DIR with fold flag maps; the full model + its flag map
# (all train rows) ship as art["dual_encoder2"].
if int(os.environ.get("RANKER_HANGUL_DE2_K") or 0) > 0:
    CAND_OPTS["hangul_de2_k"] = int(os.environ["RANKER_HANGUL_DE2_K"])
DE2_DIR = os.environ.get("RANKER_DE2_DIR")  # None: <repo>/experiments/s9_robust/kode15 (fit time only)
# S9 latde13 (experiments/s9_robust/latde13/RESULT_13.md, l10 arm), default off: RANKER_LATIN_DE_K=k Latin char dual
# encoder top-k extras for non-Hangul rows (never LLM-scored) + ld_cos / ld_grank / ld_gap. Fit: group-fold models of
# LDE_DIR keyed by target group (strictly out-of-fold); the full model ships as art["latin_dual_encoder"].
if int(os.environ.get("RANKER_LATIN_DE_K") or 0) > 0:
    CAND_OPTS["latin_de_k"] = int(os.environ["RANKER_LATIN_DE_K"])
LDE_DIR = os.environ.get("RANKER_LDE_DIR")  # None: <repo>/experiments/s9_robust/latde13 (fit time only)
# S9 hs22 (experiments/s9_robust/hs22), default off: RANKER_HANGUL_DE3_K=k third Hangul char dual encoder (charft10
# architecture + human-spelling pseudo queries; models charft10/models/de_f*<DE3_SUFFIX>.pt) top-k extras + de3_*.
# Fit: group-fold models keyed by target group; the full model ships as art["dual_encoder3"].
if int(os.environ.get("RANKER_HANGUL_DE3_K") or 0) > 0:
    CAND_OPTS["hangul_de3_k"] = int(os.environ["RANKER_HANGUL_DE3_K"])
DE3_SUFFIX = os.environ.get("RANKER_DE3_SUFFIX", "_hs")
# S9 ldwin (experiments/s9_robust/hs22/port_ldwin.py), default off: RANKER_LATIN_DE_WIN_K=k Latin DE top-k inside the
# tonnage window as extras + ld_wrank (needs RANKER_LATIN_DE_K).
if int(os.environ.get("RANKER_LATIN_DE_WIN_K") or 0) > 0:
    CAND_OPTS["latin_de_win_k"] = int(os.environ["RANKER_LATIN_DE_WIN_K"])
# S9 hs22 train24 Latin v2-tok, default off: RANKER_LATIN_DE2_K=k second Latin encoder (side tokens) top-k extras + ld2_*.
if int(os.environ.get("RANKER_LATIN_DE2_K") or 0) > 0:
    CAND_OPTS["latin_de2_k"] = int(os.environ["RANKER_LATIN_DE2_K"])


def fit_artifacts(train: pd.DataFrame, catalog: pd.DataFrame, pairs: pd.DataFrame | None = None) -> dict:
    train = train.reset_index(drop=True)
    de_full = de2_full = lde_full = de3_full = lde2_full = None
    if (CAND_OPTS.get("hangul_de_k") or CAND_OPTS.get("hangul_de2_k") or CAND_OPTS.get("latin_de_k")
            or CAND_OPTS.get("hangul_de3_k") or CAND_OPTS.get("latin_de2_k")):
        from common import ROOT
        from rank import dual_encoder as dmod
        if CAND_OPTS.get("hangul_de_k"):  # training rows: group-fold encoders (by target vessel); ship the full model
            de_dir = DE_DIR or ROOT / "experiments" / "s9_robust" / "charft10"
            router, de_full = dmod.fit_time_encoders(train, catalog, de_dir, DE_SUFFIX)
            set_dual_encoder(router)
        if CAND_OPTS.get("hangul_de2_k"):  # same, v2-tok; the synthetic rows' side tokens use their localised attributes
            from models.ranked import SYNTH_NOISE, SYNTH_ROWS, SYNTH_SEED
            from rank.finetune import synthetic_queries
            synth = synthetic_queries(catalog, SYNTH_ROWS, SYNTH_SEED, noise=SYNTH_NOISE) if USE_SYNTH_AUG else None
            de2_dir = DE2_DIR or ROOT / "experiments" / "s9_robust" / "kode15"
            router2, de2_full = dmod.fit_time_tok_encoders(train, catalog, de2_dir, synth=synth)
            set_dual_encoder2(router2)
        if CAND_OPTS.get("latin_de_k"):  # same leak control, Latin encoder
            routerL, lde_full = dmod.fit_time_latin_encoders(train, catalog, LDE_DIR or ROOT / "experiments" /
                                                             "s9_robust" / "latde13")
            set_latin_dual_encoder(routerL)
        if CAND_OPTS.get("hangul_de3_k"):  # same leak control, third Hangul encoder (charft10 dir, DE3_SUFFIX files)
            router3, de3_full = dmod.fit_time_encoders(train, catalog, DE_DIR or ROOT / "experiments" / "s9_robust" /
                                                       "charft10", DE3_SUFFIX)
            set_dual_encoder3(router3)
        if CAND_OPTS.get("latin_de2_k"):
            s9 = ROOT / "experiments" / "s9_robust"
            routerL2, lde2_full = dmod.fit_time_latin_encoders(train, catalog, LDE_DIR or s9 / "latde13",
                                                               tok_side=s9 / "kode15" / "cache" / "side15.pkl")
            set_latin_dual_encoder2(routerL2)
        try:
            return _fit_artifacts(train, catalog, pairs, de_full, de2_full, lde_full, de3_full, lde2_full)
        finally:
            set_latin_dual_encoder2(None)
            set_dual_encoder(None)
            set_dual_encoder2(None)
            set_dual_encoder3(None)
            set_latin_dual_encoder(None)
    return _fit_artifacts(train, catalog, pairs, de_full)


def _fit_artifacts(train: pd.DataFrame, catalog: pd.DataFrame, pairs: pd.DataFrame | None, de_full: dict | None,
                   de2_full: dict | None = None, lde_full: dict | None = None, de3_full: dict | None = None,
                   lde2_full: dict | None = None) -> dict:
    from common import ROOT

    pairs = candidate_generator(catalog, **CAND_OPTS).build(train) if pairs is None else pairs
    oof = pd.read_parquet(ROOT / "experiments" / "cache" / OOF_FILE)
    if LATIN_LLM_FIT_OFF and not LATIN_K:  # S9s: inference never scores Latin rows, so neither does the fit
        from common import is_hangul_name
        ko = train["vessel_name"].map(is_hangul_name).to_numpy()
        n0 = len(oof)
        oof = oof[ko[oof["q"].to_numpy()]]
        print(f"Latin LLM fit off: {n0 - len(oof)} Latin OOF pairs dropped")
    if LATIN_K:  # llmgen41: keep Latin OOF scores only on the pairs the scorer selects at inference (select_pairs)
        from common import is_hangul_name
        from rank.finetune import select_pairs
        ko = train["vessel_name"].map(is_hangul_name).to_numpy()
        sel = select_pairs(pairs, train, SCORE_K, LATIN_K)[["q", "c"]]
        keep_lat = oof[["q", "c"]].merge(sel.assign(_s=1), on=["q", "c"], how="left")["_s"].notna().to_numpy()
        is_ko = ko[oof["q"].to_numpy()]
        n0 = len(oof)
        oof = oof[is_ko | keep_lat]
        print(f"Latin LLM k={LATIN_K}: kept {int((~is_ko & keep_lat).sum())} Latin OOF pairs, dropped {n0 - len(oof)}")
    pairs = pairs.merge(oof, on=["q", "c"], how="left")
    if "src_extra" in pairs.columns:  # S9 extras: NaN LLM features at fit, as at inference (select_pairs skips them)
        extra = pairs["src_extra"].to_numpy() == 1
        n_scored = int(pairs.loc[extra, [c for c in oof.columns if c not in ("q", "c")]].notna().any(axis=1).sum())
        print(f"S9 extras: {int(extra.sum())} pairs, {n_scored} with an OOF LLM score")
        if n_scored:
            raise ValueError(f"{n_scored} S9 extra pairs carry OOF LLM scores; inference never scores extras")
    aug = synthetic_augmentation(train, catalog, cand_opts=CAND_OPTS) if USE_SYNTH_AUG else None
    # With LATIN_C3 the OOF ft_score/ft_rank get the same rename-mode mask inside RankPipeline.features
    # as the scorer output at inference.
    pipe = RankPipeline(RANKER, latin_feats=LATIN_C3, llm_mask=LATIN_C3,
                        cand_opts=CAND_OPTS).fit(train, catalog, pairs=pairs, aug=aug)
    art = {"pipeline": pipe, "backbone": BACKBONE, "score_k": SCORE_K, "latin_k": LATIN_K, "tr_cache": dict(candidate_generator(catalog, **CAND_OPTS).tr_cache)}
    if CAND_OPTS.get("pair_feats4") or CAND_OPTS.get("word_contain_k") or CAND_OPTS.get("hangul_wc2_k"):  # ship it
        art["feats4_aux"] = candidate_generator(catalog, **CAND_OPTS).feats4_aux
    if USE_SYNTH_AUG and NOISE_AUG:
        art.update(noise_aug_frac=NOISE_AUG, noise_aug_w=NOISE_AUG_W, noise_aug_seed=NOISE_AUG_SEED)
    if de_full is not None:
        art["dual_encoder"] = de_full
    if de2_full is not None:
        art["dual_encoder2"] = de2_full
    if lde_full is not None:
        art["latin_dual_encoder"] = lde_full
    if de3_full is not None:
        art["dual_encoder3"] = de3_full
    if lde2_full is not None:
        art["latin_dual_encoder2"] = lde2_full
    return art


def usable_cpus() -> int:
    """CPUs this process may use: affinity mask, capped by a cgroup v2 CPU quota (containers)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) // int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def from_artifacts(art: dict, catalog: pd.DataFrame):
    """predict(queries) for the submission.

    ``art["fast"]`` (optional, S8 speed path; absent = the S7 behaviour exactly), e.g.
    ``{"cand_jobs": "auto", "prefix_min_B": 6, "pad_multiple": 8}``:
      cand_jobs     candidate generation in that many worker processes ("auto" = min(5, usable CPUs - 1), >= 1;
                    0/None = serial), started BEFORE the LLM is loaded so the load overlaps with it. Same pairs
                    frame as the serial build (each row is still computed alone); any pool failure -> serial build.
      prefix_min_B  FinetunedScorer: shared-prefix forward for rows with >= this many scored pairs (None = off).
      pad_multiple  FinetunedScorer: pad each row's batch to its own longest text rounded up to this (None = 128).
      cand_timeout  seconds to wait for the workers before falling back to the serial build (default 300; a dead
                    worker loses its task and would otherwise stall the pool until the timeout).
    """
    from rank.finetune import FinetunedScorer

    if art.get("tr_cache"):
        set_tr_cache(art["tr_cache"])
    pipe = art["pipeline"]
    cand_opts = getattr(pipe, "cand_opts", None) or {}
    if cand_opts.get("pair_feats4") or cand_opts.get("word_contain_k") or cand_opts.get("hangul_wc2_k"):
        set_feats4_aux(art["feats4_aux"])
    if cand_opts.get("hangul_de_k"):  # shipped full model (weights + catalog embeddings in the artifacts)
        set_dual_encoder(art["dual_encoder"])
    if cand_opts.get("hangul_de2_k"):  # shipped v2-tok full model (weights, catalog embeddings, flag/type maps)
        set_dual_encoder2(art["dual_encoder2"])
    if cand_opts.get("hangul_de3_k"):  # shipped third Hangul full model
        set_dual_encoder3(art["dual_encoder3"])
    if cand_opts.get("latin_de2_k"):  # shipped Latin v2-tok full model (weights, catalog embeddings, side maps)
        set_latin_dual_encoder2(art["latin_dual_encoder2"])
    if cand_opts.get("latin_de_k"):  # shipped Latin full model (weights + catalog embeddings)
        set_latin_dual_encoder(art["latin_dual_encoder"])
    fast = art.get("fast") or {}

    def load_scorers() -> list:
        llm_opts = {k: fast[k] for k in ("prefix_min_B", "pad_multiple") if fast.get(k)}
        if llm_opts.get("prefix_min_B"):
            import transformers
            # The shared-prefix forward relies on transformers 5.19 cache internals (Qwen3.5 hybrid cache);
            # if the server resolved another version, score every row with the normal path instead.
            if transformers.__version__ != "5.19.0":
                print(f"transformers {transformers.__version__}: shared-prefix forward off")
                llm_opts.pop("prefix_min_B")
        try:
            return [FinetunedScorer(art["backbone"], path=str(LLM_DIR), score_k=art.get("score_k"),
                                    latin_k=art.get("latin_k"), **llm_opts)]
        except Exception as exc:  # no torch/transformers/GPU: rank without LLM features (NaN)
            print(f"LLM scorer unavailable ({exc!r}); predicting without LLM features")
            return []

    if not fast:
        pipe.scorers = load_scorers()
        return lambda queries: pipe.predict(queries, catalog)

    def predict(queries: pd.DataFrame) -> list[str]:
        from rank.candidates import finish_parallel_build, start_parallel_build

        queries = queries.reset_index(drop=True)
        gen = pipe.generator(catalog)
        gen._catalog_readings()  # built once here: fork workers inherit it
        jobs = fast.get("cand_jobs")
        jobs = max(1, min(5, usable_cpus() - 1)) if jobs == "auto" else int(jobs or 0)
        if fast.get("split") and jobs >= 1:
            try:
                return predict_split(queries, gen, jobs)
            except Exception as exc:  # noqa: BLE001
                print(f"split build failed ({type(exc).__name__}); parallel build")
        handle = None
        if jobs >= 1:
            opts = {k: v for k, v in (getattr(pipe, "cand_opts", None) or {}).items() if v}
            try:
                handle = start_parallel_build(gen, queries, jobs, opts)
            except Exception as exc:  # noqa: BLE001
                print(f"parallel candidates unavailable ({type(exc).__name__}); serial build")
        pipe.scorers = load_scorers()  # the LLM load overlaps with the candidate workers
        pairs = None
        if handle is not None:
            try:
                pairs = finish_parallel_build(handle, timeout=float(fast.get("cand_timeout", 300)))
                print(f"parallel candidates: {jobs} workers, {len(pairs)} pairs")
            except Exception as exc:  # noqa: BLE001
                print(f"parallel candidates failed ({type(exc).__name__}); serial build")
        if pairs is None:
            pairs = gen.build(queries)
        return pipe.predict(queries, catalog, pairs=pairs)

    def predict_split(queries: pd.DataFrame, gen, jobs: int) -> list[str]:
        """fast["split"]: Hangul rows' candidates first; the LLM scores them while the workers build the Latin rows.
        The scorer only scores Hangul rows, row by row, so its output on the Hangul pairs equals its output on the
        full frame; the full frame is gen.build(queries) (finish_split_build). Same predictions as predict()."""
        from rank.candidates import finish_split_build, split_hangul_pairs, start_split_build

        opts = {k: v for k, v in (getattr(pipe, "cand_opts", None) or {}).items() if v}
        timeout = float(fast.get("cand_timeout", 300))
        handle = start_split_build(gen, queries, jobs, opts)  # a failure here: predict() falls back, no LLM loaded yet
        scorers = load_scorers()  # the LLM load overlaps with the Hangul candidates
        try:
            scored = None
            if scorers:
                hp = split_hangul_pairs(handle, timeout=timeout)
                if len(hp):
                    s = scorers[0].score(queries, hp, catalog)
                    scored = pd.concat([hp[["q", "c"]], s], axis=1)
            pairs = finish_split_build(handle, timeout=timeout)
        except Exception as exc:  # noqa: BLE001  (LLM already loaded: serial build + the real scorer, no reload)
            handle["pool"].terminate()
            print(f"split build failed after the LLM load ({type(exc).__name__}); serial build")
            pipe.scorers = scorers
            return pipe.predict(queries, catalog, pairs=gen.build(queries))
        print(f"split candidates: {jobs} workers, {len(pairs)} pairs")
        pipe.scorers = [_Precomputed(scorers[0], scored)] if scorers else []
        return pipe.predict(queries, catalog, pairs=pairs)

    return predict


class _Precomputed:
    """PairScorer stand-in for the split path: the real scorer's output on the Hangul pairs, keyed by (q, c);
    every other pair NaN, as the real scorer leaves it."""

    def __init__(self, scorer, scored: pd.DataFrame | None):
        self.columns = tuple(scorer.columns)
        self.name = getattr(scorer, "name", "precomputed")
        self.scored = scored

    def score(self, queries: pd.DataFrame, pairs: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=pairs.index, columns=list(self.columns))
        if self.scored is not None and len(self.scored):
            m = pairs[["q", "c"]].merge(self.scored, on=["q", "c"], how="left")
            for col in self.columns:
                out[col] = m[col].to_numpy(dtype=float)
        return out
