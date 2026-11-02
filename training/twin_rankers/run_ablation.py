"""Exact selected-recipe 91-vs-97 relation-feature ablation.

Authority for the common 91 columns, candidate generation, rankers and post-hoc
logic is the unpacked immutable ``s9c4T_de10_noLLM.zip``.  The treatment arm
only appends ``NameRelationFeatures.COLUMNS`` from the current audited feature
module.  Candidate frames are built once per fold and shared by both arms.

This is intentionally a disk-cached, sequential harness.  On the local 32 GB
machine each branch matrix is loaded and fitted alone; no two large fits run in
parallel.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "output" / "ablation"
REPORT = ROOT / "reports" / "relation_ablation_20261029"
PKG = ROOT / "experiments" / "s9_robust" / "build" / "pkg" / "s9c4T_de10" / "model"
PKG_SRC = PKG / "src"
ZIP = ROOT / "submissions" / "s9c4T_de10_noLLM.zip"
EXPECTED_ZIP_SHA256 = "ccef59e410c6066d0e5359a01d0ce0419b4a9be4141339d9d87bc6e2da9ab6aa"
EXPECTED_COLUMNS_91 = [
    "name_exact", "bigram_jac", "trigram_jac", "prefix_ratio", "len_ratio", "gross_gap", "intl_gap",
    "best_gap", "q_gt_missing", "c_gt_missing", "src_window", "src_base", "src_key", "src_tr",
    "src_gt_exact", "src_typo", "osa_sim", "word_overlap", "last_word_eq", "is_hangul", "ph_sim",
    "ph_exact", "ph_prefix", "q_key_len", "c_key_len", "nm_sim", "tr_sim", "tr_rank", "lg_gross",
    "lg_min", "gt_eq", "digit_state", "heur", "heur_rank", "heur_gap_best", "n_cands",
    "tr_sim_gap_best", "ph_sim_gap_best", "nm_sim_gap_best", "gap_rank", "tr_top_gt_eq", "src_extra",
    "src_hglobal", "src_lword", "word_cos", "f_q_in_c", "f_c_in_q", "f_q_part", "f_tset", "f_tsort",
    "f_idf_share_q", "f_idf_share_c", "f_idf_rare", "f_idf_miss_c", "f_num_state", "f_num_q",
    "f_num_c", "f_ton_lr", "f_ton_bkt", "f_n_src", "f_nstr", "f_nstr_gap", "f_nstr_rank",
    "f_qin_gap", "f_idfq_gap", "f_row_bestname_bkt", "f_row_bestton_ngap", "de_cos", "de_grank",
    "de_gap", "de2_cos", "de2_grank", "de2_gap", "de3_cos", "de3_grank", "de3_gap", "ld_cos",
    "ld_grank", "ld_gap", "ld2_cos", "ld2_grank", "ld2_gap", "ft_score", "ft_rank", "flag_p",
    "flag_q_missing", "type_p", "mem_cs", "mem_cs_tot", "mem_nm", "mem_nm_tot",
]
CACHE_QUERY_COLS = ["vessel_name", "gross_tonnage", "international_gross_tonnage", "flag_name",
                    "vessel_type", "call_sign", "target_vessel_id"]
FOLD_COLS = ["flag_p", "flag_q_missing", "type_p", "mem_cs", "mem_cs_tot", "mem_nm", "mem_nm_tot"]
DE3_COLS = ["de3_cos", "de3_grank", "de3_gap"]
LD2_COLS = ["ld2_cos", "ld2_grank", "ld2_gap"]
PAIR_COLS_79 = [c for c in EXPECTED_COLUMNS_91 if c not in FOLD_COLS + DE3_COLS + ["ft_score", "ft_rank"]]

# The immutable package must win every normal import.  The relation class is
# loaded under a private name below, so importing current dirty rank modules can
# never alter the 91-column authority.
sys.path.insert(0, str(PKG_SRC))
sys.path.insert(1, str(ROOT / "src"))  # cv.py is not shipped; rank/common still resolve from immutable package first

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import common  # noqa: E402

CACHE_FLOAT_TOLERANCE_COLUMNS = {"ld_cos", "ld_gap", "ld2_cos", "ld2_gap"}
CACHE_FLOAT_ATOL = 4 * np.finfo(np.float32).eps

common.ROOT = ROOT  # packaged source is relocated; fit-only helpers need repo data/assets

from common import is_hangul_name  # noqa: E402
from cv import iter_splits  # noqa: E402
from rank.candidates import CandidateGenerator  # noqa: E402
from rank.dual_encoder import (fit_time_encoders, fit_time_latin_encoders,
                               fit_time_tok_encoders)  # noqa: E402
from rank.finetune import localize_attributes  # noqa: E402
from rank.fold_features import FoldFeatures  # noqa: E402
from rank.llm_aug import llm_copies, rule_abbr_copies  # noqa: E402
from rank.pipeline import RankPipeline, apply_posthoc, de_swap, labels, query_weights  # noqa: E402
from rank.pipeline import set_dual_encoder, set_dual_encoder2, set_dual_encoder3  # noqa: E402
from rank.pipeline import set_latin_dual_encoder, set_latin_dual_encoder2  # noqa: E402
from rank.rankers import RANKERS  # noqa: E402
from rank.train_memory import CallSignMemory, TrainTargetRule  # noqa: E402


def _relation_class():
    path = ROOT / "src" / "rank" / "name_relations.py"
    spec = importlib.util.spec_from_file_location("relation_v1_private", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.NameRelationFeatures


NameRelationFeatures = _relation_class()
REL_COLS = list(NameRelationFeatures.COLUMNS)

DATA = ROOT / "data" / "raw" / "open"
S9 = ROOT / "experiments" / "s9_robust"
OOF_FILE = ROOT / "experiments" / "cache" / "ft_oof_q35_4b_full.parquet"
SYNTH = {
    "synth_ko": (ROOT / "experiments" / "cache" / "synthetic_4000_0_n0.12_v5.parquet", 1),
    "synth_la": (ROOT / "experiments" / "cache" / "synthetic_latin_4000_0_v5.parquet", 3),
    "synth_rn": (ROOT / "experiments" / "cache" / "synthetic_rename_2000_0_v5.parquet", 5),
}
VARIANTS = {
    "lab2": S9 / "llmgen41" / "data" / "variants41_f38b_lab2.parquet",
    "lab345": S9 / "llmgen41" / "data" / "variants41_f38c_lab345.parquet",
}
NOISE = S9 / "item4" / "cache" / "port"
SHIFT_FULL = S9 / "cache" / "shift_rows.parquet"
SHIFT_RELEASE = S9 / "build" / "inputs" / "shift.csv"
LTOK = S9 / "hs22" / "cache" / "ltok"
S9K = S9 / "hs22" / "cache" / "s9k"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stable_frame_hash(df: pd.DataFrame, cols: list[str]) -> str:
    vals = pd.util.hash_pandas_object(df[cols], index=False).to_numpy(np.uint64)
    return hashlib.sha256(vals.tobytes()).hexdigest()


def load_data():
    train = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(DATA / "vessel_catalog.csv", encoding="utf-8-sig")
    return train, catalog


def authority():
    if sha256(ZIP) != EXPECTED_ZIP_SHA256:
        raise RuntimeError("selected ZIP SHA-256 mismatch")
    checked = []
    with zipfile.ZipFile(ZIP) as zf:
        members = [n for n in zf.namelist() if n == "model/artifacts.joblib" or n.startswith("model/src/")]
        for member in sorted(n for n in members if not n.endswith("/")):
            local = PKG / member.removeprefix("model/")
            if not local.is_file():
                raise RuntimeError(f"ZIP member has no extracted counterpart: {member}")
            zh = hashlib.sha256(zf.read(member)).hexdigest()
            if zh != sha256(local):
                raise RuntimeError(f"ZIP/extracted byte mismatch: {member}")
            checked.append((member, zh))
    if not any(n == "model/artifacts.joblib" for n, _ in checked):
        raise RuntimeError("ZIP artifacts.joblib was not checked")
    art = joblib.load(PKG / "artifacts.joblib")
    pipe = art["pipeline"]
    if list(pipe.columns) != EXPECTED_COLUMNS_91:
        raise RuntimeError("selected pipeline exact 91-column order mismatch")
    pipe._ablation_zip_manifest = hashlib.sha256(json.dumps(checked).encode()).hexdigest()
    return art, pipe


def prototype_cols(pipe, branch):
    if branch == "ko":
        return [list(x) for x in pipe.ranker.ko.cols]
    index = {"lab2": 0, "rabbr": 2, "lab345": 3}[branch]
    return [list(x) for x in pipe.ranker.la.members[index].cols]


def make_routers(train, catalog):
    char = S9 / "charft10"
    synth_ko = pd.read_parquet(SYNTH["synth_ko"][0])
    r1, _ = fit_time_encoders(train, catalog, char, "_v2")
    r2, _ = fit_time_tok_encoders(train, catalog, S9 / "kode15", synth=synth_ko)
    r3, _ = fit_time_encoders(train, catalog, char, "_hs")
    rl, _ = fit_time_latin_encoders(train, catalog, S9 / "latde13")
    rl2, _ = fit_time_latin_encoders(train, catalog, S9 / "latde13",
                                     tok_side=S9 / "kode15" / "cache" / "side15.pkl")
    return r1, r2, r3, rl, rl2


def make_generator(catalog, art, routers):
    r1, r2, r3, rl, rl2 = routers
    return CandidateGenerator(catalog, tr_cache=art["tr_cache"], feats4_aux=art["feats4_aux"],
                              dual_encoder=r1, dual_encoder2=r2, dual_encoder3=r3,
                              latin_dual_encoder=rl, latin_dual_encoder2=rl2,
                              **art["pipeline"].cand_opts)


def hybrid_pairs(queries, gen, cache_path=None, cache_queries=None, cache_rows=None):
    """Reuse only query-identical Latin cached rows; regenerate every other query.

    The ltok/s9k cache contains q,c plus immutable pair features.  Fold, OOF-LLM,
    de3 (Hangul-only), and relation features are deliberately recomputed later.
    """
    queries = queries.reset_index(drop=True)
    reusable = np.zeros(len(queries), dtype=bool)
    cache_rows = np.arange(len(queries)) if cache_rows is None else np.asarray(cache_rows, dtype=np.int64)
    if cache_path is not None:
        cache_queries = cache_queries.reset_index(drop=True)
        if len(cache_rows) != len(queries):
            raise RuntimeError("cache row mapping length mismatch")
        missing_q = [c for c in CACHE_QUERY_COLS if c not in queries or c not in cache_queries]
        if missing_q:
            raise RuntimeError(f"cache authority missing candidate-driving query columns: {missing_q}")
        cols = CACHE_QUERY_COLS
        if cache_rows.min(initial=0) < 0 or cache_rows.max(initial=-1) >= len(cache_queries):
            raise RuntimeError("cache query authority/mapping mismatch")
        left = queries[cols].astype(object).where(queries[cols].notna(), "__NA__").astype(str).reset_index(drop=True)
        authority_rows = cache_queries.iloc[cache_rows][cols].reset_index(drop=True)
        right = authority_rows.astype(object).where(authority_rows.notna(), "__NA__").astype(str)
        identical = (left == right).all(axis=1).to_numpy()
        latin = ~queries["vessel_name"].map(is_hangul_name).to_numpy()
        reusable = identical & latin
    frames = []
    fresh_rows = np.flatnonzero(~reusable)
    if len(fresh_rows):
        fresh = gen.build(queries.iloc[fresh_rows].reset_index(drop=True))
        fresh["q"] = fresh_rows[fresh["q"].to_numpy(np.int64)]
        frames.append(fresh)
    if reusable.any():
        cached = pd.read_parquet(cache_path)
        required = PAIR_COLS_79 if cache_path.parent.name == "ltok" else [c for c in PAIR_COLS_79 if c not in LD2_COLS]
        if list(cached.columns) != ["q", "c"] + required:
            raise RuntimeError(f"cached pair schema authority mismatch: {cache_path}")
        if not all(pd.api.types.is_numeric_dtype(cached[c]) for c in cached.columns):
            raise RuntimeError(f"cached pair dtype authority mismatch: {cache_path}")
        wanted = cache_rows[reusable]
        cached = cached[cached["q"].isin(wanted)].copy()
        if set(cached["q"].astype(int)) != set(wanted.tolist()) or cached.duplicated(["q", "c"]).any():
            raise RuntimeError(f"cached candidate coverage/uniqueness mismatch: {cache_path}")
        remap = pd.Series(np.flatnonzero(reusable), index=wanted)
        cached["q"] = cached["q"].map(remap)
        if cached["q"].isna().any():
            raise RuntimeError("cached candidate remap failed")
        cached["q"] = cached["q"].astype(np.int32)
        for c in ("de3_cos", "de3_grank", "de3_gap"):
            cached[c] = np.float32(np.nan)
        frames.append(cached)
    if not frames:
        raise RuntimeError("candidate generation produced no frames")
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values(["q", "c"], kind="stable").reset_index(drop=True)
    if out.duplicated(["q", "c"]).any() or set(out["q"].astype(int)) != set(range(len(queries))):
        raise RuntimeError("hybrid candidate output lacks exact unique per-query coverage")
    print("hybrid candidates", cache_path.name if cache_path else "fresh", len(queries),
          "reused_latin", int(reusable.sum()), "fresh", int((~reusable).sum()), "pairs", len(out), flush=True)
    return out


def cache_column_equal(column, fresh, cached):
    """Exact cache parity except a fixed cross-CPU bound for four continuous Latin-DE columns."""
    a = np.asarray(fresh, dtype=np.float32); b = np.asarray(cached, dtype=np.float32)
    if a.shape != b.shape:
        return False
    if not (np.array_equal(np.isnan(a), np.isnan(b)) and
            np.array_equal(np.isposinf(a), np.isposinf(b)) and
            np.array_equal(np.isneginf(a), np.isneginf(b))):
        return False
    if column not in CACHE_FLOAT_TOLERANCE_COLUMNS:
        return np.array_equal(a, b, equal_nan=True)
    finite = np.isfinite(a) & np.isfinite(b)
    return bool(np.all(np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64)) <= CACHE_FLOAT_ATOL))


def verify_cache_sample(gen, cache_path, queries, label):
    latin = np.flatnonzero(~queries["vessel_name"].map(is_hangul_name).to_numpy())[:20]
    fresh = gen.build(queries.iloc[latin].reset_index(drop=True)).sort_values(["q", "c"], kind="stable")
    cached = pd.read_parquet(cache_path)
    cached = cached[cached["q"].isin(latin)].copy()
    cached["q"] = cached["q"].map(pd.Series(np.arange(len(latin)), index=latin)).astype(np.int32)
    cached = cached.sort_values(["q", "c"], kind="stable")
    if not np.array_equal(fresh[["q", "c"]].to_numpy(np.int64), cached[["q", "c"]].to_numpy(np.int64)):
        raise RuntimeError(f"{label} cached candidates differ from fresh package generator")
    cols = [c for c in cached if c not in ("q", "c") and c in fresh]
    bad = [c for c in cols if not cache_column_equal(c, fresh[c], cached[c])]
    if bad:
        raise RuntimeError(f"{label} cached pair features differ from fresh package generator: {bad}")
    print("cache parity sample PASS", label, "queries", len(latin), "pairs", len(fresh), "features", len(cols),
          flush=True)


def ensure_pair_columns(pairs: pd.DataFrame):
    for c in ("ft_score", "ft_rank"):
        if c not in pairs:
            pairs[c] = np.float32(np.nan)
    return pairs


def merge_oof(pairs: pd.DataFrame, global_rows: np.ndarray, oof: pd.DataFrame) -> pd.DataFrame:
    p = pairs.copy()
    local_q = p["q"].to_numpy(np.int32)
    p["global_q"] = global_rows[local_q]
    p = p.merge(oof.rename(columns={"q": "global_q"}), on=["global_q", "c"], how="left", validate="one_to_one")
    p["q"] = local_q
    p = p.drop(columns="global_q")
    extra = p.get("src_extra", pd.Series(np.zeros(len(p), dtype=np.int8))).to_numpy() == 1
    if p.loc[extra, ["ft_score", "ft_rank"]].notna().any(axis=None):
        raise RuntimeError("S9 extra candidate unexpectedly carries OOF LLM features")
    return p


def part_paths(fold: int, name: str):
    d = OUT / f"fold{fold}" / "parts"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.parquet", d / f"{name}.json", d / f"q_{name}.parquet", d / f"pairs_{name}.parquet"


def part_provenance(queries, pairs, selected_cols, src=None):
    qcols = [c for c in ("sample_id", "target_vessel_id", "vessel_name") if c in queries]
    value = {
        "producer": sha256(Path(__file__)),
        "relation_producer": sha256(ROOT / "src" / "rank" / "name_relations.py"),
        "selected_columns": hashlib.sha256(json.dumps(selected_cols).encode()).hexdigest(),
        "queries": stable_frame_hash(queries.reset_index(drop=True), qcols),
        "pairs": stable_frame_hash(pairs.reset_index(drop=True), ["q", "c"]),
    }
    if src is not None:
        scols = [c for c in ("sample_id", "target_vessel_id", "vessel_name") if c in src]
        value["source"] = stable_frame_hash(src.reset_index(drop=True), scols)
    return value


def build_part(fold, name, queries, pairs, pipe, relation, catalog, selected_cols,
               train_part=True, src=None, weight=1.0, global_rows=None, oof=None, force=False):
    xp, mp, qp, pp = part_paths(fold, name)
    provenance = part_provenance(queries, pairs, selected_cols, src)
    if xp.exists() and mp.exists() and qp.exists() and pp.exists() and not force:
        cached = json.loads(mp.read_text(encoding="utf-8"))
        if cached.get("provenance") != provenance:
            raise RuntimeError(f"stale or foreign cache for fold={fold} part={name}; rerun with --force")
        if cached.get("schema") != selected_cols + REL_COLS:
            raise RuntimeError(f"cached schema mismatch for fold={fold} part={name}")
        return
    queries = queries.reset_index(drop=True)
    pairs = pairs.reset_index(drop=True)
    if global_rows is not None:
        pairs = merge_oof(pairs, np.asarray(global_rows), oof)
    else:
        pairs = ensure_pair_columns(pairs)
    if not pairs["q"].is_monotonic_increasing:
        pairs = pairs.sort_values(["q", "c"], kind="stable").reset_index(drop=True)
    if src is None:
        X = pipe.features(pairs, queries, catalog, exclude_self=bool(train_part))
    else:
        X = pipe.features(pairs, queries, catalog, exclude_self=False, lso_src=src.reset_index(drop=True))
    X = X.reindex(columns=selected_cols)
    if list(X.columns) != selected_cols:
        raise RuntimeError("common 91 schema mismatch")
    rel = relation.transform(pairs, queries, catalog, group_out=bool(train_part))
    X = pd.concat([X.reset_index(drop=True), rel.reset_index(drop=True)], axis=1).astype(np.float32)
    y = labels(pairs, queries, catalog).astype(np.int8)
    meta = pd.DataFrame({"q": pairs["q"].to_numpy(np.int32), "c": pairs["c"].to_numpy(np.int32), "y": y})
    if train_part:
        if name not in ("real", "ko_real"):
            ko = queries["vessel_name"].map(is_hangul_name).to_numpy()
            # Exact RankPipeline convention: max real row weight per language is
            # supplied later from fold-train. Store language only here.
            meta["is_ko"] = ko[meta["q"].to_numpy()].astype(np.int8)
        meta["part_weight"] = np.float32(weight)
    X.to_parquet(xp, index=False)
    meta.to_parquet(pp, index=False)
    queries.to_parquet(qp, index=False)
    info = {
        "fold": fold, "part": name, "queries": len(queries), "pairs": len(pairs),
        "candidate_hash": stable_frame_hash(meta, ["q", "c", "y"]),
        "common91_hash": stable_frame_hash(X, selected_cols),
        "relation6_hash": stable_frame_hash(X, REL_COLS),
        "relation_active_pairs": int((X["name_rel_support_distinct_vessels"] > 0).sum()),
        "target_present": int(meta.loc[meta.y == 1, "q"].nunique()),
        "provenance": provenance,
        "schema": selected_cols + REL_COLS,
    }
    mp.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("built", fold, name, info, flush=True)
    del X, rel, pairs, meta
    gc.collect()


def build_fold(fold, force=False):
    train, catalog = load_data()
    art, selected = authority()
    selected_cols = list(selected.columns)
    splits = list(iter_splits(train, "group"))
    tr, va = splits[fold]
    tr_df = train.iloc[tr].reset_index(drop=True)
    val_targets = set(train.iloc[va]["target_vessel_id"].astype(str))
    routers = make_routers(train, catalog)
    gen = make_generator(catalog, art, routers)
    ko_pipe = art["pipeline"]
    ko_opts = dict(ko_pipe.cand_opts)
    ko_opts["latin_de2_k"] = 0
    ko_gen = CandidateGenerator(catalog, tr_cache=art["tr_cache"], feats4_aux=art["feats4_aux"],
                                dual_encoder=routers[0], dual_encoder2=routers[1], dual_encoder3=routers[2],
                                latin_dual_encoder=routers[3], latin_dual_encoder2=routers[4], **ko_opts)
    verify_cache_sample(gen, LTOK / "clean.parquet", train, "ltok")
    verify_cache_sample(ko_gen, S9K / "clean.parquet", train, "s9k-no-ld2")
    fold_pipe = RankPipeline("lgb_binary", cand_opts=selected.cand_opts)
    fold_pipe.fold = FoldFeatures(tr_df, catalog)
    fold_pipe.scorers = []
    relation = NameRelationFeatures(tr_df, catalog)
    oof = pd.read_parquet(OOF_FILE)

    build_part(fold, "real", tr_df, hybrid_pairs(tr_df, gen, LTOK / "clean.parquet", train, tr),
               fold_pipe, relation, catalog, selected_cols,
               global_rows=np.asarray(tr), oof=oof, force=force)
    build_part(fold, "ko_real", tr_df, hybrid_pairs(tr_df, ko_gen, S9K / "clean.parquet", train, tr),
               fold_pipe, relation, catalog, selected_cols,
               global_rows=np.asarray(tr), oof=oof, force=force)
    for name, (path, seed) in SYNTH.items():
        q0 = pd.read_parquet(path)
        keep = ~q0["target_vessel_id"].astype(str).isin(val_targets)
        kept = np.flatnonzero(keep.to_numpy())
        q = localize_attributes(q0.iloc[kept].reset_index(drop=True), tr_df, catalog, seed=seed)
        build_part(fold, name, q, hybrid_pairs(q, gen, LTOK / f"{name}.parquet", q0, kept),
                   fold_pipe, relation, catalog, selected_cols,
                   weight=.5, force=force)
        build_part(fold, f"ko_{name}", q, hybrid_pairs(q, ko_gen, S9K / f"{name}.parquet", q0, kept),
                   fold_pipe, relation, catalog, selected_cols,
                   weight=.5, force=force)

    noise = pd.read_parquet(NOISE / f"copies_f{fold}.parquet")
    noise_src = pd.read_parquet(NOISE / f"sources_f{fold}.parquet")
    if set(noise["target_vessel_id"].astype(str)) & val_targets:
        raise RuntimeError("noise copies contain outer validation target")
    build_part(fold, "noise", noise, hybrid_pairs(noise, gen, LTOK / f"aug_f{fold}.parquet", noise),
               fold_pipe, relation, catalog, selected_cols,
               src=noise_src, weight=.5, force=force)
    build_part(fold, "ko_noise", noise, hybrid_pairs(noise, ko_gen, S9K / f"aug_f{fold}.parquet", noise),
               fold_pipe, relation, catalog, selected_cols,
               src=noise_src, weight=.5, force=force)

    for name in ("lab2", "lab345"):
        variants = pd.read_parquet(VARIANTS[name])
        q, src = llm_copies(tr_df, variants, "llm")
        build_part(fold, name, q, hybrid_pairs(q, gen), fold_pipe, relation, catalog, selected_cols,
                   src=src, weight=.5, force=force)
    q, src = rule_abbr_copies(tr_df)
    build_part(fold, "rabbr", q, hybrid_pairs(q, gen), fold_pipe, relation, catalog, selected_cols,
               src=src, weight=.5, force=force)

    evals = {
        "clean": train.iloc[va].reset_index(drop=True),
    }
    shift_rows = pd.read_parquet(SHIFT_FULL)
    for c in ("sample_id", "target_vessel_id"):
        if c not in shift_rows or not shift_rows[c].astype(str).reset_index(drop=True).equals(
                train[c].astype(str).reset_index(drop=True)):
            raise RuntimeError(f"shift_rows is not row-aligned with official train on {c}")
    evals["shift"] = shift_rows.iloc[va].reset_index(drop=True)
    if SHIFT_RELEASE.exists():
        sr = pd.read_csv(SHIFT_RELEASE)
        truth = train[["sample_id", "target_vessel_id"]]
        sr = sr.drop(columns=["target_vessel_id"], errors="ignore").merge(truth, on="sample_id", how="left",
                                                                           validate="one_to_one")
        fold_ids = set(train.iloc[va]["sample_id"].astype(str))
        evals["release_shift"] = sr[sr["sample_id"].astype(str).isin(fold_ids)].reset_index(drop=True)
    for name, q in evals.items():
        global_by_id = pd.Series(np.arange(len(train)), index=train["sample_id"].astype(str))
        mapped = q["sample_id"].astype(str).map(global_by_id)
        cache_path = LTOK / ("shift.parquet" if name != "clean" else "clean.parquet")
        cache_queries = shift_rows if name != "clean" else train
        if mapped.isna().any():
            raise RuntimeError(f"evaluation sample_id cannot map to cache authority: {name}")
        build_part(fold, f"eval_{name}", q,
                   hybrid_pairs(q, gen, cache_path, cache_queries, mapped.to_numpy(np.int64)),
                   fold_pipe, relation, catalog, selected_cols,
                   train_part=False, force=force)

    manifest = {
        "status": "PREPARED", "fold": fold, "train_rows": len(tr), "validation_rows": len(va),
        "selected_zip_sha256": sha256(ZIP), "selected_columns": selected_cols,
        "zip_extract_manifest_sha256": selected._ablation_zip_manifest,
        "relation_columns": REL_COLS, "candidate_options": selected.cand_opts,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [
            Path(__file__), ROOT / "src" / "rank" / "name_relations.py", OOF_FILE,
            VARIANTS["lab2"], VARIANTS["lab345"], *(v[0] for v in SYNTH.values()),
            NOISE / f"copies_f{fold}.parquet", NOISE / f"sources_f{fold}.parquet",
            LTOK / "clean.parquet", LTOK / "shift.parquet", S9K / "clean.parquet", S9K / "shift.parquet",
            *(LTOK / f"{n}.parquet" for n in SYNTH), *(S9K / f"{n}.parquet" for n in SYNTH),
            LTOK / f"aug_f{fold}.parquet", S9K / f"aug_f{fold}.parquet",
            DATA / "train.csv", DATA / "vessel_catalog.csv", SHIFT_FULL, SHIFT_RELEASE]},
        "limitations": [
            "encoder routers are target-own-fold but are not nested inside the outer ranker folds",
            "release_shift is a known-target retrospective stress input, not public-test labels",
        ],
    }
    d = OUT / f"fold{fold}"
    (d / "prepare_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("prepared fold", fold, flush=True)


COMMON_PARTS = ["real", "synth_ko", "synth_la", "synth_rn", "noise"]
BRANCHES = {
    "ko": ("lgb_colbag_de3", None),
    "lab2": ("lgb_colbag_de3_t3000", "lab2"),
    "rabbr": ("cat_yetirank_z", "rabbr"),
    "lab345": ("lgb_colbag_de3_t3000", "lab345"),
}


def load_training(fold, branch, arm, selected_cols, train):
    parts = COMMON_PARTS + ([] if branch == "ko" else [BRANCHES[branch][1]])
    frames, ys, qids, weights = [], [], [], []
    qoff = 0
    tr = list(iter_splits(train, "group"))[fold][0]
    tr_df = train.iloc[tr].reset_index(drop=True)
    rw = query_weights(tr_df)
    ko = tr_df["vessel_name"].map(is_hangul_name).to_numpy()
    base_w = {1: float(rw[ko].max()) if ko.any() else 1.0,
              0: float(rw[~ko].max()) if (~ko).any() else 1.0}
    for name in parts:
        if branch == "ko":
            name = f"ko_{name}"
        xp, _, _, pp = part_paths(fold, name)
        cols = selected_cols + (REL_COLS if arm == "97" else [])
        X = pd.read_parquet(xp, columns=cols)
        m = pd.read_parquet(pp)
        q = m["q"].to_numpy(np.int64)
        frames.append(X)
        ys.append(m["y"].to_numpy(np.int8))
        qids.append(q + qoff)
        if name in ("real", "ko_real"):
            weights.append(rw[q])
        else:
            weights.append(np.asarray([base_w[int(x)] for x in m["is_ko"]], dtype=np.float32)
                           * m["part_weight"].to_numpy(np.float32))
        qoff += int(q.max()) + 1 if len(q) else 0
    X = pd.concat(frames, ignore_index=True)
    y = np.concatenate(ys); qid = np.concatenate(qids); w = np.concatenate(weights)
    if len(X) != len(y) or not np.all(qid[1:] >= qid[:-1]):
        raise RuntimeError("assembled training matrix is inconsistent/non-contiguous")
    return X, y, qid, w


def fit_branch(fold, branch, arm, selected_cols, train, threads):
    art, selected = authority()
    expected_views = prototype_cols(selected, branch)
    base_cols = [c for c in selected_cols if any(c in view for view in expected_views)]
    X, y, qid, w = load_training(fold, branch, arm, base_cols, train)
    ranker_name = BRANCHES[branch][0]
    model = RANKERS[ranker_name]()
    # Bound thread use without changing seeds/trees or other learner settings.
    for member in getattr(model, "members", []):
        if hasattr(member, "model") and hasattr(member.model, "set_params"):
            member.model.set_params(n_jobs=threads)
        if hasattr(member, "params"):
            member.params["thread_count"] = threads
    t0 = time.time()
    model.fit(X, y, qid, w)
    elapsed = time.time() - t0
    got_views = [[c for c in cols if c not in REL_COLS] for cols in model.cols]
    if got_views != expected_views:
        raise RuntimeError(f"{branch} refit common-column projection differs from selected package")
    importance = {}
    for i, (member, cols) in enumerate(zip(getattr(model, "members", []), getattr(model, "cols", []))):
        raw = None
        if hasattr(member, "model") and hasattr(member.model, "feature_importances_"):
            raw = member.model.feature_importances_
        elif hasattr(member, "_model"):
            try:
                raw = member._model().get_feature_importance(type="FeatureImportance")
            except Exception:  # CatBoost importance is diagnostic; a missing value must not abort the fit
                raw = None
        if raw is not None:
            importance[f"member_{i}"] = {c: float(v) for c, v in zip(cols, raw) if c in REL_COLS or v > 0}
    d = OUT / f"fold{fold}" / "models"
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d / f"{arm}_{branch}.joblib", compress=1)
    meta = {"fold": fold, "arm": arm, "branch": branch, "ranker": ranker_name,
            "rows": len(X), "columns": list(X.columns), "seconds": elapsed,
            "relation_importance": {m: {c: v for c, v in x.items() if c in REL_COLS}
                                    for m, x in importance.items()}}
    (d / f"{arm}_{branch}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    del X, y, qid, w
    gc.collect()
    print("fitted", meta, flush=True)
    return model, meta


def score_metrics(q, pred, target):
    pred = np.asarray(pred, dtype=object); target = np.asarray(target, dtype=object)
    ok = pred == target
    ko = q["vessel_name"].map(is_hangul_name).to_numpy()
    ha = float(ok[ko].mean()) if ko.any() else float("nan")
    la = float(ok[~ko].mean()) if (~ko).any() else float("nan")
    return {"rows": len(q), "correct": int(ok.sum()), "errors": int((~ok).sum()),
            "acc_ko": ha, "acc_la": la, "weighted_accuracy": .7 * ha + .3 * la}


def raw_stats(pairs, score, target_pos, catalog):
    qv, cv = pairs["q"].to_numpy(), pairs["c"].to_numpy()
    f = pd.DataFrame({"q": qv, "c": cv, "s": score})
    f = f.sort_values(["q", "s", "c"], ascending=[True, False, True], kind="stable")
    f["rank"] = f.groupby("q").cumcount() + 1
    truth = f[f["c"].to_numpy() == target_pos[f["q"].to_numpy()]].drop_duplicates("q")
    n = int(qv.max()) + 1 if len(qv) else 0
    ranks = np.full(n, np.inf); ranks[truth["q"].to_numpy(int)] = truth["rank"].to_numpy(float)
    ids = catalog["vessel_id"].astype(str).to_numpy()
    top = f.drop_duplicates("q").sort_values("q")
    pred = np.full(n, "", dtype=object); pred[top["q"].to_numpy(int)] = ids[top["c"].to_numpy(int)]
    return pred, ranks


def ensemble_score(models, X, kop):
    sk = models["ko"].predict(X)
    sl2 = models["lab2"].predict(X); src = models["rabbr"].predict(X); sl3 = models["lab345"].predict(X)
    scales = [models[n].scale[0] for n in ("lab2", "rabbr", "lab345")]
    z = 2 * (sl2 - scales[0][0]) / scales[0][1]
    z += (src - scales[1][0]) / scales[1][1]
    z += (sl3 - scales[2][0]) / scales[2][1]
    sl = z / 4 * scales[0][1] + scales[0][0]
    return np.where(kop, sk, sl)


def evaluate_fold(fold, arm, models, selected, train, catalog, routers):
    rows = []
    ids = catalog["vessel_id"].astype(str).to_numpy()
    pos_of = dict(zip(ids, range(len(ids))))
    tt_tr = train.iloc[list(iter_splits(train, "group"))[fold][0]].reset_index(drop=True)
    tt = TrainTargetRule(tt_tr, catalog); cs = CallSignMemory(tt_tr, keep_names=True)
    set_dual_encoder(routers[0]); set_dual_encoder2(routers[1]); set_dual_encoder3(routers[2])
    for kind in ("clean", "shift", "release_shift"):
        xp, _, qp, pp = part_paths(fold, f"eval_{kind}")
        if not xp.exists():
            continue
        cols = list(selected.columns) + (REL_COLS if arm == "97" else [])
        X = pd.read_parquet(xp, columns=cols)
        if X[["ft_score", "ft_rank"]].notna().any(axis=None):
            raise RuntimeError(f"noLLM evaluation unexpectedly has finite ft features: fold={fold} kind={kind}")
        pairs = pd.read_parquet(pp)[["q", "c", "y"]]
        q = pd.read_parquet(qp)
        koq = q["vessel_name"].map(is_hangul_name).to_numpy()
        kop = koq[pairs["q"].to_numpy()]
        score = ensemble_score(models, X, kop)
        target = q["target_vessel_id"].astype(str).to_numpy()
        tpos = q["target_vessel_id"].astype(str).map(pos_of).to_numpy()
        raw, ranks = raw_stats(pairs, score, tpos, catalog)
        cfg = {"tt": tt, "tt_weight": 10.0, "cs_override": cs, "cs_guard": .25, "cs_need_cand": False}
        tt_pred = apply_posthoc({"tt": tt, "tt_weight": 10.0}, pairs, X, score, q, catalog,
                                log=lambda _: None)
        cs_pred = apply_posthoc(cfg, pairs, X, score, q, catalog, log=lambda _: None)
        final = de_swap(q, catalog, cs_pred, 1.0, log=lambda _: None)
        zero_raw = zero_final = np.full(len(q), "", dtype=object)
        score_changed = np.zeros(len(q), dtype=bool)
        if arm == "97":
            saved = X[REL_COLS].copy()
            X.loc[:, REL_COLS] = 0.0
            X.loc[:, ["name_rel_core_number_consistency", "name_rel_core_gt_consistency"]] = np.nan
            score0 = ensemble_score(models, X, kop)
            zero_raw, _ = raw_stats(pairs, score0, tpos, catalog)
            post0 = apply_posthoc(cfg, pairs, X, score0, q, catalog, log=lambda _: None)
            zero_final = np.asarray(de_swap(q, catalog, post0, 1.0, log=lambda _: None), dtype=object)
            delta = np.abs(score - score0) > 1e-10
            score_changed = np.bincount(pairs["q"].to_numpy(), weights=delta.astype(float), minlength=len(q)) > 0
            X.loc[:, REL_COLS] = saved.to_numpy()
        present = np.isfinite(ranks)
        rel_active = np.bincount(pairs["q"].to_numpy(),
                                 weights=(X.get("name_rel_support_distinct_vessels", 0) > 0).to_numpy(float)
                                 if arm == "97" else np.zeros(len(X)), minlength=len(q)) > 0
        truth_rel = np.full((len(q), len(REL_COLS)), np.nan, dtype=np.float32)
        raw_rel = np.full((len(q), len(REL_COLS)), np.nan, dtype=np.float32)
        if arm == "97":
            locator = pairs.reset_index(names="pair_row")[["pair_row", "q", "c"]]
            raw_c = pd.Series(raw).map(pos_of).to_numpy()
            wants = pd.DataFrame({"q": np.arange(len(q), dtype=np.int32), "truth_c": tpos, "raw_c": raw_c})
            ti = wants[["q", "truth_c"]].rename(columns={"truth_c": "c"}).merge(
                locator, on=["q", "c"], how="left", validate="one_to_one")["pair_row"]
            ri = wants[["q", "raw_c"]].rename(columns={"raw_c": "c"}).merge(
                locator, on=["q", "c"], how="left", validate="one_to_one")["pair_row"]
            ok_t = ti.notna().to_numpy(); ok_r = ri.notna().to_numpy()
            truth_rel[ok_t] = X.iloc[ti[ok_t].astype(int).to_numpy()][REL_COLS].to_numpy(np.float32)
            raw_rel[ok_r] = X.iloc[ri[ok_r].astype(int).to_numpy()][REL_COLS].to_numpy(np.float32)
        for i in range(len(q)):
            row = {"fold": fold, "arm": arm, "kind": kind, "sample_id": q.iloc[i].get("sample_id", ""),
                         "query": q.iloc[i]["vessel_name"], "target": target[i], "raw_pred": raw[i],
                         "tt_pred": tt_pred[i], "cs_pred": cs_pred[i], "posthoc_pred": cs_pred[i],
                         "final_pred": final[i], "target_rank": ranks[i],
                         "in_candidates": bool(present[i]), "relation_active": bool(rel_active[i]),
                         "zero6_raw_pred": zero_raw[i], "zero6_final_pred": zero_final[i],
                         "relation_score_changed": bool(score_changed[i]),
                         "lang": "ko" if koq[i] else "la"}
            for j, c in enumerate(REL_COLS):
                row[f"truth_{c}"] = float(truth_rel[i, j])
                row[f"raw_pred_{c}"] = float(raw_rel[i, j])
            rows.append(row)
        print("metrics", fold, arm, kind, "raw", score_metrics(q, raw, target),
              "final", score_metrics(q, final, target), flush=True)
        del X, pairs, q, score
        gc.collect()
    return pd.DataFrame(rows)


def fit_fold(fold, arms, threads=3):
    train, catalog = load_data(); art, selected = authority()
    prep = OUT / f"fold{fold}" / "prepare_manifest.json"
    if not prep.exists():
        raise RuntimeError(f"fold {fold} not prepared")
    prepared = json.loads(prep.read_text(encoding="utf-8"))
    for rel, expected in prepared["source_sha256"].items():
        if sha256(ROOT / rel) != expected:
            raise RuntimeError(f"producer/input changed after prepare: {rel}")
    routers = make_routers(train, catalog)
    all_rows, timings = [], []
    for arm in arms:
        models = {}
        for branch in BRANCHES:
            models[branch], meta = fit_branch(fold, branch, arm, list(selected.columns), train, threads)
            timings.append(meta)
        all_rows.append(evaluate_fold(fold, arm, models, selected, train, catalog, routers))
        del models
        gc.collect()
    rows = pd.concat(all_rows, ignore_index=True)
    d = OUT / f"fold{fold}"
    rows.to_parquet(d / "scores.parquet", index=False)
    (d / "fit_manifest.json").write_text(json.dumps({"status": "COMPLETE", "fold": fold, "timings": timings},
                                                       indent=2), encoding="utf-8")
    print("completed fold", fold, flush=True)


def summarize():
    files = sorted(OUT.glob("fold*/scores.parquet"))
    expected_files = [OUT / f"fold{i}" / "scores.parquet" for i in range(5)]
    if files != expected_files:
        raise RuntimeError(f"summary requires exactly folds 0..4; got {[str(p) for p in files]}")
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    release_errors_path = HERE / "output" / "release_shift_errors.csv"
    if not release_errors_path.exists():
        raise RuntimeError("frozen release error ledger is required for actual-error replay")
    release_errors = pd.read_csv(release_errors_path)
    if len(release_errors) != 247 or not release_errors["sample_id"].is_unique:
        raise RuntimeError("frozen release error ledger must contain the exact 247 unique shift errors")
    release_error_ids = set(release_errors["sample_id"].astype(str))
    df["frozen_release_error"] = ((df["kind"] == "release_shift") &
                                  df["sample_id"].astype(str).isin(release_error_ids))
    if int(df[df.arm == "91"].frozen_release_error.sum()) != 247:
        raise RuntimeError("OOF ledger does not cover all 247 frozen release errors")
    keys = ["kind", "sample_id", "fold"]
    if set(df.arm.unique()) != {"91", "97"} or set(df.fold.unique()) != set(range(5)):
        raise RuntimeError("case ledger arm/fold coverage mismatch")
    a = df[df.arm == "91"].set_index(keys, verify_integrity=True).sort_index()
    b = df[df.arm == "97"].set_index(keys, verify_integrity=True).sort_index()
    if not a.index.equals(b.index):
        raise RuntimeError("91/97 case keys differ; refusing silent intersection")
    for c in ("target", "query", "in_candidates"):
        if not a[c].astype(str).equals(b[c].astype(str)):
            raise RuntimeError(f"91/97 {c} differs on aligned cases")
    summary = {"status": "COMPLETE", "folds": len(files), "populations": {}}
    for kind in sorted(df.kind.unique()):
        ix = a.index.get_level_values("kind") == kind
        aa, bb = a[ix], b[ix]
        target = aa.target.to_numpy()
        entry = {}
        for stage in ("raw_pred", "tt_pred", "cs_pred", "final_pred"):
            ca = aa[stage].to_numpy() == target; cb = bb[stage].to_numpy() == target
            entry[stage] = {"n": len(ca), "correct_91": int(ca.sum()), "correct_97": int(cb.sum()),
                            "fixes": int((~ca & cb).sum()), "breaks": int((ca & ~cb).sum()),
                            "fix_vessels": int(aa.loc[~ca & cb, "target"].nunique()),
                            "break_vessels": int(aa.loc[ca & ~cb, "target"].nunique())}
            for lang in ("ko", "la"):
                lm = aa.lang.to_numpy() == lang
                entry[stage][f"accuracy_91_{lang}"] = float(ca[lm].mean()) if lm.any() else None
                entry[stage][f"accuracy_97_{lang}"] = float(cb[lm].mean()) if lm.any() else None
            entry[stage]["weighted_91"] = .7 * entry[stage]["accuracy_91_ko"] + .3 * entry[stage]["accuracy_91_la"]
            entry[stage]["weighted_97"] = .7 * entry[stage]["accuracy_97_ko"] + .3 * entry[stage]["accuracy_97_la"]
        entry["candidate_recall"] = float(aa.in_candidates.mean())
        finite = np.isfinite(aa.target_rank.to_numpy(float))
        ranks_a = aa.target_rank.to_numpy(float)
        ranks_b = bb.target_rank.to_numpy(float)
        entry["raw_rank"] = {
            "top5_91": float(np.mean(ranks_a <= 5)), "top5_97": float(np.mean(ranks_b <= 5)),
            "mrr_91": float(np.mean(np.where(finite, 1.0 / ranks_a, 0.0))),
            "mrr_97": float(np.mean(np.where(np.isfinite(ranks_b), 1.0 / ranks_b, 0.0))),
        }
        entry["relation_active_rows"] = int(bb.relation_active.sum())
        entry["relation_score_changed_rows"] = int(bb.relation_score_changed.sum())
        if kind == "release_shift":
            rm = aa.frozen_release_error.to_numpy(bool)
            entry["frozen_release_errors"] = {"n": int(rm.sum())}
            for stage in ("raw_pred", "tt_pred", "cs_pred", "final_pred"):
                ca = aa[stage].to_numpy()[rm] == target[rm]
                cb = bb[stage].to_numpy()[rm] == target[rm]
                entry["frozen_release_errors"][stage] = {
                    "correct_91": int(ca.sum()), "correct_97": int(cb.sum()),
                    "fixes": int((~ca & cb).sum()), "breaks": int((ca & ~cb).sum()),
                }
        entry["fold_final"] = {}
        fold_values = aa.index.get_level_values("fold").to_numpy()
        for fold in sorted(np.unique(fold_values)):
            fm = fold_values == fold
            ca = aa.final_pred.to_numpy()[fm] == target[fm]
            cb = bb.final_pred.to_numpy()[fm] == target[fm]
            entry["fold_final"][str(int(fold))] = {"n": int(fm.sum()), "correct_91": int(ca.sum()),
                                                       "correct_97": int(cb.sum()),
                                                       "fixes": int((~ca & cb).sum()),
                                                       "breaks": int((ca & ~cb).sum())}
        for active, label in ((True, "relation_active"), (False, "relation_inactive")):
            rm = bb.relation_active.to_numpy() == active
            ca = aa.final_pred.to_numpy()[rm] == target[rm]
            cb = bb.final_pred.to_numpy()[rm] == target[rm]
            entry[label] = {"n": int(rm.sum()), "correct_91": int(ca.sum()), "correct_97": int(cb.sum()),
                            "fixes": int((~ca & cb).sum()), "breaks": int((ca & ~cb).sum())}
        summary["populations"][kind] = entry
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "ablation_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_parquet(REPORT / "ablation_case_ledger.parquet", index=False)
    lines = ["# Production-recipe 91 vs 97 relation ablation", "", f"Status: {summary['status']}; folds: {len(files)}", ""]
    for k, v in summary["populations"].items():
        f = v["final_pred"]
        lines += [f"## {k}", "", f"- final correct: 91={f['correct_91']}/{f['n']}, 97={f['correct_97']}/{f['n']}",
                  f"- fixes={f['fixes']}, breaks={f['breaks']}",
                  f"- candidate recall={v['candidate_recall']:.6f}, relation-active rows={v['relation_active_rows']}", ""]
    lines += ["The encoder candidate assets are shared between arms but are not fully nested inside the outer ranker CV.",
              "release_shift is a known-target stress set; it is not public-test ground truth."]
    (REPORT / "ablation_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prepare", "fit", "all", "summarize"], default="all")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--arms", nargs="+", choices=["91", "97"], default=["91", "97"])
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    if a.phase in ("prepare", "all"):
        for f in a.folds:
            build_fold(f, force=a.force)
    if a.phase in ("fit", "all"):
        for f in a.folds:
            fit_fold(f, a.arms, threads=a.threads)
    if a.phase in ("summarize", "all"):
        summarize()


if __name__ == "__main__":
    main()
