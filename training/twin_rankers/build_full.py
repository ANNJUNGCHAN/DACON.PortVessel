"""Full-train ko parts (2026-10-31) with the relation-ablation harness logic (run_ablation.build_fold) applied to ALL
train rows (no validation fold): the shipped S9o Hangul-ranker recipe (real rows + synth_ko/la/rn at weight .5 +
noise copies frac .5 seed 4242 tag "full" at weight .5, OOF LLM columns merged on real rows, vessel-held-out encoder
routers, FoldFeatures over the full train). One part per process:
  python build_full.py --part ko_real|ko_synth_ko|ko_synth_la|ko_synth_rn|ko_noise
-> ablation_full/foldfull/parts/<part>.parquet (+ pairs_/q_ files), same layout as the fold parts."""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/relation_ablation_20261029"))
import run_ablation as RA  # noqa: E402
from rank.candidates import CandidateGenerator  # noqa: E402
from rank.finetune import localize_attributes  # noqa: E402
from rank.fold_features import FoldFeatures  # noqa: E402
from rank.pipeline import RankPipeline  # noqa: E402

RA.OUT = HERE / "ablation_full"
ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True)
a = ap.parse_args()
t0 = time.time()
train, catalog = RA.load_data()
art, selected = RA.authority()
selected_cols = list(selected.columns)
routers = RA.make_routers(train, catalog)
ko_opts = dict(art["pipeline"].cand_opts); ko_opts["latin_de2_k"] = 0
ko_gen = CandidateGenerator(catalog, tr_cache=art["tr_cache"], feats4_aux=art["feats4_aux"], dual_encoder=routers[0],
                            dual_encoder2=routers[1], dual_encoder3=routers[2], latin_dual_encoder=routers[3],
                            latin_dual_encoder2=routers[4], **ko_opts)
pipe = RankPipeline("lgb_binary", cand_opts=selected.cand_opts)
pipe.fold = FoldFeatures(train, catalog); pipe.scorers = []
relation = RA.NameRelationFeatures(train, catalog)
allrows = np.arange(len(train))
print(f"setup {time.time() - t0:.0f}s", flush=True)
if a.part == "ko_real":
    oof = pd.read_parquet(RA.OOF_FILE)
    pairs = RA.hybrid_pairs(train, ko_gen, RA.S9K / "clean.parquet", train, allrows)
    RA.build_part("full", "ko_real", train, pairs, pipe, relation, catalog, selected_cols,
                  global_rows=allrows, oof=oof, force=True)
elif a.part in ("ko_synth_ko", "ko_synth_la", "ko_synth_rn"):
    name = a.part[3:]
    path, seed = RA.SYNTH[name]
    q0 = pd.read_parquet(path)
    q = localize_attributes(q0.reset_index(drop=True), train, catalog, seed=seed)
    pairs = RA.hybrid_pairs(q, ko_gen, RA.S9K / f"{name}.parquet", q0, np.arange(len(q0)))
    RA.build_part("full", a.part, q, pairs, pipe, relation, catalog, selected_cols, weight=.5, force=True)
elif a.part == "ko_noise":
    from rank.noise_aug import noise_copies
    copies, src = noise_copies(train, catalog, frac=0.5, seed=4242, fold=None)
    pairs = RA.hybrid_pairs(copies, ko_gen)
    RA.build_part("full", "ko_noise", copies, pairs, pipe, relation, catalog, selected_cols, src=src, weight=.5,
                  force=True)
print(f"DONE {a.part} {time.time() - t0:.0f}s", flush=True)
