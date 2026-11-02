"""Fit the full-train Hangul ranker (shipped S9o recipe lgb_colbag_de3, same part weights as run_ablation.load_training)
on ablation_full/foldfull parts, with or without the 4 e5 features.
  python fit_full.py --e5 1 [--threads 8]   -> models/full_e5.joblib  (--e5 0 -> models/full_noe5.joblib)"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/relation_ablation_20261029"))
import run_ablation as RA  # noqa: E402
import joblib  # noqa: E402
from common import is_hangul_name  # noqa: E402
from rank.pipeline import query_weights  # noqa: E402
from rank.rankers import RANKERS  # noqa: E402

FULL = HERE / "ablation_full/foldfull/parts"
ap = argparse.ArgumentParser(); ap.add_argument("--e5", type=int, default=1); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--seeds", default="0,1,0,1"); ap.add_argument("--tag", default="")
ap.add_argument("--extra", default="", help="comma list of extra e5 feature file prefixes, e.g. e5b")
a = ap.parse_args()
train, catalog = RA.load_data()
art, selected = RA.authority()
shipped = art["pipeline"].ranker
base_cols = list(shipped.ko.cols[-1]) if hasattr(shipped, "ko") else None
if base_cols is None:  # s9c4T_de10 authority: the Hangul ranker of the LangSplit is in .ko
    raise SystemExit("authority ranker has no .ko")
rw = query_weights(train); ko = train["vessel_name"].map(is_hangul_name).to_numpy()
base_w = {1: float(rw[ko].max()), 0: float(rw[~ko].max())}
frames, ys, qids, ws, off = [], [], [], [], 0
for name in [f"ko_{n}" for n in RA.COMMON_PARTS]:
    X = pd.read_parquet(FULL / f"{name}.parquet", columns=base_cols)
    m = pd.read_parquet(FULL / f"pairs_{name}.parquet")
    if a.e5:
        X = pd.concat([X.reset_index(drop=True), pd.read_parquet(FULL / f"e5_{name}.parquet").reset_index(drop=True)], axis=1)
    for pre in [x for x in a.extra.split(",") if x]:
        E = pd.read_parquet(FULL / f"{pre}_{name}.parquet"); E.columns = [c.replace("e5_", f"{pre}_") for c in E.columns]
        X = pd.concat([X.reset_index(drop=True), E.reset_index(drop=True)], axis=1)
    q = m["q"].to_numpy(np.int64)
    frames.append(X); ys.append(m["y"].to_numpy(np.int8)); qids.append(q + off)
    ws.append(rw[q] if name == "ko_real" else
              np.asarray([base_w[int(x)] for x in m["is_ko"]], np.float32) * m["part_weight"].to_numpy(np.float32))
    off += int(q.max()) + 1
X = pd.concat(frames, ignore_index=True); y = np.concatenate(ys); qid = np.concatenate(qids); w = np.concatenate(ws)
from rank.rankers import ColumnEnsemble, LGBRank
seeds = [int(s) for s in a.seeds.split(",")]
model = ColumnEnsemble([LGBRank("lambdarank", seed=s) for s in seeds], [("de3_",), ("de3_",), (), ()], name="lgb_colbag_de3")
for mem in model.members:
    mem.model.set_params(n_jobs=a.threads)
t0 = time.time(); model.fit(X, y, qid, w)
out = HERE / f"models/full_{'e5' if a.e5 else 'noe5'}{('_' + a.extra.replace(',', '_')) if a.extra else ''}{a.tag}.joblib"
joblib.dump(model, out, compress=1)
print(f"fitted {len(X)} rows x {X.shape[1]} cols in {time.time() - t0:.0f}s -> {out}; shipped ko cols {len(base_cols)}",
      flush=True)
