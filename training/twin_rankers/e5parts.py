"""e5 features for the relation-ablation ko training / eval parts (2026-10-31), fold-honest: every Hangul query is
encoded with the e5 fold model of its TARGET vessel's group (data10 D["group"]), which never trained on that vessel
(as the shipped dual-encoder routers do). Latin queries get NaN. Columns, aligned row-for-row with pairs_<part>:
e5_cos, e5_grank (whole-catalog rank, capped 200), e5_gap (row best - cos), e5_rank (rank within the row).
  python e5parts.py   -> <ablation>/fold*/parts/e5_<part>.parquet"""
import pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
E5 = ROOT / "experiments/e5ko_20261030"
sys.path.insert(0, str(E5)); sys.path.insert(0, str(ROOT / "src"))
from train_e5 import E5 as Enc, qfmt, QLEN  # noqa: E402
A = ROOT / "experiments/relation_ablation_20261029/runpod_results/<pod_id>/DACON.Ship/experiments/relation_ablation_20261029/output/ablation"
PARTS = ["ko_real", "ko_synth_ko", "ko_noise", "ko_synth_la", "ko_synth_rn", "eval_clean", "eval_shift", "eval_release_shift"]
GCAP = 200


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(A)); ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--enc", default="e5"); ap.add_argument("--prefix", default="e5")
    a = ap.parse_args(); root = Path(a.root); folds = a.folds.split(",")
    D = pickle.load(open(ROOT / "experiments/s9_robust/charft10/cache/data10.pkl", "rb"))
    pos = {v: i for i, v in enumerate(D["vids"])}
    encs = {g: Enc(str(E5 / f"models/{a.enc}_f{g}")) for g in range(5)}
    cats = {g: torch.from_numpy(np.load(E5 / f"models/{a.enc}_f{g}/cat.npy")).cuda().half() for g in range(5)}
    t0 = time.time()
    for f in folds:
        for part in PARTS:
            qp, pp = root / f"fold{f}/parts/q_{part}.parquet", root / f"fold{f}/parts/pairs_{part}.parquet"
            if not qp.exists():
                continue
            q = pd.read_parquet(qp); p = pd.read_parquet(pp, columns=["q", "c"])
            ko = q.vessel_name.fillna("").str.contains("[가-힣]").to_numpy()
            grp = np.array([D["group"][pos[str(t)]] for t in q.target_vessel_id])
            cos = np.full(len(p), np.nan, np.float32); grk = np.full(len(p), np.nan, np.float32)
            pq = p.q.to_numpy(); pc = p.c.to_numpy()
            order = np.argsort(pq, kind="stable"); starts = np.searchsorted(pq[order], np.arange(len(q)))
            ends = np.searchsorted(pq[order], np.arange(len(q)), side="right")
            for g in range(5):
                qs = np.flatnonzero(ko & (grp == g))
                if not len(qs):
                    continue
                Q = torch.from_numpy(encs[g].encode([qfmt(n) for n in q.vessel_name.iloc[qs]], QLEN)).cuda().half()
                for i in range(0, len(qs), 1024):
                    S = (Q[i:i + 1024] @ cats[g].T).float()
                    for k in range(S.shape[0]):
                        qi = qs[i + k]
                        rows = order[starts[qi]:ends[qi]]
                        if not len(rows):
                            continue
                        srow = S[k]
                        cc = torch.from_numpy(pc[rows].astype(np.int64)).cuda()
                        v = srow[cc]
                        cos[rows] = v.cpu().numpy()
                        grk[rows] = np.minimum((srow[None, :] > v[:, None]).sum(1).cpu().numpy(), GCAP)
            df = pd.DataFrame({"e5_cos": cos, "e5_grank": grk, "q": pq})
            gmax = df.groupby("q").e5_cos.transform("max")
            df["e5_gap"] = (gmax - df.e5_cos).astype(np.float32)
            df["e5_rank"] = df.groupby("q").e5_cos.rank(ascending=False, method="min").astype(np.float32)
            df.drop(columns="q").to_parquet(root / f"fold{f}/parts/{a.prefix}_{part}.parquet", index=False)
            print(f"fold {f} {part}: queries {len(q)} hangul {int(ko.sum())} pairs {len(p)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
