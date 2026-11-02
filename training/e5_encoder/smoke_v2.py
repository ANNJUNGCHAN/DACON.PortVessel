"""Normal-entrypoint smoke of an e5ko package vs its base: python smoke_v2.py BASE_ZIP E5_ZIP
Input: 200 HJ validation-fold Hangul queries (LLM variants of train vessels), 150 clean Hangul train rows, 100 Latin
train rows. Runs full / reversed / 150-row subset; checks row independence, prints the e5stack log line, the stage's
wall time, and changed / fixed / broken counts vs the base package (the shipped ranker is in-sample on these vessels,
so this checks the code path, not accuracy)."""
import os, subprocess, sys, tempfile, time, zipfile
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S9 = ROOT / "experiments/s9_robust"
CAT = ROOT / "data/raw/open/vessel_catalog.csv"
COLS = ["sample_id", "vessel_name", "call_sign", "gross_tonnage", "international_gross_tonnage", "flag_name",
        "vessel_type", "arrival_year"]
tr = pd.read_csv(ROOT / "data/raw/open/train.csv")
ko = tr.vessel_name.str.contains("[가-힣]")
parts = [tr[ko].sample(150, random_state=0).assign(kind="clean_ko"), tr[~ko].sample(100, random_state=0).assign(kind="latin")]
for k in ("llmko", "f38lako", "q32lako", "f38b_krr"):
    q = pd.read_parquet(S9 / f"llmgen30/data/q_{k}.parquet")
    q = q[q.vessel_name.fillna("").str.contains("[가-힣]")].sample(50, random_state=1).copy()
    q["sample_id"] = [f"HJ_{k}_{i}" for i in q.index]; q["arrival_year"] = 2024
    parts.append(q.assign(kind=k))
inp = pd.concat(parts, ignore_index=True)
truth = dict(zip(inp.sample_id, inp.target_vessel_id.astype(str))); kind = dict(zip(inp.sample_id, inp.kind))
work = Path(tempfile.mkdtemp(prefix="e5smoke_", dir=HERE))
sc = {"full": inp[COLS], "rev": inp[COLS].iloc[::-1], "sub": inp[COLS].sample(150, random_state=2)}
for n, f in sc.items():
    f.to_csv(work / f"{n}.csv", index=False, encoding="utf-8-sig")


def run(zp):
    d = work / Path(zp).stem
    with zipfile.ZipFile(zp) as z:
        for m in z.namelist():
            if m == "script.py" or m.startswith("model/"):
                z.extract(m, d)
    res = {}
    for n in sc:
        t0 = time.time()
        r = subprocess.run([sys.executable, "-X", "utf8", "-c",
                            f"import script; script.run(r'{work / (n + '.csv')}', r'{CAT}', r'{d / (n + '_out.csv')}')"],
                           cwd=d, capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
        if r.returncode:
            raise SystemExit(f"{zp} {n} failed:\n{r.stderr[-3000:]}")
        log = [l for l in r.stdout.splitlines() if "e5stack" in l or "ranker features" in l]
        print(f"  {Path(zp).name} {n}: {time.time() - t0:.0f}s {log}", flush=True)
        o = pd.read_csv(d / f"{n}_out.csv"); res[n] = dict(zip(o.sample_id, o.target_vessel_id.astype(str)))
    for n in ("rev", "sub"):
        bad = sum(res["full"][s] != v for s, v in res[n].items())
        assert bad == 0, f"{zp}: {n} differs from full on {bad} rows"
    print(f"  {Path(zp).name}: row-independent OK", flush=True)
    return res["full"]


b = run(sys.argv[1]); p = run(sys.argv[2])
for k in sorted(set(kind.values())):
    ids = [s for s in truth if kind[s] == k]
    ch = [s for s in ids if p[s] != b[s]]
    print(f"  {k:9s} rows {len(ids)}, base err {sum(b[s] != truth[s] for s in ids)}, changed {len(ch)}, "
          f"fixed {sum(p[s] == truth[s] for s in ch)}, broken {sum(b[s] == truth[s] for s in ch)}")
