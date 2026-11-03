"""Pack a Hangul-bag variant of a shipped noLLM ZIP (LangSplit ranker): art["pipeline"].ranker.ko := BagRanker(models).
Copies every member byte-for-byte except model/artifacts.joblib (and, with --e5 1, model/src/rank/pipeline.py, which
gets one hook before ranker.predict that adds the 4 e5 ranker features); adds model/src/rank/bagrank.py (+ e5feat.py
and model/e5ko/* with --e5 1). Unpickling uses the base ZIP's own src snapshot.
  python pack_bag.py --base E08.zip --out hbag5_e08_noLLM.zip --models f0.joblib,...,f4.joblib [--e5 1]
"""
import argparse, hashlib, io, json, sys, tempfile, zipfile
from pathlib import Path
import joblib

HERE = Path(__file__).resolve().parent
HOOK = "        score = self.ranker.predict(X)\n"
E5_PATCH = ("        if getattr(self, \"e5_feats\", None):  # 10/31 e5 ranker features (rank/e5feat.py)\n"
         "            from pathlib import Path as _P\n"
         "            from rank.e5feat import add_e5\n"
         "            _ef = self.e5_feats if isinstance(self.e5_feats, (list, tuple)) else [(\"e5ko\", \"e5\")]\n"
         "            for _d, _p in _ef:\n"
         "                X = add_e5(X, pairs, queries, _P(__file__).resolve().parents[2] / _d, prefix=_p)\n")
DE4_PATCH = ("        if getattr(self, \"de4_feats\", None):  # 10/31 DE4 ranker features (rank/de4feat.py)\n"
             "            from pathlib import Path as _P4\n"
             "            from rank.de4feat import add_de4\n"
             "            X = add_de4(X, pairs, queries, _P4(__file__).resolve().parents[2] / \"de4\")\n")
CP_PATCH = ("        if getattr(self, \"cp_feats\", None):  # 10/31 catalog-prior ranker features (rank/cpfeat.py)\n"
            "            from pathlib import Path as _PC\n"
            "            from rank.cpfeat import add_cp\n"
            "            X = add_cp(X, pairs, _PC(__file__).resolve().parents[2] / \"cp\")\n")
CRLF, LF = b"\r\n", "\n"


def patch_pipeline(data: bytes, e5: bool, de4: bool, cp: bool) -> bytes:
    """Insert only the enabled feature hooks (the final TwinDelta ZIP has the e5 hook alone)."""
    crlf = CRLF in data
    s = data.decode("utf-8").replace("\r\n", LF)
    assert s.count(HOOK) == 1, "ranker.predict hook not found exactly once"
    s = s.replace(HOOK, (E5_PATCH if e5 else "") + (DE4_PATCH if de4 else "") + (CP_PATCH if cp else "") + HOOK)
    return (s.replace(LF, "\r\n") if crlf else s).encode("utf-8")


ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--models", required=True, help="comma list of fitted Hangul ranker joblib files")
ap.add_argument("--e5", type=int, default=0, help="1: add the 4 e5 ranker features before ranker.predict")
ap.add_argument("--cp", type=int, default=0, help="1: catalog-prior features (model/cp/prior.npy, cp_*)")
ap.add_argument("--de4", type=int, default=0, help="1: DE4 ranker features (model/de4/payload.pkl, de4_*)")
ap.add_argument("--e5b", type=int, default=0, help="1: also the second encoder e5b (model/e5kob, features e5b_*)")
ap.add_argument("--e5more", default="", help="comma list of further e5 encoders, e.g. e5c,e5d (model/e5ko<x>/, features <name>_*)")
ap.add_argument("--weights", default=None, help="comma list aligned with --models: WeightedRanker; model SHIPPED = the base ZIP's own Hangul ranker")
a = ap.parse_args()
zb = zipfile.ZipFile(a.base)
tmp = Path(tempfile.mkdtemp(prefix="bag_pack_src_"))
for i in zb.infolist():
    if i.filename.startswith("model/src/"):
        zb.extract(i, tmp)
(tmp / "model/src/rank/bagrank.py").write_bytes((HERE / "bagrank.py").read_bytes())
sys.path.insert(0, str(tmp / "model" / "src"))
from rank.bagrank import BagRanker, WeightedRanker  # noqa: E402
art = joblib.load(io.BytesIO(zb.read("model/artifacts.joblib")))
rk = art["pipeline"].ranker
assert type(rk).__name__ == "LangSplit", type(rk).__name__
members = [rk.ko if p == "SHIPPED" else joblib.load(p) for p in a.models.split(",")]
rk.ko = (WeightedRanker(members, [float(x) for x in a.weights.split(",")]) if a.weights else BagRanker(members))
if a.de4:
    art["pipeline"].de4_feats = True
if a.cp:
    art["pipeline"].cp_feats = True
if a.e5:
    art["pipeline"].e5_feats = ([("e5ko", "e5")] + ([("e5kob", "e5b")] if a.e5b else [])
                                 + [(f"e5ko{x[2:]}", x) for x in a.e5more.split(",") if x])
buf = io.BytesIO(); joblib.dump(art, buf)
out = Path(a.out)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as o:
    for i in zb.infolist():
        if i.filename == "model/artifacts.joblib":
            continue
        data = zb.read(i.filename)
        if (a.e5 or a.de4 or a.cp) and i.filename == "model/src/rank/pipeline.py":
            data = patch_pipeline(data, bool(a.e5), bool(a.de4), bool(a.cp))
        o.writestr(i, data)
    o.writestr("model/src/rank/bagrank.py", (HERE / "bagrank.py").read_bytes())
    if a.e5:
        E5 = HERE.parent / "e5ko_20261030/models/e5_f-1"
        o.writestr("model/src/rank/e5feat.py", (HERE / "e5feat.py").read_bytes())
        for f in sorted(E5.iterdir()):
            if f.name == "cat.npy":
                o.write(f, "model/e5ko/cat.npy", compress_type=zipfile.ZIP_STORED)
            elif f.is_file():
                o.write(f, f"model/e5ko/model/{f.name}")
        for x in [y for y in a.e5more.split(",") if y]:
            EX = HERE.parent / f"e5ko_20261030/models/{x}_f-1"
            for f in sorted(EX.iterdir()):
                if f.name == "cat.npy":
                    o.write(f, f"model/e5ko{x[2:]}/cat.npy", compress_type=zipfile.ZIP_STORED)
                elif f.is_file():
                    o.write(f, f"model/e5ko{x[2:]}/model/{f.name}")
        if a.e5b:
            E5B = HERE.parent / "e5ko_20261030/models/e5b_f-1"
            for f in sorted(E5B.iterdir()):
                if f.name == "cat.npy":
                    o.write(f, "model/e5kob/cat.npy", compress_type=zipfile.ZIP_STORED)
                elif f.is_file():
                    o.write(f, f"model/e5kob/model/{f.name}")
    if a.de4:
        o.writestr("model/src/rank/de4feat.py", (HERE / "de4feat.py").read_bytes())
        o.write(HERE / "models/de4_payload.pkl", "model/de4/payload.pkl")
    if a.cp:
        o.writestr("model/src/rank/cpfeat.py", (HERE / "cpfeat.py").read_bytes())
        o.write(HERE / "models/catprior_full.npy", "model/cp/prior.npy")
    o.writestr("model/artifacts.joblib", buf.getvalue())
h = hashlib.sha256()
with open(out, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
print(json.dumps({"out": str(out), "bytes": out.stat().st_size, "sha256": h.hexdigest(), "members": len(members),
                  "e5": bool(a.e5)}))
