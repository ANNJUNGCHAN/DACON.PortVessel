"""e5ko (2026-10-30): multilingual-e5-small (MIT, pretrained subword transformer) fine-tuned as a Hangul query ->
catalog Latin name dual encoder. Same data and vessel folds as the charft10 / DE4 encoders (charft10/cache/data10.pkl:
real Hangul train pairs, 4,000 synthetic, rule-rendered pseudo Hangul of every catalog vessel, token postings for hard
negatives) + Flash-Next Hangul variants of the train vessels (fold models: round-1 sources kor kof ko ko2; the full
model adds krr krs, as DE4). Fold j excludes every pair whose vessel is in group j.
Evaluation = DE4's (llmgen41/train41de.py evaluate): whole-catalog R@k of the fold's validation Hangul rows (clean,
shift, q32lako), comparable to llmgen41/results/recall_f{j}_f41.json.
  python train_e5.py --fold 0 [--steps 2500]      (fold -1 = full model)
"""
import argparse, json, math, pickle, random, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S9 = ROOT / "experiments/s9_robust"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(S9 / "charft10"))
MODELS = HERE / "models"; RES = HERE / "results"
BASE = "intfloat/multilingual-e5-small"
QLEN, DLEN = 24, 24


def qfmt(name: str) -> str:
    return "query: " + re.sub(r"\s+", " ", str(name)).strip()


def dfmt(name: str) -> str:
    return "passage: " + re.sub(r"\s+", " ", str(name)).strip()


class E5:
    def __init__(self, path=BASE, dev="cuda"):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(dev)
        self.dev = dev

    def __call__(self, texts, maxlen):
        b = self.tok(texts, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(self.dev)
        h = self.model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)

    @torch.no_grad()
    def encode(self, texts, maxlen, bs=1024):
        self.model.eval()
        out = []
        for i in range(0, len(texts), bs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(self(texts[i:i + bs], maxlen).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def llm_pairs(vids, srcs):
    pos_of = {v: i for i, v in enumerate(vids)}
    out = []
    for f in ("variants41_f38.parquet", "variants41_f38b.parquet"):
        V = pd.read_parquet(S9 / "llmgen41/data" / f)
        V = V[(V.lang == "ko") & V.src.isin(srcs) & ~V.op.isin({"mix", "drop", "other"})]
        out += [(v, pos_of[str(i)]) for v, i in zip(V.variant, V.vessel_id) if str(i) in pos_of]
    return out


NAME = "e5"


def evaluate(enc, D, j, cat, names):
    from common import is_hangul_name, load_train
    tr = load_train(); sh = pd.read_parquet(S9 / "cache/shift_rows.parquet")
    va = np.flatnonzero(D["row_fold"] == j)
    pos_of = {v: i for i, v in enumerate(D["vids"])}
    C = torch.from_numpy(cat).cuda().half()
    q32 = pd.read_parquet(S9 / "llmgen30/data/q_q32lako.parquet")
    out = {}
    for kind, df in (("clean", tr), ("shift", sh), ("q32lako", q32)):
        d = df.iloc[va]; d = d[d.vessel_name.map(is_hangul_name)]
        if kind == "q32lako":
            d = d[d.family != "llm_none"]
        Q = torch.from_numpy(enc.encode([qfmt(n) for n in d.vessel_name], QLEN)).cuda().half()
        top = (Q @ C.T).topk(50, dim=1).indices.cpu().numpy()
        t = d.target_vessel_id.astype(str).map(pos_of).to_numpy()
        rk = np.array([np.flatnonzero(r == x)[0] if (r == x).any() else 99 for r, x in zip(top, t)])
        out[kind] = {f"R@{k}": round(float((rk < k).mean()), 4) for k in (1, 5, 10, 20, 50)}
        out[kind]["n"] = int(len(d))
    print(json.dumps(out), flush=True)
    RES.mkdir(exist_ok=True)
    json.dump(out, open(RES / f"recall_{NAME}_f{j}.json", "w"), indent=1)


def main(a):
    D = pickle.load(open(S9 / "charft10/cache/data10.pkl", "rb"))
    from common import load_catalog
    names = load_catalog()["vessel_name"].fillna("").tolist()
    docs_txt = [dfmt(n) for n in names]
    j = a.fold; seed = 5300 + j + 1000 * a.seedoff
    rng = random.Random(seed); np.random.seed(seed); torch.manual_seed(seed)
    grp = D["group"]
    ok = grp != j if j >= 0 else np.ones(len(grp), bool)
    real = [(n, p) for n, p in D["real"] if ok[p]]
    synth = [(n, p) for n, p in D["synth"] if ok[p]]
    pseudo = [(s, p) for p, v in enumerate(D["pseudo"]) if ok[p] and names[p] for s in v]
    srcs = ["kor", "kof", "ko", "ko2"] + (["krr", "krs"] if j < 0 else [])
    llm = [(n, p) for n, p in llm_pairs(D["vids"], srcs) if ok[p]]
    post = {t: v[ok[v]] for t, v in D["post"].items()}
    post = {t: v for t, v in post.items() if len(v) > 1}
    toks = [[t for t in set(re.findall(r"[A-Z]{2,}", d)) if t in post] for d in D["docs"]]
    dh = np.array([hash(n.upper()) for n in names])
    allowed = np.flatnonzero(ok)
    print(f"fold {j}: real {len(real)} synth {len(synth)} pseudo {len(pseudo)} llm {len(llm)}", flush=True)
    from de10 import augment
    enc = E5()
    opt = torch.optim.AdamW(enc.model.parameters(), lr=a.lr, weight_decay=0.01)
    scale = 1.0 / a.temp
    sched = lambda s: min(1, s / 150) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps)))
    mined = {}

    def mine():
        cat = enc.encode([docs_txt[i] for i in allowed], DLEN)
        catt = torch.from_numpy(cat).cuda().half()
        qs = real + synth + llm[::3] + rng.sample(pseudo, min(len(pseudo), 20000))
        Q = enc.encode([qfmt(n) for n, _ in qs], QLEN)
        for i in range(0, len(qs), 4096):
            s = torch.from_numpy(Q[i:i + 4096]).cuda().half() @ catt.T
            top = s.topk(12, dim=1).indices.cpu().numpy()
            for (n, p), row in zip(qs[i:i + 4096], top):
                c = allowed[row]; mined[p] = c[dh[c] != dh[p]][:8]
        enc.model.train()

    def hard(p):
        r = rng.random()
        if p in mined and r < 0.5 and len(mined[p]):
            m = mined[p]; return int(m[rng.randrange(len(m))])
        if toks[p] and r < 0.9:
            pp = post[rng.choice(toks[p])]; c = int(pp[rng.randrange(len(pp))])
            return c if dh[c] != dh[p] else int(allowed[rng.randrange(len(allowed))])
        return int(allowed[rng.randrange(len(allowed))])

    t0 = time.time(); enc.model.train(); B = a.bs
    for step in range(a.steps):
        if step in (int(a.steps * 0.4), int(a.steps * 0.7)):
            tm = time.time(); mine(); print(f"  mined {len(mined)} in {time.time() - tm:.0f}s", flush=True)
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        nr, ns, nl = int(B * 0.25), int(B * 0.15), int(B * 0.3)
        batch = rng.choices(real, k=nr) + rng.choices(synth, k=ns) + rng.choices(llm, k=nl) + \
            rng.choices(pseudo, k=B - nr - ns - nl)
        pos = [p for _, p in batch]
        negs = [hard(p) for p in pos]
        dpos = pos + negs
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eq = enc([qfmt(augment(n, rng, a.augv) or n) for n, _ in batch], QLEN)
            ed = enc([docs_txt[i] for i in dpos], DLEN)
        logits = scale * eq.float() @ ed.float().T
        h = torch.from_numpy(dh[np.array(dpos)]).cuda()
        same = h[None, :] == torch.from_numpy(dh[np.array(pos)]).cuda()[:, None]
        tgt = torch.arange(B, device="cuda")
        same[tgt, tgt] = False
        loss = F.cross_entropy(logits.masked_fill(same, -1e4), tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.model.parameters(), 1.0); opt.step()
        if step % 250 == 0 or step == a.steps - 1:
            print(f"  step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    out = MODELS / f"{a.name}_f{j}"
    out.mkdir(parents=True, exist_ok=True)
    enc.model.half().save_pretrained(out); enc.tok.save_pretrained(out); enc.model.float()
    cat = enc.encode(docs_txt, DLEN)
    np.save(out / "cat.npy", cat.astype(np.float16))
    if j >= 0:
        evaluate(enc, D, j, cat, names)
    print(f"DONE fold {j} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--name", default="e5"); ap.add_argument("--seedoff", type=int, default=0)
    ap.add_argument("--augv", type=int, default=1)
    args = ap.parse_args(); NAME = args.name
    main(args)
