"""e5 ranker features at inference (2026-10-31), the same four columns the e5ko fold rankers were trained with
(e5parts.py): for Hangul queries, e5_cos (fine-tuned multilingual-e5-small cosine of the query name and the candidate),
e5_grank (whole-catalog rank, capped 200), e5_gap (row best - cos), e5_rank (rank within the row's candidates);
NaN on Latin rows. Row-local. Packaged as rank/e5feat.py; model files under model/e5ko/."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

GCAP = 200
_ENC = {}


def _encoder(root: Path):
    key = str(root)
    if key not in _ENC:
        import torch
        from transformers import AutoModel, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(root / "model")
        model = AutoModel.from_pretrained(root / "model", torch_dtype=torch.float16 if dev == "cuda"
                                          else torch.float32).to(dev).eval()
        _ENC[key] = (tok, model, dev, np.load(root / "cat.npy").astype(np.float32))
    return _ENC[key]


def _encode(root: Path, names: list[str]) -> np.ndarray:
    import torch
    tok, model, dev, _ = _encoder(root)
    texts = ["query: " + re.sub(r"\s+", " ", str(n)).strip() for n in names]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            b = tok(texts[i:i + 256], padding=True, truncation=True, max_length=24, return_tensors="pt").to(dev)
            h = model(**b).last_hidden_state.float()
            m = b["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 384), np.float32)


def add_e5(X: pd.DataFrame, pairs: pd.DataFrame, queries: pd.DataFrame, root: Path, log=print,
           prefix: str = "e5") -> pd.DataFrame:
    from common import is_hangul_name
    root = Path(root)
    pq = pairs["q"].to_numpy(); pc = pairs["c"].to_numpy()
    cos = np.full(len(pairs), np.nan, np.float32); grk = np.full(len(pairs), np.nan, np.float32)
    ko = queries["vessel_name"].map(is_hangul_name).to_numpy(dtype=bool)
    qs = np.flatnonzero(ko)
    if len(qs):
        cat = _encoder(root)[3]
        Q = _encode(root, queries["vessel_name"].iloc[qs].tolist())
        order = np.argsort(pq, kind="stable"); sq = pq[order]
        for qi, qv in zip(qs, Q):
            rows = order[np.searchsorted(sq, qi):np.searchsorted(sq, qi, side="right")]
            if not len(rows):
                continue
            sims = cat @ qv
            v = sims[pc[rows]]
            cos[rows] = v
            # count(sims > v) capped at GCAP, from the GCAP+1 largest values only (identical to the full count)
            top = np.sort(np.partition(sims, len(sims) - (GCAP + 1))[-(GCAP + 1):])
            cnt = len(top) - np.searchsorted(top, v, side="right")
            grk[rows] = np.where(v < top[0], GCAP, np.minimum(cnt, GCAP))
    df = pd.DataFrame({"e5_cos": cos, "e5_grank": grk, "q": pq})
    df["e5_gap"] = (df.groupby("q").e5_cos.transform("max") - df.e5_cos).astype(np.float32)
    df["e5_rank"] = df.groupby("q").e5_cos.rank(ascending=False, method="min").astype(np.float32)
    X = X.copy()
    for c in ("e5_cos", "e5_grank", "e5_gap", "e5_rank"):
        X[c.replace("e5_", f"{prefix}_")] = df[c].to_numpy(np.float32)
    log(f"e5 ranker features ({prefix}): Hangul rows {len(qs)}")
    return X
