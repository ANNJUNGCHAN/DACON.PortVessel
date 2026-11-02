"""S9x: Latin cross-encoder re-scoring of the ranker's top-k (experiments/s9_robust/llmgen30/xe33.py, xeval33.py).

For every NON-Hangul query, the ranker's top-``k`` candidates are re-scored by a small cross-encoder
(intfloat/multilingual-e5-small + 1-logit head, fine-tuned listwise on train Latin rows and their noise / abbreviation
copies) and the score becomes ``score + alpha * xe`` for those candidates (others unchanged), before the post-hoc
rules. Each row is scored from its own fields and candidates only (row independence). Hangul queries are untouched.
Model files: model/<subdir>/ (HF save_pretrained, fp16). Group-fold OOF on the harness (seed-0 ranker, top 5):
SHIFT Latin 198 -> 180 errors (26 fixed / 8 broken) at alpha 0.75 x scale, CLEAN 3 -> 3.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ROOT = Path(__file__).resolve().parents[2]  # model/src/rank -> model/


def _num(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "na"
    return "na" if math.isnan(v) else f"{v:.0f}"


def _s(x) -> str:
    return x if isinstance(x, str) and x else "na"


def record_text(r: dict) -> str:
    return (f"{r['vessel_name']} | GT {_num(r['gross_tonnage'])} | IGT {_num(r['international_gross_tonnage'])} | "
            f"{_s(r.get('flag_name'))} | {_s(r.get('vessel_type'))}")


def candidate_text(record_gt, name: str, gt, dwt, flag, vtype) -> str:
    try:
        rg, cg = float(record_gt), float(gt)
        ratio = f"x{rg / cg:.2f}" if rg > 0 and cg > 0 else "na"
    except (TypeError, ValueError):
        ratio = "na"
    if ratio != "na" and ("nan" in ratio or "inf" in ratio):
        ratio = "na"
    return f"{name} | GT {_num(gt)} ({ratio}) | DWT {_num(dwt)} | {_s(flag)} | {_s(vtype)}"


class LatinXE:
    """Picklable config; the model is loaded lazily from model/<subdir> at the first call."""

    def __init__(self, subdir: str = "xe33", alpha: float = 0.75, k: int = 5, max_length: int = 96, bs: int = 256):
        self.subdir, self.alpha, self.k, self.max_length, self.bs = subdir, float(alpha), int(k), max_length, bs
        self._m = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_m"] = None
        return d

    def _load(self):
        if self._m is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            path = MODEL_ROOT / self.subdir
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(
                path, dtype=torch.float16 if dev.type == "cuda" else torch.float32).to(dev).eval()
            self._m = (tok, model, dev)
        return self._m

    def scores(self, recs: list[str], cands: list[str]) -> np.ndarray:
        import torch
        tok, model, dev = self._load()
        out = []
        with torch.no_grad():
            for i in range(0, len(recs), self.bs):
                enc = tok(recs[i:i + self.bs], cands[i:i + self.bs], truncation=True, max_length=self.max_length,
                          padding=True, return_tensors="pt").to(dev)
                out.append(model(**enc).logits.float().squeeze(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def blend(self, pairs: pd.DataFrame, score: np.ndarray, queries: pd.DataFrame, catalog: pd.DataFrame,
              is_hangul) -> np.ndarray:
        score = np.asarray(score, dtype=np.float64).copy()
        q = pairs["q"].to_numpy()
        lat_q = ~np.asarray(is_hangul, dtype=bool)
        idx = np.flatnonzero(lat_q[q])
        if not len(idx):
            return score
        order = idx[np.lexsort((-score[idx], q[idx]))]
        qs = q[order]
        first = np.r_[True, qs[1:] != qs[:-1]]
        grp_start = np.maximum.accumulate(np.where(first, np.arange(len(qs)), 0))
        rank = np.arange(len(qs)) - grp_start
        sel = order[rank < self.k]
        c = pairs["c"].to_numpy()[sel]
        recs_all = queries.to_dict("records")
        rec_txt = {}
        recs, cands = [], []
        names = catalog["vessel_name"].fillna("").astype(str).to_numpy()
        gts = pd.to_numeric(catalog["gross_tonnage"], errors="coerce").to_numpy()
        dwts = pd.to_numeric(catalog["deadweight_tonnage"], errors="coerce").to_numpy()
        flags, types = catalog["flag_name"].to_numpy(), catalog["vessel_type"].to_numpy()
        for qi, ci in zip(q[sel], c):
            if qi not in rec_txt:
                rec_txt[qi] = record_text(recs_all[qi])
            recs.append(rec_txt[qi])
            cands.append(candidate_text(recs_all[qi]["gross_tonnage"], names[ci], gts[ci], dwts[ci], flags[ci], types[ci]))
        try:
            xe = self.scores(recs, cands)
        except Exception as e:  # a missing / unloadable model must not cost the submission: keep the ranker scores
            print(f"LatinXE unavailable ({e!r}); Latin scores unchanged", flush=True)
            return score
        score[sel] += self.alpha * xe
        print(f"LatinXE: {len(sel)} Latin top-{self.k} pairs re-scored (alpha {self.alpha:g})", flush=True)
        return score
