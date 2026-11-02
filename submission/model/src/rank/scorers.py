"""Pluggable external pair scorers (e.g. an LLM) that add feature columns.

A scorer receives the query rows and the candidate pairs and returns extra
columns aligned to ``pairs.index``. Unscored pairs get NaN, which every
ranker here handles natively.

Row independence: a scorer must score each query row's candidates on their
own. ``CausalLMScorer`` therefore runs exactly one forward batch per query
row, so batch composition (and fp16 numerics) never depend on other rows.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from common import is_hangul_name


class PairScorer(Protocol):
    name: str
    columns: tuple[str, ...]

    def score(self, queries: pd.DataFrame, pairs: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
        ...


PROMPT = ("항만 입출항 기록에 한글로 적힌 선박명을 선박 등록부의 영문 선박명과 연결한다.\n"
          "한글 선박명: {ko}\n영문 선박명:")


class CausalLMScorer:
    """Log-likelihood of the catalog (English) name given the Hangul record name.

    Scores the ``top_k`` candidates per Hangul row (by the heuristic rank from
    candidate generation). Results are cached on disk per (model, prompt).
    """

    columns = ("llm_lp_mean", "llm_lp_sum", "llm_rank")

    def __init__(self, model_path: str, top_k: int = 10, hangul_only: bool = True,
                 device: str | None = None, dtype: str = "float16", cache_dir: str | Path | None = None):
        self.model_path = model_path
        self.name = "llm_" + Path(model_path).name
        self.top_k, self.hangul_only = top_k, hangul_only
        self.device, self.dtype = device, dtype
        self.cache_path = (Path(cache_dir) / f"{self.name}_{self._prompt_hash()}.pkl") if cache_dir else None
        self.cache: dict[tuple[str, str], tuple[float, float]] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = pickle.loads(self.cache_path.read_bytes())
        self._model = None

    @staticmethod
    def _prompt_hash() -> str:
        return hashlib.sha1(PROMPT.encode("utf-8")).hexdigest()[:8]

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.torch = torch
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tok = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=getattr(torch, self.dtype)).to(dev).eval()
            self.dev = dev
        return self._model

    def _score_row(self, ko: str, names: list[str]) -> list[tuple[float, float]]:
        missing = [n for n in names if (ko, n) not in self.cache]
        if missing:
            model = self._load()
            torch = self.torch
            prefix_ids = self.tok(PROMPT.format(ko=ko), add_special_tokens=False)["input_ids"]
            seqs = [prefix_ids + self.tok(" " + n, add_special_tokens=False)["input_ids"] for n in missing]
            width = max(map(len, seqs))
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, : len(s)] = torch.tensor(s)
                mask[i, : len(s)] = 1
            with torch.no_grad():
                logits = model(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev)).logits.float()
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            target = ids[:, 1:].to(self.dev)
            tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).cpu().numpy()
            for i, (n, s) in enumerate(zip(missing, seqs)):
                cont = tok_lp[i, len(prefix_ids) - 1 : len(s) - 1]
                self.cache[(ko, n)] = (float(cont.sum()), float(cont.mean()))
        return [self.cache[(ko, n)] for n in names]

    def score(self, queries: pd.DataFrame, pairs: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=pairs.index, columns=list(self.columns))
        names = catalog["vessel_name"].fillna("").astype(str).to_numpy()
        qnames = queries["vessel_name"].to_numpy()
        selected = pairs[pairs["heur_rank"] < self.top_k]
        for q, grp in selected.groupby("q", sort=True):
            ko = qnames[q]
            if not isinstance(ko, str) or (self.hangul_only and not is_hangul_name(ko)):
                continue
            res = np.array(self._score_row(ko, names[grp["c"].to_numpy()].tolist()))
            out.loc[grp.index, "llm_lp_sum"] = res[:, 0]
            out.loc[grp.index, "llm_lp_mean"] = res[:, 1]
            out.loc[grp.index, "llm_rank"] = np.argsort(np.argsort(-res[:, 1], kind="stable"))
        return out

    def save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(pickle.dumps(self.cache))
