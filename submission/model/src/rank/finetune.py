"""Fine-tuned pair scorer for Hangul rows (Ditto-style record/candidate text).

Training signal
  - real Hangul train rows (fold-internal): the true vessel vs. the hard
    negatives our candidate generator retrieved for that row
  - synthetic rows: a catalog vessel's English name rendered to Hangul by the
    rule-based transliterator (en2ko), with its tonnage, run through the same
    candidate generator for hard negatives. Uses only official catalog data.

Objective: listwise softmax over each row's candidate group, matching the
argmax decision used at prediction time.

Backbones (all public general-purpose weights, loaded from the local cache):
  minilm     sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, full FT
  qwen0.6b   Qwen/Qwen3-0.6B + LoRA, sequence-classification head
  qwen1.7b   Qwen/Qwen3-1.7B + LoRA, sequence-classification head

Row independence: at scoring time each query row's candidates form one batch,
padded to a fixed length, so no other row can influence its numbers.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import is_hangul_name

SELECT_K = 8          # top-K by heuristic rank ∪ top-K by transliteration rank
GROUP_SIZE = 16       # candidates per row in a training group (positive kept)
LATIN_GROUP_SIZE = 8  # Latin rows have fewer selected candidates
MAX_LEN = 96
DECODER_MAX_LEN = 128  # record + candidate text is ~100 tokens for Qwen

BACKBONES = {
    "minilm": dict(repo="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                   lora=False, lr=3e-5, batch_groups=8, pair_input=True),
    "qwen0.6b": dict(repo="Qwen/Qwen3-0.6B", lora=True, lr=2e-4, batch_groups=1, pair_input=False),
    "qwen1.7b": dict(repo="Qwen/Qwen3-1.7B", lora=True, lr=1.5e-4, batch_groups=1, pair_input=False),
    "qwen4b": dict(repo="Qwen/Qwen3-4B", lora=True, lr=1e-4, batch_groups=1, pair_input=False),
    # Qwen3.5 (2026-03): hybrid linear/full attention, so LoRA targets every linear layer.
    "qwen35_0.8b": dict(repo="Qwen/Qwen3.5-0.8B", lora=True, lr=2e-4, batch_groups=1, pair_input=False,
                        lora_targets="all-linear"),
    "qwen35_2b": dict(repo="Qwen/Qwen3.5-2B", lora=True, lr=1.5e-4, batch_groups=1, pair_input=False,
                      lora_targets="all-linear"),
    "qwen35_4b": dict(repo="Qwen/Qwen3.5-4B", lora=True, lr=1e-4, batch_groups=1, pair_input=False,
                      lora_targets="all-linear"),
    "qwen35_9b": dict(repo="Qwen/Qwen3.5-9B", lora=True, lr=8e-5, batch_groups=1, pair_input=False,
                      lora_targets="all-linear"),
}


# ---------------------------------------------------------------------------
# Text serialization (Ditto style)
# ---------------------------------------------------------------------------
def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)) or value == "":
        return "unknown"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def query_text(row: dict) -> str:
    return (f"name: {_fmt(row.get('vessel_name'))}; gross tonnage: {_fmt(row.get('gross_tonnage'))}; "
            f"international gross tonnage: {_fmt(row.get('international_gross_tonnage'))}; "
            f"flag: {_fmt(row.get('flag_name'))}; type: {_fmt(row.get('vessel_type'))}")


def candidate_text(row: dict) -> str:
    return (f"official name: {_fmt(row.get('vessel_name'))}; gross tonnage: {_fmt(row.get('gross_tonnage'))}; "
            f"deadweight tonnage: {_fmt(row.get('deadweight_tonnage'))}; "
            f"flag: {_fmt(row.get('flag_name'))}; type: {_fmt(row.get('vessel_type'))}")


def select_pairs(pairs: pd.DataFrame, queries: pd.DataFrame, k: int | None = SELECT_K,
                 latin_k: int | None = None, hangul_extra: bool = False) -> pd.DataFrame:
    """Pairs worth scoring.

    Hangul rows: top-k by heuristic or transliteration rank (k=None: every candidate).
    Latin rows (only when latin_k is set): rows whose normalized name does not match
    exactly one catalog vessel (renames, typos, sister ships), top-latin_k by heuristic
    or OSA rank. Every rule is row-local.
    Candidates added by the S9 retrieval options (column src_extra = 1, rank.candidates) are never selected;
    without that column (all S9 options off) nothing changes.
    ``hangul_extra`` (S9 item 5, default False = the rule above exactly): additionally select EVERY src_extra
    candidate of a Hangul row (the hangul_global_k extras); the result is then sorted by (q, c). Latin extras are
    never selected.
    """
    if hangul_extra and "src_extra" in pairs.columns:
        base = select_pairs(pairs, queries, k, latin_k)
        hangul = queries["vessel_name"].map(is_hangul_name).to_numpy()
        add = pairs[(pairs["src_extra"].to_numpy() == 1) & hangul[pairs["q"].to_numpy()]]
        return pd.concat([base, add]).sort_values(["q", "c"], kind="stable")
    if "src_extra" in pairs.columns:
        pairs = pairs[pairs["src_extra"].to_numpy() == 0]
    hangul = queries["vessel_name"].map(is_hangul_name).to_numpy()
    is_ko = hangul[pairs["q"].to_numpy()]
    sub = pairs[is_ko]
    if k is not None:
        sub = sub[(sub["heur_rank"] < k) | (sub["tr_rank"] < k)]
    if not latin_k:
        return sub
    lat = pairs[~is_ko]
    exact = lat.groupby("q")["name_exact"].transform("sum")
    lat = lat[exact != 1]
    osa_rank = lat.groupby("q")["osa_sim"].rank(ascending=False, method="first") - 1
    lat = lat[(lat["heur_rank"] < latin_k) | (osa_rank < latin_k)]
    return pd.concat([sub, lat]).sort_values(["q", "c"], kind="stable")


# ---------------------------------------------------------------------------
# Synthetic Hangul rows from the catalog
# ---------------------------------------------------------------------------
_CONFUSIONS = [set("ㅓㅗ"), set("ㅐㅔ"), set("ㅕㅖ"), set("ㅅㅆ"), set("ㅈㅊㅉ"), set("ㄱㅋ"),
               set("ㄷㅌ"), set("ㅂㅍ"), set("ㅜㅠ")]
_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"


def jamo_noise(text: str, rng: random.Random, p: float = 0.12) -> str:
    """Perturb Hangul the way real records drift from rule-based readings."""
    out = []
    for ch in text:
        code = ord(ch) - 0xAC00
        if not (0 <= code < 11172) or rng.random() > p:
            out.append(ch)
            continue
        cho, jung, jong = code // 588, (code % 588) // 28, code % 28
        kind = rng.random()
        if kind < 0.45:
            for group in _CONFUSIONS:
                if _JUNG[jung] in group:
                    jung = _JUNG.index(rng.choice(sorted(group - {_JUNG[jung]})))
                    break
        elif kind < 0.85:
            for group in _CONFUSIONS:
                if _CHO[cho] in group:
                    cho = _CHO.index(rng.choice(sorted(group - {_CHO[cho]})))
                    break
        else:
            jong = 0 if jong else 8  # drop a coda, or add ㄹ
        out.append(chr(0xAC00 + cho * 588 + jung * 28 + jong))
    return "".join(out)


def synthetic_queries(catalog: pd.DataFrame, n: int, seed: int = 0, noise: float = 0.0) -> pd.DataFrame:
    """Catalog vessels rendered as Hangul records (name only; flag/type filled per fold).

    ``noise`` > 0 perturbs the jamo so the record is not an exact rule-based reading,
    which keeps transliteration similarity from being trivially 1.0 for the target.
    """
    from en2ko import name_variants

    rng = random.Random(seed)
    gt = pd.to_numeric(catalog["gross_tonnage"], errors="coerce")
    names = catalog["vessel_name"].fillna("")
    eligible = np.flatnonzero((gt >= 1000).to_numpy() & names.str.contains("[A-Za-z]{3}").to_numpy())
    picks = sorted(rng.sample(list(eligible), n))
    rows = []
    for i, pos in enumerate(picks):
        variants = name_variants(names.iat[pos])
        # Mostly the primary reading; sometimes an alternative, sometimes spaces removed.
        ko = variants[0] if rng.random() < 0.6 or len(variants) == 1 else rng.choice(variants[1:])
        if rng.random() < 0.3:
            ko = ko.replace(" ", "")
        if noise:
            ko = jamo_noise(ko, rng, noise)
            if rng.random() < 0.1:
                ko += "호"
        rows.append({"sample_id": f"SYN_{i:05d}", "vessel_name": ko, "call_sign": "",
                     "gross_tonnage": float(gt.iat[pos]), "international_gross_tonnage": float(gt.iat[pos]),
                     "flag_name": catalog["flag_name"].iat[pos], "vessel_type": catalog["vessel_type"].iat[pos],
                     "arrival_year": 2024, "target_vessel_id": catalog["vessel_id"].iat[pos]})
    return pd.DataFrame(rows)


def typo(name: str, rng: random.Random, n_edits: int = 1) -> str:
    """Uppercase record-style spelling with transposition / deletion / doubling / substitution."""
    s = list(name.upper())
    letters = [i for i, ch in enumerate(s) if ch.isalpha()]
    for _ in range(n_edits):
        if len(letters) < 4:
            break
        i = rng.choice(letters[1:-1])
        op = rng.random()
        if op < 0.35 and i + 1 < len(s) and s[i + 1].isalpha():
            s[i], s[i + 1] = s[i + 1], s[i]                      # KESTRAVEN -> KETSRAVEN
        elif op < 0.65:
            del s[i]                                             # DORSET -> DORET
        elif op < 0.85:
            s.insert(i, s[i])                                    # VASILIKO -> VASIILKO
        else:
            s[i] = rng.choice("ABCDEFGHIJKLMNOPRSTUVWY")         # substitution
        letters = [j for j, ch in enumerate(s) if ch.isalpha()]
    return "".join(s)


def synthetic_latin_queries(catalog: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Catalog vessels as misspelled Latin records; about half without tonnage."""
    rng = random.Random(seed)
    gt = pd.to_numeric(catalog["gross_tonnage"], errors="coerce")
    names = catalog["vessel_name"].fillna("")
    eligible = np.flatnonzero((gt >= 500).to_numpy() & names.str.fullmatch(r"[A-Za-z .\-']{5,}").to_numpy())
    picks = sorted(rng.sample(list(eligible), n))
    rows = []
    for i, pos in enumerate(picks):
        name = typo(names.iat[pos], rng, n_edits=1 if rng.random() < 0.8 else 2)
        keep_gt = rng.random() < 0.5
        g = float(gt.iat[pos]) if keep_gt else np.nan
        rows.append({"sample_id": f"SYNL_{i:05d}", "vessel_name": name, "call_sign": "",
                     "gross_tonnage": g, "international_gross_tonnage": g,
                     "flag_name": catalog["flag_name"].iat[pos], "vessel_type": catalog["vessel_type"].iat[pos],
                     "arrival_year": 2024, "target_vessel_id": catalog["vessel_id"].iat[pos]})
    return pd.DataFrame(rows)


def synthetic_rename_queries(catalog: pd.DataFrame, n: int, seed: int = 0, min_prefix_count: int = 15) -> pd.DataFrame:
    """Fleet renames: the record keeps the vessel's own name but a different leading word
    (OCEAN FINCH for catalog 'BLUE Finch'); tonnage, flag and type stay the same."""
    rng = random.Random(seed)
    names = catalog["vessel_name"].fillna("").astype(str)
    words = names.str.upper().str.split()
    multi = words.str.len() >= 2
    first = words.str[0]
    counts = first[multi].value_counts()
    prefixes = sorted(counts[counts >= min_prefix_count].index)
    gt = pd.to_numeric(catalog["gross_tonnage"], errors="coerce")
    eligible = np.flatnonzero((multi & first.isin(prefixes) & (gt >= 500)).to_numpy())
    picks = sorted(rng.sample(list(eligible), n))
    rows = []
    for i, pos in enumerate(picks):
        w = words.iat[pos]
        new_first = rng.choice([p for p in prefixes if p != w[0]])
        rows.append({"sample_id": f"SYNR_{i:05d}", "vessel_name": " ".join([new_first] + w[1:]), "call_sign": "",
                     "gross_tonnage": float(gt.iat[pos]), "international_gross_tonnage": float(gt.iat[pos]),
                     "flag_name": catalog["flag_name"].iat[pos], "vessel_type": catalog["vessel_type"].iat[pos],
                     "arrival_year": 2024, "target_vessel_id": catalog["vessel_id"].iat[pos]})
    return pd.DataFrame(rows)


def localize_attributes(synth: pd.DataFrame, train_fold: pd.DataFrame, catalog: pd.DataFrame,
                        seed: int = 0) -> pd.DataFrame:
    """Rewrite English catalog flag/type into the Korean record vocabulary seen in the train fold."""
    cat = catalog.set_index("vessel_id")
    joined = train_fold.join(cat[["flag_name", "vessel_type"]], on="target_vessel_id", rsuffix="_cat")
    flag_map = {k: Counter(g.dropna()).most_common(1)[0][0]
                for k, g in joined.groupby("flag_name_cat")["flag_name"] if g.notna().any()}
    type_map = {k: Counter(g.dropna()).most_common(1)[0][0]
                for k, g in joined.groupby("vessel_type_cat")["vessel_type"] if g.notna().any()}
    rng = np.random.default_rng(seed)
    out = synth.copy()
    out["flag_name"] = [flag_map.get(f) if rng.random() > 0.16 else None for f in out["flag_name"]]
    out["vessel_type"] = [type_map.get(t) for t in out["vessel_type"]]
    return out


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------
def build_groups(queries: pd.DataFrame, pairs: pd.DataFrame, catalog: pd.DataFrame,
                 seed: int = 0, k: int | None = SELECT_K,
                 latin_k: int | None = None, hangul_extra: bool = False) -> list[tuple[str, list[str], int]]:
    """(query text, candidate texts, index of the positive) per row whose target is retrieved.
    ``hangul_extra``: see select_pairs (default False = unchanged)."""
    rng = random.Random(seed)
    pos_of = {v: i for i, v in enumerate(catalog["vessel_id"].astype(str))}
    cat_records = catalog.to_dict("records")
    rows = queries.to_dict("records")
    groups = []
    sel = select_pairs(pairs, queries, k, latin_k, hangul_extra=hangul_extra)
    for q, grp in sel.groupby("q", sort=True):
        row = rows[q]
        target = pos_of[str(row["target_vessel_id"])]
        cands = grp["c"].tolist()
        if target not in cands:
            continue
        negs = [c for c in cands if c != target]
        rng.shuffle(negs)
        size = GROUP_SIZE if is_hangul_name(row.get("vessel_name")) else LATIN_GROUP_SIZE
        group = [target] + negs[: size - 1]
        rng.shuffle(group)
        groups.append((query_text(row), [candidate_text(cat_records[c]) for c in group], group.index(target)))
    return groups


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _resolve(repo: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_files_only=True)


class FinetunedScorer:
    """PairScorer plugin: adds ``ft_score`` for the selected Hangul pairs."""

    columns = ("ft_score", "ft_rank")

    def __init__(self, backbone: str, path: str | None = None, device: str | None = None,
                 batch_groups: int | None = None, grad_ckpt: bool = True, lr: float | None = None,
                 score_k: int | None = SELECT_K, latin_k: int | None = None,
                 prefix_min_B: int | None = None, pad_multiple: int | None = None, hangul_extra: bool = False):
        """``prefix_min_B`` / ``pad_multiple`` (inference speed, S8; both None = fixed 128 padding, bitwise the
        production path): rows with >= prefix_min_B scored pairs run the shared-prefix forward
        (_prefix_scores); other rows pad their batch to the row's own longest text rounded up to pad_multiple.
        Both are row-local: shapes come from the row's own texts only.
        ``hangul_extra`` (S9 item 5, default False = production): score() also scores the src_extra candidates
        of Hangul rows (select_pairs hangul_extra). Row-local."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.backbone = backbone
        self.score_k = score_k  # None = score every candidate of a Hangul row
        self.latin_k = latin_k  # None = Latin rows are not scored
        self.prefix_min_B, self.pad_multiple = prefix_min_B, pad_multiple
        self.hangul_extra = bool(hangul_extra)
        self.cfg = dict(BACKBONES[backbone])
        if batch_groups:
            self.cfg["batch_groups"] = batch_groups
        if lr:
            self.cfg["lr"] = lr
        self.name = f"ft_{backbone}"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        src = path or _resolve(self.cfg["repo"])
        self.tok = AutoTokenizer.from_pretrained(src)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if (self.cfg["lora"] and self.device == "cuda") else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=1, dtype=dtype)
        model.config.pad_token_id = self.tok.pad_token_id
        if hasattr(model.config, "text_config"):  # Qwen3.5: composite config
            model.config.text_config.pad_token_id = self.tok.pad_token_id
        if self.cfg["lora"] and path is None:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(
                task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=self.cfg.get("lora_targets", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                modules_to_save=["score"]))
        if self.cfg["lora"] and path is None and grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        self.model = model.to(self.device)

    def _encode(self, left: list[str], right: list[str], pad_multiple: int | None = None):
        if self.cfg["pair_input"]:
            enc = self.tok(left, right, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        else:
            texts = [f"record: {a}\ncandidate: {b}\nsame vessel?" for a, b in zip(left, right)]
            if pad_multiple:  # right padding to the batch's own longest text (cap 128); real tokens keep positions
                enc = self.tok(texts, padding="longest", truncation=True, max_length=DECODER_MAX_LEN,
                               pad_to_multiple_of=pad_multiple, return_tensors="pt")
                enc = {k: v[:, :DECODER_MAX_LEN] for k, v in enc.items()}
            else:
                enc = self.tok(texts, padding="max_length", truncation=True, max_length=DECODER_MAX_LEN,
                               return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def _prefix_scores(self, left: str, rights: list[str]) -> np.ndarray | None:
        """Shared-prefix forward of one row (port of experiments/diag/speed/scorer519.Scorer.forward_prefix).

        Every text is "record: <left>\ncandidate:" + " <right>\nsame vessel?". The prefix runs once (batch 1,
        use_cache); the cache (conv state, gated-delta state, full-attention KV) is copied to the B candidates
        and only the B suffixes run, right-padded to their longest rounded up to pad_multiple. The score is the
        head at the last non-pad token, as the sequence-classification head pools. Same tokens and positions as
        the full text, so exact in real arithmetic (bf16 kernel-shape noise only). Returns None (the caller then
        uses the full path) when prefix tokens + suffix tokens != full-text tokens for any text of the row, or a
        full text exceeds 128 tokens.
        """
        torch = self.torch
        prefix = f"record: {left}\ncandidate:"
        suffixes = [f" {r}\nsame vessel?" for r in rights]
        p_ids = self.tok(prefix)["input_ids"]
        s_ids = self.tok(suffixes)["input_ids"]
        full = self.tok([prefix + x for x in suffixes])["input_ids"]
        if not p_ids or any(not s or p_ids + s != f or len(f) > DECODER_MAX_LEN for s, f in zip(s_ids, full)):
            return None
        pad_id, m = self.tok.pad_token_id, self.pad_multiple or 1
        B, Ls = len(s_ids), -(-max(len(s) for s in s_ids) // m) * m
        ids = torch.full((B, Ls), pad_id, dtype=torch.long)
        am = torch.zeros((B, Ls), dtype=torch.long)
        for i, s in enumerate(s_ids):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            am[i, : len(s)] = 1
        dev = self.device
        ids, am = ids.to(dev), am.to(dev)
        base = getattr(self.model, self.model.base_model_prefix)
        lm = getattr(base, "language_model", base)  # Qwen3.5: the text model inside the composite model
        p = torch.tensor([p_ids], dtype=torch.long, device=dev)
        cache = lm(input_ids=p, use_cache=True).past_key_values
        cache.reorder_cache(torch.zeros(B, dtype=torch.long, device=dev))
        am = torch.cat([torch.ones(B, p.shape[1], dtype=am.dtype, device=dev), am], dim=1)
        h = lm(input_ids=ids, attention_mask=am, past_key_values=cache, use_cache=True).last_hidden_state
        logits = self.model.score(h)
        non_pad = (ids != pad_id).to(torch.int32)
        last = (torch.arange(Ls, device=dev, dtype=torch.int32) * non_pad).argmax(-1)
        return logits[torch.arange(B, device=dev), last].squeeze(-1).float().cpu().numpy()

    def fit(self, groups, epochs: int = 2, seed: int = 0, log_every: int = 200) -> dict:
        torch = self.torch
        torch.manual_seed(seed)
        rng = random.Random(seed)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.cfg["lr"], weight_decay=0.01)
        bg = self.cfg["batch_groups"]
        total = epochs * math.ceil(len(groups) / bg)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(1, total // 20)) * max(0.0, 1 - s / total))
        use_amp = self.device == "cuda" and not self.cfg["lora"]
        scaler = torch.amp.GradScaler(enabled=use_amp)
        self.model.train()
        step, losses = 0, []
        for _ in range(epochs):
            order = list(range(len(groups)))
            rng.shuffle(order)
            for start in range(0, len(order), bg):
                batch = [groups[i] for i in order[start: start + bg]]
                left = [g[0] for g in batch for _ in g[1]]
                right = [c for g in batch for c in g[1]]
                with torch.autocast(self.device, dtype=torch.float16, enabled=use_amp):
                    logits = self.model(**self._encode(left, right)).logits.squeeze(-1).float()
                loss, offset = 0.0, 0
                for g in batch:
                    seg = logits[offset: offset + len(g[1])]
                    loss = loss + torch.nn.functional.cross_entropy(seg[None], torch.tensor([g[2]], device=self.device))
                    offset += len(g[1])
                loss = loss / len(batch)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(float(loss.detach()))
                step += 1
                if step % log_every == 0:
                    print(f"    step {step}/{total} loss {np.mean(losses[-log_every:]):.4f}", flush=True)
        self.model.eval()
        return {"steps": step, "mean_loss": float(np.mean(losses)) if losses else float("nan")}

    def score(self, queries: pd.DataFrame, pairs: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
        torch = self.torch
        out = pd.DataFrame(np.nan, index=pairs.index, columns=list(self.columns))
        cat_records = catalog.to_dict("records")
        rows = queries.to_dict("records")
        self.model.eval()
        min_b, pad = getattr(self, "prefix_min_B", None), getattr(self, "pad_multiple", None)
        use_prefix = bool(min_b) and not self.cfg["pair_input"]
        n_prefix = n_fallback = n_full = 0
        with torch.inference_mode():
            for q, grp in select_pairs(pairs, queries, self.score_k, self.latin_k,
                                       hangul_extra=getattr(self, "hangul_extra", False)).groupby("q", sort=True):
                left = [query_text(rows[q])] * len(grp)
                right = [candidate_text(cat_records[c]) for c in grp["c"]]
                s = None
                if use_prefix and len(grp) >= min_b:
                    s = self._prefix_scores(left[0], right)
                    n_prefix += s is not None
                    n_fallback += s is None
                if s is None:
                    s = self.model(**self._encode(left, right, pad)).logits.squeeze(-1).float().cpu().numpy()
                    n_full += 1
                out.loc[grp.index, "ft_score"] = s
                out.loc[grp.index, "ft_rank"] = np.argsort(np.argsort(-s, kind="stable"))
        if use_prefix or pad:  # counts only (no per-row values)
            print(f"LLM rows: shared-prefix {n_prefix}, full {n_full} (prefix split fallback {n_fallback}); "
                  f"prefix_min_B={min_b} pad_multiple={pad}")
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model = self.model.merge_and_unload() if self.cfg["lora"] else self.model
        model.save_pretrained(path)
        self.tok.save_pretrained(path)
