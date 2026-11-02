"""S9 charft10 char-level dual encoder (Hangul record name -> catalog name) for CandidateGenerator.hangul_de_k.

Port of experiments/s9_robust/charft10/de10.py (Enc, qtext, Vocab.enc; RESULT_10.md section 3, v2 model):
one shared character transformer (4 layers, d 256, 4 heads, ff 768, pre-norm, masked mean pool, linear -> 128, L2
norm; a tower embedding marks query vs document). The query is the NFKC upper-case record name decomposed to jamo
(translit.to_jamo), at most 64 characters; the catalog side is shipped precomputed (float16, catalog order).

Inference is CPU fp32 with ``threads`` torch threads (default 1), ONE query per forward (a batch of one has no
padding, so a row's embedding never depends on other rows). A model is a plain "payload" dict (numpy weights, the
character list, the catalog embeddings), so artifacts load without torch; torch is imported on the first encode.

``DualEncoder(models, route)``: ``models`` {gid: payload}. Inference: one model (route None). Fit time (charft10 leak
control): the 5 group-fold encoders, ``route`` = {vessel_id: group}; a row is encoded by the model of its TARGET
vessel's group, which never saw that vessel (fit_time_encoders). Encodings are cached per (gid, query text);
``prepare(rows)`` fills the cache in the calling process (the parent, before forking candidate workers, so the
workers never run torch).

S9 kode15 "v2-tok" (experiments/s9_robust/kode15/RESULT_15.md; CandidateGenerator.hangul_de2_k): the same network
with 3 side-token characters before the jamo query, [TON][FLAG][TYPE] (side_tokens). A payload with a "side" entry
(plain lists/dicts: the model's flag / type vocabularies and the flag localisation map of its fold) gets these
tokens on every query; payloads without it (v2) are unchanged.

S9 latde13 Latin encoder (experiments/s9_robust/latde13/RESULT_13.md; CandidateGenerator.latin_de_k): the same network
on Latin record names. Its payload has kind "latin" (query text qtext_latin: NFKC upper-case, whitespace collapsed, at
most 48 characters, no jamo); such an encoder serves and prepares only non-Hangul rows.
"""

from __future__ import annotations

import unicodedata

import numpy as np

QLEN = 64
ARCH = {"d": 256, "layers": 4, "heads": 4, "ff": 768, "out": 128}


def qtext(name) -> str:
    """Query text of a record name (charft10 de10.qtext)."""
    from translit import to_jamo
    return to_jamo(unicodedata.normalize("NFKC", str(name)).upper().strip())


def qtext_latin(name) -> str:
    """Query text of a Latin record name (latde13 de13.qtext)."""
    import re
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(name)).upper()).strip()


def _net(n_vocab: int):
    """charft10 de10.Enc (same module names, so its state dict loads)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Enc(nn.Module):
        def __init__(self, V, d=256, layers=4, heads=4, ff=768, out=128, L=QLEN):
            super().__init__()
            self.emb = nn.Embedding(V, d, padding_idx=0)
            self.pos = nn.Embedding(L, d)
            self.tower = nn.Embedding(2, d)
            lay = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(lay, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.proj = nn.Linear(d, out)
            self.logit_scale = nn.Parameter(torch.tensor(np.log(20.0), dtype=torch.float32))

        def forward(self, x, tower):
            m = x != 0
            m[:, 0] |= ~m.any(1)  # empty string guard
            h = self.emb(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None] + self.tower.weight[tower]
            h = self.tf(h, src_key_padding_mask=~m)
            h = self.norm(h)
            h = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
            return F.normalize(self.proj(h).float(), dim=-1)

    return Enc(n_vocab, **ARCH)


def catalog_crc(catalog) -> int:
    import zlib
    return zlib.crc32("\n".join(catalog["vessel_id"].astype(str)).encode())


def make_payload(state: dict, itos: list[str], cat_emb: np.ndarray, catalog, name: str = "", qlen: int = QLEN,
                 kind: str = "hangul") -> dict:
    """Shippable model: float32 numpy weights, the character list (index = id), float16 catalog embeddings."""
    state = {k: np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=np.float32).copy()
             for k, v in state.items()}
    cat_emb = np.ascontiguousarray(cat_emb, dtype=np.float16)
    if len(cat_emb) != len(catalog):
        raise ValueError(f"catalog embeddings {cat_emb.shape} vs catalog {len(catalog)} rows")
    p = {"state": state, "itos": list(itos), "cat": cat_emb, "qlen": int(qlen), "arch": dict(ARCH),
         "n_catalog": len(catalog), "catalog_crc": catalog_crc(catalog), "name": name}
    if kind != "hangul":  # hangul payloads stay key-for-key as before
        p["kind"] = kind
    return p


class DualEncoder:
    def __init__(self, models: dict[int, dict], route: dict[str, int] | None = None, threads: int = 1) -> None:
        if not models:
            raise ValueError("no dual-encoder model")
        if route is None and len(models) != 1:
            raise ValueError("several models need a route (vessel_id -> model)")
        self.models = dict(models)
        self.route = route
        self.threads = int(threads)
        self._q: dict[tuple[int, str], np.ndarray] = {}
        self._nets: dict = {}
        self._cat32: dict[int, np.ndarray] = {}
        self._stoi: dict[int, dict[str, int]] = {}
        # fit time only (kode15 frames15.tok_router): (vessel_name, target_vessel_id, gross_tonnage) -> attributes
        # replacing the row's flag/type in the side tokens (localised synthetic rows); never set at inference
        self.override: dict[tuple, dict] = {}

    def __getstate__(self):  # torch modules / float32 catalog are rebuilt on use (spawn workers)
        st = dict(self.__dict__)
        st["_nets"], st["_cat32"] = {}, {}
        return st

    def check_catalog(self, catalog) -> None:
        crc = catalog_crc(catalog)
        for g, p in self.models.items():
            if p["n_catalog"] != len(catalog) or p["catalog_crc"] != crc:
                raise ValueError(f"dual encoder {g}: catalog embeddings were built for another vessel catalog")

    def gid(self, row: dict) -> int:
        if self.route is None:
            return next(iter(self.models))
        return self.route[str(row["target_vessel_id"])]

    def cat32(self, g: int) -> np.ndarray:
        if g not in self._cat32:
            self._cat32[g] = self.models[g]["cat"].astype(np.float32)
        return self._cat32[g]

    def _encode(self, g: int, text: str) -> np.ndarray:
        import torch
        net = self._nets.get(g)
        p = self.models[g]
        if net is None:
            net = _net(len(p["itos"]))
            net.load_state_dict({k: torch.from_numpy(v) for k, v in p["state"].items()})
            net.eval()
            self._nets[g] = net
        stoi = self._stoi.get(g)
        if stoi is None:
            stoi = self._stoi[g] = {c: i for i, c in enumerate(p["itos"])}
        ids = [stoi.get(c, 1) for c in text[:p["qlen"]]] or [0]
        old = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        try:
            with torch.no_grad():
                return net(torch.tensor([ids], dtype=torch.long), 0)[0].numpy().astype(np.float32)
        finally:
            torch.set_num_threads(old)

    def text(self, row: dict, g: int) -> str:
        """Query text: qtext(name), after the side tokens of model g's fold map when its payload has "side"."""
        if self.models[g].get("kind") == "latin":  # Latin; with "side": the hs22 train24 Latin v2-tok tokens first
            side = self.models[g].get("side")
            return (side_tokens(row, side) if side else "") + qtext_latin(row.get("vessel_name"))
        side = self.models[g].get("side")
        if side is None:
            return qtext(row.get("vessel_name"))
        ov = getattr(self, "override", None)
        if ov:
            key = (row.get("vessel_name"), str(row.get("target_vessel_id")), row.get("gross_tonnage"))
            if key in ov:
                row = {**row, **ov[key]}
        return side_tokens(row, side) + qtext(row.get("vessel_name"))

    def query(self, row: dict) -> tuple[np.ndarray, np.ndarray]:
        """(query embedding float32 (128,), float32 catalog embeddings of the row's model)."""
        g = self.gid(row)
        key = (g, self.text(row, g))
        q = self._q.get(key)
        if q is None:
            q = self._q[key] = self._encode(g, key[1])
        return q, self.cat32(g)

    @property
    def latin(self) -> bool:
        return next(iter(self.models.values())).get("kind") == "latin"

    def prepare(self, rows) -> int:
        """Encode every row of ``rows`` (dicts) of this encoder's language (Hangul, or non-Hangul for a Latin
        encoder) not cached yet, in this process; returns the count."""
        from common import is_hangul_name
        n = 0
        latin = self.latin
        for row in rows:
            if is_hangul_name(row.get("vessel_name")) != latin:
                g = self.gid(row)
                key = (g, self.text(row, g))
                if key not in self._q:
                    self._q[key] = self._encode(g, key[1])
                    n += 1
        return n

    def embeddings(self, rows) -> dict:
        """prepare(rows), then {cache key: embedding} of these rows (the split build ships them to the workers)."""
        from common import is_hangul_name
        self.prepare(rows)
        out = {}
        for row in rows:
            if is_hangul_name(row.get("vessel_name")) != self.latin:
                g = self.gid(row)
                key = (g, self.text(row, g))
                out[key] = self._q[key]
        return out


def from_payload(payload: dict) -> DualEncoder:
    return DualEncoder({-1: payload})


# ---------------------------------------------------------------------------------------- v2-tok side tokens
# kode15 de15.py: TON = gross-tonnage bucket clip(round(2 log2 t), 10, 40) (record: international_gross_tonnage, else
# gross_tonnage); FLAG = English catalog flag (a record flag through the fold's localisation map); TYPE = one
# vocabulary over every type string (the query carries the raw record type). Private-use characters.
TON0, TON_NONE = 0xE000, chr(0xE001)
FLAG0, FLAG_NONE, FLAG_UNK = 0xE100, chr(0xE0FF), chr(0xE0FE)
TYPE0, TYPE_NONE, TYPE_UNK = 0xE400, chr(0xE3FF), chr(0xE3FE)


def _isna(x) -> bool:
    import math
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return isinstance(x, str) and not x.strip()


def ton_bucket(t):
    import math
    try:
        t = float(t)
    except (TypeError, ValueError):
        return None
    if math.isnan(t) or not t > 0:
        return None
    return int(min(40, max(10, round(2 * math.log2(t)))))


def side_tokens(row: dict, side: dict) -> str:
    """[TON][FLAG][TYPE] of a record row; ``side`` = payload["side"] (make_side)."""
    t = row.get("international_gross_tonnage")
    b = None if _isna(t) else ton_bucket(t)
    if b is None:
        g = row.get("gross_tonnage")
        b = None if _isna(g) else ton_bucket(g)
    ton = TON_NONE if b is None else chr(TON0 + b)
    f = row.get("flag_name")
    if _isna(f):
        flag = FLAG_NONE
    else:
        e = side["fmap"].get(str(f).strip())
        flag = FLAG_UNK if e is None or e not in side["fi"] else chr(FLAG0 + side["fi"][e])
    y = row.get("vessel_type")
    if _isna(y):
        typ = TYPE_NONE
    else:
        i = side["ti"].get(str(y).strip())
        typ = TYPE_UNK if i is None else chr(TYPE0 + i)
    return ton + flag + typ


def flag_map(train, catalog, flags: list[str], keep) -> dict[str, str]:
    """kode15 Side.fit_maps flag map: record flag -> the most common catalog flag of the target among the ``keep``
    train rows (ties: first seen), then every catalog flag -> itself."""
    from collections import Counter
    cat_flag = dict(zip(catalog["vessel_id"].astype(str), catalog["flag_name"].astype(str)))
    tflag = [cat_flag[v] for v in train["target_vessel_id"].astype(str)]
    rflag = train["flag_name"].astype(object).where(train["flag_name"].notna(), None).tolist()
    pairs = Counter((a, b) for a, b, k in zip(rflag, tflag, keep) if k and not _isna(a))
    fm: dict[str, str] = {}
    for (rf, cf), _ in pairs.most_common():
        fm.setdefault(str(rf).strip(), cf)
    for f in flags:
        fm.setdefault(f, f)
    return fm


def make_side(flags: list[str], types: list[str], fmap: dict[str, str]) -> dict:
    """payload["side"]: the model's flag / type vocabularies (index = token), their lookups and the flag map."""
    return {"flags": list(flags), "types": list(types), "fmap": dict(fmap),
            "fi": {f: i for i, f in enumerate(flags)}, "ti": {t: i for i, t in enumerate(types)}}


# ---------------------------------------------------------------------------------------------- fit time only
def _load_plain(path):
    """A pickled charft10 de10.Vocab / kode15 de15.Side -> a plain object with its attributes (no experiment module
    imported)."""
    import pickle

    class _Plain:
        pass

    class _U(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) in (("de10", "Vocab"), ("de15", "Side")):
                return _Plain
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return _U(f).load()


def _load_itos(path) -> list[str]:
    """charft10 cache/vocab10.pkl (a pickled de10.Vocab) -> its itos list, without importing de10."""
    return list(_load_plain(path).itos)


def vessel_groups(train, catalog) -> np.ndarray:
    """charft10 prep10 groups per catalog position: train vessels keep their group-CV fold (cv.iter_splits(train,
    "group")), every other vessel sha1(vessel_id) % 5."""
    import hashlib
    from cv import iter_splits
    vids = catalog["vessel_id"].astype(str).tolist()
    pos_of = {v: i for i, v in enumerate(vids)}
    group = np.array([int(hashlib.sha1(v.encode()).hexdigest(), 16) % 5 for v in vids], dtype=np.int8)
    for k, (_, va) in enumerate(iter_splits(train, "group")):
        for v in train["target_vessel_id"].iloc[va].astype(str).unique():
            group[pos_of[v]] = k
    return group


def fit_time_encoders(train, catalog, de_dir, suffix: str = "_v2") -> tuple[DualEncoder, dict]:
    """(fold router for the ranker-training rows, full-model payload to ship). Reads the charft10 files:
    models/de_f{0..4,-1}<suffix>.pt, models/cat_f{..}<suffix>.npy, cache/vocab10.pkl."""
    from pathlib import Path

    import torch
    de_dir = Path(de_dir)
    itos = _load_itos(de_dir / "cache" / "vocab10.pkl")

    def load(j: int) -> dict:
        state = torch.load(de_dir / "models" / f"de_f{j}{suffix}.pt", map_location="cpu")
        return make_payload(state, itos, np.load(de_dir / "models" / f"cat_f{j}{suffix}.npy"), catalog,
                            name=f"de_f{j}{suffix}")
    group = vessel_groups(train, catalog)
    route = dict(zip(catalog["vessel_id"].astype(str), group.astype(int).tolist()))
    return DualEncoder({j: load(j) for j in range(5)}, route=route), load(-1)


def fit_time_latin_encoders(train, catalog, de_dir, tok_side=None) -> tuple[DualEncoder, dict]:
    """latde13 counterpart of fit_time_encoders for the Latin encoder: (fold router, full-model payload to ship).
    Reads models/lde_f{0..4,-1}.pt, models/lcat_f{..}.npy, cache/vocab13.pkl; query length 48, kind "latin".
    ``tok_side`` (path of kode15 cache/side15.pkl): the hs22 Latin v2-tok encoder instead (models/lde_f*_tok.pt,
    lcat_f*_tok.npy, cache/vocab24.pkl, query length 51 = 3 side tokens + 48); each payload gets the side vocabularies
    and its fold's flag map (train rows whose target is not in the fold's group; the full model: every row)."""
    from pathlib import Path

    import torch
    de_dir = Path(de_dir)
    suf = "_tok" if tok_side else ""
    itos = _load_itos(de_dir / "cache" / ("vocab24.pkl" if tok_side else "vocab13.pkl"))
    group = vessel_groups(train, catalog)
    if tok_side:
        sd = _load_plain(tok_side)
        flags, types = list(sd.flags), list(sd.types)
        pos_of = {v: i for i, v in enumerate(catalog["vessel_id"].astype(str))}
        tg = group[train["target_vessel_id"].astype(str).map(pos_of).to_numpy()]

    def load(j: int) -> dict:
        state = torch.load(de_dir / "models" / f"lde_f{j}{suf}.pt", map_location="cpu")
        p = make_payload(state, itos, np.load(de_dir / "models" / f"lcat_f{j}{suf}.npy"), catalog,
                         name=f"lde_f{j}{suf}", qlen=51 if tok_side else 48, kind="latin")
        if tok_side:
            keep = (tg != j) if j >= 0 else np.ones(len(train), bool)
            p["side"] = make_side(flags, types, flag_map(train, catalog, flags, keep))
        return p
    route = dict(zip(catalog["vessel_id"].astype(str), group.astype(int).tolist()))
    return DualEncoder({j: load(j) for j in range(5)}, route=route), load(-1)


def fit_time_tok_encoders(train, catalog, de_dir, suffix: str = "_tok", synth=None) -> tuple[DualEncoder, dict]:
    """v2-tok (kode15) counterpart of fit_time_encoders: (fold router, full-model payload to ship). Reads
    models/de_f{0..4,-1}<suffix>.pt, models/cat_f{..}<suffix>.npy, cache/vocab15.pkl and the flag / type vocabularies
    of cache/side15.pkl (the models' token ids). Flag maps are built here from ``train``: fold j from the rows whose
    target is not in group j (the ranker-training rows of group j get fold j's model), the shipped full model from
    every train row. ``synth``: the ranker's synthetic Hangul rows (English attributes); their side tokens use
    localize_attributes(synth, train, catalog, seed=1), the attributes the ranker sees (kode15 frames15.tok_router)."""
    from pathlib import Path

    import torch
    de_dir = Path(de_dir)
    itos = _load_itos(de_dir / "cache" / "vocab15.pkl")
    sd = _load_plain(de_dir / "cache" / "side15.pkl")
    flags, types = list(sd.flags), list(sd.types)
    group = vessel_groups(train, catalog)
    pos_of = {v: i for i, v in enumerate(catalog["vessel_id"].astype(str))}
    tg = group[train["target_vessel_id"].astype(str).map(pos_of).to_numpy()]

    def load(j: int) -> dict:
        state = torch.load(de_dir / "models" / f"de_f{j}{suffix}.pt", map_location="cpu")
        p = make_payload(state, itos, np.load(de_dir / "models" / f"cat_f{j}{suffix}.npy"), catalog,
                         name=f"de_f{j}{suffix}")
        keep = (tg != j) if j >= 0 else np.ones(len(train), bool)
        p["side"] = make_side(flags, types, flag_map(train, catalog, flags, keep))
        return p
    route = dict(zip(catalog["vessel_id"].astype(str), group.astype(int).tolist()))
    router = DualEncoder({j: load(j) for j in range(5)}, route=route)
    if synth is not None and len(synth):
        from rank.finetune import localize_attributes
        lk = localize_attributes(synth, train, catalog, seed=1)
        router.override = {(n, str(v), g): {"flag_name": f, "vessel_type": t} for n, v, g, f, t in
                           zip(synth.vessel_name, synth.target_vessel_id, synth.gross_tonnage, lk.flag_name,
                               lk.vessel_type)}
    return router, load(-1)
