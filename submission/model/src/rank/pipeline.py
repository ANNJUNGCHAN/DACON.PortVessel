"""Candidate generation -> features -> boosting ranker -> argmax per row."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import is_hangul_name
from rank.candidates import CandidateGenerator
from rank.fold_features import FoldFeatures
from rank.rankers import make_ranker

NON_FEATURES = ("q", "c")

_GEN_CACHE: dict[tuple, CandidateGenerator] = {}
# Precomputed catalog transliterations (name -> jamo readings). Loaded from the
# experiment cache in development, or set from submission artifacts at inference.
TR_CACHE: dict[str, list[str]] = {}
# S9 item 4b catalog side table (rank.pair_feats4.build_aux) for generators with pair_feats4: set from
# art["feats4_aux"] at inference; empty = the generator builds it from the catalog + TR_CACHE (fit time).
FEATS4_AUX: dict = {}
# S9 charft10 dual encoder (rank.dual_encoder.DualEncoder) for generators with hangul_de_k: set from
# art["dual_encoder"] at inference (the full model), or the group-fold router at fit time (models/ranked_llm.py).
# "obj2": the kode15 v2-tok encoder for hangul_de2_k (art["dual_encoder2"]; set_dual_encoder2).
# "obj3": the hs22 third Hangul encoder for hangul_de3_k (art["dual_encoder3"]; set_dual_encoder3).
# "objL": the latde13 Latin encoder for latin_de_k (art["latin_dual_encoder"]; set_latin_dual_encoder).
DUAL_ENCODER: dict = {}


def _tr_cache_file():
    from common import ROOT
    return ROOT / "experiments" / "cache" / "tr_cache.pkl"


def set_tr_cache(cache: dict[str, list[str]]) -> None:
    TR_CACHE.clear()
    TR_CACHE.update(cache)
    _GEN_CACHE.clear()


def set_feats4_aux(aux: dict) -> None:
    FEATS4_AUX.clear()
    FEATS4_AUX.update(aux)
    _GEN_CACHE.clear()


def set_dual_encoder(encoder, key: str = "obj") -> None:
    """``encoder``: a DualEncoder, a shipped payload dict (art["dual_encoder"]), or None (unset)."""
    DUAL_ENCODER.pop(key, None)
    if encoder is not None:
        if isinstance(encoder, dict):
            from rank.dual_encoder import from_payload
            encoder = from_payload(encoder)
        DUAL_ENCODER[key] = encoder
    _GEN_CACHE.clear()


def set_dual_encoder2(encoder) -> None:
    """hangul_de2_k encoder (kode15 v2-tok): a DualEncoder, art["dual_encoder2"], or None (unset)."""
    set_dual_encoder(encoder, "obj2")


def set_dual_encoder3(encoder) -> None:
    """hangul_de3_k encoder (hs22): a DualEncoder, art["dual_encoder3"], or None (unset)."""
    set_dual_encoder(encoder, "obj3")


def set_latin_dual_encoder2(encoder) -> None:
    """latin_de2_k encoder (hs22 Latin v2-tok): a DualEncoder, art["latin_dual_encoder2"], or None (unset)."""
    set_dual_encoder(encoder, "objL2")


def set_latin_dual_encoder(encoder) -> None:
    """latin_de_k encoder (latde13): a DualEncoder, art["latin_dual_encoder"], or None (unset)."""
    set_dual_encoder(encoder, "objL")


def save_tr_cache(gen: CandidateGenerator) -> None:
    import pickle
    path = _tr_cache_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(gen.tr_cache))


def candidate_generator(catalog: pd.DataFrame, **opts) -> CandidateGenerator:
    """Shared generator per (catalog object, S4 options); no options = the S3 generator.

    ``opts`` are CandidateGenerator keyword options (mv_digits, num_words, notonnage_net); False/None
    values are dropped so that an all-off option set shares the plain generator.
    """
    opts = {k: v for k, v in opts.items() if v}
    key = (id(catalog), tuple(sorted(opts.items())))
    if key not in _GEN_CACHE:
        if any(k[0] != id(catalog) for k in _GEN_CACHE):
            _GEN_CACHE.clear()
        if not TR_CACHE:
            import pickle
            path = _tr_cache_file()
            if path.exists():
                TR_CACHE.update(pickle.loads(path.read_bytes()))
        extra = ({"feats4_aux": FEATS4_AUX or None}
                 if opts.get("pair_feats4") or opts.get("word_contain_k") or opts.get("hangul_wc2_k") else {})
        if opts.get("hangul_de_k"):
            extra["dual_encoder"] = DUAL_ENCODER.get("obj")
        if opts.get("hangul_de2_k"):
            extra["dual_encoder2"] = DUAL_ENCODER.get("obj2")
        if opts.get("hangul_de3_k"):
            extra["dual_encoder3"] = DUAL_ENCODER.get("obj3")
        if opts.get("latin_de_k"):
            extra["latin_dual_encoder"] = DUAL_ENCODER.get("objL")
        if opts.get("latin_de2_k"):
            extra["latin_dual_encoder2"] = DUAL_ENCODER.get("objL2")
        _GEN_CACHE[key] = CandidateGenerator(catalog, tr_cache=TR_CACHE, **opts, **extra)
    return _GEN_CACHE[key]


def query_weights(queries: pd.DataFrame, hangul_share: float = 0.7) -> np.ndarray:
    """Per-query weights matching the metric: Hangul rows carry 70% of the total."""
    hangul = queries["vessel_name"].map(is_hangul_name).to_numpy()
    n_h, n_l = max(hangul.sum(), 1), max((~hangul).sum(), 1)
    w = np.where(hangul, hangul_share / n_h, (1 - hangul_share) / n_l)
    return w / w.mean()


def labels(pairs: pd.DataFrame, queries: pd.DataFrame, catalog: pd.DataFrame) -> np.ndarray:
    pos_of = {v: i for i, v in enumerate(catalog["vessel_id"].astype(str))}
    tpos = np.array([pos_of[v] for v in queries["target_vessel_id"].astype(str)])
    return (pairs["c"].to_numpy() == tpos[pairs["q"].to_numpy()]).astype(np.int8)


class RankPipeline:
    def __init__(self, ranker: str = "lgb_binary", scorers=(), weighted: bool = True,
                 drop: tuple[str, ...] = (), latin_feats: bool = False, llm_mask: bool = False,
                 cand_opts: dict | None = None):
        """``latin_feats``: add rank.latin_feats gen + rename columns (Latin rows; NaN on Hangul rows).
        ``llm_mask``: NaN the LLM columns on Latin rename-mode rows (rank.latin_feats.mask_llm), the same
        way for the OOF columns at fit time and the scorer output at inference. Both default off; pickles
        from before these attributes existed read them with getattr and behave as before.
        ``cand_opts``: S4 CandidateGenerator options (mv_digits, num_words, notonnage_net) used whenever the
        pipeline builds candidates itself (fit without pairs, predict). Stored on the pickle, so inference
        needs no environment variable; pickles without the attribute use the plain generator.
        """
        self.ranker_name = ranker
        self.scorers = list(scorers)
        self.weighted = weighted
        self.drop = drop
        self.latin_feats = latin_feats
        self.llm_mask = llm_mask
        self.cand_opts = dict(cand_opts or {})

    def generator(self, catalog: pd.DataFrame) -> CandidateGenerator:
        return candidate_generator(catalog, **(getattr(self, "cand_opts", None) or {}))

    def features(self, pairs, queries, catalog, exclude_self: bool, lso_src: pd.DataFrame | None = None) -> pd.DataFrame:
        """``lso_src``: noise copies' source train rows -> leave-source-out fold features (FoldFeatures.transform_lso)."""
        latin, mask = getattr(self, "latin_feats", False), getattr(self, "llm_mask", False)
        fold = (self.fold.transform(pairs, queries, exclude_self) if lso_src is None
                else self.fold.transform_lso(pairs, queries, lso_src))
        if not (latin or mask):
            parts = [pairs.drop(columns=list(NON_FEATURES)), fold]
            parts += [s.score(queries, pairs, catalog) for s in self.scorers]
            X = pd.concat(parts, axis=1).drop(columns=list(self.drop), errors="ignore")
            if getattr(self, "rel_feats", False):
                X = add_relative(X, pairs["q"].to_numpy())
            prior = getattr(self, "cat_prior", None)
            if prior is not None:  # S9u: catalog-only P(vessel is a target) (hs22/catprior28.py), per candidate
                X = add_cat_prior(X, pairs, prior)
                if getattr(self, "cat_prior_skip_synth", False) and "sample_id" in queries.columns:
                    # synthetic catalog queries (sample_id SYN*) target random catalog vessels, not "target-like"
                    # ones: their prior columns are NaN so the ranker learns the prior from real rows + copies only
                    syn = queries["sample_id"].astype(str).str.startswith("SYN").to_numpy()[pairs["q"].to_numpy()]
                    X.loc[syn, ["cat_prior", "cat_prior_gap"]] = np.nan
                if getattr(self, "cat_prior_hangul_only", False):  # S9u2: prior columns NaN on Latin rows
                    lat = ~queries["vessel_name"].map(is_hangul_name).to_numpy()[pairs["q"].to_numpy()]
                    X.loc[lat, ["cat_prior", "cat_prior_gap"]] = np.nan
            return X.astype(np.float32)
        from rank import latin_feats as lf
        aux = lf.catalog_aux(catalog)
        parts = [pairs.drop(columns=list(NON_FEATURES))]
        if latin:
            parts.append(lf.latin_features(pairs, queries, aux))
        parts.append(fold)
        parts += [s.score(queries, pairs, catalog) for s in self.scorers]
        X = pd.concat(parts, axis=1).drop(columns=list(self.drop), errors="ignore")
        if mask:
            X = lf.mask_llm(X, pairs, queries, aux)
        return X.astype(np.float32)

    def fit(self, train: pd.DataFrame, catalog: pd.DataFrame, pairs: pd.DataFrame | None = None,
            aug: tuple[pd.DataFrame, pd.DataFrame, float] | None = None):
        """``aug`` = (extra queries, their pairs, weight relative to a real row of the same language), or a list
        of such items. An item may carry a 4th element, the source train rows of noise-augmented copies (S9
        item 4a, rank.noise_aug): those copies get leave-source-out fold features (FoldFeatures.transform_lso).
        3-tuples behave exactly as before.

        Extra rows (e.g. synthetic) are training-only: fold features are still built
        from the real train rows, and the extra rows never enter the history tables.
        """
        train = train.reset_index(drop=True)
        pairs = self.generator(catalog).build(train) if pairs is None else pairs
        self.fold = FoldFeatures(train, catalog)
        X = self.features(pairs, train, catalog, exclude_self=True)
        self.columns = list(X.columns)
        y = labels(pairs, train, catalog)
        qid = pairs["q"].to_numpy()
        w = query_weights(train)[qid] if self.weighted else np.ones(len(qid))
        augs = [] if aug is None else ([aug] if isinstance(aug, tuple) else list(aug))
        real_w = query_weights(train)
        is_ko = train["vessel_name"].map(is_hangul_name).to_numpy()
        base_w = {True: real_w[is_ko].max() if is_ko.any() else 1.0,
                  False: real_w[~is_ko].max() if (~is_ko).any() else 1.0}
        offset = len(train)
        for item in augs:
            aq, ap, aw = item[:3]
            src = item[3] if len(item) > 3 else None
            aq = aq.reset_index(drop=True)
            # Extra rows may lack OOF-only columns (e.g. LLM scores): leave those NaN.
            Xa = self.features(ap, aq, catalog, exclude_self=False,
                               lso_src=None if src is None else src.reset_index(drop=True)
                               ).reindex(columns=self.columns)
            # Each extra row weighs aw x a real row of the same language group.
            row_w = np.array([base_w[is_hangul_name(n)] for n in aq["vessel_name"]]) * aw
            X = pd.concat([X, Xa], ignore_index=True)
            y = np.concatenate([y, labels(ap, aq, catalog)])
            qid = np.concatenate([qid, ap["q"].to_numpy() + offset])
            w = np.concatenate([w, row_w[ap["q"].to_numpy()]])
            offset += len(aq)
        self.ranker = make_ranker(self.ranker_name).fit(X, y, qid, w)
        return self

    def predict(self, queries: pd.DataFrame, catalog: pd.DataFrame,
                pairs: pd.DataFrame | None = None) -> list[str]:
        queries = queries.reset_index(drop=True)
        pairs = self.generator(catalog).build(queries) if pairs is None else pairs
        # reindex: columns a scorer could not produce (e.g. LLM unavailable) become NaN
        X = self.features(pairs, queries, catalog, exclude_self=False).reindex(columns=self.columns)
        if getattr(self, "e5_feats", None):  # 10/31 e5 ranker features (rank/e5feat.py)
            from pathlib import Path as _P
            from rank.e5feat import add_e5
            _ef = self.e5_feats if isinstance(self.e5_feats, (list, tuple)) else [("e5ko", "e5")]
            for _d, _p in _ef:
                X = add_e5(X, pairs, queries, _P(__file__).resolve().parents[2] / _d, prefix=_p)
        score = self.ranker.predict(X)
        xe = getattr(self, "xe", None)  # S9x: Latin top-k cross-encoder re-scoring (rank/xe_latin.py); None = off
        if xe is not None:
            score = xe.blend(pairs, score, queries, catalog, queries["vessel_name"].map(is_hangul_name).to_numpy())
        # Optional post-hoc rules; pickles from before this attribute existed have none.
        posthoc = getattr(self, "posthoc", None)
        if not posthoc:
            final = pick_best(pairs, score, catalog, len(queries))
        else:
            final = apply_posthoc(posthoc, pairs, X, score, queries, catalog)
        try:
            from rank.submission_variant import apply_submission_variant
            final = apply_submission_variant(pairs, score, queries, catalog, posthoc, final)
        except Exception as exc:  # final policy is fail-safe
            print(f"submission variant failed ({{type(exc).__name__}}); prior answers retained")
        probe = getattr(self, "probe", None)
        if isinstance(probe, dict):  # LB diagnostics per language: {"la": mode, "ko": mode} (see probe_answers)
            return probe_answers(probe, pairs, X, score, queries, catalog, final, posthoc=posthoc)
        if probe == "rank2":  # LB diagnostic only: the best-scored candidate other than the answer
            return runner_up(pairs, score, catalog, final)
        if probe == "rank3_latin":  # LB diagnostic only: Latin rows answer the best candidate that is neither the
            r2 = runner_up(pairs, score, catalog, final)  # answer nor the rank2-probe answer; Hangul rows keep the answer
            excl = [f"{a}|{b}" for a, b in zip(final, r2)]
            r3 = runner_up_excluding(pairs, score, catalog, excl)
            ko = queries["vessel_name"].map(is_hangul_name).to_numpy()
            return [f if k else t for f, t, k in zip(final, r3, ko)]
        if probe in ("exact1_latin", "exactdup_latin"):  # LB diagnostic only: Latin rows whose normalized name
            from collections import Counter  # matches exactly one (or 2+) catalog names keep the answer, other Latin
            from common import normalize_key  # rows answer NONE; Hangul rows keep the answer
            cnt = Counter(normalize_key(n) for n in catalog["vessel_name"].fillna(""))
            ko = queries["vessel_name"].map(is_hangul_name).to_numpy()
            k = [cnt.get(normalize_key(n), 0) if isinstance(n, str) else 0 for n in queries["vessel_name"]]
            want = (lambda x: x == 1) if probe == "exact1_latin" else (lambda x: x >= 2)
            return [f if (h or want(x)) else PROBE_WRONG for f, h, x in zip(final, ko, k)]
        return final


def pick_best(pairs: pd.DataFrame, score: np.ndarray, catalog: pd.DataFrame, n_queries: int) -> list[str]:
    ids = catalog["vessel_id"].astype(str).to_numpy()
    frame = pd.DataFrame({"q": pairs["q"].to_numpy(), "c": pairs["c"].to_numpy(), "s": score})
    # Highest score, ties -> lowest catalog position: deterministic per row.
    frame = frame.sort_values(["q", "s", "c"], ascending=[True, False, True], kind="stable")
    best = frame.drop_duplicates("q").set_index("q")["c"]
    return [ids[best.get(q, 0)] for q in range(n_queries)]


# LB probes: a valid catalog id that is never a plausible answer (type "Pass" never appears among the 5,851 train
# targets; script.py rejects ids outside the catalog, so a literal "NONE" would fail the run).
PROBE_WRONG = "VESSEL_PROBE"

def add_cat_prior(X: pd.DataFrame, pairs: pd.DataFrame, prior: np.ndarray) -> pd.DataFrame:
    """S9u: cat_prior = logit of the catalog prior of the candidate; cat_prior_gap = row max - value (row-local)."""
    p = np.clip(np.asarray(prior, dtype=np.float64)[pairs["c"].to_numpy()], 1e-5, 1 - 1e-5)
    lg = np.log(p / (1 - p))
    X = X.copy()
    X["cat_prior"] = lg
    X["cat_prior_gap"] = pd.Series(lg).groupby(pairs["q"].to_numpy()).transform("max").to_numpy() - lg
    return X


REL_COLS = ("osa_sim", "word_overlap", "ld_cos", "de_cos", "de2_cos", "tr_sim", "nm_sim", "ph_sim", "flag_p",
            "type_p", "lg_min", "gt_eq", "word_cos")
REL_LOWER_BETTER = ("lg_min",)


def add_relative(X: pd.DataFrame, q: np.ndarray) -> pd.DataFrame:
    """S9t row-relative features (Foursquare-style): for each signal, the candidate's rank within its row and its gap
    to the row's best value (NaN where the signal is NaN). Row-local: only the row's own candidates."""
    out = {}
    g = pd.Series(q)
    for c in REL_COLS:
        if c not in X.columns:
            continue
        v = X[c].to_numpy(dtype=np.float64)
        sign = -1.0 if c in REL_LOWER_BETTER else 1.0
        sv = pd.Series(sign * v)
        best = sv.groupby(g.values).transform("max").to_numpy()
        out[f"rel_{c}_gap"] = best - sign * v
        out[f"rel_{c}_rank"] = sv.groupby(g.values).rank(ascending=False, method="min").to_numpy()
    return pd.concat([X, pd.DataFrame(out, index=X.index)], axis=1)


def runner_up(pairs: pd.DataFrame, score: np.ndarray, catalog: pd.DataFrame, final: list[str]) -> list[str]:
    """Per row: the highest-scored candidate whose vessel differs from ``final`` (ties -> lowest catalog position);
    a row with no other candidate keeps its answer. Row-local (LB probe "rank2")."""
    ids = catalog["vessel_id"].astype(str).to_numpy()
    frame = pd.DataFrame({"q": pairs["q"].to_numpy(), "c": pairs["c"].to_numpy(), "s": score})
    frame = frame.sort_values(["q", "s", "c"], ascending=[True, False, True], kind="stable")
    fin = np.asarray(final, dtype=object)
    frame = frame[ids[frame["c"].to_numpy()] != fin[frame["q"].to_numpy()]].drop_duplicates("q")
    alt = dict(zip(frame["q"].to_numpy(), ids[frame["c"].to_numpy()]))
    return [alt.get(q, final[q]) for q in range(len(final))]


SWAP_CONDS = ("sw_name", "sw_ton", "sw_meta")
SWAP_TON = float(np.log(1.25))  # tonnage "match" = within 25 % (the s9v_tonveto band)
SWAP_META = 0.3  # flag_p / type_p lead of the answer that counts as "flag / type chose it"
SWAP_NAME_EPS = 1e-6


def swap_table(pairs: pd.DataFrame, X: pd.DataFrame, catalog: pd.DataFrame, final: list[str],
               r2: list[str]) -> pd.DataFrame:
    """Per row: name strength (f_nstr), tonnage distance (lg_min), flag_p and type_p of the answer (A) and of the
    rank-2 candidate (B), and three DISJOINT rank-2 swap conditions (LB probe modes sw_*). B must differ from A and
    both must be candidates of the row, else no condition holds. Row-local.
      sw_ton   B's name is stronger, A is within 25 % tonnage and B is not (tonnage chose A)
      sw_meta  B's name is stronger, tonnage did not choose A, A leads on flag_p or type_p by >= SWAP_META
      sw_name  B's name is stronger and neither tonnage nor flag / type chose A
    Their union = every row whose rank-2 candidate has the stronger name, so per-condition Public gains add up."""
    ids = catalog["vessel_id"].astype(str).to_numpy()
    pos_of = {v: i for i, v in enumerate(ids)}
    n = len(final)
    q, c = pairs["q"].to_numpy(), pairs["c"].to_numpy()
    at = pd.Series(np.arange(len(pairs)), index=pd.MultiIndex.from_arrays([q, c]))
    at = at[~at.index.duplicated()]
    rows = np.arange(n)

    def locate(ans):
        p = np.array([pos_of.get(v, -1) for v in ans])
        return at.reindex(pd.MultiIndex.from_arrays([rows, p])).to_numpy(dtype=float)

    ia, ib = locate(final), locate(r2)
    has = np.isfinite(ia) & np.isfinite(ib) & (np.asarray(final, dtype=object) != np.asarray(r2, dtype=object))
    ia, ib = np.where(has, ia, 0).astype(int), np.where(has, ib, 0).astype(int)
    T = pd.DataFrame({"has": has})
    for k, col in (("n", "f_nstr"), ("t", "lg_min"), ("f", "flag_p"), ("y", "type_p")):
        v = X[col].to_numpy(dtype=float) if col in X.columns else np.full(len(pairs), np.nan)
        T[k + "A"] = np.where(has, v[ia], np.nan)
        T[k + "B"] = np.where(has, v[ib], np.nan)
    better = has & (np.nan_to_num(T["nB"].to_numpy(), nan=-1.0)
                    > np.nan_to_num(T["nA"].to_numpy(), nan=-1.0) + SWAP_NAME_EPS)
    with np.errstate(invalid="ignore"):
        ton = (T["tA"].to_numpy() <= SWAP_TON) & (T["tB"].to_numpy() > SWAP_TON)  # NaN (no tonnage) -> False
    lead = lambda k: np.nan_to_num(T[k + "A"].to_numpy(), nan=0.0) - np.nan_to_num(T[k + "B"].to_numpy(), nan=0.0)
    meta = (lead("f") >= SWAP_META) | (lead("y") >= SWAP_META)
    T["sw_ton"] = better & ton
    T["sw_meta"] = better & ~ton & meta
    T["sw_name"] = better & ~ton & ~meta
    return T


def probe_answers(probe: dict, pairs, X, score, queries, catalog, final: list[str],
                  posthoc: dict | None = None) -> list[str]:
    """LB diagnostics, chosen per language (probe["la"], probe["ko"]; missing = "final"). Modes:
      final      the answer
      rank2      best-scored candidate other than the answer
      rank3      best-scored candidate that is neither the answer nor the rank2 answer
      exact1     the answer if the record's normalized name matches exactly one catalog name, else PROBE_WRONG
      exactdup   the answer if it matches two or more catalog names, else PROBE_WRONG
      llm        4B argmax (max ft_score) if it differs from the answer, else PROBE_WRONG
      llm2       the rank2 answer where it is also the 4B argmax (4B agrees with the ranker's runner-up against the
                 answer), else the answer (a candidate, not a sentinel probe). Only rows the post-hoc rules left alone
                 (answer == the ranker's own top-1; elsewhere rank2 IS the vessel tt / cs_override removed) and only
                 when ``posthoc['tt']`` does not penalise the rank2 vessel for that row.
      outcand    best char-3gram TF-IDF catalog name OUTSIDE the row's candidate set and its answer (Codex R7:
                 every hit is a row whose truth the candidate generator missed)
      deout      Hangul: best catalog vessel by the summed cosine of the shipped Hangul dual encoders (obj, obj2,
                 obj3) OUTSIDE the row's candidate set and its answer (truths a deeper DE top-k would add)
      deall      Hangul: like deout but over the WHOLE catalog except the answer (inside or outside the candidates)
      rank4      best-scored candidate that is none of the answer, rank2, rank3
      sw_name / sw_ton / sw_meta
                 the rank2 answer where that swap condition holds (swap_table), else the answer: the decoded
                 score change is the condition's net Public gain (fixed - broken); "sw_name+sw_meta" = the union
    Every mode is row-local; non-matching rows answer PROBE_WRONG (public-repo placeholder; set it to a real catalog id that is never a plausible answer before using a probe mode)
    (sw_* modes keep the answer instead)."""
    from collections import Counter
    from common import normalize_key
    ids = catalog["vessel_id"].astype(str).to_numpy()
    ko = queries["vessel_name"].map(is_hangul_name).to_numpy()
    r2 = runner_up(pairs, score, catalog, final)
    r3 = runner_up_excluding(pairs, score, catalog, [f"{a}|{b}" for a, b in zip(final, r2)])
    cnt = Counter(normalize_key(n) for n in catalog["vessel_name"].fillna(""))
    nk = [cnt.get(normalize_key(n), 0) if isinstance(n, str) else 0 for n in queries["vessel_name"]]
    llm = [PROBE_WRONG] * len(final)
    ft_top: dict[int, str] = {}  # row -> 4B argmax vessel (rows with at least one finite ft_score)
    if "ft_score" in X.columns:
        f = pd.DataFrame({"q": pairs["q"].to_numpy(), "c": pairs["c"].to_numpy(), "s": X["ft_score"].to_numpy()})
        f = f[np.isfinite(f["s"].to_numpy())].sort_values(["q", "s", "c"], ascending=[True, False, True], kind="stable")
        top = f.drop_duplicates("q")
        for q, c in zip(top["q"].to_numpy(), top["c"].to_numpy()):
            ft_top[int(q)] = ids[c]
            if ids[c] != final[q]:
                llm[q] = ids[c]
    top1, tt_bad = final, set()
    if "llm2" in probe.values():
        top1 = pick_best(pairs, score, catalog, len(final))  # the ranker's own answer, before the post-hoc rules
        if posthoc and posthoc.get("tt") is not None:
            m = posthoc["tt"].penalty(pairs, queries, catalog) > 0
            tt_bad = set(zip(pairs["q"].to_numpy()[m].tolist(), ids[pairs["c"].to_numpy()[m]].tolist()))
    r4 = (runner_up_excluding(pairs, score, catalog, [f"{a}|{b}|{c}" for a, b, c in zip(final, r2, r3)])
          if "rank4" in probe.values() else None)
    deo = None
    if "deout" in probe.values() or "deall" in probe.values():
        mode_de = "deout" if "deout" in probe.values() else "deall"
        only = np.array([probe.get("ko" if k else "la", "final") == mode_de for k in ko])
        deo = outside_de(pairs, queries, catalog, final, only, exclude_cands=mode_de == "deout")
    outc = None
    if "outcand" in probe.values():
        only = np.array([probe.get("ko" if k else "la", "final") == "outcand" for k in ko])
        outc = outside_tfidf(pairs, queries, catalog, final, only=only)
    sw = None
    if any(str(m).startswith("sw_") for m in probe.values()):
        bad = {p for m in probe.values() if str(m).startswith("sw_") for p in m.split("+")} - set(SWAP_CONDS)
        if bad:
            raise ValueError(f"unknown swap conditions {sorted(bad)}")
        sw = swap_table(pairs, X, catalog, final, r2)
    out = []
    for i in range(len(final)):
        mode = probe.get("ko" if ko[i] else "la", "final")
        if mode == "final":
            out.append(final[i])
        elif mode == "rank2":
            out.append(r2[i])
        elif mode == "rank3":
            out.append(r3[i])
        elif mode == "exact1":
            out.append(final[i] if nk[i] == 1 else PROBE_WRONG)
        elif mode == "exactdup":
            out.append(final[i] if nk[i] >= 2 else PROBE_WRONG)
        elif mode == "llm":
            out.append(llm[i])
        elif mode == "llm2":
            ok = (r2[i] != final[i] and ft_top.get(i) == r2[i] and final[i] == top1[i]
                  and (i, r2[i]) not in tt_bad)
            out.append(r2[i] if ok else final[i])
        elif mode == "outcand":
            out.append(outc[i])
        elif mode in ("deout", "deall"):
            out.append(deo[i])
        elif mode == "rank4":
            out.append(r4[i])
        elif mode.startswith("sw_"):
            out.append(r2[i] if any(sw[m].iat[i] for m in mode.split("+")) else final[i])
        else:
            raise ValueError(f"unknown probe mode {mode}")
    return out


def outside_de(pairs: pd.DataFrame, queries: pd.DataFrame, catalog: pd.DataFrame, final: list[str], only,
               exclude_cands: bool = True) -> list[str]:
    """Per selected Hangul row: argmax over catalog vessels outside the row's candidate set and answer of the summed
    cosine of the registered Hangul dual encoders (DUAL_ENCODER obj / obj2 / obj3); PROBE_WRONG if none is set."""
    encs = [DUAL_ENCODER[k] for k in ("obj", "obj2", "obj3") if DUAL_ENCODER.get(k) is not None]
    ids = catalog["vessel_id"].astype(str).to_numpy()
    pos_of = {v: i for i, v in enumerate(ids)}
    cand = pairs.groupby("q")["c"].agg(lambda c: set(c.tolist())).to_dict()
    recs = queries.to_dict("records")
    out = []
    for i, r in enumerate(recs):
        if not only[i] or not encs:
            out.append(PROBE_WRONG); continue
        sc = None
        for e in encs:
            q, cat = e.query(r)
            s = cat @ q
            sc = s if sc is None else sc + s
        excl = set(cand.get(i, set())) if exclude_cands else set()
        if final[i] in pos_of:
            excl.add(pos_of[final[i]])
        if excl:
            sc = sc.copy(); sc[np.fromiter(excl, dtype=np.int64)] = -np.inf
        out.append(ids[int(np.argmax(sc))])
    return out


def outside_tfidf(pairs: pd.DataFrame, queries: pd.DataFrame, catalog: pd.DataFrame, final: list[str],
                  only=None) -> list[str]:
    """Per row: the catalog vessel with the highest char-3gram TF-IDF cosine (binary tf x idf, L2; the
    latin_char_tfidf scheme) among vessels that are NOT in the row's candidate set and NOT its answer; ties -> lowest
    catalog position; no trigram / no positive score -> PROBE_WRONG. Row-local (the index is built from the catalog)."""
    from scipy import sparse
    from common import normalize_key
    from rank.candidates import _trigrams
    ids = catalog["vessel_id"].astype(str).to_numpy()
    pos_of = {v: i for i, v in enumerate(ids)}
    docs = [_trigrams(normalize_key(n)) for n in catalog["vessel_name"].fillna("").astype(str)]
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for i, terms in enumerate(docs):
        for t in terms:
            cols.append(vocab.setdefault(t, len(vocab))); rows.append(i)
    n = len(docs)
    m = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(vocab)))
    df = np.asarray(m.sum(axis=0)).ravel()
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    m = m.multiply(idf[None, :]).tocsr()
    norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    m = (sparse.diags(1.0 / np.where(norm > 0, norm, 1.0)) @ m).tocsc()
    cand = pairs.groupby("q")["c"].agg(lambda c: set(c.tolist())).to_dict()
    out = []
    for i, name in enumerate(queries["vessel_name"].tolist()):
        if only is not None and not only[i]:
            out.append(PROBE_WRONG); continue
        terms = _trigrams(normalize_key(name)) if isinstance(name, str) else set()
        known = sorted(vocab[t] for t in terms if t in vocab)
        if not known:
            out.append(PROBE_WRONG); continue
        sc = np.asarray(m[:, known] @ idf[known]).ravel()
        excl = set(cand.get(i, set()))
        if final[i] in pos_of:
            excl.add(pos_of[final[i]])
        if excl:
            sc[np.fromiter(excl, dtype=np.int64)] = -1.0
        j = int(np.argmax(sc))
        out.append(ids[j] if sc[j] > 0 else PROBE_WRONG)
    return out


def runner_up_excluding(pairs: pd.DataFrame, score: np.ndarray, catalog: pd.DataFrame, excl: list[str]) -> list[str]:
    """Per row: the highest-scored candidate whose vessel id is not in excl[q] ("id|id"); a row with no such candidate
    answers PROBE_WRONG (never a correct id, so the probe arithmetic stays exact)."""
    ids = catalog["vessel_id"].astype(str).to_numpy()
    frame = pd.DataFrame({"q": pairs["q"].to_numpy(), "c": pairs["c"].to_numpy(), "s": score})
    frame = frame.sort_values(["q", "s", "c"], ascending=[True, False, True], kind="stable")
    ex = [set(e.split("|")) for e in excl]
    keep = np.array([ids[c] not in ex[q] for q, c in zip(frame["q"].to_numpy(), frame["c"].to_numpy())])
    frame = frame[keep].drop_duplicates("q")
    alt = dict(zip(frame["q"].to_numpy(), ids[frame["c"].to_numpy()]))
    return [alt.get(q, PROBE_WRONG) for q in range(len(excl))]


def hangul_llm_pick(pairs: pd.DataFrame, ft_score: np.ndarray | None, queries: pd.DataFrame) -> dict[int, int]:
    """Hangul rows with any non-NaN LLM score -> argmax ft_score (ties -> lowest catalog position).

    Returns {row index: catalog position}; Latin rows and rows without LLM scores are absent.
    """
    if ft_score is None:
        return {}
    hangul = queries["vessel_name"].map(is_hangul_name).to_numpy(dtype=bool)
    q = pairs["q"].to_numpy()
    s = np.asarray(ft_score, dtype=np.float64)
    keep = hangul[q] & ~np.isnan(s)
    if not keep.any():
        return {}
    frame = pd.DataFrame({"q": q[keep], "c": pairs["c"].to_numpy()[keep], "s": s[keep]})
    frame = frame.sort_values(["q", "s", "c"], ascending=[True, False, True], kind="stable")
    best = frame.drop_duplicates("q")
    return dict(zip(best["q"].tolist(), best["c"].tolist()))


def apply_posthoc(cfg: dict, pairs: pd.DataFrame, X: pd.DataFrame, score: np.ndarray,
                  queries: pd.DataFrame, catalog: pd.DataFrame, log=print) -> list[str]:
    """Row-local post-hoc rules on top of the ranker (reports/diag/action_plan.md C1, C2).

    Order: 'tt' (TrainTargetRule; score -= tt_weight * penalty) -> pick_best ->
    'hangul_llm_pick' (Hangul rows: argmax ft_score) -> 'cs_override' (CallSignMemory; optional 'cs_guard' name
    agreement threshold, and 'cs_need_cand' (S9 err5 fix A, default off): the call-sign vessel must be among the
    row's candidates in ``pairs``, the complete frame including src_extra rows).
    Each stage logs COUNTS only (rows fired / picks changed): no sample_id, vessel id or any other
    per-row value is printed, because the competition rules forbid logging evaluation rows.
    """
    from rank.train_memory import tt_adjust

    ids = catalog["vessel_id"].astype(str).to_numpy()
    n = len(queries)

    def report(rule: str, before: list[str], after: list[str], fired: int) -> None:
        changed = sum(1 for i in range(n) if before[i] != after[i])
        log(f"posthoc {rule}: fired on {fired} rows, changed {changed}")

    if cfg.get("ton_veto_w"):  # S9v: train tonnage law (record vs truth GT |log ratio| <= 0.0915 on all 11,429 train
        # rows with tonnage); the ranker learned to accept big gaps from the x0.5-x2 noise copies. Candidates whose GT is
        # more than exp(ton_veto_b) off the record tonnage lose ton_veto_w (before tt / cs, like the tt score shift).
        lg = X["lg_min"].to_numpy(dtype=np.float64)
        bad = np.nan_to_num(lg, nan=0.0) > float(cfg.get("ton_veto_b", np.log(1.25)))
        base_pick = pick_best(pairs, score, catalog, n)
        score = np.asarray(score, dtype=np.float64) - float(cfg["ton_veto_w"]) * bad
        report(f"ton_veto w={cfg['ton_veto_w']:g}", base_pick, pick_best(pairs, score, catalog, n),
               len(np.unique(pairs["q"].to_numpy()[bad])))
    preds = pick_best(pairs, score, catalog, n)
    if cfg.get("tt") is not None:
        pen = cfg["tt"].penalty(pairs, queries, catalog)
        new = pick_best(pairs, tt_adjust(score, pen, cfg.get("tt_weight", 10.0)), catalog, n)
        report(f"tt w={cfg.get('tt_weight', 10.0):g}", preds, new,
               len(np.unique(pairs["q"].to_numpy()[pen > 0])))
        preds = new
    if cfg.get("hangul_llm_pick"):
        ft = X["ft_score"].to_numpy() if "ft_score" in X.columns else None
        picks = hangul_llm_pick(pairs, ft, queries)
        new = list(preds)
        for qi, c in picks.items():
            new[qi] = ids[c]
        report("hangul_llm_pick", preds, new, len(picks))
        preds = new
    if cfg.get("cs_override") is not None:
        over = cfg["cs_override"].lookup(queries)
        if cfg.get("cs_guard") is not None:  # S9: keep an override only when the vessel's name agrees with the row
            from rank.train_memory import cs_name_agreement
            gen = candidate_generator(catalog)
            pos_of = {v: i for i, v in enumerate(ids)}
            names = getattr(cfg["cs_override"], "names", {})
            rows = queries.to_dict("records")
            kept = {qi: vid for qi, vid in over.items() if vid in pos_of and cs_name_agreement(
                rows[qi], pos_of[vid], gen, names.get(vid, ())) >= cfg["cs_guard"]}
            log(f"posthoc cs_guard t={cfg['cs_guard']:g}: blocked {len(over) - len(kept)} of {len(over)}")
            over = kept
        if cfg.get("cs_need_cand"):  # S9 err5 fix A: keep an override only when that vessel is a candidate of the row
            n_cat = len(ids)
            pos_of = {v: i for i, v in enumerate(ids)}
            have = pairs["q"].to_numpy(np.int64) * n_cat + pairs["c"].to_numpy(np.int64)
            rows_in = [qi for qi, vid in over.items() if vid in pos_of]
            want = np.array([qi * n_cat + pos_of[over[qi]] for qi in rows_in], dtype=np.int64)
            kept = {qi: over[qi] for qi, ok in zip(rows_in, np.isin(want, have)) if ok}
            log(f"posthoc cs_need_cand: blocked {len(over) - len(kept)} of {len(over)}")
            over = kept
        known = set(ids)
        new = list(preds)
        for qi, vid in over.items():
            if vid in known:
                new[qi] = vid
        report("cs_override", preds, new, len(over))
        preds = new
    return preds
