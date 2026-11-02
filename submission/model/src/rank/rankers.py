"""Uniform wrappers over gradient-boosting rankers.

Each ranker: ``fit(X, y, qid, weight)`` with rows grouped contiguously by qid,
``predict(X) -> score`` (higher = more likely the true vessel). Only the
argmax within each query matters downstream.
"""

from __future__ import annotations

import numpy as np

SEED = 0


class _Base:
    name = "base"

    def fit(self, X, y, qid, weight):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


def _group_sizes(qid: np.ndarray) -> np.ndarray:
    # qid must be sorted/contiguous.
    _, sizes = np.unique(qid, return_counts=True)
    return sizes


def _query_weight(qid: np.ndarray, weight: np.ndarray) -> np.ndarray:
    first = np.r_[True, qid[1:] != qid[:-1]]
    return weight[first]


class LGBBinary(_Base):
    name = "lgb_binary"

    def __init__(self, **params):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
            verbose=-1, **params)

    def fit(self, X, y, qid, weight):
        self.model.fit(X, y, sample_weight=weight)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class LGBRank(_Base):
    def __init__(self, objective="lambdarank", seed: int | None = None, **params):
        """``seed`` None -> the module-level SEED at construction time (the old behaviour)."""
        import lightgbm as lgb
        self.name = f"lgb_{objective}"
        base = dict(objective=objective, n_estimators=600, learning_rate=0.05, num_leaves=31,
                    min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    random_state=SEED if seed is None else seed, verbose=-1)
        self.model = lgb.LGBMRanker(**{**base, **params})  # params may override (llmgen41: n_estimators)

    def fit(self, X, y, qid, weight):
        self.model.fit(X, y, group=_group_sizes(qid), sample_weight=weight)
        return self

    def predict(self, X):
        return self.model.predict(X)


class XGB(_Base):
    def __init__(self, objective="binary:logistic", **params):
        import xgboost as xgb
        self.name = "xgb_" + objective.split(":")[1]
        self.objective = objective
        common = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                      colsample_bytree=0.8, tree_method="hist", random_state=SEED, **params)
        if objective.startswith("rank:"):
            self.model = xgb.XGBRanker(objective=objective, lambdarank_pair_method="topk", **common)
        else:
            self.model = xgb.XGBClassifier(objective=objective, eval_metric="logloss", **common)

    def fit(self, X, y, qid, weight):
        if self.objective.startswith("rank:"):
            # XGBoost ranking takes one weight per query group.
            self.model.fit(X, y, qid=qid, sample_weight=_query_weight(qid, weight))
        else:
            self.model.fit(X, y, sample_weight=weight)
        return self

    def predict(self, X):
        if self.objective.startswith("rank:"):
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]


class Cat(_Base):
    def __init__(self, loss="Logloss", **params):
        self.name = f"cat_{loss.lower()}"
        self.loss = loss
        self.params = dict(iterations=800, learning_rate=0.08, depth=6, random_seed=SEED,
                           verbose=0, thread_count=-1, **params)

    def fit(self, X, y, qid, weight):
        from catboost import CatBoost, Pool
        if self.loss == "Logloss":
            pool = Pool(X, y, weight=weight)
        else:
            # Pairwise/ranking losses take one weight per group, not per object.
            pool = Pool(X, y, group_id=qid, group_weight=weight)
        self.model = CatBoost({"loss_function": self.loss, **self.params})
        self.model.fit(pool)
        return self

    def predict(self, X):
        from catboost import Pool
        kind = "Probability" if self.loss == "Logloss" else "RawFormulaVal"
        pred = self._model().predict(Pool(X), prediction_type=kind)
        return pred[:, 1] if pred.ndim == 2 else pred

    # llmgen41 C1: pickle the fitted model as CatBoost's own bytes and rebuild it on first use, so loading the
    # artifacts never needs catboost (a server without it loses only this member, see FittedZAvg.predict).
    def _model(self):
        if getattr(self, "model", None) is None and getattr(self, "_cbm", None) is not None:
            import os
            import tempfile
            from catboost import CatBoost
            fd, path = tempfile.mkstemp(suffix=".cbm")
            os.close(fd)
            try:
                with open(path, "wb") as f:
                    f.write(self._cbm)
                self.model = CatBoost().load_model(path)
            finally:
                os.remove(path)
        return self.model

    def __getstate__(self):
        st = dict(self.__dict__)
        m = st.pop("model", None)
        if m is not None:
            import os
            import tempfile
            fd, path = tempfile.mkstemp(suffix=".cbm")
            os.close(fd)
            try:
                m.save_model(path)
                with open(path, "rb") as f:
                    st["_cbm"] = f.read()
            finally:
                os.remove(path)
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self.model = None


class HGB(_Base):
    name = "sk_hgb"

    def __init__(self, **params):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, max_leaf_nodes=31, random_state=SEED, **params)

    def fit(self, X, y, qid, weight):
        self.model.fit(X, y, sample_weight=weight)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class RankEnsemble(_Base):
    """Average of member scores, each standardized with constants from its training predictions.

    The constants are fixed at fit time, so a row's ensemble score never depends on
    the other rows being predicted.
    """

    def __init__(self, members, name: str | None = None):
        self.members = members
        self.name = name or "ens_" + "+".join(m.name for m in members)

    def fit(self, X, y, qid, weight):
        self.scale = []
        for m in self.members:
            m.fit(X, y, qid, weight)
            p = m.predict(X)
            self.scale.append((float(p.mean()), float(p.std()) or 1.0))
        return self

    def predict(self, X):
        z = sum((m.predict(X) - mu) / sd for m, (mu, sd) in zip(self.members, self.scale)) / len(self.members)
        if getattr(self, "rescale", False):  # back on member 0's score scale (post-hoc tt_weight was tuned on it)
            mu0, sd0 = self.scale[0]
            z = z * sd0 + mu0
        return z


def lgb_lambdarank_bag(n: int = 3) -> RankEnsemble:
    """C4.5: n LightGBM lambdarank models that differ only in the seed, z-averaged with the mean/sd of each
    member's training scores fixed at fit (RankEnsemble). Seeds n*SEED .. n*SEED+n-1, read when the ranker
    is made, so SEED=0 gives seeds 0,1,2 and a CV over SEED in {0,1,2} sees 3 disjoint bags."""
    return RankEnsemble([LGBRank("lambdarank", seed=n * SEED + i) for i in range(n)], name=f"lgb_lambdarank_bag{n}")


class ColumnEnsemble(RankEnsemble):
    """RankEnsemble whose members see different column subsets: member i drops the columns starting with any prefix
    in ``drops[i]`` (S9o: rankers with and without the de3_* features on the same frame). Output on member 0's score
    scale (rescale), as the post-hoc tt weight expects."""

    def __init__(self, members, drops, name: str | None = None):
        super().__init__(members, name)
        self.drops = [tuple(d) for d in drops]
        self.rescale = True

    def _cols(self, X, d):
        return [c for c in X.columns if not any(c.startswith(p) for p in d)]

    def fit(self, X, y, qid, weight):
        self.scale, self.cols = [], []
        for m, d in zip(self.members, self.drops):
            cols = self._cols(X, d)
            m.fit(X[cols], y, qid, weight)
            p = m.predict(X[cols])
            self.scale.append((float(p.mean()), float(p.std()) or 1.0))
            self.cols.append(cols)
        return self

    def predict(self, X):
        z = sum((m.predict(X[c]) - mu) / sd for m, c, (mu, sd) in zip(self.members, self.cols, self.scale)) / len(
            self.members)
        mu0, sd0 = self.scale[0]
        return z * sd0 + mu0


def _colbag_de3() -> ColumnEnsemble:
    seeds = [0, 1, 0, 1]
    drops = [("de3_",), ("de3_",), (), ()]
    return ColumnEnsemble([LGBRank("lambdarank", seed=s) for s in seeds], drops, name="lgb_colbag_de3")


def _colbag_de3_t3000() -> ColumnEnsemble:
    """llmgen41: lgb_colbag_de3 with 3000 trees per member (tP3 5-fold harness: P3 Latin errors -3.6%, every fold
    equal or better, shift / clean not worse; the Optuna-tuned parameters did not hold up beyond the tuning fold)."""
    seeds = [0, 1, 0, 1]
    drops = [("de3_",), ("de3_",), (), ()]
    return ColumnEnsemble([LGBRank("lambdarank", seed=s, n_estimators=3000) for s in seeds], drops,
                          name="lgb_colbag_de3_t3000")


def _colbag_de3_trees(n_trees: int, seeds: tuple[int, ...], name: str):
    """llmgen41 10/27 follow-ups of t3000: more trees (tune41 fold 0 still improved at 5000) or other seeds (a second
    bag to z-average with the t3000 member)."""
    drops = [("de3_",), ("de3_",), (), ()]
    return lambda: ColumnEnsemble([LGBRank("lambdarank", seed=s, n_estimators=n_trees) for s in seeds], drops,
                                  name=name)


def _colbag_de3_ld2() -> ColumnEnsemble:
    """S9p: the S9o idea over the de3 + Latin v2-tok frame: members without both new feature groups, without one, and
    with all features."""
    spec = [(("de3_", "ld2_"), 0), (("de3_", "ld2_"), 1), (("ld2_",), 0), (("ld2_",), 1), (("de3_",), 0), ((), 0)]
    return ColumnEnsemble([LGBRank("lambdarank", seed=s) for _, s in spec], [d for d, _ in spec],
                          name="lgb_colbag_de3_ld2")


def _colbag_obj() -> ColumnEnsemble:
    """S9q: S9o's two column views x three LightGBM objectives (loss diversity: lambdarank, xendcg, binary)."""
    spec = [(LGBRank("lambdarank", seed=0), ("de3_",)), (LGBRank("lambdarank", seed=0), ()),
            (LGBRank("rank_xendcg", seed=0), ("de3_",)), (LGBRank("rank_xendcg", seed=0), ()),
            (LGBBinary(), ("de3_",)), (LGBBinary(), ())]
    return ColumnEnsemble([m for m, _ in spec], [d for _, d in spec], name="lgb_colbag_obj")


class LangSplit(_Base):
    """S9w-L: two FITTED rankers over the same feature matrix, routed by the query language column ``is_hangul``
    (constant within a query): Hangul rows -> ``ko``, other rows -> ``la``. Assembled post hoc from two pipelines
    with identical columns (experiments/s9_robust/llmgen30/compose31.py); it is never fitted itself."""
    name = "langsplit"

    def __init__(self, ko: _Base, la: _Base):
        self.ko, self.la = ko, la

    def fit(self, X, y, qid, weight):  # pragma: no cover - assembled from fitted rankers only
        raise NotImplementedError("LangSplit is assembled from two fitted rankers")

    def predict(self, X) -> np.ndarray:
        h = X["is_hangul"].to_numpy() > 0.5
        out = np.empty(len(X), dtype=np.float64)
        if h.any():
            out[h] = self.ko.predict(X[h])
        if (~h).any():
            out[~h] = self.la.predict(X[~h])
        return out


class FittedZAvg(_Base):
    """S9w-E: z-average of FITTED rankers trained on different augmentation sets (llmgen30/run32 "E_aug3"), mapped back
    to the first member's scale so the post-hoc tt weight keeps its meaning. ``scales[i]`` = (mean, std) of member i's
    training predictions (a ColumnEnsemble's own ``scale[0]``, since its output is on its member-0 scale)."""
    name = "fitted_zavg"

    def __init__(self, members, scales):
        self.members, self.scales = list(members), [(float(m), float(s) or 1.0) for m, s in scales]

    def fit(self, X, y, qid, weight):  # pragma: no cover - assembled from fitted rankers only
        raise NotImplementedError("FittedZAvg is assembled from fitted rankers")

    def predict(self, X) -> np.ndarray:
        z, n = 0.0, 0
        for m, (mu, sd) in zip(self.members, self.scales):
            try:
                p = m.predict(X)
            except ImportError:  # e.g. a CatBoost member on a server without catboost: drop it, keep the others
                continue
            z, n = z + (p - mu) / sd, n + 1
        if n == 0:
            raise ImportError("no FittedZAvg member could predict")
        mu0, sd0 = self.scales[0]
        return z / n * sd0 + mu0


def _rescaled(ens: RankEnsemble) -> RankEnsemble:
    ens.rescale = True
    ens.name += "r"
    return ens


RANKERS = {
    "lgb_binary": lambda: LGBBinary(),
    "lgb_lambdarank": lambda: LGBRank("lambdarank"),
    "lgb_xendcg": lambda: LGBRank("rank_xendcg"),
    "xgb_logistic": lambda: XGB("binary:logistic"),
    "xgb_pairwise": lambda: XGB("rank:pairwise"),
    "xgb_ndcg": lambda: XGB("rank:ndcg"),
    "cat_logloss": lambda: Cat("Logloss"),
    "cat_yetirank": lambda: Cat("YetiRank"),
    "cat_pairlogit": lambda: Cat("PairLogit"),
    "sk_hgb": lambda: HGB(),
    "ens_lgb_xgb": lambda: RankEnsemble([LGBRank("lambdarank"), XGB("rank:pairwise")]),
    "ens_lgb_xgb_cat": lambda: RankEnsemble([LGBRank("lambdarank"), XGB("rank:pairwise"), Cat("YetiRank")]),
    "lgb_lambdarank_bag3": lambda: lgb_lambdarank_bag(3),
    "lgb_lambdarank_bag5": lambda: lgb_lambdarank_bag(5),
    "lgb_lambdarank_bag5r": lambda: _rescaled(lgb_lambdarank_bag(5)),
    "lgb_colbag_de3": _colbag_de3,
    "lgb_colbag_de3_t3000": _colbag_de3_t3000,
    "lgb_colbag_de3_t6000": _colbag_de3_trees(6000, (0, 1, 0, 1), "lgb_colbag_de3_t6000"),
    "lgb_colbag_de3_t3000s23": _colbag_de3_trees(3000, (2, 3, 2, 3), "lgb_colbag_de3_t3000s23"),
    "lgb_colbag_de3_ld2": _colbag_de3_ld2,
    "lgb_colbag_obj": _colbag_obj,
    # llmgen41 C1: one CatBoost YetiRank model with fit-time score scale (a FittedZAvg member next to LightGBM ones)
    "cat_yetirank_z": lambda: ColumnEnsemble([Cat("YetiRank")], [()], name="cat_yetirank_z"),
}


def make_ranker(name: str) -> _Base:
    return RANKERS[name]()
