from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogCPM(BaseEstimator, TransformerMixin):
    def __init__(self, prior: float = 1.0):
        self.prior = prior

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        lib = X.sum(axis=1, keepdims=True)
        return np.log2((X + self.prior) / (lib + 2 * self.prior) * 1e6)


class TopVariance(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 2000):
        self.k = k

    def fit(self, X, y=None):
        var = np.asarray(X).var(axis=0)
        k = min(self.k, var.size)
        self.support_ = np.sort(np.argpartition(var, -k)[-k:])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


def build_pipeline(k: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("logcpm", LogCPM()),
            ("select", TopVariance(k=k)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(solver="saga", max_iter=5000, tol=1e-3, random_state=seed)),
        ]
    )


def _aligned(counts: pd.DataFrame, meta: pd.DataFrame, cfg):
    if counts.index.has_duplicates or set(counts.index) != set(meta.index):
        raise ValueError("counts and metadata sample indices differ")
    idx = counts.index[meta["etiology"].reindex(counts.index).isin([cfg.positive, cfg.negative]).to_numpy()]
    m = meta.loc[idx]
    X = counts.loc[idx].to_numpy(dtype=float)
    y = (m["etiology"] == cfg.positive).astype(int).to_numpy()
    return X, m, y


@dataclass
class ClassifierResult:
    oof: pd.DataFrame
    metrics: dict
    roc: pd.DataFrame
    coefficients: pd.DataFrame
    final_model: Pipeline


def nested_cv(counts: pd.DataFrame, meta: pd.DataFrame, cfg, seed: int, n_jobs: int = 1) -> ClassifierResult:
    X, m, y = _aligned(counts, meta, cfg)
    genes = counts.columns.to_numpy()
    grid = {"clf__C": cfg.C_grid, "clf__l1_ratio": cfg.l1_ratio_grid}
    outer = StratifiedKFold(cfg.outer_folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    oof_base = np.full(len(y), np.nan)
    base_idx = [np.where(genes == g)[0][0] for g in cfg.baseline_genes if g in genes]
    lc = LogCPM().fit_transform(X)

    for f, (tr, te) in enumerate(outer.split(X, y)):
        inner = StratifiedKFold(cfg.inner_folds, shuffle=True, random_state=seed + f + 1)
        search = GridSearchCV(build_pipeline(cfg.n_top_var_genes, seed), grid, scoring="roc_auc", cv=inner, n_jobs=n_jobs)
        search.fit(X[tr], y[tr])
        oof[te] = search.predict_proba(X[te])[:, 1]
        if base_idx:
            base = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
            base.fit(lc[tr][:, base_idx], y[tr])
            oof_base[te] = base.predict_proba(lc[te][:, base_idx])[:, 1]

    metrics = {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, oof)),
        "auprc": float(average_precision_score(y, oof)),
        "brier": float(brier_score_loss(y, oof)),
    }
    fpr, tpr, _ = roc_curve(y, oof)
    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "model": "elastic-net"})
    if base_idx:
        metrics["baseline_genes"] = [g for g in cfg.baseline_genes if g in genes]
        metrics["baseline_auroc"] = float(roc_auc_score(y, oof_base))
        bf, bt, _ = roc_curve(y, oof_base)
        roc = pd.concat([roc, pd.DataFrame({"fpr": bf, "tpr": bt, "model": "+".join(metrics["baseline_genes"])})])

    final = GridSearchCV(
        build_pipeline(cfg.n_top_var_genes, seed),
        grid,
        scoring="roc_auc",
        cv=StratifiedKFold(cfg.inner_folds, shuffle=True, random_state=seed),
        n_jobs=n_jobs,
    ).fit(X, y)
    best = final.best_estimator_
    sel = best.named_steps["select"].support_
    coef = pd.DataFrame({"gene": genes[sel], "coefficient": best.named_steps["clf"].coef_[0]})
    coef = coef[coef["coefficient"] != 0].assign(abs=lambda d: d["coefficient"].abs())
    coef = coef.sort_values("abs", ascending=False).drop(columns="abs")
    metrics["final_params"] = {k: float(v) for k, v in final.best_params_.items()}

    oof_df = pd.DataFrame({"label": y, "prob": oof, "baseline_prob": oof_base}, index=m.index)
    return ClassifierResult(oof_df, metrics, roc, coef, best)
