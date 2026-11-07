"""Utility regressor (mean, 10% and 90% quantiles) and harm classifier (Phase 4, P4.2-P4.3).

Gradient-boosted trees (LightGBM) with strong regularisation are the main
models; ridge / logistic regression are the guards against over-fitting on a
few hundred rows. Hyperparameters are chosen by grouped cross-validation over
training targets only (all rows of a target fall in one fold).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, QuantileRegressor, Ridge
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="X does not have valid feature names")

LGB_GRID = {"num_leaves": [4, 8], "min_child_samples": [10, 30], "n_estimators": [150, 400]}
LGB_FIXED = dict(learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                 verbose=-1, n_jobs=1)


def grouped_folds(groups: np.ndarray, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))


def _lgb_reg(objective: str, alpha: float | None, seed: int, **hp):
    kw = dict(LGB_FIXED, random_state=seed, objective=objective, **hp)
    if alpha is not None:
        kw["alpha"] = alpha
    return lgb.LGBMRegressor(**kw)


def _lgb_clf(seed: int, **hp):
    return lgb.LGBMClassifier(**dict(LGB_FIXED, random_state=seed, class_weight="balanced", **hp))


def _grid():
    return [dict(zip(LGB_GRID, v)) for v in product(*LGB_GRID.values())]


def _cv_select(make, X, y, groups, score, seed) -> dict:
    """Pick the hyperparameters with the best grouped-CV score (lower is better)."""
    folds = grouped_folds(groups)
    best, best_s = None, np.inf
    for hp in _grid():
        s = []
        for tr, va in folds:
            m = make(seed=seed, **hp).fit(X[tr], y[tr])
            s.append(score(m, X[va], y[va]))
        if np.mean(s) < best_s:
            best, best_s = hp, float(np.mean(s))
    return best


@dataclass
class UtilityModel:
    """Predicts mean utility and its 10% / 90% quantiles."""

    kind: str = "lgbm"  # "lgbm" | "linear" | "constant"
    seed: int = 0
    hp: dict = field(default_factory=dict)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "UtilityModel":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        if self.kind == "constant" or X.shape[1] == 0:
            self.kind = "constant"
            self.c = (y.mean(), np.quantile(y, 0.1), np.quantile(y, 0.9))
            return self
        if self.kind == "linear":
            self.mean_ = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, y)
            self.q_ = {a: make_pipeline(StandardScaler(), QuantileRegressor(quantile=a, alpha=0.01, solver="highs"))
                       .fit(X, y) for a in (0.1, 0.9)}
            return self
        self.hp = _cv_select(lambda seed, **hp: _lgb_reg("regression", None, seed, **hp), X, y, groups,
                             lambda m, Xv, yv: mean_absolute_error(yv, m.predict(Xv)), self.seed)
        self.mean_ = _lgb_reg("regression", None, self.seed, **self.hp).fit(X, y)
        self.q_ = {a: _lgb_reg("quantile", a, self.seed, **self.hp).fit(X, y) for a in (0.1, 0.9)}
        return self

    def predict(self, X: np.ndarray) -> pd.DataFrame:
        X = np.asarray(X, float)
        if self.kind == "constant":
            n = len(X)
            return pd.DataFrame({"u_mean": np.full(n, self.c[0]), "u_q10": np.full(n, self.c[1]),
                                 "u_q90": np.full(n, self.c[2])})
        mean = self.mean_.predict(X)
        q10, q90 = self.q_[0.1].predict(X), self.q_[0.9].predict(X)
        return pd.DataFrame({"u_mean": mean, "u_q10": np.minimum(q10, mean), "u_q90": np.maximum(q90, mean)})


@dataclass
class HarmModel:
    """Predicts Pr(harm). Platt recalibration fitted on grouped out-of-fold predictions (training data only)."""

    kind: str = "lgbm"  # "lgbm" | "logistic" | "constant"
    seed: int = 0
    calibrate: bool = True
    hp: dict = field(default_factory=dict)

    def _make(self, **hp):
        if self.kind == "logistic":
            return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
        return _lgb_clf(self.seed, **hp)

    def fit(self, X: np.ndarray, h: np.ndarray, groups: np.ndarray) -> "HarmModel":
        X = np.asarray(X, float)
        h = np.asarray(h, int)
        if self.kind == "constant" or X.shape[1] == 0 or h.min() == h.max():
            self.kind = "constant"
            self.p = float(h.mean())
            return self
        if self.kind == "lgbm":
            self.hp = _cv_select(lambda seed, **hp: _lgb_clf(seed, **hp), X, h, groups,
                                 lambda m, Xv, yv: log_loss(yv, m.predict_proba(Xv)[:, 1], labels=[0, 1]),
                                 self.seed)
        self.model_ = self._make(**self.hp).fit(X, h)
        self.platt_ = None
        if self.calibrate:
            oof = np.zeros(len(h))
            for tr, va in grouped_folds(groups):
                if h[tr].min() == h[tr].max():
                    oof[va] = h[tr].mean()
                    continue
                oof[va] = self._make(**self.hp).fit(X[tr], h[tr]).predict_proba(X[va])[:, 1]
            z = np.log(np.clip(oof, 1e-4, 1 - 1e-4) / (1 - np.clip(oof, 1e-4, 1 - 1e-4)))
            self.platt_ = LogisticRegression(C=1e6).fit(z[:, None], h)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        if self.kind == "constant":
            return np.full(len(X), self.p)
        p = self.model_.predict_proba(X)[:, 1]
        if self.platt_ is not None:
            z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
            p = self.platt_.predict_proba(z[:, None])[:, 1]
        return p
