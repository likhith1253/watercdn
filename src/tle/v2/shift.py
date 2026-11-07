"""V2-B: shift-aware gate (docs/v2_plan.md).

Covariate-shift correction of the calibration harm rate. A logistic domain classifier separates calibration
targets (label 0) from the new, unlabelled targets (label 1) using their decision features; the density ratio
w(x) = p(new | x) / p(cal | x) * n_cal / n_new re-weights each calibration point (not clipped). For each
threshold the weighted harm rate is tested against alpha with a binomial tail evaluated at the Kish effective
sample size n_eff = min((sum w)^2 / sum w^2, sum w) and k_eff = n_eff * weighted harm rate (continuous extension through the
regularised incomplete beta function). Under strong shift n_eff collapses and the gate abstains.

This is an approximation of weighted conformal risk control, not an exact finite-sample guarantee; it is
reported as such.
"""
from __future__ import annotations

import numpy as np
from scipy.special import betainc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..models.gate import min_certifiable


def domain_weights(X_cal: np.ndarray, X_new: np.ndarray, seed: int, w_max: float | None = None) -> tuple[np.ndarray, float]:
    """Density-ratio weights for calibration points and the cross-validated domain AUC (0.5 = no shift)."""
    X = np.vstack([X_cal, X_new])
    y = np.r_[np.zeros(len(X_cal)), np.ones(len(X_new))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))
    k = int(min(5, len(X_cal), len(X_new)))
    p_cv = cross_val_predict(clf, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                             method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, p_cv))
    p = np.clip(clf.fit(X, y).predict_proba(X_cal)[:, 1], 1e-6, 1 - 1e-6)
    w = p / (1 - p) * len(X_cal) / len(X_new)
    # no clipping by default: clipping would hide a lack of overlap, which n_eff must reveal
    return (np.clip(w, 0.0, w_max) if w_max is not None else w), auc


def binom_cdf_cont(k: float, n: float, a: float) -> float:
    """P(X <= k) for X ~ Bin(n, a), extended to real k, n through I_{1-a}(n - k, k + 1)."""
    if k >= n:
        return 1.0
    return float(betainc(n - k, k + 1, 1 - a))


def weighted_threshold(p_hat: np.ndarray, harm: np.ndarray, w: np.ndarray, alpha: float, delta: float) -> float | None:
    """Fixed-sequence threshold on the weighted harm rate with Kish effective sample size."""
    p_hat, harm, w = np.asarray(p_hat, float), np.asarray(harm, float), np.asarray(w, float)
    n_min = min_certifiable(alpha, delta)
    best = None
    for lam in np.sort(np.unique(p_hat)):
        acc = p_hat <= lam
        ww = w[acc]
        if ww.sum() <= 0:
            continue
        # Kish effective size, capped by the weight mass: with a correctly normalised density ratio (mean 1 under
        # the calibration distribution) sum(w) estimates how many calibration points stand in for new-domain
        # cases, so many tiny weights cannot pose as a large sample when overlap is poor
        n_eff = min(ww.sum() ** 2 / (ww ** 2).sum(), ww.sum())
        if n_eff < n_min:
            continue
        k_eff = n_eff * float((ww * harm[acc]).sum() / ww.sum())
        if binom_cdf_cont(k_eff, n_eff, alpha) <= delta:
            best = float(lam)
        else:
            break
    return best
