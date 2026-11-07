"""Risk-controlled transfer gate (Phase 5).

Decision for a target: compute the predicted harm probability p_j and the
utility lower bound u_j (10% quantile) of every candidate source; take the
source j* with the largest u_j; transfer if p_j* <= lambda, otherwise abstain
(fresh start).

lambda is chosen on calibration targets (one pair per target, so calibration
points are independent) with Learn-then-Test fixed-sequence testing: thresholds
are tested from strict to loose; at each, H0 "harm rate among accepted > alpha"
is rejected when the binomial tail P(X <= k), X ~ Bin(n, alpha), is <= delta.
The last threshold before the first non-rejection is chosen. With probability
>= 1 - delta over the calibration data, the harm rate among accepted transfers
on exchangeable new targets is at most alpha (conditional version).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom


def min_certifiable(alpha: float, delta: float) -> int:
    """Smallest number of accepted points that can certify alpha (zero harms): ceil(ln delta / ln(1 - alpha))."""
    return int(np.ceil(np.log(delta) / np.log(1 - alpha) - 1e-12))


def choose_threshold(p_hat: np.ndarray, harm: np.ndarray, alpha: float = 0.20, delta: float = 0.05) -> float | None:
    """Learn-then-Test fixed-sequence threshold. Returns lambda, or None (never transfer).

    The sequence starts at the first threshold that accepts at least
    ``min_certifiable(alpha, delta)`` calibration points: stricter thresholds
    can never be certified (even with zero harms their p-value exceeds delta),
    so testing them would stop the fixed sequence before any testable
    hypothesis. The order depends on p_hat only, never on the harm labels, so
    the fixed-sequence guarantee is unchanged (amendment A-001).
    """
    p_hat = np.asarray(p_hat, float)
    harm = np.asarray(harm, int)
    n_min = min_certifiable(alpha, delta)
    best = None
    for lam in np.sort(np.unique(p_hat)):  # strict -> loose
        acc = p_hat <= lam
        n, k = int(acc.sum()), int(harm[acc].sum())
        if n < n_min:
            continue
        pval = binom.cdf(k, n, alpha)  # H0: P(harm | accept) > alpha
        if pval <= delta:
            best = float(lam)
        else:
            break  # fixed-sequence stopping rule
    return best


def top_source(pred: pd.DataFrame, rank_col: str = "u_q10") -> pd.DataFrame:
    """One row per target: the candidate source with the largest rank score (ties: lowest p_harm)."""
    p = pred.sort_values(["target_id", rank_col, "p_harm"], ascending=[True, False, True])
    return p.groupby("target_id", as_index=False).head(1).reset_index(drop=True)


def decide(top: pd.DataFrame, lam: float | None, score_col: str = "p_harm", accept_if_leq: bool = True) -> np.ndarray:
    if lam is None:
        return np.zeros(len(top), bool)
    s = top[score_col].to_numpy()
    return s <= lam if accept_if_leq else s >= lam


def evaluate_decisions(top: pd.DataFrame, accept: np.ndarray, oracle: pd.Series, eps: float) -> dict:
    """Offline evaluation by lookup of measured pair outcomes (abstain = fresh = utility 0)."""
    U = np.where(accept, top.U_mean.to_numpy(), 0.0)
    harm_acc = top.harm.to_numpy()[accept]
    orc = oracle.loc[top.target_id].to_numpy()
    n_acc = int(accept.sum())
    return {
        "n_targets": int(len(top)),
        "n_accepted": n_acc,
        "abstention_rate": float(1 - accept.mean()),
        "harm_rate_among_accepted": float(harm_acc.mean()) if n_acc else 0.0,
        "n_harm_accepted": int(harm_acc.sum()),
        "mean_utility": float(U.mean()),
        "oracle_mean_utility": float(orc.mean()),
        "regret_to_oracle": float((orc - U).mean()),
        "share_oracle_captured": float(U.sum() / orc.sum()) if orc.sum() > 0 else float("nan"),
        "joint_harm_rate": float((accept & (top.harm.to_numpy() == 1)).mean()),
    }


def oracle_utility(pairs: pd.DataFrame) -> pd.Series:
    """Best measured action per target with hindsight: max(0 (fresh), max_j U_j)."""
    return pairs.groupby("target_id").U_mean.max().clip(lower=0.0)


def resplit_experiment(top_pool: pd.DataFrame, oracle: pd.Series, alphas, delta: float, n_cal: int,
                       n_splits: int, rng: np.random.Generator, eps: float, score_col: str = "p_harm") -> pd.DataFrame:
    """Pool calibration + test targets, re-split at random, calibrate on one part, evaluate on the other."""
    rows = []
    idx = np.arange(len(top_pool))
    for s in range(n_splits):
        perm = rng.permutation(idx)
        cal, ev = top_pool.iloc[perm[:n_cal]], top_pool.iloc[perm[n_cal:]]
        for a in alphas:
            lam = choose_threshold(cal[score_col].to_numpy(), cal.harm.to_numpy(), a, delta)
            res = evaluate_decisions(ev.reset_index(drop=True), decide(ev, lam, score_col), oracle, eps)
            rows.append({"split": s, "alpha": a, "lambda": lam, **res})
    return pd.DataFrame(rows)
