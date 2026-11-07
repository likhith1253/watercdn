"""Label definitions (Section 1.2 of the build guide). Defined once, imported everywhere.

Sign convention: utility U > 0 means the warm start was better (lower J) than fresh.
"""
from __future__ import annotations

import numpy as np


def sigma_fresh(Q_fresh: np.ndarray, floor_frac: float = 0.01) -> float:
    """Seed-to-seed std of fresh final quality, floored at floor_frac * |mean|."""
    Q = np.asarray(Q_fresh, float)
    sd = Q.std(ddof=1) if len(Q) > 1 else 0.0
    return float(max(sd, floor_frac * abs(Q.mean()), 1e-12))


def utility(Q_fresh, Q_warm, sigma_floor_frac: float = 0.01) -> tuple[float, float]:
    """Normalised matched-repeat utility and its standard error.

    Q_*: final penalised objective of the R matched repeats (lower is better).
    """
    Q_fresh, Q_warm = np.asarray(Q_fresh, float), np.asarray(Q_warm, float)
    if Q_fresh.shape != Q_warm.shape:
        raise ValueError("fresh and warm repeats must be matched")
    sigma = sigma_fresh(Q_fresh, sigma_floor_frac)
    diff = Q_fresh - Q_warm  # positive = warm-start better
    U = diff.mean() / sigma
    se = diff.std(ddof=1) / (np.sqrt(len(diff)) * sigma) if len(diff) > 1 else float("nan")
    return float(U), float(se)


def raw_utility(Q_fresh, Q_warm) -> float:
    return float(np.mean(np.asarray(Q_fresh, float) - np.asarray(Q_warm, float)))


def harm_label(U: float, eps: float = 0.5) -> int:
    return int(U < -eps)


def target_quality(bsf_fresh: np.ndarray, fraction: float = 0.8) -> float:
    """Q* = median over fresh repeats of the best-so-far at fraction * B (fresh runs only)."""
    bsf_fresh = np.atleast_2d(bsf_fresh)
    B = bsf_fresh.shape[1]
    return float(np.median(bsf_fresh[:, int(round(fraction * B)) - 1]))


def evals_to_target(best_so_far: np.ndarray, target_Q: float, B: int | None = None) -> int:
    """First evaluation count at which best-so-far <= Q*; B if never reached (censored)."""
    bsf = np.asarray(best_so_far)
    B = len(bsf) if B is None else B
    hit = np.flatnonzero(bsf <= target_Q)
    return int(hit[0]) + 1 if hit.size else int(B)


def anytime_utility(bsf_fresh: np.ndarray, bsf_warm: np.ndarray, sigma: float) -> float:
    """Area between mean best-so-far curves (fresh minus warm) over evaluations 1..B, / (B * sigma)."""
    f = np.atleast_2d(bsf_fresh).mean(0)
    w = np.atleast_2d(bsf_warm).mean(0)
    return float((f - w).sum() / (len(f) * sigma))


def transfer_gain(E_fresh: float, E_warm: float) -> float:
    return float((E_fresh - E_warm) / E_fresh)


def pair_labels(bsf_fresh: np.ndarray, bsf_warm: np.ndarray, eps: float = 0.5, sigma_floor_frac: float = 0.01,
                target_fraction: float = 0.8) -> dict:
    """All labels of one pair from the R x B best-so-far matrices of matched repeats."""
    bsf_fresh, bsf_warm = np.atleast_2d(bsf_fresh), np.atleast_2d(bsf_warm)
    B = bsf_fresh.shape[1]
    Qf, Qw = bsf_fresh[:, -1], bsf_warm[:, -1]
    U, se = utility(Qf, Qw, sigma_floor_frac)
    sigma = sigma_fresh(Qf, sigma_floor_frac)
    Qstar = target_quality(bsf_fresh, target_fraction)
    Ef = np.mean([evals_to_target(b, Qstar, B) for b in bsf_fresh])
    Ew = np.mean([evals_to_target(b, Qstar, B) for b in bsf_warm])
    return {
        "n_repeats": int(len(Qf)),
        "U_mean": U,
        "U_se": se,
        "U_raw": raw_utility(Qf, Qw),
        "harm": harm_label(U, eps),
        "anytime_U": anytime_utility(bsf_fresh, bsf_warm, sigma),
        "E_fresh": float(Ef),
        "E_warm": float(Ew),
        "TG": transfer_gain(Ef, Ew),
        "Q_star": Qstar,
        "sigma_fresh": sigma,
        "Q_fresh_mean": float(Qf.mean()),
        "Q_warm_mean": float(Qw.mean()),
    }
