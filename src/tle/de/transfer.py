"""The transfer action: elite selection from a source run and injection into a target population."""
from __future__ import annotations

import numpy as np


def normalised_distance(a: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """RMS distance per variable as a fraction of each variable's range (0 = identical, 1 = opposite corners)."""
    return np.sqrt(np.mean(((a - b) / (hi - lo)) ** 2, axis=-1))


def select_elites(X: np.ndarray, J: np.ndarray, feasible: np.ndarray, k: int, min_dist_frac: float,
                  lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Top-k distinct elites from a run archive.

    Candidates are all evaluated solutions (the archive contains the final
    population). Feasible solutions come first, ordered by J; infeasible ones
    only fill remaining slots. A candidate is skipped if it is closer than
    ``min_dist_frac`` (normalised RMS distance) to an elite already chosen.

    Returns (elites, their J, number of feasible elites).
    """
    order = np.lexsort((J, ~feasible))  # feasible first, then by J
    chosen: list[int] = []
    for i in order:
        if len(chosen) == k:
            break
        if chosen and np.min(normalised_distance(X[chosen], X[i], lo, hi)) < min_dist_frac:
            continue
        chosen.append(int(i))
    idx = np.array(chosen)
    return X[idx].copy(), J[idx].copy(), int(feasible[idx].sum())


def n_injected(N: int, rho: float) -> int:
    return int(round(rho * N))


def inject_elites(pop0: np.ndarray, elites: np.ndarray, rho: float, sigma_frac: float, lo: np.ndarray,
                  hi: np.ndarray, rng_inj: np.random.Generator) -> np.ndarray:
    """Overwrite the first round(rho*N) members of pop0 with perturbed elites (cycled), clipped to bounds.

    The noise draw has a fixed shape, so the injection stream is consumed
    identically whatever the elites are.
    """
    m = n_injected(len(pop0), rho)
    if m == 0:
        return pop0.copy()
    idx = np.arange(m) % len(elites)
    noise = sigma_frac * (hi - lo) * rng_inj.standard_normal((m, pop0.shape[1]))
    out = pop0.copy()
    out[:m] = np.clip(elites[idx] + noise, lo, hi)
    return out
