"""Bootstrap confidence intervals."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .labels import utility


def bootstrap_ci(values: np.ndarray, stat: Callable = np.mean, n: int = 10_000, level: float = 0.95,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Percentile bootstrap CI of ``stat`` over the first axis of ``values``."""
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    boots = np.array([stat(values[i]) for i in idx])
    a = (1 - level) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def utility_ci(Q_fresh, Q_warm, sigma_floor_frac: float = 0.01, n: int = 10_000,
               rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Bootstrap CI of the normalised utility, resampling matched repeats (pairs stay matched)."""
    pairs = np.column_stack([Q_fresh, Q_warm])
    return bootstrap_ci(pairs, lambda p: utility(p[:, 0], p[:, 1], sigma_floor_frac)[0], n=n, rng=rng)
