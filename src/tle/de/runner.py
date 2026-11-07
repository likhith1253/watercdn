"""Run execution helpers: per-process simulator cache and parallel map."""
from __future__ import annotations

import os
from typing import Callable, Iterable

import numpy as np
from joblib import Parallel, delayed

from ..config import config_hash, problem_spec
from ..hydraulics.scenario import Scenario
from ..hydraulics.simulator import Simulator
from ..seeds import run_streams
from .optimizer import DEResult, Evaluator, initial_population, run_de

_SIMS: dict[str, Simulator] = {}


def get_simulator(cfg: dict) -> Simulator:
    """One simulator per (process, problem definition)."""
    key = config_hash({k: cfg[k] for k in ("network", "decision", "objective")})
    if key not in _SIMS:
        _SIMS[key] = Simulator(problem_spec(cfg))
    return _SIMS[key]


def de_settings(cfg: dict) -> tuple[int, float, float, int]:
    de = cfg["de"]
    missing = [k for k in ("N", "F", "CR", "budget_B") if de.get(k) is None]
    if missing:
        raise ValueError(f"DE settings not frozen yet: {missing}")
    return int(de["N"]), float(de["F"]), float(de["CR"]), int(de["budget_B"])


def run_fresh(cfg: dict, scenario: Scenario, ss: np.random.SeedSequence, N: int, F: float, CR: float,
              B: int) -> tuple[Evaluator, DEResult]:
    sim = get_simulator(cfg)
    r_init, r_de, _ = run_streams(ss)
    ev = Evaluator(sim, scenario, B)
    pop0 = initial_population(sim.lo, sim.hi, N, r_init)
    out = run_de(ev, N, F, CR, pop0, r_de)
    return ev, out


def n_jobs() -> int:
    return int(os.environ.get("TLE_JOBS", max(1, (os.cpu_count() or 2) - 1)))


def parallel_map(fn: Callable, items: Iterable, jobs: int | None = None, verbose: int = 0) -> list:
    """Process-parallel map (loky). Each worker owns its own EPANET project."""
    items = list(items)
    jobs = jobs or n_jobs()
    if jobs == 1:
        return [fn(it) for it in items]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return Parallel(n_jobs=jobs, backend="loky", verbose=verbose)(delayed(fn)(it) for it in items)
