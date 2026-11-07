"""Differential Evolution (DE/rand/1/bin) with exact evaluation budgets.

Budget counting lives in :class:`Evaluator`; no method can exceed B. All random
draws come from explicitly passed ``numpy.random.Generator`` objects. The
number of draws per trial vector does not depend on objective values, so a
fresh and a warm run that share the DE stream consume it identically (common
random numbers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..hydraulics.scenario import Scenario
from ..hydraulics.simulator import Simulator


class BudgetExceeded(RuntimeError):
    pass


class Evaluator:
    """Counts every simulation and keeps the archive and best-so-far trace."""

    def __init__(self, sim: Simulator, scenario: Scenario, budget: int):
        self.sim = sim
        self.scenario = scenario
        self.budget = int(budget)
        self.count = 0
        self.n_failed = 0
        D = len(sim.lo)
        self.X = np.empty((self.budget, D))
        self.J = np.empty(self.budget)
        self.feasible = np.zeros(self.budget, bool)
        self.violation = np.empty(self.budget)
        self.best_so_far = np.empty(self.budget)
        self._best = np.inf
        self.best_result: dict | None = None
        self.best_x: np.ndarray | None = None
        sim.set_scenario(scenario)

    @property
    def remaining(self) -> int:
        return self.budget - self.count

    def __call__(self, x: np.ndarray) -> dict:
        if self.count >= self.budget:
            raise BudgetExceeded(f"budget {self.budget} exceeded")
        if self.sim.scenario is not self.scenario:
            self.sim.set_scenario(self.scenario)
        r = self.sim.simulate(x)
        i = self.count
        self.X[i] = x
        self.J[i] = r["J"]
        self.feasible[i] = r["feasible"]
        self.violation[i] = r["violation"]
        self.n_failed += int(r["failed"])
        if r["J"] < self._best:
            self._best = r["J"]
            self.best_result = r
            self.best_x = np.array(x, float)
        self.best_so_far[i] = self._best
        self.count += 1
        return r


def reflect(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Reflect out-of-bound components back into [lo, hi] (then clip for safety)."""
    v = np.where(v < lo, lo + (lo - v), v)
    v = np.where(v > hi, hi - (v - hi), v)
    return np.clip(v, lo, hi)


def initial_population(lo: np.ndarray, hi: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    return lo + rng.random((N, len(lo))) * (hi - lo)


def population_diversity(pop: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Mean distance to the centroid in range-normalised space, divided by sqrt(D)."""
    z = (pop - lo) / (hi - lo)
    return float(np.linalg.norm(z - z.mean(0), axis=1).mean() / np.sqrt(pop.shape[1]))


@dataclass
class DEResult:
    pop: np.ndarray
    fit: np.ndarray
    results: list
    generations: list = field(default_factory=list)  # one dict per generation


def run_de(
    ev: Evaluator,
    N: int,
    F: float,
    CR: float,
    pop0: np.ndarray,
    rng_de: np.random.Generator,
    on_generation: Callable[[int, np.ndarray], None] | None = None,
    trial_hook: Callable[[np.ndarray, int, np.ndarray, np.ndarray], np.ndarray] | None = None,
    after_eval: Callable[[np.ndarray, dict], None] | None = None,
) -> DEResult:
    """DE/rand/1/bin. Generation 0 evaluates all N members of ``pop0``; the run stops at exactly B.

    ``trial_hook(trial, i, pop, fit)`` may replace a trial vector before it is
    evaluated (used by the BCKT-style baseline); ``after_eval(x, result)`` is
    called after every evaluation (generation 0 included). Neither
    changes the DE random stream.
    """
    lo, hi = ev.sim.lo, ev.sim.hi
    D = len(lo)
    if pop0.shape != (N, D):
        raise ValueError("pop0 has the wrong shape")
    if ev.budget < N:
        raise ValueError("budget smaller than the population size")
    pop = pop0.copy()
    res = []
    for x in pop:
        res.append(ev(x))
        if after_eval is not None:
            after_eval(x, res[-1])
    fit = np.array([r["J"] for r in res])
    out = DEResult(pop, fit, res)
    _log_generation(out, 0, ev, lo, hi)
    if on_generation:
        on_generation(0, pop)
    gen = 0
    others = [np.array([j for j in range(N) if j != i]) for i in range(N)]
    while ev.count < ev.budget:
        for i in range(N):
            if ev.count >= ev.budget:
                break
            a, b, c = rng_de.choice(others[i], 3, replace=False)
            mutant = reflect(pop[a] + F * (pop[b] - pop[c]), lo, hi)
            cross = rng_de.random(D) < CR
            cross[rng_de.integers(D)] = True
            trial = np.where(cross, mutant, pop[i])
            if trial_hook is not None:
                trial = trial_hook(trial, i, pop, fit)
            r = ev(trial)
            if after_eval is not None:
                after_eval(trial, r)
            if r["J"] <= fit[i]:
                pop[i], fit[i], res[i] = trial, r["J"], r
        gen += 1
        _log_generation(out, gen, ev, lo, hi)
        if on_generation:
            on_generation(gen, pop)
    return out


def _log_generation(out: DEResult, gen: int, ev: Evaluator, lo, hi) -> None:
    fit = out.fit
    feas = np.array([r["feasible"] for r in out.results])
    viol = np.array([r["violation"] for r in out.results])
    ib = int(np.argmin(fit))
    best = out.results[ib]
    prev = out.generations[-5]["best_J"] if len(out.generations) >= 5 else (
        out.generations[0]["best_J"] if out.generations else fit[ib])
    out.generations.append(
        {
            "gen": gen,
            "evals": ev.count,
            "best_J": float(fit[ib]),
            "mean_J": float(fit.mean()),
            "std_J": float(fit.std()),
            "diversity": population_diversity(out.pop, lo, hi),
            "share_feasible": float(feas.mean()),
            "mean_violation": float(viol.mean()),
            "improvement_rate": float((prev - fit[ib]) / max(abs(prev), 1e-12)),
            "best_min_pressure": best["p_min"],
            "best_leakage": best["leak_m3"],
            "best_energy": best["energy_cost"],
        }
    )
