"""Real-coded genetic algorithm for the optimiser-robustness check (build guide P7.1 step 6).

Binary tournament selection, simulated binary crossover (SBX, eta_c = 15, p_c = 0.9), polynomial mutation
(eta_m = 20, p_m = 1/D), elitist (mu + lambda) survival. Generation 0 evaluates the N members of pop0, then
offspring are evaluated one by one until exactly B evaluations are spent. The number of random draws per
offspring does not depend on objective values except through tournament outcomes, which use fixed-shape draws,
so fresh and warm runs that share the stream stay aligned in their consumption.
"""
from __future__ import annotations

import numpy as np

from .optimizer import DEResult, Evaluator, _log_generation


def _sbx(a, b, lo, hi, eta, rng):
    u = rng.random(len(a))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * a + (1 - beta) * b)
    c2 = 0.5 * ((1 - beta) * a + (1 + beta) * b)
    return np.clip(c1, lo, hi), np.clip(c2, lo, hi)


def _pm(x, lo, hi, eta, pm, rng):
    u = rng.random(len(x))
    mask = rng.random(len(x)) < pm
    d = np.where(u < 0.5, (2 * u) ** (1 / (eta + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta + 1)))
    return np.clip(np.where(mask, x + d * (hi - lo), x), lo, hi)


def run_ga(ev: Evaluator, N: int, pop0: np.ndarray, rng: np.random.Generator, pc: float = 0.9,
           eta_c: float = 15.0, eta_m: float = 20.0) -> DEResult:
    lo, hi = ev.sim.lo, ev.sim.hi
    D = len(lo)
    pm = 1.0 / D
    pop = pop0.copy()
    res = [ev(x) for x in pop]
    fit = np.array([r["J"] for r in res])
    out = DEResult(pop, fit, res)
    _log_generation(out, 0, ev, lo, hi)
    gen = 0
    while ev.count < ev.budget:
        kids, kres = [], []
        while len(kids) < N and ev.count < ev.budget:
            i, j, k, l = rng.integers(0, N, 4)
            p1 = pop[i] if fit[i] <= fit[j] else pop[j]
            p2 = pop[k] if fit[k] <= fit[l] else pop[l]
            do_x = rng.random() < pc
            c1, c2 = _sbx(p1, p2, lo, hi, eta_c, rng)
            if not do_x:
                c1, c2 = p1.copy(), p2.copy()
            for c in (_pm(c1, lo, hi, eta_m, pm, rng), _pm(c2, lo, hi, eta_m, pm, rng)):
                if ev.count < ev.budget and len(kids) < N:
                    kids.append(c)
                    kres.append(ev(c))
        allx = np.vstack([pop, np.array(kids)])
        allr = res + kres
        allf = np.array([r["J"] for r in allr])
        keep = np.argsort(allf, kind="stable")[:N]
        pop, fit, res = allx[keep], allf[keep], [allr[t] for t in keep]
        out.pop, out.fit, out.results = pop, fit, res
        gen += 1
        _log_generation(out, gen, ev, lo, hi)
    return out
