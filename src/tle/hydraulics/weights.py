"""Objective-weight normalisation rule (Phase 1, step 7), applied once and frozen.

1. Evaluate n random decisions on the base scenario with unit weights.
2. w_leak = 1 / median(leakage), w_energy = 1 / median(energy cost): each of the
   two operating terms contributes about 1 for a typical decision.
3. Penalty weights: let R be the range (max - min) of the weighted operating
   objective over the random sample. For each violation term v (shortfall,
   tank deficit), w_v = 2 R / q10(v | v > tol_v), where q10 is the 10th
   percentile of the values above the feasibility tolerance tol_v (values
   below it are numerical noise and count as feasible). Any violation larger
   than that small quantile costs more than twice the whole spread of the
   operating objective, so feasible solutions beat all but marginally
   infeasible ones. If fewer than 10 random samples violate a term (on Net3
   the tank balance is almost never violated by random decisions, which run
   the pumps fast enough to fill the tanks), the reference violation is
   10 x the feasibility tolerance of that term instead of the quantile.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .scenario import Scenario
from .simulator import ObjectiveWeights, ProblemSpec, Simulator


def normalise_weights(spec: ProblemSpec, scenario: Scenario, n: int, seed: int) -> dict:
    unit = replace(spec, weights=ObjectiveWeights(1.0, 1.0, 1.0, 1.0))
    sim = Simulator(unit)
    try:
        sim.set_scenario(scenario)
        rng = np.random.default_rng(seed)
        lo, hi = sim.lo, sim.hi
        rows = [sim.simulate(lo + rng.random(len(lo)) * (hi - lo)) for _ in range(n)]
    finally:
        sim.close()
    ok = [r for r in rows if not r["failed"]]
    leak = np.array([r["leak_m3"] for r in ok])
    energy = np.array([r["energy_cost"] for r in ok])
    short = np.array([r["shortfall_mh"] for r in ok])
    tank = np.array([r["tank_deficit_m3"] for r in ok])
    w_leak = 1.0 / np.median(leak)
    w_energy = 1.0 / np.median(energy)
    op = w_leak * leak + w_energy * energy
    R = float(op.max() - op.min())
    s_pos = short[short > spec.shortfall_tol_mh]
    t_pos = tank[tank > spec.tank_tol_m3]
    q_short = float(np.quantile(s_pos, 0.10)) if len(s_pos) >= 10 else 10 * spec.shortfall_tol_mh
    q_tank = float(np.quantile(t_pos, 0.10)) if len(t_pos) >= 10 else 10 * spec.tank_tol_m3
    weights = {
        "leak": float(w_leak),
        "energy": float(w_energy),
        "shortfall": float(2 * R / q_short),
        "tank": float(2 * R / q_tank),
    }
    stats = {
        "n_samples": n,
        "n_failed": n - len(ok),
        "median_leak_m3": float(np.median(leak)),
        "median_energy_cost": float(np.median(energy)),
        "median_shortfall_mh": float(np.median(short)),
        "median_tank_deficit_m3": float(np.median(tank)),
        "operating_range_R": R,
        "q10_violating_shortfall_mh": q_short,
        "q10_violating_tank_deficit_m3": q_tank,
        "n_violating_shortfall": int(len(s_pos)),
        "n_violating_tank": int(len(t_pos)),
        "share_feasible": float(np.mean([r["feasible"] for r in ok])),
    }
    return {"weights": weights, "stats": stats}
