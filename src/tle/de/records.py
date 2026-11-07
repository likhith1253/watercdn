"""Run records: hydraulic signatures and per-run summaries (Appendix D schemas)."""
from __future__ import annotations

import numpy as np

from ..hydraulics.scenario import Scenario
from ..hydraulics.simulator import Simulator
from .optimizer import DEResult, Evaluator


def hydraulic_signature(sim: Simulator, scenario: Scenario, x: np.ndarray) -> dict:
    """Network-normalised hydraulic signature of a decision (one extra simulation, not budgeted).

    Pressures are expressed as fractions of the required pressure, volumes as
    fractions of the delivered consumer demand, so that signatures are
    comparable across networks (E4).
    """
    sim.set_scenario(scenario)
    r = sim.simulate(x, keep_pressures=True)
    if r["failed"]:
        return {"sig_failed": 1.0}
    preq = scenario.required_pressure_m
    W = sim.spec.windows
    S = sim.decode(x)
    n_p = len(sim.pump_idx)
    speeds, prv = S[:n_p], S[n_p:]
    lo, hi = sim.decode(sim.lo), sim.decode(sim.hi)
    at_bound = (np.isclose(S, lo, atol=0.01 * (hi - lo))) | (np.isclose(S, hi, atol=0.01 * (hi - lo)))
    dem = max(r["demand_m3"], 1e-9)
    n_dem = r["pressures"].shape[1]
    window_h = sim.spec.horizon_h / W
    sig = {
        "sig_failed": 0.0,
        "sig_p_min": r["p_min"] / preq,
        "sig_p_p5": r["p_p5"] / preq,
        "sig_p_median": r["p_median"] / preq,
        "sig_leak_frac": r["leak_m3"] / dem,
        "sig_energy_per_m3": r["energy_kwh"] / dem,
        "sig_energy_cost_per_m3": r["energy_cost"] / dem,
        "sig_shortfall_norm": r["shortfall_mh"] / (n_dem * sim.spec.horizon_h * preq),
        "sig_tank_deficit_frac": r["tank_deficit_m3"] / dem,
        "sig_speed_mean": float(speeds.mean()),
        "sig_speed_std": float(speeds.std()),
        "sig_prv_mean": float(prv.mean() / preq) if prv.size else 0.0,
        "sig_prv_std": float(prv.std() / preq) if prv.size else 0.0,
        "sig_frac_at_bounds": float(at_bound.mean()),
        "sig_feasible": float(r["feasible"]),
    }
    for w in range(W):
        sig[f"sig_shortfall_w{w}"] = r["shortfall_by_window"][w] / (n_dem * window_h * preq)
        sig[f"sig_leak_w{w}"] = r["leak_by_window"][w] / dem
    return sig


def run_summary(ev: Evaluator, out: DEResult) -> dict:
    """Final-quality fields of the run table (one row per run)."""
    b = ev.best_result
    return {
        "final_J": float(ev.best_so_far[-1]),
        "J_leak": b["J_leak"], "J_energy": b["J_energy"], "J_shortfall": b["J_shortfall"], "J_tank": b["J_tank"],
        "leak_m3": b["leak_m3"], "energy_cost": b["energy_cost"], "shortfall_mh": b["shortfall_mh"],
        "tank_deficit_m3": b["tank_deficit_m3"], "best_p_min": b["p_min"],
        "feasible": bool(b["feasible"]),
        "n_failed_sims": int(ev.n_failed),
        "n_evals": int(ev.count),
        "n_generations": len(out.generations) - 1,
    }
