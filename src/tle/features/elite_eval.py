"""Tier e: direct evaluation of the source elites under the target scenario (cost k evaluations, counted).

For each pair, the k elites of the source run are simulated in the target scenario and compared with the
target's reference decision (already evaluated for the state vector). Features describe how good, feasible and
how similar in operating character the source's best solutions are where they would be injected.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from ..config import repo_path
from ..de.jobs import library_elites, reference_decision
from ..de.runner import get_simulator, parallel_map
from ..scenarios.generator import read_scenarios, to_scenario


def _pair_job(args):
    cfg, target_row, src_id, elites = args
    sim = get_simulator(cfg)
    sc = to_scenario(target_row)
    sim.set_scenario(sc)
    ref = sim.simulate(reference_decision(sim))
    res = [sim.simulate(np.clip(e, sim.lo, sim.hi)) for e in elites]
    J = np.array([r["J"] for r in res])
    lr = np.log(J) - np.log(ref["J"])
    feas = np.array([r["feasible"] for r in res], float)
    short = np.array([r["shortfall_mh"] for r in res])
    tank = np.array([r["tank_deficit_m3"] for r in res])
    best = res[int(np.argmin(J))]
    return {
        "pair_id": f"{sc.scenario_id}__{src_id}",
        "e_logratio_min": float(lr.min()), "e_logratio_mean": float(lr.mean()), "e_logratio_spread": float(np.ptp(lr)),
        "e_feasible_share": float(feas.mean()),
        "e_shortfall_log_min": float(np.log1p(short.min())), "e_tank_log_min": float(np.log1p(tank.min())),
        "e_leak_ratio_best": float(best["leak_m3"] / max(ref["leak_m3"], 1e-9)),
        "e_energy_ratio_best": float(best["energy_cost"] / max(ref["energy_cost"], 1e-9)),
        "e_pmin_best": float(best["p_min"] / sc.required_pressure_m),
        "e_ref_feasible": float(ref["feasible"]),
    }


def elite_features(cfg: dict, pairs: pd.DataFrame, library_jobs_dir: str | None = None) -> pd.DataFrame:
    root = repo_path(library_jobs_dir or (repo_path(cfg["results_dir"]) / "jobs" / "library"))
    scen = read_scenarios(cfg["scenarios"]["table"]).set_index("scenario_id", drop=False)
    el = {s: library_elites(cfg, joblib.load(root / f"{s}.joblib"))[0] for s in pairs.source_id.unique()}
    jobs = [(cfg, scen.loc[p.target_id], p.source_id, el[p.source_id]) for _, p in pairs.iterrows()]
    return pd.DataFrame(parallel_map(_pair_job, jobs))
