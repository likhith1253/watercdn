"""Run-based baseline jobs and the inputs they need (built from the library, before any target decision)."""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from ..config import provenance, repo_path
from ..de.jobs import _execute, scenario_int
from ..de.optimizer import Evaluator, initial_population
from ..de.records import run_summary
from ..de.runner import de_settings, get_simulator
from ..features.tiers import cosine, node_features
from ..scenarios.generator import to_scenario
from ..scenarios.space import network_info
from ..seeds import run_streams, seed_sequence
from .run_based import TaskModel, pmto_init, run_bckt, seeto_init

SEETO_GAMMA = 5          # top-gamma sources (SEETO paper)
SEETO_TEMPERATURE = 0.065  # softmax temperature Gamma (SEETO paper)
PMTO_KEEP = 0.7          # top 70% tasks train the task model (PMTO paper)
BCKT_DELTA = 10          # evaluations between competitions (BCKT paper)


def theta_matrix(cfg: dict, scen: pd.DataFrame) -> pd.DataFrame:
    """PMTO task parameters: scenario factors plus physical leak-node descriptors (no node ids)."""
    nf = node_features(network_info(cfg))
    th = pd.DataFrame({
        "log_area": np.log(scen.leak_area_m2.to_numpy()), "demand_mult": scen.demand_mult.to_numpy(),
        "peak_shift_h": scen.peak_shift_h.to_numpy(), "pattern_amp": scen.pattern_amp.to_numpy(),
        "tank_level_frac": scen.tank_level_frac.to_numpy(),
    }, index=scen.scenario_id.to_numpy())
    for c in ("elev_z", "dist_feeder_rel", "in_prv_zone"):
        th[f"node_{c}"] = nf.loc[scen.leak_node, c].to_numpy()
    return th


def fit_task_model(cfg: dict, scen: pd.DataFrame, ref: pd.DataFrame) -> TaskModel:
    root = repo_path(cfg["results_dir"]) / "jobs" / "library"
    lib = scen[scen.split == "library"]
    th = theta_matrix(cfg, lib)
    att = ref.set_index("scenario_id")
    X, gap = [], []
    for sid in lib.scenario_id:
        rec = joblib.load(root / f"{sid}.joblib")
        X.append(rec["best_x"])
        q_long = att.loc[sid, "attain_Q_long"]
        gap.append((rec["summary"]["final_J"] - q_long) / abs(q_long))
    return TaskModel(th.loc[lib.scenario_id].to_numpy(), np.array(X), np.array(gap), PMTO_KEEP, int(cfg["seed"]))


def baseline_job(cfg: dict, row: pd.Series, repeat: int, archives: list[tuple], sims: np.ndarray,
                 x_hat: np.ndarray) -> dict:
    sim = get_simulator(cfg)
    N, F, CR, B = de_settings(cfg)
    t = cfg["transfer"]
    sc = to_scenario(row)
    ss = seed_sequence(cfg["seed"], "target_run", scenario_int(sc.scenario_id), repeat)  # same as paired runs
    r_init, _, _ = run_streams(ss)
    pop0 = initial_population(sim.lo, sim.hi, N, r_init)
    runs = {}
    inits = {"B2g": seeto_init(archives, sims, SEETO_GAMMA, SEETO_TEMPERATURE, float(t["rho"]), sim.lo, sim.hi,
                               float(t["min_elite_distance_frac"])),
             "B4": pmto_init(x_hat, float(t["rho"]), float(t["sigma_frac"]), sim.lo, sim.hi)}
    for name, init in inits.items():
        runs[name] = _execute(Evaluator(sim, sc, B), init(pop0, run_streams(ss)[2]), run_streams(ss)[1], N, F, CR)
    ev = Evaluator(sim, sc, B)
    rng_b = np.random.default_rng(seed_sequence(cfg["seed"], "baseline", scenario_int(sc.scenario_id), repeat))
    t0 = time.perf_counter()
    out, log = run_bckt(ev, N, F, CR, pop0, run_streams(ss)[1], archives, rng_b, BCKT_DELTA, int(cfg["seed"]))
    summ = run_summary(ev, out)
    summ["wall_s"] = time.perf_counter() - t0
    summ.update({f"bckt_{k}": v for k, v in log.items() if k != "transfers_by_source"})
    runs["B3"] = {"summary": summ, "bsf": ev.best_so_far.copy()}
    res = {k: ({"summary": v.summary, "bsf": v.bsf} if not isinstance(v, dict) else v) for k, v in runs.items()}
    return {"target_id": sc.scenario_id, "repeat": repeat, "runs": res, "seeto_sims": np.asarray(sims),
            "pmto_x_hat": x_hat, "provenance": provenance(cfg)}


def target_inputs(cfg: dict, pairs: pd.DataFrame, ref: pd.DataFrame, tid: str) -> tuple[list, np.ndarray, list]:
    root = repo_path(cfg["results_dir"]) / "jobs" / "library"
    R = ref.set_index("scenario_id")
    src = list(pairs[pairs.target_id == tid].source_id)
    archives = []
    for s in src:
        rec = joblib.load(root / f"{s}.joblib")
        archives.append((rec["archive_X"], rec["archive_J"], rec["archive_feasible"]))
    sims = np.array([cosine(R.loc[tid, "state_vector"], R.loc[s, "state_vector"]) for s in src])
    return archives, sims, src
