"""Resumable execution engine for library runs and paired target runs.

Each job writes one compressed file atomically (write to a temporary name,
then rename). Re-running skips jobs whose file exists, so a crashed or
interrupted campaign is resumed by running the same command again.

Library job  : one fresh DE run of one library scenario (the source run).
Target job   : one (target, repeat) with a shared fresh run and one warm run per
               candidate source; all runs share the initial population draw and
               the DE stream (common random numbers).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from ..config import provenance
from ..hydraulics.scenario import Scenario
from ..scenarios.generator import to_scenario
from ..seeds import run_streams, seed_sequence
from .optimizer import Evaluator, initial_population, run_de
from .records import hydraulic_signature, run_summary
from .runner import de_settings, get_simulator, parallel_map
from .transfer import inject_elites, n_injected, select_elites


def atomic_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    joblib.dump(obj, tmp, compress=3)
    os.replace(tmp, path)


def scenario_int(scenario_id: str) -> int:
    return int(scenario_id.rsplit("_", 1)[-1])


@dataclass
class RunOutput:
    summary: dict
    bsf: np.ndarray            # best-so-far after each evaluation (float64, length B)
    gen0_J: np.ndarray         # objective of every generation-0 member, in population order
    gen0_feasible: np.ndarray
    generations: list          # generation log rows
    best_x: np.ndarray
    final_pop: np.ndarray
    final_fit: np.ndarray


def _execute(ev: Evaluator, pop0: np.ndarray, r_de, N, F, CR) -> RunOutput:
    t0 = time.perf_counter()
    out = run_de(ev, N, F, CR, pop0, r_de)
    wall = time.perf_counter() - t0
    summ = run_summary(ev, out)
    summ["wall_s"] = wall
    return RunOutput(summ, ev.best_so_far.copy(), ev.J[:N].copy(), ev.feasible[:N].copy(), out.generations, ev.best_x.copy(),
                     out.pop.copy(), out.fit.copy())


# ---------------------------------------------------------------- library runs
def library_job(cfg: dict, row: pd.Series) -> dict:
    sim = get_simulator(cfg)
    N, F, CR, B = de_settings(cfg)
    sc = to_scenario(row)
    key = ("library_run", scenario_int(sc.scenario_id))
    ss = seed_sequence(cfg["seed"], *key)
    r_init, r_de, _ = run_streams(ss)
    ev = Evaluator(sim, sc, B)
    run = _execute(ev, initial_population(sim.lo, sim.hi, N, r_init), r_de, N, F, CR)
    return {
        "scenario": sc.to_dict(),
        "seed_key": [int(cfg["seed"]), *key],
        "de": {"N": N, "F": F, "CR": CR, "B": B},
        "summary": run.summary,
        "generations": run.generations,
        "archive_X": ev.X.copy(),
        "archive_J": ev.J.copy(),
        "archive_feasible": ev.feasible.copy(),
        "bsf": run.bsf,
        "best_x": run.best_x,
        "final_pop": run.final_pop,
        "final_fit": run.final_fit,
        "signature": hydraulic_signature(sim, sc, ev.best_x),
        "provenance": provenance(cfg),
    }


def library_elites(cfg: dict, rec: dict) -> tuple[np.ndarray, np.ndarray, int]:
    sim = get_simulator(cfg)
    t = cfg["transfer"]
    return select_elites(rec["archive_X"], rec["archive_J"], rec["archive_feasible"], int(t["k"]),
                         float(t["min_elite_distance_frac"]), sim.lo, sim.hi)


# ---------------------------------------------------------------- target runs
def paired_runs(cfg: dict, target: Scenario, ss: np.random.SeedSequence, sources: dict[str, np.ndarray],
                transfer: dict | None = None, extra_inits: dict[str, Callable] | None = None) -> dict:
    """Fresh run plus one warm run per source, with common random numbers.

    ``extra_inits`` maps a method name to ``f(pop0, rng_inj) -> pop`` for
    baselines that build their own initial population (B2-gamma, B4).
    """
    sim = get_simulator(cfg)
    N, F, CR, B = de_settings(cfg)
    t = transfer or cfg["transfer"]
    r_init, _, _ = run_streams(ss)
    pop0 = initial_population(sim.lo, sim.hi, N, r_init)

    def de_stream():
        return run_streams(ss)[1]

    def inj_stream():
        return run_streams(ss)[2]

    runs = {"fresh": _execute(Evaluator(sim, target, B), pop0, de_stream(), N, F, CR)}
    for sid, elites in sources.items():
        pop_w = inject_elites(pop0, elites, float(t["rho"]), float(t["sigma_frac"]), sim.lo, sim.hi, inj_stream())
        runs[f"warm:{sid}"] = _execute(Evaluator(sim, target, B), pop_w, de_stream(), N, F, CR)
    for name, init in (extra_inits or {}).items():
        runs[name] = _execute(Evaluator(sim, target, B), init(pop0, inj_stream()), de_stream(), N, F, CR)
    return {"runs": runs, "m_injected": n_injected(N, float(t["rho"]))}


def target_job(cfg: dict, row: pd.Series, repeat: int, sources: dict[str, np.ndarray]) -> dict:
    sc = to_scenario(row)
    key = ("target_run", scenario_int(sc.scenario_id), repeat)
    ss = seed_sequence(cfg["seed"], *key)
    res = paired_runs(cfg, sc, ss, sources)
    res.update({"target_id": sc.scenario_id, "repeat": repeat, "seed_key": [int(cfg["seed"]), *key],
                "provenance": provenance(cfg)})
    return res


def run_jobs(jobs: list[tuple[Path, Callable, tuple]], jobs_parallel: int | None = None) -> int:
    """Execute (path, fn, args) jobs whose output file does not exist yet. Returns the number run."""
    todo = [j for j in jobs if not j[0].exists()]
    if not todo:
        return 0

    def work(job):
        path, fn, args = job
        atomic_dump(fn(*args), path)
        return str(path)

    parallel_map(work, todo, jobs=jobs_parallel)
    return len(todo)
