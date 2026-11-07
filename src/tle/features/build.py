"""Assemble the pair-level feature table (tiers a-c) and the per-repeat probe table (tier d)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from ..config import repo_path
from ..de.jobs import library_elites
from ..de.runner import de_settings, get_simulator
from ..scenarios.generator import ScenarioDistance, read_scenarios
from ..scenarios.space import network_info
from .tiers import tier_a, tier_b, tier_c, tier_d

PROBE_SIZES = (1, 2, 4, 8)


def source_elites(cfg: dict, source_ids) -> tuple[dict, dict]:
    root = repo_path(cfg["results_dir"]) / "jobs" / "library"
    E, EJ = {}, {}
    for sid in source_ids:
        e, ej, _ = library_elites(cfg, joblib.load(root / f"{sid}.joblib"))
        E[sid], EJ[sid] = e, ej
    return E, EJ


def build_features(cfg: dict, pairs: pd.DataFrame, lib_runs: pd.DataFrame, gens: pd.DataFrame,
                   curves: pd.DataFrame, ref: pd.DataFrame, shuffle_rng: np.random.Generator | None = None
                   ) -> pd.DataFrame:
    scen = read_scenarios(cfg["scenarios"]["table"])
    sim = get_simulator(cfg)
    N = de_settings(cfg)[0]
    E, EJ = source_elites(cfg, lib_runs.scenario_id)
    fa = tier_a(pairs, scen, ScenarioDistance(cfg), network_info(cfg), ref)
    fb = tier_b(pairs, lib_runs, gens, curves, E, EJ, sim.lo, sim.hi, N, shuffle_rng)
    fc = tier_c(pairs, lib_runs, ref)
    return pd.concat([fa, fb, fc], axis=1).reset_index()


def build_probe(pairs: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for q in PROBE_SIZES:
        d = tier_d(pairs, runs, q)
        d.insert(1, "probe_q", q)
        out.append(d)
    return pd.concat(out, ignore_index=True)
