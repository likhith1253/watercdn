"""Scenario space: leak candidates, node features, the base (centre) scenario."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import repo_path
from ..hydraulics.network_prep import network_constants
from ..hydraulics.scenario import Scenario


@lru_cache(maxsize=4)
def _constants(inp: str, meta_file: str) -> dict:
    meta = json.loads(Path(meta_file).read_text())
    return network_constants(inp, meta)


def network_info(cfg: dict) -> dict:
    net = cfg["network"]
    meta = json.loads(repo_path(net["meta"]).read_text())
    const = _constants(str(repo_path(net["inp"])), str(repo_path(net["meta"])))
    return {"meta": meta, **const}


def leak_candidates(cfg: dict) -> list[str]:
    """All junctions except bulk offtakes and the artificial PRV outlet nodes."""
    info = network_info(cfg)
    exclude = set(info["meta"].get("bulk_nodes", []))
    exclude |= {p["outlet_node"] for p in info["meta"]["edits"]["prvs"]}
    return sorted(j for j in info["junctions"] if j not in exclude)


def base_scenario(cfg: dict) -> Scenario:
    """Centre of the scenario space; used for weight normalisation and DE tuning checks."""
    space = cfg["scenario_space"]
    info = network_info(cfg)
    cands = [j for j in leak_candidates(cfg) if info["junctions"][j]["zone"] == "main"]
    elev = np.array([info["junctions"][j]["elevation"] for j in cands])
    node = cands[int(np.argmin(np.abs(elev - np.median(elev))))]
    a_lo, a_hi = space["leak_area_m2"]
    return Scenario(
        scenario_id="base",
        network=cfg["network"]["name"],
        leak_node=node,
        leak_area_m2=float(np.sqrt(a_lo * a_hi)),
        demand_mult=float(np.mean(space["demand_mult"])),
        peak_shift_h=0,
        pattern_amp=float(np.mean(space["pattern_amp"])),
        tank_level_frac=float(np.mean(space["tank_level_frac"])),
        required_pressure_m=float(cfg.get("required_pressure_m", 20.0)),
    )
