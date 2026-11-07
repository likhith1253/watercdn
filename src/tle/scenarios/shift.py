"""Shifted target scenarios for E2 (goal change) and E3 (layout change).

Sources always come from the unchanged E1 library; only the targets change.
E2: required service pressure 20 -> 25 m (G1) or energy weight x2 (G2).
E3: shift level l = number of closed pipes (1, 2, 3), drawn from the 40 pipes
with the largest mean flow under the reference decision whose closure (jointly)
keeps every junction connected to a tank or reservoir. The decision vector keeps
its dimension. Pipe additions and moved PRVs are not used (closures only).
"""
from __future__ import annotations

import json
from dataclasses import replace

import networkx as nx
import numpy as np
import pandas as pd
import wntr

from ..config import repo_path
from ..hydraulics.scenario import Scenario
from ..seeds import rng
from .generator import sample_scenarios

GOALS = {"G1": {"required_pressure_m": 25.0}, "G2": {"energy_weight_mult": 2.0}}


def closure_candidates(cfg: dict, n_top: int = 40) -> list[str]:
    wn = wntr.network.WaterNetworkModel(str(repo_path(cfg["network"]["inp"])))
    meta = json.loads(repo_path(cfg["network"]["meta"]).read_text())
    protected = set(meta["edits"]["closed_pipes"]) | {p["prv"] for p in meta["edits"]["prvs"]}
    protected |= {name for p in meta["edits"]["prvs"] for name in wn.get_links_for_node(p["outlet_node"])}
    for n in wn.tank_name_list + wn.reservoir_name_list:
        protected |= set(wn.get_links_for_node(n))
    res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(repo_path("results/tmp/e3_flows")))
    flow = res.link["flowrate"].abs().mean()
    open_pipes = [p for p in wn.pipe_name_list if p not in protected and wn.get_link(p).initial_status != 0]
    ranked = sorted(open_pipes, key=lambda p: -flow[p])
    g = _graph(wn, set(meta["edits"]["closed_pipes"]))
    return [p for p in ranked if _connected_after(wn, g, {p})][:n_top]


def _graph(wn, closed: set[str]) -> nx.MultiGraph:
    g = nx.MultiGraph()
    g.add_nodes_from(wn.node_name_list)
    for name, link in wn.links():
        if name not in closed:
            g.add_edge(link.start_node_name, link.end_node_name, key=name)
    return g


def _connected_after(wn, g: nx.MultiGraph, close: set[str]) -> bool:
    h = g.copy()
    for p in close:
        L = wn.get_link(p)
        h.remove_edge(L.start_node_name, L.end_node_name, key=p)
    src = set(wn.tank_name_list) | set(wn.reservoir_name_list)
    reach = set()
    for s in src:
        reach |= nx.node_connected_component(h, s)
    return set(wn.junction_name_list) <= reach


def layout_edits(cfg: dict, n: int, level: int, master: int, idx: int) -> list[tuple[str, ...]]:
    wn = wntr.network.WaterNetworkModel(str(repo_path(cfg["network"]["inp"])))
    meta = json.loads(repo_path(cfg["network"]["meta"]).read_text())
    g = _graph(wn, set(meta["edits"]["closed_pipes"]))
    cands = closure_candidates(cfg)
    r = rng(master, "scenarios", 100 + idx)
    out = []
    while len(out) < n:
        pick = tuple(sorted(r.choice(cands, size=level, replace=False)))
        if _connected_after(wn, g, set(pick)):
            out.append(pick)
    return out


def shifted_targets(cfg: dict, kind: str, n_per_group: int, master: int, start: int,
                    cal_share: float) -> pd.DataFrame:
    """Targets for E2 (kind='goal') or E3 (kind='layout'); groups are goals or shift levels."""
    groups = list(GOALS) if kind == "goal" else [1, 2, 3]
    frames = []
    for gi, grp in enumerate(groups):
        base = sample_scenarios(cfg, n_per_group, master + 1 + gi, prefix=f"{cfg['experiment'].lower()}_",
                                start=start + gi * 1000)
        rows = []
        edits = layout_edits(cfg, n_per_group, grp, master, gi) if kind == "layout" else None
        for i, (_, row) in enumerate(base.iterrows()):
            sc = Scenario.from_dict(row.to_dict())
            if kind == "goal":
                sc = replace(sc, **GOALS[grp])
            else:
                sc = replace(sc, closed_pipes=edits[i])
            d = sc.to_dict()
            d["closed_pipes"] = ";".join(d["closed_pipes"])
            d["zone"] = row.zone
            d["group"] = grp
            rows.append(d)
        df = pd.DataFrame(rows)
        n_cal = int(round(cal_share * len(df)))
        df["split"] = ["calibration"] * n_cal + ["test"] * (len(df) - n_cal)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
