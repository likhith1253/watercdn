"""Scenario sampling, disjoint splits and the stratified pair manifest (Phase 3)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import wntr
from scipy.stats import qmc

from ..config import repo_path
from ..hydraulics.network_prep import network_graph
from ..hydraulics.scenario import Scenario
from ..seeds import rng
from .space import leak_candidates, network_info

SPLITS = ("library", "train", "calibration", "test")
ZONE_SHARES = {"main": 0.5, "A": 0.3, "B": 0.2}


def _largest_remainder(total: int, shares: dict[str, float]) -> dict[str, int]:
    raw = {k: total * v / sum(shares.values()) for k, v in shares.items()}
    out = {k: int(np.floor(v)) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def sample_scenarios(cfg: dict, n: int, master_seed: int, prefix: str = "s", start: int = 0) -> pd.DataFrame:
    """Scrambled Sobol over the continuous factors, stratified draw over leak zone and node."""
    space = cfg["scenario_space"]
    info = network_info(cfg)
    cands = leak_candidates(cfg)
    shares = space.get("zone_shares", ZONE_SHARES)
    by_zone = {z: [j for j in cands if info["junctions"][j]["zone"] == z] for z in shares}
    quotas = _largest_remainder(n, shares)
    r = rng(master_seed, "scenarios", 0)
    zones = np.array([z for z, q in quotas.items() for _ in range(q)])
    r.shuffle(zones)
    m = int(np.ceil(np.log2(max(n, 2))))
    U = qmc.Sobol(d=6, scramble=True, seed=rng(master_seed, "scenarios", 1)).random_base2(m)[:n]
    a_lo, a_hi = space["leak_area_m2"]
    rows = []
    for i in range(n):
        u = U[i]
        zone = zones[i]
        nodes = by_zone[zone]
        sh_lo, sh_hi = space["peak_shift_h"]
        rows.append(
            Scenario(
                scenario_id=f"{prefix}{start + i:04d}",
                network=cfg["network"]["name"],
                leak_node=nodes[min(int(u[5] * len(nodes)), len(nodes) - 1)],
                leak_area_m2=float(np.exp(np.log(a_lo) + u[0] * (np.log(a_hi) - np.log(a_lo)))),
                demand_mult=float(space["demand_mult"][0] + u[1] * np.ptp(space["demand_mult"])),
                peak_shift_h=int(min(sh_lo + int(u[2] * (sh_hi - sh_lo + 1)), sh_hi)),
                pattern_amp=float(space["pattern_amp"][0] + u[3] * np.ptp(space["pattern_amp"])),
                tank_level_frac=float(space["tank_level_frac"][0] + u[4] * np.ptp(space["tank_level_frac"])),
                required_pressure_m=float(cfg.get("required_pressure_m", 20.0)),
            ).to_dict()
        )
    df = pd.DataFrame(rows)
    df["zone"] = [info["junctions"][j]["zone"] for j in df.leak_node]
    df["closed_pipes"] = df.closed_pipes.map(lambda v: ";".join(v))
    return df


def assign_splits(df: pd.DataFrame, sizes: dict[str, int], master_seed: int) -> pd.DataFrame:
    """Four disjoint scenario sets, stratified by leak zone (largest-remainder allocation)."""
    if sum(sizes.values()) != len(df):
        raise ValueError("split sizes must add up to the number of scenarios")
    r = rng(master_seed, "splits", 0)
    split = pd.Series(index=df.index, dtype=object)
    remaining = dict(sizes)
    zones = list(df.zone.value_counts().index)
    for zi, zone in enumerate(zones):
        idx = df.index[df.zone == zone].to_numpy().copy()
        r.shuffle(idx)
        if zi == len(zones) - 1:
            alloc = dict(remaining)
        else:
            alloc = _largest_remainder(len(idx), {k: sizes[k] / len(df) for k in SPLITS})
            alloc = {k: min(alloc[k], remaining[k]) for k in SPLITS}
            short = len(idx) - sum(alloc.values())
            for k in SPLITS:
                while short > 0 and alloc[k] < remaining[k]:
                    alloc[k] += 1
                    short -= 1
        pos = 0
        for k in SPLITS:
            split.loc[idx[pos: pos + alloc[k]]] = k
            remaining[k] -= alloc[k]
            pos += alloc[k]
    out = df.copy()
    out["split"] = split.values
    assert not out.split.isna().any()
    return out


class ScenarioDistance:
    """Distance in scenario space (equal-weight RMS of normalised factor differences).

    Components: log leak area, demand multiplier, peak shift, pattern amplitude,
    tank level (each divided by its range), hydraulic distance between the leak
    nodes (shortest-path pipe length divided by the largest one between leak
    candidates) and a zone-mismatch indicator.
    """

    def __init__(self, cfg: dict):
        space = cfg["scenario_space"]
        self.ranges = {
            "log_area": np.log(space["leak_area_m2"][1]) - np.log(space["leak_area_m2"][0]),
            "demand_mult": float(np.ptp(space["demand_mult"])),
            "peak_shift_h": float(np.ptp(space["peak_shift_h"])),
            "pattern_amp": float(np.ptp(space["pattern_amp"])),
            "tank_level_frac": float(np.ptp(space["tank_level_frac"])),
        }
        wn = wntr.network.WaterNetworkModel(str(repo_path(cfg["network"]["inp"])))
        g = network_graph(wn)
        cands = leak_candidates(cfg)
        self.sp = {n: nx.single_source_dijkstra_path_length(g, n, weight="weight") for n in cands}
        self.diam = max(max(self.sp[a][b] for b in cands) for a in cands)
        info = network_info(cfg)
        self.zone = {j: v["zone"] for j, v in info["junctions"].items()}

    def hydraulic_distance(self, a: str, b: str) -> float:
        return float(self.sp[a][b] / self.diam)

    def components(self, a: pd.Series, b: pd.Series) -> dict[str, float]:
        r = self.ranges
        return {
            "d_log_area": (np.log(a.leak_area_m2) - np.log(b.leak_area_m2)) / r["log_area"],
            "d_demand_mult": (a.demand_mult - b.demand_mult) / r["demand_mult"],
            "d_peak_shift": (a.peak_shift_h - b.peak_shift_h) / r["peak_shift_h"],
            "d_pattern_amp": (a.pattern_amp - b.pattern_amp) / r["pattern_amp"],
            "d_tank_level": (a.tank_level_frac - b.tank_level_frac) / r["tank_level_frac"],
            "hyd_dist": self.hydraulic_distance(a.leak_node, b.leak_node),
            "zone_mismatch": float(self.zone[a.leak_node] != self.zone[b.leak_node]),
        }

    def __call__(self, a: pd.Series, b: pd.Series) -> float:
        c = self.components(a, b)
        return float(np.sqrt(np.mean([v * v for v in c.values()])))


def pair_manifest(scen: pd.DataFrame, dist: ScenarioDistance, per_tertile: int, master_seed: int,
                  target_splits=("train", "calibration", "test")) -> pd.DataFrame:
    """For every target choose ``per_tertile`` sources from each distance tertile of the library."""
    lib = scen[scen.split == "library"]
    rows = []
    for ti, (_, t) in enumerate(scen[scen.split.isin(target_splits)].iterrows()):
        d = np.array([dist(t, s) for _, s in lib.iterrows()])
        order = np.argsort(d, kind="stable")
        tert = np.array_split(order, 3)
        r = rng(master_seed, "pairs", ti)
        for band, idx in zip(("near", "mid", "far"), tert):
            for k in r.choice(idx, size=per_tertile, replace=False):
                s = lib.iloc[k]
                rows.append({"target_id": t.scenario_id, "source_id": s.scenario_id, "split": t.split,
                             "distance_band": band, "scenario_distance": float(d[k])})
    out = pd.DataFrame(rows)
    out.insert(0, "pair_id", [f"{a}__{b}" for a, b in zip(out.target_id, out.source_id)])
    return out


def frame_sha256(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def write_manifest(df: pd.DataFrame, path: str | Path) -> str:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return frame_sha256(df)


def read_scenarios(path: str | Path) -> pd.DataFrame:
    """Load a scenario table with node and pipe ids kept as strings."""
    df = pd.read_csv(repo_path(path), dtype={"leak_node": str, "closed_pipes": str, "scenario_id": str})
    df["closed_pipes"] = df["closed_pipes"].fillna("")
    return df


def read_pairs(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(repo_path(path), dtype={"target_id": str, "source_id": str})


def to_scenario(row: pd.Series) -> Scenario:
    return Scenario.from_dict(row.to_dict())
