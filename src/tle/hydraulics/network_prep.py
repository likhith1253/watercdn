"""Build the edited benchmark networks used in the study.

Net3 (EPANET example network shipped with WNTR) has two pumps, three tanks and
no valves. The study needs decision variables that change leakage physically,
so two pressure-reducing-valve (PRV) zones are created:

* each zone is a part of the network that is fed through exactly two pipes
  (found by a search over 2-pipe cuts that isolate a source-free sub-network);
* the larger-capacity feed pipe receives a PRV at its upstream end, the other
  feed pipe is closed, so the zone is supplied only through the PRV.

Other edits (all recorded in ``<name>.meta.json`` next to the .inp file):

* every original rule/simple control is removed: pump speeds and PRV settings
  are the decision variables, applied per time window;
* pipe 330 (the pump-335 bypass that the original controls close whenever the
  pump runs) is closed permanently because the pump always runs at a decided
  speed;
* 24 h horizon, 1 h hydraulic, pattern and report steps;
* pressure-driven demand (PDA) with minimum pressure 0 m and required pressure
  20 m, exponent 0.5;
* the file is written in SI units (flow in L/s, pressure in m).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import wntr

REQUIRED_PRESSURE_M = 20.0


@dataclass(frozen=True)
class PRVZoneSpec:
    name: str
    prv_pipe: str  # feed pipe that receives the PRV
    closed_pipe: str  # second feed pipe, closed


NET3_ZONES = (
    PRVZoneSpec("A", prv_pipe="238", closed_pipe="289"),
    PRVZoneSpec("B", prv_pipe="135", closed_pipe="161"),
)
NET3_EXTRA_CLOSED = ("330",)


def wntr_example(name: str) -> Path:
    import wntr as _w

    return Path(_w.__file__).parent / "library" / "networks" / name


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _zone_nodes(wn: wntr.network.WaterNetworkModel, cut: tuple[str, str]) -> set[str]:
    """Junctions of the source-free component created by removing the two cut pipes."""
    g = nx.MultiGraph()
    for name, link in wn.links():
        if name in cut:
            continue
        g.add_edge(link.start_node_name, link.end_node_name, key=name)
    sources = set(wn.tank_name_list) | set(wn.reservoir_name_list)
    for comp in nx.connected_components(g):
        if not comp & sources:
            return {n for n in comp if n in set(wn.junction_name_list)}
    raise ValueError(f"cut {cut} does not isolate a source-free component")


def _insert_prv(wn, pipe_name: str, zone: set[str], prv_name: str, init_setting: float) -> dict:
    """Replace the upstream end of ``pipe_name`` by a PRV that feeds ``zone``."""
    pipe = wn.get_link(pipe_name)
    a, b = pipe.start_node_name, pipe.end_node_name
    if b in zone and a not in zone:
        up, down = a, b
    elif a in zone and b not in zone:
        up, down = b, a
    else:
        raise ValueError(f"pipe {pipe_name} is not a zone boundary pipe")
    props = dict(length=pipe.length, diameter=pipe.diameter, roughness=pipe.roughness, minor_loss=pipe.minor_loss)
    up_node = wn.get_node(up)
    mid = f"{prv_name}_out"
    x, y = up_node.coordinates
    wn.remove_link(pipe_name)
    wn.add_junction(mid, base_demand=0.0, elevation=up_node.elevation, coordinates=(x + 1.0, y + 1.0))
    wn.add_valve(prv_name, up, mid, diameter=props["diameter"], valve_type="PRV", minor_loss=0.0,
                 initial_setting=init_setting, initial_status="Active")
    wn.add_pipe(pipe_name, mid, down, length=props["length"], diameter=props["diameter"],
                roughness=props["roughness"], minor_loss=props["minor_loss"])
    return {"prv": prv_name, "upstream_node": up, "outlet_node": mid, "zone_entry_node": down}


def build_net3_prv(out_dir: str | Path) -> Path:
    """Write ``net3_prv.inp`` and ``net3_prv.meta.json`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = wntr_example("Net3.inp")
    wn = wntr.network.WaterNetworkModel(str(src))

    zones = {}
    for spec in NET3_ZONES:
        zones[spec.name] = sorted(_zone_nodes(wn, (spec.prv_pipe, spec.closed_pipe)))

    for name in list(wn.control_name_list):
        wn.remove_control(name)

    prv_info = []
    for spec in NET3_ZONES:
        wn.get_link(spec.closed_pipe).initial_status = wntr.network.LinkStatus.Closed
        info = _insert_prv(wn, spec.prv_pipe, set(zones[spec.name]), f"PRV_{spec.name}", init_setting=35.0)
        info.update(zone=spec.name, closed_pipe=spec.closed_pipe)
        prv_info.append(info)
    for name in NET3_EXTRA_CLOSED:
        wn.get_link(name).initial_status = wntr.network.LinkStatus.Closed
    for name, pump in wn.pumps():
        pump.initial_status = wntr.network.LinkStatus.Open

    t = wn.options.time
    t.duration = 24 * 3600
    t.hydraulic_timestep = 3600
    t.pattern_timestep = 3600
    t.report_timestep = 3600
    t.quality_timestep = 3600
    wn.options.quality.parameter = "NONE"
    h = wn.options.hydraulic
    h.demand_model = "PDD"
    h.minimum_pressure = 0.0
    h.required_pressure = REQUIRED_PRESSURE_M
    h.pressure_exponent = 0.5
    h.emitter_exponent = 0.5

    inp = out_dir / "net3_prv.inp"
    wntr.network.write_inpfile(wn, str(inp), units="LPS", version=2.2)

    meta = {
        "name": "net3_prv",
        "source": "WNTR example Net3.inp",
        "source_sha256": file_sha256(src),
        "inp_sha256": file_sha256(inp),
        "wntr_version": wntr.__version__,
        "edits": {
            "controls_removed": True,
            "closed_pipes": [s.closed_pipe for s in NET3_ZONES] + list(NET3_EXTRA_CLOSED),
            "prvs": prv_info,
        },
        "zones": zones,
        "pumps": list(wn.pump_name_list),
        "prvs": [p["prv"] for p in prv_info],
        "required_pressure_m": REQUIRED_PRESSURE_M,
        # Bulk-supply offtakes (industrial patterns 2-5 at dead ends). Node 15 is below
        # 20 m even in the original Net3 at full pump speed, so the service-pressure
        # requirement is applied to consumer nodes only; bulk nodes keep PDD demand.
        "bulk_nodes": sorted(
            j for j in wn.junction_name_list
            if any(d.base_value > 0 and d.pattern_name not in (None, "1") for d in wn.get_node(j).demand_timeseries_list)
        ),
    }
    (out_dir / "net3_prv.meta.json").write_text(json.dumps(meta, indent=2))
    return inp


def network_graph(wn: wntr.network.WaterNetworkModel) -> nx.Graph:
    """Undirected graph weighted by pipe length (valves and pumps get length 1 m)."""
    g = nx.Graph()
    for _, link in wn.links():
        g.add_edge(link.start_node_name, link.end_node_name, weight=float(getattr(link, "length", 1.0)))
    return g


def network_constants(inp: str | Path, meta: dict) -> dict:
    """Static per-network quantities used for feature normalisation and leak candidates."""
    wn = wntr.network.WaterNetworkModel(str(inp))
    wg = network_graph(wn)
    feeders = list(wn.pump_name_list) + list(wn.tank_name_list) + list(wn.reservoir_name_list)
    feeder_nodes = set(wn.tank_name_list) | {wn.get_link(p).end_node_name for p in wn.pump_name_list}
    dist = nx.multi_source_dijkstra_path_length(wg, feeder_nodes, weight="weight")
    zone_of = {}
    for z, nodes in meta["zones"].items():
        for n in nodes:
            zone_of[n] = z
    junction_info = {}
    for j in wn.junction_name_list:
        node = wn.get_node(j)
        base = sum(d.base_value for d in node.demand_timeseries_list)
        local_d = [wn.get_link(l).diameter for l in wn.get_links_for_node(j) if hasattr(wn.get_link(l), "diameter")]
        junction_info[j] = {
            "elevation": float(node.elevation),
            "base_demand": float(base),
            "dist_to_feeder_m": float(dist.get(j, np.nan)),
            "local_diameter_m": float(np.mean(local_d)) if local_d else float("nan"),
            "zone": zone_of.get(j, "main"),
        }
    return {"junctions": junction_info, "feeders": feeders}


LTOWN_FILE = "downloads/L-TOWN_v2_Model.inp"
LTOWN_URL = ("https://media.githubusercontent.com/media/KIOS-Research/BattLeDIM/master/"
             "Dataset%20Generator/L-TOWN_v2_Model.inp")


def build_ltown_prv(out_dir: str | Path) -> Path:
    """Edited L-Town (BattLeDIM, EUPL-1.2) for E4.

    L-Town already has one pump and three PRVs, so no valves are added. Edits:
    controls removed (pump speed and PRV settings are decisions), 24 h horizon with
    1 h steps, pressure-driven demand (required 20 m), SI units (L/s). The PRV
    zones are the source-free components created by removing each PRV.
    """
    out_dir = Path(out_dir)
    src = out_dir / LTOWN_FILE
    if not src.exists():
        raise FileNotFoundError(f"download L-Town first: {LTOWN_URL} -> {src}")
    wn = wntr.network.WaterNetworkModel(str(src))
    for name in list(wn.control_name_list):
        wn.remove_control(name)
    for name, pump in wn.pumps():
        pump.initial_status = wntr.network.LinkStatus.Open
    zones = {}
    g_all = nx.MultiGraph()
    for name, link in wn.links():
        g_all.add_edge(link.start_node_name, link.end_node_name, key=name)
    sources = set(wn.tank_name_list) | set(wn.reservoir_name_list)
    for name in wn.valve_name_list:
        v = wn.get_link(name)
        g = g_all.copy()
        g.remove_edge(v.start_node_name, v.end_node_name, key=name)
        comp = nx.node_connected_component(g, v.end_node_name)
        zones[name] = sorted(n for n in comp if n in set(wn.junction_name_list)) if not comp & sources else []
    t = wn.options.time
    t.duration, t.hydraulic_timestep, t.pattern_timestep, t.report_timestep, t.quality_timestep = (
        24 * 3600, 3600, 3600, 3600, 3600)
    t.pattern_start = 0
    wn.options.quality.parameter = "NONE"
    h = wn.options.hydraulic
    h.demand_model, h.minimum_pressure, h.required_pressure = "PDD", 0.0, REQUIRED_PRESSURE_M
    h.pressure_exponent, h.emitter_exponent = 0.5, 0.5
    inp = out_dir / "ltown_prv.inp"
    wntr.network.write_inpfile(wn, str(inp), units="LPS", version=2.2)
    meta = {
        "name": "ltown_prv", "source": "BattLeDIM L-TOWN_v2_Model.inp (EUPL-1.2)", "source_url": LTOWN_URL,
        "source_sha256": file_sha256(src), "inp_sha256": file_sha256(inp), "wntr_version": wntr.__version__,
        "edits": {"controls_removed": True, "closed_pipes": [], "prvs": []},
        "zones": {k: v for k, v in zones.items() if v},
        "pumps": list(wn.pump_name_list), "prvs": list(wn.valve_name_list),
        "required_pressure_m": REQUIRED_PRESSURE_M, "bulk_nodes": [],
        "residential_pattern": "P-Residential",
    }
    (out_dir / "ltown_prv.meta.json").write_text(json.dumps(meta, indent=2))
    return inp
