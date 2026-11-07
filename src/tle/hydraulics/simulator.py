"""Scenario-configurable hydraulic simulator and the penalised objective J(x; theta).

Decision vector x (dimension D = W * (n_pumps + n_prvs)), laid out link by link:

    [speed(pump_1, w_0..w_{W-1}), ..., speed(pump_P, ...), setting(prv_1, w_0..), ...]

Pump speeds are relative speed multipliers, PRV settings are outlet pressure
heads in metres. Window w covers hours [w*H/W, (w+1)*H/W). Window 0 is applied
as the initial link setting, windows 1..W-1 through EPANET timer controls that
are created once and only re-parameterised per evaluation.

Objective (lower is better), all terms kept separately in the output:

    J = w_L * leakage volume [m3]
      + w_E * energy_weight_mult * pump energy cost [currency]
      + w_P * pressure shortfall [m h]   (sum over demand nodes and time of max(0, p_req - p))
      + w_T * tank-balance deficit [m3]  (max(0, sum V_start - sum V_end); per-tank variant optional)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import epanet as en
from .scenario import Scenario

FAILURE_J = 1e9


@dataclass(frozen=True)
class ObjectiveWeights:
    leak: float = 1.0
    energy: float = 1.0
    shortfall: float = 50.0
    tank: float = 10.0


@dataclass(frozen=True)
class ProblemSpec:
    """Everything that defines the optimisation problem on one network."""

    inp_file: str
    meta_file: str
    windows: int = 6
    horizon_h: int = 24
    pump_speed: tuple[float, float] = (0.4, 1.0)
    prv_setting_m: tuple[float, float] = (10.0, 60.0)
    tariff: tuple[float, ...] = field(default_factory=lambda: tuple([0.10] * 7 + [0.20] * 16 + [0.10]))
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    shortfall_tol_mh: float = 1e-3
    tank_tol_m3: float = 1.0
    tank_balance: str = "system"  # "system": total storage; "per_tank": every tank separately

    def meta(self) -> dict:
        return json.loads(Path(self.meta_file).read_text())

    @property
    def n_pumps(self) -> int:
        return len(self.meta()["pumps"])

    @property
    def n_prvs(self) -> int:
        return len(self.meta()["prvs"])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        m = self.meta()
        W = self.windows
        lo = [self.pump_speed[0]] * (W * len(m["pumps"])) + [self.prv_setting_m[0]] * (W * len(m["prvs"]))
        hi = [self.pump_speed[1]] * (W * len(m["pumps"])) + [self.prv_setting_m[1]] * (W * len(m["prvs"]))
        return np.asarray(lo, float), np.asarray(hi, float)


class Simulator:
    """Owns one EPANET project; reconfigured per scenario, evaluated per decision."""

    RESIDENTIAL_PATTERN = "1"

    def __init__(self, spec: ProblemSpec):
        self.spec = spec
        self.meta = spec.meta()
        self.p = en.Project(spec.inp_file)
        p = self.p
        W, H = spec.windows, spec.horizon_h
        if H % W:
            raise ValueError("horizon must be a multiple of the number of windows")
        self.window_s = int(H * 3600 // W)
        p.delete_all_controls()
        p.set_time_param(en.EN_DURATION, H * 3600)

        n_nodes = p.count(en.EN_NODECOUNT)
        n_links = p.count(en.EN_LINKCOUNT)
        bulk = set(self.meta.get("bulk_nodes", []))
        self.junction_ids, self.junction_idx, self.demand_mask = [], [], []
        self.tank_idx = []
        for i in range(1, n_nodes + 1):
            t = p.node_type(i)
            if t == en.EN_JUNCTION:
                nid = p.node_id(i)
                self.junction_ids.append(nid)
                self.junction_idx.append(i)
                # service-pressure nodes: consumer demand nodes (bulk offtakes excluded)
                self.demand_mask.append(p.get_node(i, en.EN_BASEDEMAND) > 0 and nid not in bulk)
            elif t == en.EN_TANK:
                self.tank_idx.append(i)
        self.demand_mask = np.asarray(self.demand_mask, bool)
        self.demand_idx = [i for i, m in zip(self.junction_idx, self.demand_mask) if m]
        self._jpos = {nid: k for k, nid in enumerate(self.junction_ids)}

        self.pump_idx = [p.link_index(l) for l in self.meta["pumps"]]
        self.prv_idx = [p.link_index(l) for l in self.meta["prvs"]]
        self.control_links = self.pump_idx + self.prv_idx
        # timer controls for windows 1..W-1 of every controlled link
        self.controls = []  # (control index, link index)
        for link in self.control_links:
            for w in range(1, W):
                ci = p.add_control(en.EN_TIMER, link, 1.0 if link in self.pump_idx else 30.0, 0, w * self.window_s)
                self.controls.append((ci, link, w))

        # originals restored on every scenario change
        self.orig_status = {i: p.get_link(i, en.EN_INITSTATUS) for i in range(1, n_links + 1)}
        self.pattern_res = p.pattern_index(self.RESIDENTIAL_PATTERN)
        self.pattern_res_orig = np.array(
            [p.get_pattern_value(self.pattern_res, k) for k in range(1, p.pattern_len(self.pattern_res) + 1)]
        )
        self.tank_min = np.array([p.get_node(i, en.EN_MINLEVEL) for i in self.tank_idx])
        self.tank_max = np.array([p.get_node(i, en.EN_MAXLEVEL) for i in self.tank_idx])
        self.tariff = np.asarray(spec.tariff, float)
        if len(self.tariff) != H:
            raise ValueError("tariff must give one price per hour of the horizon")
        self.lo, self.hi = spec.bounds()
        self._leak_node_idx: int | None = None
        self.scenario: Scenario | None = None
        p.open_h()

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.p.close()

    def node_index(self, node_id: str) -> int:
        return self.p.node_index(node_id)

    def set_scenario(self, sc: Scenario) -> None:
        p = self.p
        if self._leak_node_idx is not None:
            p.set_node(self._leak_node_idx, en.EN_EMITTER, 0.0)
        self._leak_node_idx = p.node_index(sc.leak_node)
        p.set_node(self._leak_node_idx, en.EN_EMITTER, sc.emitter_coeff_lps)
        self._leak_coeff = sc.emitter_coeff_lps
        p.set_option(en.EN_DEMANDMULT, sc.demand_mult)
        # residential pattern: shift the peak and scale the amplitude around the mean
        base = self.pattern_res_orig
        mean = base.mean()
        shaped = mean + sc.pattern_amp * (np.roll(base, sc.peak_shift_h) - mean)
        shaped = np.maximum(shaped, 0.0)
        for k, v in enumerate(shaped, start=1):
            p.set_pattern_value(self.pattern_res, k, float(v))
        levels = self.tank_min + sc.tank_level_frac * (self.tank_max - self.tank_min)
        for i, lev in zip(self.tank_idx, levels):
            p.set_node(i, en.EN_TANKLEVEL, float(lev))
        closed = {p.link_index(l) for l in sc.closed_pipes}
        for i, st in self.orig_status.items():
            want = en.EN_CLOSED if i in closed else st
            if i in self.prv_idx and i not in closed:
                continue  # valve status follows its setting
            p.set_link(i, en.EN_INITSTATUS, want)
        self.scenario = sc

    # ------------------------------------------------------------------
    def decode(self, x: np.ndarray) -> np.ndarray:
        """Return settings as an array of shape (n_controlled_links, W)."""
        W = self.spec.windows
        return np.asarray(x, float).reshape(len(self.control_links), W)

    def _apply_decision(self, x: np.ndarray) -> None:
        p = self.p
        S = self.decode(x)
        for k, link in enumerate(self.control_links):
            p.set_link(link, en.EN_INITSETTING, float(S[k, 0]))
            if link in self.pump_idx:
                p.set_link(link, en.EN_INITSTATUS, en.EN_OPEN)
        for (ci, link, w) in self.controls:
            k = self.control_links.index(link)
            p.set_control(ci, en.EN_TIMER, link, float(S[k, w]), 0, float(w * self.window_s))

    def simulate(self, x: np.ndarray, keep_pressures: bool = False) -> dict:
        """Run one 24 h extended-period simulation; never raises on hydraulic failure."""
        if self.scenario is None:
            raise RuntimeError("set_scenario() must be called first")
        x = np.asarray(x, float)
        if x.shape != self.lo.shape or np.any(x < self.lo - 1e-9) or np.any(x > self.hi + 1e-9):
            raise ValueError("decision vector outside bounds or of wrong dimension")
        sc, spec, p = self.scenario, self.spec, self.p
        W, H = spec.windows, spec.horizon_h
        preq = sc.required_pressure_m
        n_dem = len(self.demand_idx)
        steps_p = []
        steps_dt = []
        steps_t = []
        leak_m3 = 0.0
        energy_kwh = 0.0
        energy_cost = 0.0
        shortfall_w = np.zeros(W)
        leak_w = np.zeros(W)
        unbalanced = False
        demand_m3 = 0.0  # consumer volume delivered (detail mode only)
        get_node, get_link = p.get_node, p.get_link
        try:
            self._apply_decision(x)
            p.init_h()
            v_start = np.array([get_node(i, en.EN_TANKVOLUME) for i in self.tank_idx])
            while True:
                t, code = p.run_h()
                if code == 1:  # system hydraulically unbalanced (did not converge)
                    unbalanced = True
                dt = p.next_h()
                if t >= H * 3600:
                    v_end = np.array([get_node(i, en.EN_TANKVOLUME) for i in self.tank_idx])
                    break
                if dt <= 0:
                    raise en.EpanetError(-1, "simulation ended before the horizon")
                dt_h = dt / 3600.0
                pres = np.fromiter((get_node(i, en.EN_PRESSURE) for i in self.demand_idx), float, n_dem)
                w = min(int(t // self.window_s), W - 1)
                sf = np.maximum(preq - pres, 0.0).sum() * dt_h
                shortfall_w[w] += sf
                pl = get_node(self._leak_node_idx, en.EN_PRESSURE)
                q_leak_lps = self._leak_coeff * math.sqrt(pl) if pl > 0 else 0.0
                lv = q_leak_lps / 1000.0 * dt
                leak_m3 += lv
                leak_w[w] += lv
                hour = min(int(t // 3600), H - 1)
                e = sum(get_link(i, en.EN_ENERGY) for i in self.pump_idx) * dt_h
                energy_kwh += e
                energy_cost += e * self.tariff[hour]
                if keep_pressures:
                    # EN_DEMAND includes emitter (leak) outflow; subtract it
                    q_all = sum(get_node(i, en.EN_DEMAND) for i in self.junction_idx)
                    demand_m3 += (q_all - q_leak_lps) / 1000.0 * dt
                steps_p.append(pres)
                steps_dt.append(dt_h)
                steps_t.append(t)
        except en.EpanetError:
            return self._failure(x)
        if unbalanced:
            return self._failure(x)

        P = np.vstack(steps_p)  # (steps, demand nodes)
        dts = np.asarray(steps_dt)
        if spec.tank_balance == "system":
            tank_def = float(max(v_start.sum() - v_end.sum(), 0.0))
        else:
            tank_def = float(np.maximum(v_start - v_end, 0.0).sum())
        shortfall = float(shortfall_w.sum())
        wts = spec.weights
        terms = {
            "J_leak": wts.leak * leak_m3,
            "J_energy": wts.energy * sc.energy_weight_mult * energy_cost,
            "J_shortfall": wts.shortfall * shortfall,
            "J_tank": wts.tank * tank_def,
        }
        J = float(sum(terms.values()))
        # time-weighted pressure distribution over demand nodes
        flatP = P.ravel()
        wflat = np.repeat(dts, P.shape[1])
        order = np.argsort(flatP)
        cw = np.cumsum(wflat[order]) / wflat.sum()
        p5 = float(flatP[order][np.searchsorted(cw, 0.05)])
        p50 = float(flatP[order][np.searchsorted(cw, 0.50)])
        out = {
            "J": J,
            **terms,
            "leak_m3": float(leak_m3),
            "energy_kwh": float(energy_kwh),
            "energy_cost": float(energy_cost),
            "shortfall_mh": shortfall,
            "tank_deficit_m3": tank_def,
            "violation": shortfall + tank_def,
            "feasible": bool(shortfall <= spec.shortfall_tol_mh and tank_def <= spec.tank_tol_m3),
            "failed": False,
            "p_min": float(flatP.min()),
            "p_p5": p5,
            "p_median": p50,
            "shortfall_by_window": shortfall_w.tolist(),
            "leak_by_window": leak_w.tolist(),
        }
        if keep_pressures:
            out["demand_m3"] = float(demand_m3)
            out["pressures"] = P
            out["step_hours"] = dts
            out["step_times"] = np.asarray(steps_t)
        return out

    def _failure(self, x: np.ndarray) -> dict:
        W = self.spec.windows
        nan = float("nan")
        return {
            "J": FAILURE_J, "J_leak": nan, "J_energy": nan, "J_shortfall": nan, "J_tank": nan,
            "leak_m3": nan, "energy_kwh": nan, "energy_cost": nan, "shortfall_mh": nan,
            "tank_deficit_m3": nan, "violation": FAILURE_J, "feasible": False, "failed": True,
            "p_min": nan, "p_p5": nan, "p_median": nan,
            "shortfall_by_window": [nan] * W, "leak_by_window": [nan] * W
        }
