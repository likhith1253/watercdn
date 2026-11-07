"""Simulator, objective terms and evaluator tests (Phase 1 deliverables)."""
import math
from dataclasses import replace

import numpy as np
import pytest
import wntr

from tle.de.optimizer import BudgetExceeded, Evaluator, initial_population, reflect, run_de
from tle.hydraulics import epanet as en
from tle.hydraulics.scenario import LEAK_DISCHARGE_COEFF, Scenario


def mid_decision(sim):
    return (sim.lo + sim.hi) / 2


def test_matches_wntr_epanet_simulator(sim, spec, tmp_path):
    """Same decision, no leak, original tank levels: pressures equal WNTR's EPANET run."""
    wn = wntr.network.WaterNetworkModel(spec.inp_file)
    tanks = list(wn.tanks())
    frac = (tanks[0][1].init_level - tanks[0][1].min_level) / (tanks[0][1].max_level - tanks[0][1].min_level)
    for _, t in tanks:
        t.init_level = t.min_level + frac * (t.max_level - t.min_level)
    sim.set_scenario(Scenario("chk", "net3_prv", "111", 0.0, tank_level_frac=frac))
    x = np.r_[np.full(12, 1.0), np.full(12, 35.0)]
    for name, v in wn.valves():
        v.initial_setting = 35.0
    r = sim.simulate(x, keep_pressures=True)
    res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(tmp_path / "wntr"))
    pw = res.node["pressure"][sim.junction_ids]
    keep = [i for i, t in enumerate(r["step_times"]) if t % 3600 == 0]
    ref = pw.loc[[r["step_times"][i] for i in keep]].values[:, sim.demand_mask]
    assert np.abs(ref - r["pressures"][keep]).max() < 1e-3


def test_leak_volume_is_orifice_law(sim, base):
    """Leak volume equals sum_t Cd*A*sqrt(2 g p)*dt, and grows with area."""
    x = mid_decision(sim)
    sim.set_scenario(base)
    r = sim.simulate(x, keep_pressures=True)
    assert r["leak_m3"] > 0
    big = replace(base, leak_area_m2=base.leak_area_m2 * 3)
    sim.set_scenario(big)
    r_big = sim.simulate(x)
    assert r_big["leak_m3"] > 2.0 * r["leak_m3"]
    assert math.isclose(base.emitter_coeff_lps, LEAK_DISCHARGE_COEFF * base.leak_area_m2 * math.sqrt(2 * 9.81) * 1000)


def test_leak_responds_to_pressure(sim, base):
    """Lower pump speed lowers pressure, which lowers leakage (pressure-dependent leak)."""
    sim.set_scenario(base)
    fast = np.r_[np.full(12, 1.0), np.full(12, 40.0)]
    slow = np.r_[np.full(12, 0.6), np.full(12, 40.0)]
    assert sim.simulate(slow)["leak_m3"] < sim.simulate(fast)["leak_m3"]


def test_prv_controls_zone_leak(sim, cfg):
    """A leak inside PRV zone A leaks less when the PRV setting is lower."""
    import json
    from tle.config import repo_path

    meta = json.loads(repo_path(cfg["network"]["meta"]).read_text())
    node = meta["zones"]["A"][len(meta["zones"]["A"]) // 2]
    sim.set_scenario(Scenario("zA", "net3_prv", node, 2e-3))
    hi = np.r_[np.full(12, 0.8), np.full(12, 40.0)]
    lo = np.r_[np.full(12, 0.8), np.full(6, 25.0), np.full(6, 40.0)]
    assert sim.simulate(lo)["leak_m3"] < sim.simulate(hi)["leak_m3"]


def test_objective_terms_add_up(sim, base, spec):
    sim.set_scenario(base)
    r = sim.simulate(mid_decision(sim))
    w = spec.weights
    assert math.isclose(r["J"], r["J_leak"] + r["J_energy"] + r["J_shortfall"] + r["J_tank"], rel_tol=1e-12)
    assert math.isclose(r["J_leak"], w.leak * r["leak_m3"], rel_tol=1e-12)
    assert math.isclose(r["J_shortfall"], w.shortfall * r["shortfall_mh"], rel_tol=1e-12)
    assert math.isclose(sum(r["shortfall_by_window"]), r["shortfall_mh"], rel_tol=1e-9, abs_tol=1e-12)


def test_shortfall_hand_case(sim, base):
    """Very low PRV settings starve the PRV zones: positive shortfall, infeasible."""
    sim.set_scenario(base)
    r = sim.simulate(np.r_[np.full(12, 1.0), np.full(12, 15.0)])
    assert r["shortfall_mh"] > 0 and not r["feasible"]


def test_simulation_is_deterministic(sim, base):
    sim.set_scenario(base)
    x = mid_decision(sim)
    a, b = sim.simulate(x), sim.simulate(x)
    assert a["J"] == b["J"]


def test_scenario_switch_resets_state(sim, base):
    other = replace(base, scenario_id="o", leak_node="211", demand_mult=1.2, closed_pipes=("20",))
    x = mid_decision(sim)
    sim.set_scenario(base)
    a = sim.simulate(x)
    sim.set_scenario(other)
    sim.simulate(x)
    sim.set_scenario(base)
    assert sim.simulate(x)["J"] == a["J"]


def test_out_of_bounds_rejected(sim, base):
    sim.set_scenario(base)
    with pytest.raises(ValueError):
        sim.simulate(sim.hi + 1.0)


def test_failure_returns_penalty_not_exception(sim, base, monkeypatch):
    sim.set_scenario(base)

    def boom(*a, **k):
        raise en.EpanetError(110, "runH")

    monkeypatch.setattr(sim.p, "run_h", boom)
    r = sim.simulate(mid_decision(sim))
    assert r["failed"] and r["J"] >= 1e9 and not r["feasible"]


def test_evaluator_counts_and_enforces_budget(sim, base):
    ev = Evaluator(sim, base, budget=3)
    for _ in range(3):
        ev(mid_decision(sim))
    assert ev.count == 3
    with pytest.raises(BudgetExceeded):
        ev(mid_decision(sim))
    assert np.all(np.diff(ev.best_so_far) <= 0)


def test_reflect_keeps_bounds():
    lo, hi = np.zeros(3), np.ones(3)
    v = reflect(np.array([-0.2, 0.5, 1.3]), lo, hi)
    assert np.allclose(v, [0.2, 0.5, 0.7])


def test_de_uses_exact_budget_and_is_reproducible(sim, base):
    def run(seed):
        s_init, s_de = np.random.SeedSequence(seed).spawn(2)
        ev = Evaluator(sim, base, budget=137)
        pop0 = initial_population(sim.lo, sim.hi, 20, np.random.default_rng(s_init))
        out = run_de(ev, 20, 0.6, 0.9, pop0, np.random.default_rng(s_de))
        return ev, out

    ev1, out1 = run(5)
    ev2, _ = run(5)
    ev3, _ = run(6)
    assert ev1.count == 137 and out1.generations[0]["evals"] == 20
    assert np.array_equal(ev1.best_so_far, ev2.best_so_far)
    assert not np.array_equal(ev1.best_so_far, ev3.best_so_far)


def test_leak_formula_matches_epanet_emitter(sim):
    """EPANET's emitter outflow (EN_DEMAND with leak minus without) equals Cd*A*sqrt(2 g p)."""
    x = np.r_[np.full(12, 0.8), np.full(12, 35.0)]
    vals = {}
    for area in (0.0, 1e-3):
        sc = Scenario("t", "net3_prv", "111", area)
        sim.set_scenario(sc)
        sim._apply_decision(x)
        sim.p.init_h()
        sim.p.run_h()
        i = sim.p.node_index("111")
        vals[area] = (sim.p.get_node(i, en.EN_DEMAND), sim.p.get_node(i, en.EN_PRESSURE), sc.emitter_coeff_lps)
    emitter = vals[1e-3][0] - vals[0.0][0]
    assert emitter == pytest.approx(vals[1e-3][2] * math.sqrt(vals[1e-3][1]), rel=1e-3)
