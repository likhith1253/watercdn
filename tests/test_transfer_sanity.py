"""Phase 2 sanity tests T1-T5 (T6, the cross-environment re-run, is scripts/check_pilot_reproduction.py)."""
from dataclasses import replace

import numpy as np
import pytest

from tle.analysis.labels import evals_to_target, harm_label, pair_labels, utility
from tle.config import load_config
from tle.de.jobs import _execute, paired_runs
from tle.de.optimizer import Evaluator, initial_population
from tle.de.runner import get_simulator
from tle.de.transfer import inject_elites, normalised_distance, select_elites
from tle.seeds import run_streams


@pytest.fixture(scope="module")
def small_cfg():
    cfg = load_config("configs/e1.yaml")
    cfg["de"]["budget_B"] = 200
    return cfg


def source_elites(cfg, scenario, seed):
    sim = get_simulator(cfg)
    N, B = cfg["de"]["N"], cfg["de"]["budget_B"]
    r_init, r_de, _ = run_streams(np.random.SeedSequence(seed))
    ev = Evaluator(sim, scenario, B)
    _execute(ev, initial_population(sim.lo, sim.hi, N, r_init), r_de, N, cfg["de"]["F"], cfg["de"]["CR"])
    t = cfg["transfer"]
    return select_elites(ev.X, ev.J, ev.feasible, t["k"], t["min_elite_distance_frac"], sim.lo, sim.hi)[0]


def test_T1_budget_equality(small_cfg, base):
    el = source_elites(small_cfg, base, 1)
    res = paired_runs(small_cfg, base, np.random.SeedSequence(7), {"s": el})
    for run in res["runs"].values():
        assert run.summary["n_evals"] == 200
        assert len(run.gen0_J) == small_cfg["de"]["N"] and len(run.bsf) == 200


def test_T2_determinism(small_cfg, base):
    el = source_elites(small_cfg, base, 1)
    a = paired_runs(small_cfg, base, np.random.SeedSequence(7), {"s": el})["runs"]
    b = paired_runs(small_cfg, base, np.random.SeedSequence(7), {"s": el})["runs"]
    for k in a:
        assert np.array_equal(a[k].bsf, b[k].bsf)


def test_common_random_numbers_share_non_injected_members(small_cfg, base):
    """Warm and fresh generation 0 differ only in the injected slots."""
    el = source_elites(small_cfg, base, 1)
    res = paired_runs(small_cfg, base, np.random.SeedSequence(3), {"s": el})
    m = res["m_injected"]
    f, w = res["runs"]["fresh"].gen0_J, res["runs"]["warm:s"].gen0_J
    assert np.array_equal(f[m:], w[m:]) and not np.array_equal(f[:m], w[:m])


def test_T3_self_transfer_helps_on_average(small_cfg, base):
    el = source_elites(small_cfg, base, 99)  # same scenario, different seed
    Qf, Qw = [], []
    for r in range(8):
        runs = paired_runs(small_cfg, base, np.random.SeedSequence(1000 + r), {"self": el})["runs"]
        Qf.append(runs["fresh"].bsf[-1])
        Qw.append(runs["warm:self"].bsf[-1])
    U, se = utility(Qf, Qw)
    assert U > 0


def test_T4_random_source_control_is_neutral(small_cfg, base):
    sim = get_simulator(small_cfg)
    rng = np.random.default_rng(5)
    Us = []
    for r in range(8):
        fake = sim.lo + rng.random((3, len(sim.lo))) * (sim.hi - sim.lo)
        runs = paired_runs(small_cfg, base, np.random.SeedSequence(2000 + r), {"rand": fake})["runs"]
        Us.append(runs["fresh"].bsf[-1] - runs["warm:rand"].bsf[-1])
    # a biased comparison would give a mean difference far from zero; random injection must not
    assert abs(np.mean(Us)) < 2.5 * np.std(Us, ddof=1) / np.sqrt(8) + 1e-12


def test_T5_sign_convention():
    U, _ = utility([10.0, 11.0, 12.0], [9.0, 10.0, 11.0])
    assert U > 0 and harm_label(U) == 0
    U, _ = utility([10.0, 11.0, 12.0], [12.0, 13.0, 14.0])
    assert U < 0 and harm_label(U, 0.5) == 1


def test_labels_hand_case():
    bsf_f = np.array([[5, 4, 3, 2.0], [5, 4, 3, 3.0]])
    bsf_w = np.array([[3, 2, 2, 2.0], [4, 3, 2, 1.0]])
    lab = pair_labels(bsf_f, bsf_w, target_fraction=0.75)
    assert lab["Q_star"] == 3.0
    assert lab["E_fresh"] == 3.0 and lab["E_warm"] == 1.5
    assert lab["TG"] == pytest.approx(0.5)
    assert lab["U_raw"] == pytest.approx(1.0)
    assert evals_to_target(np.array([5.0, 4.0]), 1.0) == 2  # censored at B


def test_elites_distinct_and_feasible_first():
    lo, hi = np.zeros(2), np.ones(2)
    X = np.array([[0.1, 0.1], [0.1, 0.11], [0.9, 0.9], [0.5, 0.5]])
    J = np.array([1.0, 0.5, 3.0, 0.1])
    feas = np.array([True, True, True, False])
    E, EJ, nf = select_elites(X, J, feas, 3, 0.05, lo, hi)
    # point with J=1.0 is skipped (too close to the J=0.5 point); the infeasible one fills the last slot
    assert nf == 2 and list(EJ) == [0.5, 3.0, 0.1]
    assert np.all(normalised_distance(E[0], E[1:], lo, hi) >= 0.05)


def test_injection_clips_and_keeps_rest():
    lo, hi = np.zeros(3), np.ones(3)
    pop0 = np.full((10, 3), 0.5)
    el = np.array([[1.0, 1.0, 1.0]])
    out = inject_elites(pop0, el, 0.2, 0.5, lo, hi, np.random.default_rng(0))
    assert np.all(out >= lo) and np.all(out <= hi)
    assert np.array_equal(out[2:], pop0[2:]) and not np.array_equal(out[:2], pop0[:2])
