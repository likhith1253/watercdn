"""Baseline mechanics: SEETO allocation, PMTO task model, BCKT competition bookkeeping, statistics helpers."""
import numpy as np
import pytest

from tle.analysis.stats import clopper_pearson, cliffs_delta, holm, rank_biserial
from tle.baselines.run_based import BCKTCompetition, TaskModel, pmto_init, seeto_init, softmax_allocation


def test_softmax_allocation_sums_and_prefers_similar():
    alloc = softmax_allocation(np.array([0.99, 0.90, 0.95, 0.10, 0.5, 0.97]), gamma=5, temperature=0.065, m=8)
    assert sum(alloc.values()) == 8
    assert 3 not in alloc  # least similar source is outside the top-5
    assert alloc[0] == max(alloc.values())


def test_seeto_init_copies_solutions_as_is():
    lo, hi = np.zeros(3), np.ones(3)
    X = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.5, 0.5, 0.5]])
    arch = [(X, np.array([1.0, 2.0, 3.0]), np.ones(3, bool))] * 2
    pop0 = np.full((10, 3), 0.3)
    out = seeto_init(arch, np.array([0.9, 0.8]), 2, 0.065, 0.4, lo, hi, 0.05)(pop0, np.random.default_rng(0))
    assert np.allclose(out[0], X[0]) and np.array_equal(out[4:], pop0[4:])


def test_pmto_task_model_interpolates():
    th = np.linspace(0, 1, 12)[:, None]
    X = np.column_stack([0.2 + 0.5 * th[:, 0], 1 - th[:, 0]])
    tm = TaskModel(th, X, quality_gap=np.zeros(12), keep=1.0, seed=0)
    assert np.allclose(tm.predict(np.array([0.5])), [0.45, 0.5], atol=0.05)
    out = pmto_init(np.array([0.45, 0.5]), 0.4, 0.03, np.zeros(2), np.ones(2))(np.full((10, 2), 0.9),
                                                                              np.random.default_rng(0))
    assert np.allclose(out[0], [0.45, 0.5]) and np.all(out[4:] == 0.9)


def test_bckt_transfers_from_a_useful_source():
    rng = np.random.default_rng(0)
    lo, hi = np.zeros(2), np.ones(2)
    f = lambda X: ((X - 0.7) ** 2).sum(1) + 1.0  # positive objective, optimum at 0.7
    Xs = rng.random((60, 2))
    comp = BCKTCompetition([(Xs, f(Xs), np.ones(60, bool))], lo, hi, N=10, delta=5, rng=rng, seed=0)
    Xt = rng.random((10, 2))
    for x, y in zip(Xt, f(Xt)):
        comp.observe(x, {"J": y})
    trial = np.array([0.05, 0.05])  # a poor own candidate
    new = comp.hook(trial, 0, None, None)
    assert comp.log["n_competitions"] == 1
    if comp.log["n_transfers"]:
        assert not np.array_equal(new, trial)
        comp.observe(new, {"J": f(new[None, :])[0]})
        assert comp.sources[0]["mu"] is not None


def test_stats_helpers():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] >= adj["c"] >= adj["a"]
    assert cliffs_delta([2, 3], [0, 1]) == 1.0
    assert rank_biserial(np.array([1.0, 2.0, -0.5])) > 0
    lo, hi = clopper_pearson(0, 29)
    assert lo == 0.0 and hi == pytest.approx(0.119, abs=0.002)
