"""Phase 5: threshold procedure on synthetic data with a known harm rate."""
import numpy as np
import pandas as pd

from tle.models.gate import choose_threshold, decide, evaluate_decisions, top_source


def synthetic(n, rng):
    """p_hat informative: harm probability equals p_hat."""
    p = rng.random(n)
    harm = (rng.random(n) < p).astype(int)
    return p, harm


def test_guarantee_holds_in_repeated_draws():
    rng = np.random.default_rng(0)
    alpha, delta = 0.2, 0.05
    viol = 0
    trials = 400
    for _ in range(trials):
        p, h = synthetic(80, rng)
        lam = choose_threshold(p, h, alpha, delta)
        if lam is None:
            continue
        # true harm rate among accepted for p <= lam with p ~ U(0,1) and P(harm|p)=p is lam/2
        if lam / 2 > alpha:
            viol += 1
    assert viol / trials <= delta + 0.02


def test_never_transfer_when_everything_harms():
    p = np.linspace(0, 1, 50)
    assert choose_threshold(p, np.ones(50, int), 0.2, 0.05) is None


def test_zero_harm_needs_enough_points():
    # with zero harm, certification of alpha=0.1 needs n >= ln(0.05)/ln(0.9) = 28.4 -> 29 accepted
    p = np.arange(40) / 40
    lam = choose_threshold(p, np.zeros(40, int), 0.10, 0.05)
    assert lam is not None and (p <= lam).sum() == 40
    lam_small = choose_threshold(p[:28], np.zeros(28, int), 0.10, 0.05)
    assert lam_small is None


def test_top_source_and_offline_evaluation():
    pred = pd.DataFrame({"target_id": ["t1", "t1", "t2", "t2"], "source_id": ["a", "b", "a", "b"],
                         "u_q10": [0.1, 0.5, -1.0, -0.2], "p_harm": [0.1, 0.3, 0.9, 0.6],
                         "U_mean": [1.0, 2.0, -2.0, -1.0], "harm": [0, 0, 1, 1]})
    top = top_source(pred)
    assert list(top.source_id) == ["b", "b"]
    acc = decide(top, 0.5)
    assert list(acc) == [True, False]
    oracle = pd.Series({"t1": 2.0, "t2": 0.0})
    r = evaluate_decisions(top, acc, oracle, 0.5)
    assert r["mean_utility"] == 1.0 and r["harm_rate_among_accepted"] == 0.0 and r["abstention_rate"] == 0.5
