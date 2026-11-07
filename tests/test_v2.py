"""V2 components: weighted threshold reduces to the unweighted one, shift detection, binomial extension."""
import numpy as np
from scipy.stats import binom

from tle.models.gate import choose_threshold
from tle.v2.shift import binom_cdf_cont, domain_weights, weighted_threshold


def test_binom_extension_matches_integer_cdf():
    for n, k in ((20, 0), (20, 3), (50, 10)):
        assert abs(binom_cdf_cont(k, n, 0.2) - binom.cdf(k, n, 0.2)) < 1e-10


def test_unit_weights_equal_unweighted_threshold():
    rng = np.random.default_rng(1)
    p = rng.random(80)
    h = (rng.random(80) < p * 0.3).astype(int)
    assert weighted_threshold(p, h, np.ones(80), 0.2, 0.05) == choose_threshold(p, h, 0.2, 0.05)


def test_domain_weights_detect_shift():
    rng = np.random.default_rng(0)
    same = domain_weights(rng.normal(size=(80, 3)), rng.normal(size=(40, 3)), 0)[1]
    far = domain_weights(rng.normal(size=(80, 3)), rng.normal(3, 1, size=(40, 3)), 0)[1]
    assert same < 0.7 and far > 0.95


def test_strong_shift_makes_gate_abstain():
    rng = np.random.default_rng(2)
    Xc, Xn = rng.normal(size=(80, 3)), rng.normal(4, 1, size=(40, 3))
    w, _ = domain_weights(Xc, Xn, 0)
    p = rng.random(80)
    assert weighted_threshold(p, np.zeros(80, int), w, 0.2, 0.05) is None
