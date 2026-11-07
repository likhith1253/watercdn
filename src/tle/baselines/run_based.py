"""Baselines that need their own DE runs (B2-gamma SEETO-style, B3 BCKT-style, B4 PMTO-style).

All use the frozen DE, budget B, population N and the same (target, repeat)
seed as the paired runs, so their initial random members and DE stream are the
fresh run's (common random numbers). Adaptations are listed in
docs/baselines.md (Table T1).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from ..de.optimizer import Evaluator, run_de
from ..de.transfer import n_injected, select_elites

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _gp(dim: int, seed: int) -> GaussianProcessRegressor:
    k = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(dim) * 0.5, (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-8, 1e0))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=0, random_state=seed)


def _gp_iso(seed: int) -> GaussianProcessRegressor:
    k = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.5, (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-8, 1e0))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=0, random_state=seed)


# ---------------------------------------------------------------- B2-gamma: SEETO-style
def softmax_allocation(sims: np.ndarray, gamma: int, temperature: float, m: int) -> dict[int, int]:
    """Top-gamma sources by similarity, softmax weights, m slots by largest remainder."""
    order = np.argsort(-sims, kind="stable")[:gamma]
    z = sims[order] / temperature
    w = np.exp(z - z.max())
    w /= w.sum()
    raw = w * m
    n = np.floor(raw).astype(int)
    for j in np.argsort(-(raw - n), kind="stable")[: m - n.sum()]:
        n[j] += 1
    return {int(order[j]): int(n[j]) for j in range(len(order)) if n[j] > 0}


def seeto_init(archives: list[tuple], sims: np.ndarray, gamma: int, temperature: float, rho: float,
               lo: np.ndarray, hi: np.ndarray, min_dist: float):
    """Returns f(pop0, rng) injecting the allocated best distinct solutions of each source, copied as-is."""

    def init(pop0: np.ndarray, rng_inj: np.random.Generator) -> np.ndarray:
        m = n_injected(len(pop0), rho)
        alloc = softmax_allocation(np.asarray(sims, float), gamma, temperature, m)
        sols = []
        for j, nj in alloc.items():
            X, J, feas = archives[j]
            sols.extend(select_elites(X, J, feas, nj, min_dist, lo, hi)[0])
        out = pop0.copy()
        out[: len(sols)] = np.clip(np.asarray(sols), lo, hi)
        return out

    return init


# ---------------------------------------------------------------- B4: PMTO-style task model
class TaskModel:
    """PMTO online task model: one GP per decision variable, theta -> best solution (posterior mean)."""

    def __init__(self, theta: np.ndarray, X_best: np.ndarray, quality_gap: np.ndarray, keep: float, seed: int):
        keep_idx = np.argsort(quality_gap, kind="stable")[: max(2, int(np.ceil(keep * len(quality_gap))))]
        self.t_lo, self.t_hi = theta.min(0), theta.max(0)
        T = self._norm(theta[keep_idx])
        self.models = [_gp(T.shape[1], seed).fit(T, X_best[keep_idx, v]) for v in range(X_best.shape[1])]
        self.n_train = len(keep_idx)

    def _norm(self, theta: np.ndarray) -> np.ndarray:
        return (theta - self.t_lo) / np.where(self.t_hi > self.t_lo, self.t_hi - self.t_lo, 1.0)

    def predict(self, theta: np.ndarray) -> np.ndarray:
        T = self._norm(np.atleast_2d(theta))
        return np.array([m.predict(T)[0] for m in self.models])


def pmto_init(x_hat: np.ndarray, rho: float, sigma_frac: float, lo: np.ndarray, hi: np.ndarray):
    """Inject round(rho*N) perturbed copies of the task-model prediction (variant b of the reading note)."""

    def init(pop0: np.ndarray, rng_inj: np.random.Generator) -> np.ndarray:
        m = n_injected(len(pop0), rho)
        noise = sigma_frac * (hi - lo) * rng_inj.standard_normal((m, pop0.shape[1]))
        out = pop0.copy()
        out[:m] = np.clip(np.clip(x_hat, lo, hi) + noise, lo, hi)
        if m:
            out[0] = np.clip(x_hat, lo, hi)  # the prediction itself (pure online-mode answer)
        return out

    return init


# ---------------------------------------------------------------- B3: BCKT-style competition
class BCKTCompetition:
    """Bayesian competitive knowledge transfer, adapted to sequential DE with frozen source archives.

    Kept from BCKT: Spearman a-priori criterion R with variance 0.05^2, evidence
    T = realised / predicted improvement, Gaussian posterior of the
    transferability tau (precision-weighted updates of the winner only, on
    transfer events), Thompson sample of tau at each competition, argmax
    competition with ties to the target, exclusion of sources whose predicted
    improvement is negative, delta = 10 evaluations between competitions,
    code-version normalisation and likelihood-variance schedule.
    Adapted: see docs/baselines.md.
    """

    SIGMA_I2 = 0.05 ** 2

    def __init__(self, archives: list[tuple], lo: np.ndarray, hi: np.ndarray, N: int, delta: int,
                 rng: np.random.Generator, seed: int, max_points: int = 400):
        self.lo, self.hi, self.N, self.delta, self.rng, self.seed = lo, hi, N, delta, rng, seed
        self.sources = []
        for X, J, feas in archives:
            idx = np.argsort(J, kind="stable")[:max_points]
            Z = self._z(X[idx])
            gp = _gp_iso(seed).fit(Z, J[idx])
            order = np.lexsort((J, ~feas))  # candidates: feasible first, then by J
            self.sources.append({"gp": gp, "y": J, "X": X, "order": list(order), "sent": 0,
                                 "mu": None, "s2": None})
        self.tX, self.ty = [], []
        self.fe0 = None
        self.pending = None
        self.log = {"n_competitions": 0, "n_transfers": 0, "transfers_by_source": [0] * len(archives)}

    def _z(self, X):
        return (np.asarray(X) - self.lo) / (self.hi - self.lo)

    def observe(self, x, r):
        self.tX.append(np.asarray(x, float))
        self.ty.append(float(r["J"]))
        if self.pending is not None:
            i, dp_raw, ybest_before = self.pending
            T = (ybest_before - r["J"]) / dp_raw if dp_raw > 0 else 0.0
            fe = len(self.ty)
            s2T = self.SIGMA_I2 * np.exp(-(fe - self.fe0) / self.delta)
            self._update(self.sources[i], T, s2T)
            self.pending = None

    @staticmethod
    def _update(src, x, s2_new):
        if src["mu"] is None:
            src["mu"], src["s2"] = x, s2_new
            return
        mu, s2 = src["mu"], src["s2"]
        src["mu"] = (mu * s2_new + x * s2) / (s2 + s2_new)
        src["s2"] = s2 * s2_new / (s2 + s2_new)

    def hook(self, trial, i, pop, fit):
        n = len(self.ty)
        if n < self.N or n % self.delta:
            return trial
        if self.fe0 is None:
            self.fe0 = n
        self.log["n_competitions"] += 1
        tX, ty = self._z(np.array(self.tX)), np.array(self.ty)
        tgp = _gp_iso(self.seed).fit(tX, ty)
        mu, sd = tgp.predict(np.vstack([tX, self._z(trial[None, :])]), return_std=True)
        lcb = mu - 2 * sd
        d_in = lcb[:-1].min() - lcb[-1]
        best, best_val, best_info = None, d_in, None
        for k, src in enumerate(self.sources):
            if src["sent"] >= len(src["order"]):
                continue
            pred = src["gp"].predict(tX)
            impn = (pred.min() - src["y"].min()) / max(np.max(np.r_[src["y"], pred]), 1e-12)
            if impn < 0:
                continue
            R = spearmanr(pred, ty).statistic
            R = 0.0 if not np.isfinite(R) else R
            if src["mu"] is None:
                mu_k, s2_k = R, self.SIGMA_I2
            else:
                mu_k = (src["mu"] * self.SIGMA_I2 + R * src["s2"]) / (src["s2"] + self.SIGMA_I2)
                s2_k = src["s2"] * self.SIGMA_I2 / (src["s2"] + self.SIGMA_I2)
            tau = self.rng.normal(mu_k, np.sqrt(s2_k))
            d_ex = tau * impn * ty.max()
            if d_ex > best_val:
                best, best_val, best_info = k, d_ex, (R, impn * ty.max())
        if best is None:
            return trial
        src = self.sources[best]
        x_new = np.asarray(src["X"][src["order"][src["sent"]]], float)
        src["sent"] += 1
        self._update(src, best_info[0], self.SIGMA_I2)  # a-priori criterion, committed for the winner only
        self.pending = (best, best_info[1], ty.min())
        self.log["n_transfers"] += 1
        self.log["transfers_by_source"][best] += 1
        return np.clip(x_new, self.lo, self.hi)


def run_bckt(ev: Evaluator, N, F, CR, pop0, rng_de, archives, rng_bckt, delta: int, seed: int):
    comp = BCKTCompetition(archives, ev.sim.lo, ev.sim.hi, N, delta, rng_bckt, seed)
    out = run_de(ev, N, F, CR, pop0, rng_de, trial_hook=comp.hook, after_eval=comp.observe)
    return out, comp.log
