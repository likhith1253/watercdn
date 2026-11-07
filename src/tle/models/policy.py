"""Repeat-level outcomes for policy evaluation, including probe mode (C4).

Probe mode. The warm run's generation 0 contains m injected elites and N-m
random members that are identical to the fresh run's (common random numbers).
The shared random members are needed whatever is decided, so they are
evaluated first; then q of the injected elites are evaluated (the probe). If
the probe accepts, the run continues exactly as the warm run. If it rejects,
the injected slots get their original random members back and the run
continues exactly as the fresh run, having wasted q evaluations: its final
quality is the fresh run's best-so-far at B - q. Both outcomes are exact
look-ups of logged runs, no new simulation is needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .gate import choose_threshold
from .predictors import HarmModel


def repeat_outcomes(runs: pd.DataFrame, curves: pd.DataFrame, pairs: pd.DataFrame,
                    probe_sizes=(1, 2, 4, 8)) -> pd.DataFrame:
    """One row per (pair, repeat): Q_fresh, Q_warm, Q_fresh at B - q, sigma_fresh of the target."""
    t = runs[runs.role == "target"]
    cur = curves.set_index("run_id").bsf
    fresh = t[t.method == "fresh"].set_index(["scenario_id", "repeat"])
    warm = t[t.method == "warm"].set_index(["scenario_id", "source_id", "repeat"])
    sig = pairs.groupby("target_id").sigma_fresh.first()
    rows = []
    for _, p in pairs.iterrows():
        for rep in sorted(fresh.loc[p.target_id].index):
            fr = fresh.loc[(p.target_id, rep)]
            wr = warm.loc[(p.target_id, p.source_id, rep)]
            bsf = np.asarray(cur.loc[fr.run_id], float)
            row = {"pair_id": p.pair_id, "target_id": p.target_id, "source_id": p.source_id, "repeat": rep,
                   "Q_fresh": float(fr.final_J), "Q_warm": float(wr.final_J), "sigma": float(sig[p.target_id])}
            for q in probe_sizes:
                row[f"Q_fresh_minus_{q}"] = float(bsf[len(bsf) - q - 1])
            rows.append(row)
    out = pd.DataFrame(rows)
    out["U_r"] = (out.Q_fresh - out.Q_warm) / out.sigma
    for q in probe_sizes:
        out[f"U_reject_{q}"] = (out.Q_fresh - out[f"Q_fresh_minus_{q}"]) / out.sigma
    return out


def probe_policy(train: pd.DataFrame, cal: pd.DataFrame, test: pd.DataFrame, probe: pd.DataFrame,
                 outcomes: pd.DataFrame, cols_abc: list[str], top_cal: pd.DataFrame, top_test: pd.DataFrame,
                 q: int, alpha: float, delta: float, seed: int) -> tuple[dict, pd.DataFrame]:
    """Probe decision with q probed elites for the zero-shot top source j* of each target.

    M6_q (tiers a-d, d = probe features at size q averaged over repeats) is fitted on
    training pairs; lambda_q is chosen on calibration targets using repeat 0 (one
    point per target); on test targets every repeat is decided from its own probe.
    Returns (summary, per-target realised utility).
    """
    pq = probe[probe.probe_q == q]
    pm = pq.drop(columns=["probe_q", "repeat"]).groupby("pair_id").mean()
    dcols = list(pm.columns)
    tr = train.drop(columns=[c for c in train.columns if c.startswith("d_")]).merge(pm, on="pair_id")
    hm = HarmModel("lgbm", seed).fit(tr[cols_abc + dcols].to_numpy(float), tr.harm.to_numpy(int),
                                     tr.target_id.to_numpy())
    pr = pq.set_index(["pair_id", "repeat"])

    def scored(top: pd.DataFrame, df: pd.DataFrame, reps) -> pd.DataFrame:
        base = df.set_index("pair_id")
        rows = []
        for _, t in top.iterrows():
            for r in reps(t):
                x = np.r_[base.loc[t.pair_id, cols_abc].to_numpy(float), pr.loc[(t.pair_id, r), dcols].to_numpy(float)]
                rows.append({"target_id": t.target_id, "pair_id": t.pair_id, "repeat": r, "harm": int(t.harm),
                             "p_probe": float(hm.predict_proba(x[None, :])[0])})
        return pd.DataFrame(rows)

    sc_cal = scored(top_cal, cal, lambda t: [0])
    lam = choose_threshold(sc_cal.p_probe.to_numpy(), sc_cal.harm.to_numpy(), alpha, delta)
    reps_of = outcomes.groupby("pair_id")["repeat"].apply(list)
    sc = scored(top_test, test, lambda t: reps_of[t.pair_id])
    o = outcomes.set_index(["pair_id", "repeat"])
    acc = (sc.p_probe <= lam).to_numpy() if lam is not None else np.zeros(len(sc), bool)
    sc["accept"] = acc
    sc["U_real"] = [o.loc[(p, r), "U_r"] if a else o.loc[(p, r), f"U_reject_{q}"]
                    for p, r, a in zip(sc.pair_id, sc["repeat"], acc)]
    per_t = sc.groupby("target_id").agg(U=("U_real", "mean"), accept_share=("accept", "mean"),
                                        harm=("harm", "first")).reset_index()
    summary = {"decision_time_evals": q, "lambda": lam, "mean_utility": float(per_t.U.mean()),
               "abstention_rate": float(1 - acc.mean()),
               "harm_rate_among_accepted": float(sc[acc].harm.mean()) if acc.any() else 0.0,
               "n_decisions": int(len(sc))}
    return summary, per_t
