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
