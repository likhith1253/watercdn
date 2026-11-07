"""Per-target outcomes of every decision policy (P6.1, P8.2).

Look-up policies choose among the measured candidate sources of a target (or
abstain); their outcome is read from the pair table (utility, harm,
evaluations to target) and the run table (water outcomes). Methods that
abstain reuse the fresh run of the target (fairness rule P6.2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import evals_to_target

WATER = ["leak_m3", "energy_cost", "best_p_min", "shortfall_mh", "feasible"]


def water_outcomes(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = runs[runs.role == "target"]
    fresh = t[t.method == "fresh"].groupby("scenario_id")[WATER].mean()
    warm = t[t.method == "warm"].groupby(["scenario_id", "source_id"])[WATER].mean()
    return fresh, warm


def lookup_policy(name: str, choice: pd.DataFrame, pairs: pd.DataFrame, runs_water) -> pd.DataFrame:
    """choice: target_id, source_id (None = abstain)."""
    fresh_w, warm_w = runs_water
    P = pairs.set_index(["target_id", "source_id"])
    E_fresh = pairs.groupby("target_id").E_fresh.first()
    rows = []
    for _, c in choice.iterrows():
        acc = c.source_id is not None and isinstance(c.source_id, str)
        if acc:
            pr = P.loc[(c.target_id, c.source_id)]
            w = warm_w.loc[(c.target_id, c.source_id)]
            row = {"U": pr.U_mean, "harm": int(pr.harm), "E": pr.E_warm}
        else:
            w = fresh_w.loc[c.target_id]
            row = {"U": 0.0, "harm": 0, "E": E_fresh[c.target_id]}
        rows.append({"method": name, "target_id": c.target_id, "accepted": acc,
                     "source_id": c.source_id if acc else None, **row,
                     "E_fresh": E_fresh[c.target_id], **{f"w_{k}": w[k] for k in WATER}})
    return pd.DataFrame(rows)


def choose_by(pred: pd.DataFrame, score: str, accept=None) -> pd.DataFrame:
    """Pick the candidate with the largest score per target; ``accept(row) -> bool`` gates it."""
    top = pred.sort_values(["target_id", score], ascending=[True, False]).groupby("target_id").head(1)
    out = top[["target_id", "source_id"]].copy()
    if accept is not None:
        keep = top.apply(accept, axis=1).to_numpy()
        out["source_id"] = np.where(keep, out.source_id, None)
    return out.reset_index(drop=True)


def historical_bins(train: pd.DataFrame, cols=("a_abs_d_log_area", "a_abs_d_demand_mult"), q: int = 3):
    """B5: mean training utility per bin of parameter differences (+ zone mismatch)."""
    edges = {c: np.unique(np.quantile(train[c], np.linspace(0, 1, q + 1))) for c in cols}

    def key(df):
        k = [np.clip(np.searchsorted(edges[c], df[c], side="right") - 1, 0, len(edges[c]) - 2) for c in cols]
        return list(zip(*k, df.a_zone_mismatch.astype(int)))

    table = pd.Series(train.U_mean.to_numpy(), index=pd.MultiIndex.from_tuples(key(train))).groupby(level=[0, 1, 2]).mean()
    glob = float(train.U_mean.mean())

    def predict(df):
        return np.array([table.get(k, glob) for k in key(df)])

    return predict


def run_based_policy(name: str, base_runs: pd.DataFrame, pairs: pd.DataFrame, runs: pd.DataFrame,
                     eps: float) -> pd.DataFrame:
    """Utility of a run-based baseline vs the target's fresh runs (same repeats, same sigma_fresh)."""
    t = runs[(runs.role == "target") & (runs.method == "fresh")]
    sig = pairs.groupby("target_id").sigma_fresh.first()
    qstar = pairs.groupby("target_id").Q_star.first()
    E_fresh = pairs.groupby("target_id").E_fresh.first()
    rows = []
    for tid, g in base_runs[base_runs.method == name].groupby("target_id"):
        f = t[t.scenario_id == tid].set_index("repeat").final_J
        q = g.set_index("repeat").final_J
        common = f.index.intersection(q.index)
        U = float((f.loc[common] - q.loc[common]).mean() / sig[tid])
        E = float(np.mean([evals_to_target(b, qstar[tid]) for b in g.bsf]))
        rows.append({"method": name, "target_id": tid, "accepted": True, "source_id": None, "U": U,
                     "harm": int(U < -eps), "E": E, "E_fresh": E_fresh[tid],
                     **{f"w_{k}": float(g[k].mean()) for k in WATER}})
    return pd.DataFrame(rows)
