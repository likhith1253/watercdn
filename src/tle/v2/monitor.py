"""V2-A: trajectory-monitored transfer (docs/v2_plan.md).

Stage 1 picks the source with the largest predicted utility lower bound (zero-shot ranking). Stage 2 watches
the warm run and, at checkpoint generation g, either continues or aborts to a fresh restart.

Exact offline outcomes (common random numbers):
  continue -> Q_warm (the warm run's final quality)
  abort    -> Q_fresh(B - c), the fresh run's best-so-far after B - c evaluations, c = evaluations spent by the
              warm run at the end of generation g (the restart repeats the fresh run with c fewer evaluations).

Trajectory features at g (all available at decision time):
  mon_gap      log best_warm(g) - log(anchor * r_fresh(g)): warm progress against the expected fresh trajectory,
               anchor = best of the N - m random generation-0 members (shared with the fresh run),
               r_fresh(g) = predicted fresh progress ratio best_fresh(g)/anchor (regressor fitted on training
               targets' fresh runs from scenario features only)
  mon_impr     log best_warm(0) - log best_warm(g)
  mon_div      diversity(g) / diversity(0)
  mon_feas     feasible share of the warm population at g
  mon_impr_rate improvement rate of the warm best over the last generations
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..models.gate import choose_threshold
from ..models.predictors import HarmModel

CHECKPOINTS = (1, 2, 4, 8)


def target_columns(features: pd.DataFrame) -> list[str]:
    """Scenario/hydraulic features of the target alone (no source information)."""
    return [c for c in features.columns if c.startswith(("a_t_", "c_tref_"))]


def _gen_table(gens: pd.DataFrame) -> dict:
    return {rid: g.sort_values("gen") for rid, g in gens.groupby("run_id")}


def fresh_progress(runs: pd.DataFrame, gens: pd.DataFrame, features: pd.DataFrame, pairs: pd.DataFrame,
                   m: int, train_targets: set[str]) -> dict[int, object]:
    """Ridge regressors of log(best_fresh(g) / anchor) from target features, one per checkpoint."""
    G = _gen_table(gens)
    tf = features.merge(pairs[["pair_id", "target_id"]], on="pair_id").groupby("target_id")[
        target_columns(features)].first()
    fr = runs[(runs.role == "target") & (runs.method == "fresh") & runs.scenario_id.isin(train_targets)]
    models = {}
    for g in CHECKPOINTS:
        X, y = [], []
        for _, r in fr.iterrows():
            gl = G[r.run_id]
            if len(gl) <= g:
                continue
            anchor = float(np.min(np.asarray(r.gen0_J)[m:]))
            X.append(tf.loc[r.scenario_id].to_numpy(float))
            y.append(np.log(gl.best_J.iloc[g]) - np.log(anchor))
        models[g] = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(np.array(X), np.array(y))
    return models


def monitor_rows(pairs: pd.DataFrame, runs: pd.DataFrame, gens: pd.DataFrame, curves: pd.DataFrame,
                 features: pd.DataFrame, progress: dict, eps: float) -> pd.DataFrame:
    """One row per (pair, repeat, checkpoint) with trajectory features and exact continue/abort outcomes."""
    G = _gen_table(gens)
    cur = curves.set_index("run_id").bsf
    tcols = target_columns(features)
    tf = features.merge(pairs[["pair_id", "target_id"]], on="pair_id").groupby("target_id")[tcols].first()
    t = runs[runs.role == "target"]
    fresh = t[t.method == "fresh"].set_index(["scenario_id", "repeat"])
    warm = t[t.method == "warm"].set_index(["scenario_id", "source_id", "repeat"])
    sig = pairs.set_index("pair_id").sigma_fresh
    rows = []
    for _, p in pairs.iterrows():
        rhat = {g: float(progress[g].predict(tf.loc[[p.target_id]].to_numpy(float))[0]) for g in CHECKPOINTS}
        for rep in fresh.loc[p.target_id].index:
            fr = fresh.loc[(p.target_id, rep)]
            wr = warm.loc[(p.target_id, p.source_id, rep)]
            m = int(wr.m_injected)
            anchor = float(np.min(np.asarray(wr.gen0_J)[m:]))
            gl = G[wr.run_id]
            bsf_f = np.asarray(cur.loc[fr.run_id], float)
            B = len(bsf_f)
            qf, qw = float(fr.final_J), float(wr.final_J)
            s = float(sig[p.pair_id])
            for g in CHECKPOINTS:
                if len(gl) <= g:
                    continue
                row_g = gl.iloc[g]
                c = int(row_g.evals)
                q_abort = float(bsf_f[B - c - 1])
                bw = float(row_g.best_J)
                rows.append({
                    "pair_id": p.pair_id, "target_id": p.target_id, "source_id": p.source_id, "split": p.split,
                    "repeat": int(rep), "g": g, "spent": c,
                    "mon_gap": np.log(bw) - (np.log(anchor) + rhat[g]),
                    "mon_impr": float(np.log(gl.best_J.iloc[0]) - np.log(bw)),
                    "mon_div": float(row_g.diversity / max(gl.diversity.iloc[0], 1e-12)),
                    "mon_feas": float(row_g.share_feasible),
                    "mon_impr_rate": float(row_g.improvement_rate),
                    "U_continue": (qf - qw) / s, "U_abort": (qf - q_abort) / s,
                    "harm_continue": int((qf - qw) / s < -eps),
                })
    return pd.DataFrame(rows)


MON_COLS = ["mon_gap", "mon_impr", "mon_div", "mon_feas", "mon_impr_rate"]


def score_monitor(mon: pd.DataFrame, design_df: pd.DataFrame, pred: pd.DataFrame, cols: list[str], g: int,
                  seed: int, use_pair_features: bool = True) -> pd.DataFrame:
    """Fit the checkpoint-g harm model on training rows; score the stage-1 top source of every other target."""
    fcols = (cols if use_pair_features else []) + MON_COLS
    d = mon[mon.g == g].merge(design_df[["pair_id"] + cols], on="pair_id")
    tr = d[d.split == "train"]
    hm = HarmModel("lgbm", seed).fit(tr[fcols].to_numpy(float), tr.harm_continue.to_numpy(int),
                                     tr.target_id.to_numpy())
    top = pred.sort_values(["target_id", "u_q10", "p_harm"], ascending=[True, False, True]).groupby("target_id").head(1)
    sel = d[d.pair_id.isin(set(top.pair_id))].copy()
    sel["p"] = hm.predict_proba(sel[fcols].to_numpy(float))
    return sel


def evaluate_monitor(sel: pd.DataFrame, cal_ids, test_ids, alpha: float, delta: float) -> tuple[dict, pd.DataFrame]:
    """lambda_g by LTT on calibration targets (first repeat of each, independent points); test per repeat."""
    cal = sel[sel.target_id.isin(set(cal_ids))]
    cal = cal[cal.repeat == cal.groupby("target_id").repeat.transform("min")]
    lam = choose_threshold(cal.p.to_numpy(), cal.harm_continue.to_numpy(), alpha, delta)
    te = sel[sel.target_id.isin(set(test_ids))].copy()
    te["continue"] = (te.p <= lam).to_numpy() if lam is not None else np.zeros(len(te), bool)
    te["U"] = np.where(te["continue"], te.U_continue, te.U_abort)
    cont = te[te["continue"]]
    per_t = te.groupby("target_id").agg(U=("U", "mean"), U_no_monitor=("U_continue", "mean"),
                                        abort_share=("continue", lambda s: 1 - s.mean())).reset_index()
    best = te[["U_continue", "U_abort"]].max(axis=1)
    return {"lambda": lam, "n_decisions": int(len(te)), "abort_rate": float(1 - te["continue"].mean()),
            "harm_rate_among_continued": float(cont.harm_continue.mean()) if len(cont) else 0.0,
            "n_continued": int(len(cont)),
            "harm_rate_no_monitor": float(te.harm_continue.mean()),
            "mean_U": float(per_t.U.mean()), "mean_U_no_monitor": float(per_t.U_no_monitor.mean()),
            "oracle_repeat_U": float(best.groupby(te.target_id).mean().mean())}, per_t


def utility_threshold(sel: pd.DataFrame, cal_ids) -> float:
    """Exploratory, no guarantee: abort threshold that maximises mean realised utility on calibration targets."""
    cal = sel[sel.target_id.isin(set(cal_ids))]
    cands = np.r_[np.unique(cal.p), np.inf]
    best, best_u = np.inf, -np.inf
    for lam in cands:
        u = np.where(cal.p <= lam, cal.U_continue, cal.U_abort).mean()
        if u > best_u:
            best, best_u = float(lam), float(u)
    return best


def evaluate_fixed(sel: pd.DataFrame, test_ids, lam: float) -> dict:
    te = sel[sel.target_id.isin(set(test_ids))]
    cont = (te.p <= lam).to_numpy()
    U = np.where(cont, te.U_continue, te.U_abort)
    return {"lambda": lam, "abort_rate": float(1 - cont.mean()), "mean_U": float(pd.Series(U).groupby(
        te.target_id.to_numpy()).mean().mean()), "harm_rate_among_continued": float(te.harm_continue[cont].mean())
        if cont.any() else 0.0, "mean_U_no_monitor": float(te.groupby("target_id").U_continue.mean().mean())}
