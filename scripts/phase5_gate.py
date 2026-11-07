"""Phase 5: calibrated gate, 200 re-splits, trade-off curve, operator check, probe mode, final test evaluation.

Models are fitted on training targets only; lambda is chosen on calibration
targets; test targets are read once for the final evaluation (offline look-up
of measured pair outcomes).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from tle.analysis.tables import read_table
from tle.config import load_config, provenance, repo_path
from tle.features.build import PROBE_SIZES
from tle.models.ablation import design, model_specs
from tle.models.gate import (choose_threshold, decide, evaluate_decisions, oracle_utility, resplit_experiment,
                             top_source)
from tle.models.policy import probe_policy, repeat_outcomes
from tle.models.fit import fit_predict
from tle.seeds import rng

N_SPLITS = 200
ALPHA_CURVE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e1.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed, eps = int(cfg["seed"]), float(cfg["labels"]["epsilon"])
    alphas, delta = [float(a) for a in cfg["gate"]["alpha"]], float(cfg["gate"]["delta"])
    out = repo_path(cfg["results_dir"]) / "phase5"
    out.mkdir(parents=True, exist_ok=True)
    pairs, feats, probe = read_table(cfg, "pairs"), read_table(cfg, "features"), read_table(cfg, "probe")
    runs, curves = read_table(cfg, "runs"), read_table(cfg, "curves")
    df = design(feats, pairs, probe)
    train, cal, test = (df[df.split == s].reset_index(drop=True) for s in ("train", "calibration", "test"))
    specs = model_specs(train)
    cols = specs["M5"]
    oracle = oracle_utility(pairs)

    pred, um, hm = fit_predict(train, pd.concat([cal, test], ignore_index=True), cols, seed)
    pred.to_csv(out / "predictions_cal_test_M5.csv", index=False)
    top_all = top_source(pred)
    top_cal = top_all[top_all.split == "calibration"].reset_index(drop=True)

    # ---- re-split experiment on pooled calibration + test targets (empirical check of the guarantee)
    rs = resplit_experiment(top_all, oracle, sorted(set(ALPHA_CURVE) | set(alphas)), delta, n_cal=len(cal.target_id.unique()),
                            n_splits=N_SPLITS, rng=rng(seed, "resplit", 0), eps=eps)
    rs.to_csv(out / "resplits.csv", index=False)
    rs_sum = rs.groupby("alpha").agg(
        mean_harm_rate=("harm_rate_among_accepted", "mean"), mean_abstention=("abstention_rate", "mean"),
        mean_utility=("mean_utility", "mean"), share_never_transfer=("lambda", lambda s: float(s.isna().mean())),
    ).reset_index()
    rs_sum["share_splits_harm_le_alpha"] = [float((rs[rs.alpha == a].harm_rate_among_accepted <= a + 1e-12).mean())
                                            for a in rs_sum.alpha]
    rs_sum.to_csv(out / "resplit_summary.csv", index=False)
    print(rs_sum.round(3).to_string())

    # ---- operator check (same predictor): threshold on p_harm, on u_q10, combined
    top_all["neg_u_q10"] = -top_all.u_q10
    top_all["combined"] = np.where(top_all.u_mean > 0, top_all.p_harm, 1.0 + top_all.p_harm)
    ops = []
    for name, col in (("p_harm", "p_harm"), ("u_q10", "neg_u_q10"), ("combined", "combined")):
        r = resplit_experiment(top_all, oracle, alphas, delta, len(cal.target_id.unique()), N_SPLITS,
                               rng(seed, "resplit", 1), eps, score_col=col)
        for a, g in r.groupby("alpha"):
            ops.append({"rule": name, "alpha": a, "share_splits_harm_le_alpha": float((g.harm_rate_among_accepted <= a).mean()),
                        "mean_utility": g.mean_utility.mean(), "mean_abstention": g.abstention_rate.mean()})
    pd.DataFrame(ops).to_csv(out / "operator_check.csv", index=False)

    # ---- final evaluation on the untouched test targets (read once)
    first_access = dt.datetime.now().isoformat(timespec="seconds")
    top_test = top_all[top_all.split == "test"].reset_index(drop=True)
    final = []
    for a in alphas:
        lam = choose_threshold(top_cal.p_harm.to_numpy(), top_cal.harm.to_numpy(), a, delta)
        res = evaluate_decisions(top_test, decide(top_test, lam), oracle, eps)
        k, n = res["n_harm_accepted"], res["n_accepted"]
        lo = float(beta_dist.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
        hi = float(beta_dist.ppf(0.975, k + 1, n - k)) if n > k else 1.0
        final.append({"alpha": a, "lambda": lam, **res, "harm_rate_CP_lo": lo, "harm_rate_CP_hi": hi})
    final = pd.DataFrame(final)
    final.to_csv(out / "final_test_gate.csv", index=False)
    models_dir = repo_path(cfg["results_dir"]) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"utility_model": um, "harm_model": hm, "feature_columns": cols,
                 "lambda": {float(a): l for a, l in zip(final.alpha, final["lambda"])},
                 "delta": delta, "provenance": provenance(cfg)}, models_dir / "gate_M5.joblib")
    print(final.round(3).to_string())

    # ---- probe mode (C4) at the main alpha
    outcomes = repeat_outcomes(runs, curves, pairs)
    outcomes.to_parquet(out / "repeat_outcomes.parquet", index=False)
    main_alpha = max(alphas)
    probe_rows, per_target = [], []
    for q in PROBE_SIZES:
        summ, pt = probe_policy(train, cal, test, probe, outcomes, cols, top_cal, top_test, q, main_alpha, delta, seed)
        probe_rows.append(summ)
        per_target.append(pt.assign(probe_q=q))
    pd.concat(per_target).to_csv(out / "probe_per_target.csv", index=False)
    pm = pd.DataFrame(probe_rows)
    zs = final[final.alpha == main_alpha].iloc[0]
    pm = pd.concat([pd.DataFrame([{"decision_time_evals": 0, "lambda": zs["lambda"], "mean_utility": zs.mean_utility,
                                   "abstention_rate": zs.abstention_rate,
                                   "harm_rate_among_accepted": zs.harm_rate_among_accepted,
                                   "n_decisions": zs.n_targets}]), pm], ignore_index=True)
    pm.to_csv(out / "probe_mode.csv", index=False)
    print(pm.round(3).to_string())

    # ---- figures F4, F5, F6
    fig, ax = plt.subplots(figsize=(5, 4))
    data = [rs[rs.alpha == a].harm_rate_among_accepted for a in ALPHA_CURVE]
    ax.boxplot(data, positions=range(len(ALPHA_CURVE)), widths=0.5)
    ax.plot(range(len(ALPHA_CURVE)), ALPHA_CURVE, "r--", label="α (target)")
    ax.set_xticks(range(len(ALPHA_CURVE)))
    ax.set_xticklabels(ALPHA_CURVE)
    ax.set_xlabel("α")
    ax.set_ylabel("realised harm rate among accepted (held-out)")
    ax.set_title(f"F4: {N_SPLITS} random calibration/test re-splits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(repo_path("figures") / "f4_harm_vs_alpha.png", dpi=150)

    s = rs_sum[rs_sum.alpha.isin(ALPHA_CURVE)]
    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(s.alpha, s.mean_abstention, "o-", label="abstention rate")
    ax1.set_xlabel("α")
    ax1.set_ylabel("abstention rate")
    ax2 = ax1.twinx()
    ax2.plot(s.alpha, s.mean_utility, "s-", color="C1", label="mean utility")
    ax2.set_ylabel("mean realised utility (σ units)")
    ax1.set_title("F5: abstention and utility vs α")
    fig.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(repo_path("figures") / "f5_abstention_utility_vs_alpha.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(pm.decision_time_evals, pm.mean_utility, "o-")
    ax.set_xlabel("decision time (probed injected elites = evaluations spent)")
    ax.set_ylabel("mean realised utility on test targets")
    ax.set_title(f"F6: zero-shot (0) vs probe decisions, α={main_alpha}")
    fig.tight_layout()
    fig.savefig(repo_path("figures") / "f6_probe_decision_time.png", dpi=150)

    summary = {"test_targets_first_accessed": first_access, "alphas": alphas, "delta": delta,
               "resplit_summary": rs_sum.to_dict(orient="records"), "final_test": final.to_dict(orient="records"),
               "provenance": provenance(cfg)}
    (out / "phase5_summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
