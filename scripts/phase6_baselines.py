"""Phase 6: every comparison method on the same test targets, budget and seeds; table T2 and figure F7.

Requires Phase 5 outputs (M5 predictions for calibration/test pairs, probe per-target outcomes).
Run-based baselines (B2-gamma, B3, B4) use their papers' settings (no tuning), so they are run
on test targets only; look-up policies are evaluated from the measured pair table.
"""
from __future__ import annotations

import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tle.analysis.policies import (WATER, choose_by, historical_bins, lookup_policy, run_based_policy,
                                   water_outcomes)
from tle.analysis.stats import clopper_pearson, compare_to_reference, friedman
from tle.analysis.tables import read_table
from tle.baselines.jobs import baseline_job, fit_task_model, target_inputs, theta_matrix
from tle.config import load_config, provenance, repo_path
from tle.de.jobs import run_jobs
from tle.models.ablation import design
from tle.models.gate import choose_threshold, oracle_utility
from tle.scenarios.generator import read_scenarios
from tle.seeds import rng

PROBE_Q_HEADLINE = 8  # pre-specified probe size for B7 (all injected elites probed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e1.yaml")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed, eps = int(cfg["seed"]), float(cfg["labels"]["epsilon"])
    alphas, delta = [float(a) for a in cfg["gate"]["alpha"]], float(cfg["gate"]["delta"])
    res_dir = repo_path(cfg["results_dir"])
    out = res_dir / "phase6"
    out.mkdir(parents=True, exist_ok=True)
    scen = read_scenarios(cfg["scenarios"]["table"])
    pairs, feats, probe, runs, ref = (read_table(cfg, n) for n in ("pairs", "features", "probe", "runs", "reference"))
    df = design(feats, pairs, probe)
    test_ids = sorted(df[df.split == "test"].target_id.unique())

    # ---------------- run-based baselines on test targets
    tm = fit_task_model(cfg, scen, ref)
    th = theta_matrix(cfg, scen.set_index("scenario_id", drop=False).loc[test_ids].reset_index(drop=True))
    by_id = scen.set_index("scenario_id", drop=False)
    jobs = []
    for tid in test_ids:
        archives, sims, _ = target_inputs(cfg, pairs, ref, tid)
        x_hat = tm.predict(th.loc[tid].to_numpy())
        for rep in range(int(cfg["pairs"]["test_repeats"])):
            jobs.append((res_dir / "jobs" / "baselines" / f"{tid}_r{rep:02d}.joblib", baseline_job,
                         (cfg, by_id.loc[tid], rep, archives, sims, x_hat)))
    print("baseline jobs run:", run_jobs(jobs, args.jobs))
    rows = []
    for path, *_ in jobs:
        rec = joblib.load(path)
        for m, r in rec["runs"].items():
            rows.append({"method": m, "target_id": rec["target_id"], "repeat": rec["repeat"], "bsf": r["bsf"],
                         **r["summary"]})
    base_runs = pd.DataFrame(rows)
    base_runs.drop(columns=["bsf"]).to_csv(out / "baseline_runs.csv", index=False)

    # ---------------- look-up policies
    water = water_outcomes(runs)
    tp = pairs[pairs.target_id.isin(test_ids)]
    test = df[df.split == "test"]
    pred = pd.read_csv(res_dir / "phase5" / "predictions_cal_test_M5.csv", dtype={"target_id": str, "source_id": str})
    pred_t = pred[pred.split == "test"].merge(test[["pair_id", "a_scenario_distance", "c_state_cosine"]], on="pair_id")
    pred_c = pred[pred.split == "calibration"]
    top_c = pred_c.sort_values(["target_id", "u_q10", "p_harm"], ascending=[True, False, True]).groupby("target_id").head(1)
    hist = historical_bins(df[df.split == "train"])
    test_h = test.assign(u_hist=hist(test))

    pol = [
        lookup_policy("B0 fresh", pd.DataFrame({"target_id": test_ids, "source_id": [None] * len(test_ids)}), tp, water),
        lookup_policy("B1 nearest", choose_by(pred_t.assign(s=-pred_t.a_scenario_distance), "s"), tp, water),
        lookup_policy("B2 similarity top-1", choose_by(pred_t, "c_state_cosine"), tp, water),
        lookup_policy("B5 historical", choose_by(test_h, "u_hist", lambda r: r.u_hist > 0), tp, water),
        lookup_policy("M5 ungated (argmax u_q10)", choose_by(pred_t, "u_q10"), tp, water),
    ]
    for a in alphas:
        lam = choose_threshold(top_c.p_harm.to_numpy(), top_c.harm.to_numpy(), a, delta)
        acc = (lambda r, lam=lam: lam is not None and r.p_harm <= lam)
        top_t = pred_t.sort_values(["target_id", "u_q10", "p_harm"], ascending=[True, False, True]).groupby(
            "target_id").head(1)
        pol.append(lookup_policy(f"B6 gate a={a:.2f}", choose_by(top_t.assign(s=1.0), "s", acc), tp, water))
    orc = tp.loc[tp.groupby("target_id").U_mean.idxmax(), ["target_id", "source_id", "U_mean"]]
    orc["source_id"] = np.where(orc.U_mean > 0, orc.source_id, None)
    pol.append(lookup_policy("Oracle", orc[["target_id", "source_id"]], tp, water))
    for m in ("B2g", "B3", "B4"):
        pol.append(run_based_policy(m, base_runs, pairs, runs, eps).replace({"method": {
            "B2g": "B2g SEETO-style (top-5 softmax)", "B3": "B3 BCKT-style", "B4": "B4 PMTO-style"}}))
    pt = pd.read_csv(res_dir / "phase5" / "probe_per_target.csv", dtype={"target_id": str})
    pt = pt[pt.probe_q == PROBE_Q_HEADLINE]
    pol.append(pd.DataFrame({"method": f"B7 probe q={PROBE_Q_HEADLINE}", "target_id": pt.target_id,
                             "accepted": pt.accept_share > 0, "U": pt.U, "harm": pt.harm * (pt.accept_share > 0)}))
    per_target = pd.concat(pol, ignore_index=True)
    per_target.to_csv(out / "per_target_outcomes.csv", index=False)

    # ---------------- table T2
    wide = per_target.pivot(index="target_id", columns="method", values="U")
    ref_method = f"B6 gate a={max(alphas):.2f}"
    comp = compare_to_reference(wide, ref_method, rng=rng(seed, "bootstrap", 6))
    oracle = oracle_utility(tp)
    extra = []
    for m, g in per_target.groupby("method"):
        acc = g[g.accepted.astype(bool)]
        k, n = int(acc.harm.sum()), len(acc)
        lo, hi = clopper_pearson(k, n)
        row = {"method": m, "abstention_rate": 1 - n / len(g), "harm_rate_among_accepted": k / n if n else 0.0,
               "harm_CP_lo": lo, "harm_CP_hi": hi, "n_harmful": k,
               "regret_to_oracle": float((oracle.loc[g.target_id].to_numpy() - g.U.to_numpy()).mean()),
               "share_oracle_captured": float(g.U.sum() / oracle.loc[g.target_id].sum())}
        if "E" in g and g.E.notna().all():
            row["mean_evals_to_target"] = float(g.E.mean())
            row["mean_transfer_gain"] = float(((g.E_fresh - g.E) / g.E_fresh).mean())
            for w in WATER:
                row[f"mean_{w}"] = float(g[f"w_{w}"].mean())
        extra.append(row)
    t2 = comp.merge(pd.DataFrame(extra), on="method").sort_values("mean_U", ascending=False)
    walls = base_runs.groupby("method").wall_s.mean().rename(lambda m: {"B2g": "B2g SEETO-style (top-5 softmax)",
                                                                       "B3": "B3 BCKT-style",
                                                                       "B4": "B4 PMTO-style"}[m])
    t2["mean_wall_s_per_run"] = t2.method.map(walls)
    t2.to_csv(out / "t2_main_results.csv", index=False)
    print(t2.round(3).to_string())
    fr = friedman(wide.dropna(axis=1))

    order = list(t2.method)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot([wide[m].dropna() for m in order], vert=True, widths=0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("realised utility per test target (σ_fresh units)")
    ax.set_title(f"F7: policy comparison on {len(wide)} test targets (Friedman p={fr['pvalue']:.2g})")
    fig.tight_layout()
    fig.savefig(repo_path("figures") / "f7_policy_comparison.png", dpi=150)

    summary = {"reference_method": ref_method, "friedman": fr, "n_test_targets": len(test_ids),
               "pmto_task_model_training_tasks": tm.n_train,
               "bckt_mean_transfers_per_run": float(base_runs[base_runs.method == "B3"].bckt_n_transfers.mean()),
               "provenance": provenance(cfg)}
    (out / "phase6_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
