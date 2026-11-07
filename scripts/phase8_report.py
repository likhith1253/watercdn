"""Phase 8: amortisation (F9), pipeline diagram (F1), reproducibility cross-check, RESULTS.md.

Every number written here is read from result files produced by the earlier scripts.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tle.config import load_config, provenance, repo_path

FIG = repo_path("figures")


def pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axis("off")
    boxes = [
        (0.02, 0.62, "New leak scenario θ*\n(leak node, size, demand,\nshape, tank level)"),
        (0.02, 0.12, "Library of past runs\nθ_j, trajectory τ_j,\nelites E_j, signature h_j"),
        (0.27, 0.37, "Pair features\n(tiers a, b, c; probe d)"),
        (0.47, 0.37, "Predictor\nû_j (10% quantile),\np̂_harm,j"),
        (0.67, 0.37, "Conformal gate\nLearn-then-Test λ:\nharm | accept ≤ α"),
        (0.86, 0.62, "Transfer: inject\ntop-k elites of j*"),
        (0.86, 0.12, "Abstain:\nfresh DE"),
    ]
    for x, y, t in boxes:
        ax.text(x, y, t, fontsize=8, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="#eef3fb", ec="#4a6fa5"))
    arrows = [((0.17, 0.70), (0.27, 0.47)), ((0.17, 0.20), (0.27, 0.42)), ((0.40, 0.45), (0.47, 0.45)),
              ((0.60, 0.45), (0.67, 0.45)), ((0.80, 0.50), (0.86, 0.68)), ((0.80, 0.42), (0.86, 0.20))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(0.42, 0.02, "Differential Evolution + EPANET 2.2 (same budget B for every method); run logged into the library",
            fontsize=8, ha="center")
    ax.set_title("F1: the decision layer sits in front of an unchanged optimiser, simulator and transfer action", fontsize=9)
    fig.savefig(FIG / "f1_pipeline.png", dpi=150, bbox_inches="tight")


def amortisation(cfg: dict) -> dict:
    res = repo_path(cfg["results_dir"])
    B = int(cfg["de"]["budget_B"])
    S = 3 * int(cfg["pairs"]["per_distance_tertile"])  # sources per target: near, mid, far tertiles
    R = int(cfg["pairs"]["repeats"])
    sp = cfg["splits"]
    per = pd.read_csv(res / "phase6" / "per_target_outcomes.csv", dtype={"target_id": str})
    ref = f"B6 gate a={max(cfg['gate']['alpha']):.2f}"
    g = per[per.method == ref]
    savings = float((g.E_fresh - g.E).mean())
    c_lib = sp["library"] * B
    c_lab_train = sp["train"] * R * (S + 1) * B
    c_lab_all = (sp["train"] + sp["calibration"]) * R * (S + 1) * B
    out = {"policy": ref, "B": B, "mean_evals_saved_per_task": savings,
           "C_library": c_lib, "C_labels_train": c_lab_train, "C_labels_train_and_calibration": c_lab_all}
    if savings > 0:
        out["break_even_dedicated_train_only"] = (c_lib + c_lab_train) / savings
        out["break_even_byproduct_train_only"] = c_lab_train / savings
        out["break_even_dedicated_with_calibration"] = (c_lib + c_lab_all) / savings
        out["break_even_byproduct_with_calibration"] = c_lab_all / savings
        n = np.arange(0, int(out["break_even_dedicated_with_calibration"] * 1.3) + 10)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(n, n * float(g.E_fresh.mean()), label="fresh DE (evaluations to Q*)")
        ax.plot(n, c_lib + c_lab_all + n * float(g.E.mean()), label="policy + library + labels (dedicated)")
        ax.plot(n, c_lab_all + n * float(g.E.mean()), label="policy + labels (library is a by-product)")
        ax.set_xlabel("number of future leak tasks")
        ax.set_ylabel("cumulative evaluations")
        ax.set_title("F9: does the library pay for itself?")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "f9_break_even.png", dpi=150)
    return out


def cross_check(cfg: dict) -> dict:
    """T6-style check: the sensitivity 'base' setting re-ran the first 5 repeats of 20 training targets in fresh
    worker processes; their best-so-far curves must equal the main campaign's bit for bit."""
    res = repo_path(cfg["results_dir"]) / "jobs"
    files = sorted((res / "sensitivity" / "base").glob("*.joblib"))
    n_eq = n = 0
    for f in files:
        a = joblib.load(f)
        b = joblib.load(res / "targets" / f.name)
        for m, bsf in a["bsf"].items():
            n += 1
            n_eq += int(np.array_equal(bsf, b["runs"][m].bsf.astype(np.float32)))
    return {"runs_compared": n, "identical": n_eq}


def md_table(df: pd.DataFrame, cols: list[str], digits: int = 3) -> str:
    d = df[cols].copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)])


def results_md(cfg: dict, amort: dict, xcheck: dict) -> str:
    r = repo_path("results")
    q = json.loads((r / "e1/reports/phase3_quality.json").read_text())
    L = ["# Results", "", "Generated by `scripts/phase8_report.py` from the files under `results/`; do not edit by hand.",
         "Utilities are in units of the fresh-run seed standard deviation (σ_fresh); positive = transfer helped.", ""]
    qq = q["quality"]
    L += ["## E1 paired dataset (Phase 3)", "",
          f"* {qq['n_pairs']} pairs on {qq['n_targets']} targets; helpful (U > ε) {qq['share_helpful_U_gt_eps']:.1%}, "
          f"neutral {qq['share_neutral']:.1%}, harmful (U < −ε) {qq['share_harmful_U_lt_minus_eps']:.1%}.",
          f"* Mean U {qq['U_mean']:.2f}, spread across pairs {qq['U_std_across_pairs']:.2f}, median standard error "
          f"{qq['U_se_median']:.2f} (noise ratio {qq['noise_floor_ratio_se_over_spread']:.2f}).",
          f"* Unattainable targets (no feasible operation found with 5B): {qq['unattainable_target_share']:.1%}; fresh-run "
          f"feasibility on attainable targets {qq['fresh_feasible_share_on_attainable_targets']:.1%}.",
          f"* Split audit passed: {q['audit']['all_passed']}.", ""]
    t3 = pd.read_csv(r / "e1/phase4/t3_tier_ablation.csv")
    L += ["## T3 feature-tier ablation (training targets, grouped out-of-fold; 95% bootstrap CI over targets)", "",
          md_table(t3, ["model", "spearman", "spearman_lo", "spearman_hi", "mae", "auroc", "auroc_lo", "auroc_hi", "brier"]), ""]
    td = pd.read_csv(r / "e1/phase4/tier_differences.csv")
    L += ["### Paired differences", "", md_table(td, ["comparison", "metric", "diff", "ci_lo", "ci_hi"]), ""]
    rs = pd.read_csv(r / "e1/phase5/resplit_summary.csv")
    L += ["## Gate: 200 random calibration/test re-splits (pooled calibration + test targets)", "",
          md_table(rs, list(rs.columns)), ""]
    fg = pd.read_csv(r / "e1/phase5/final_test_gate.csv")
    L += ["## Gate on the untouched test targets (read once)", "",
          md_table(fg, ["alpha", "lambda", "n_targets", "n_accepted", "harm_rate_among_accepted", "harm_rate_CP_lo",
                        "harm_rate_CP_hi", "abstention_rate", "mean_utility", "share_oracle_captured"]), ""]
    pm = pd.read_csv(r / "e1/phase5/probe_mode.csv")
    L += ["## Probe mode (decision time in evaluations)", "", md_table(pm, list(pm.columns)), ""]
    t2 = pd.read_csv(r / "e1/phase6/t2_main_results.csv")
    cols = [c for c in ["method", "mean_U", "mean_U_ci_lo", "mean_U_ci_hi", "holm_p_vs_ref", "rank_biserial_ref_minus_m",
                        "abstention_rate", "harm_rate_among_accepted", "harm_CP_hi", "share_oracle_captured",
                        "mean_transfer_gain", "mean_leak_m3", "mean_energy_cost", "mean_feasible"] if c in t2]
    L += ["## T2 policy comparison on E1 test targets", "", md_table(t2, cols), ""]
    for e in ("e2", "e3", "e4"):
        f = r / e / "phase7" / "gate_results.csv"
        if f.exists():
            g = pd.read_csv(f)
            L += [f"## {e.upper()}: frozen E1 gate on shifted targets", "",
                  md_table(g, ["group", "lambda_source", "alpha", "n_targets", "n_accepted", "harm_rate_among_accepted",
                               "harm_CP_hi", "abstention_rate", "mean_utility", "share_oracle_captured"]), ""]
            a = pd.read_csv(r / e / "phase7" / "predictor_accuracy.csv")
            L += [md_table(a, ["group", "n_pairs", "harm_share", "U_mean", "spearman", "auroc", "mae"]), ""]
    f = r / "e4/phase7/e4_retrained_gate.csv"
    if f.exists():
        L += ["## E4: gate retrained on L-Town", "", md_table(pd.read_csv(f), ["method", "lambda", "n_targets", "n_accepted",
              "harm_rate_among_accepted", "abstention_rate", "mean_utility", "share_oracle_captured"]), ""]
    f = r / "e1/sensitivity/t4_sensitivity.csv"
    if f.exists():
        L += ["## T4 sensitivity (no retuning)", "", md_table(pd.read_csv(f), list(pd.read_csv(f).columns)), ""]
    L += ["## Amortisation", "", "```", json.dumps(amort, indent=2, default=float), "```", "",
          "## Reproducibility cross-check", "",
          f"{xcheck['identical']} of {xcheck['runs_compared']} re-run best-so-far curves identical to the main campaign.", ""]
    return "\n".join(L)


def main() -> None:
    cfg = load_config("configs/e1.yaml")
    FIG.mkdir(exist_ok=True)
    pipeline_figure()
    amort = amortisation(cfg)
    xcheck = cross_check(cfg)
    out = repo_path(cfg["results_dir"]) / "reports"
    (out / "amortisation.json").write_text(json.dumps(amort, indent=2, default=float))
    (out / "cross_check.json").write_text(json.dumps(xcheck, indent=2))
    Path(repo_path("RESULTS.md")).write_text(results_md(cfg, amort, xcheck))
    print(json.dumps({"amortisation": amort, "cross_check": xcheck, "provenance": provenance(cfg)}, indent=2, default=float))


if __name__ == "__main__":
    main()
