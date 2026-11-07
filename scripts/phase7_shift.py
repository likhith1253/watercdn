"""Phase 7 (E2, E3, E4-zero-shot part): apply the frozen E1 predictor and gate to shifted targets.

For each group (goal for E2, shift level for E3): predictor accuracy, realised
harm among accepted transfers vs alpha, abstention, utility and oracle share
with the frozen E1 lambda. When the experiment has calibration targets, lambda
is also recalibrated on them and a few-shot recalibration curve (m calibration
targets, random subsets) is reported on the test targets.
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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from tle.analysis.stats import clopper_pearson
from tle.analysis.tables import read_table
from tle.config import load_config, provenance, repo_path
from tle.models.ablation import design
from tle.models.gate import choose_threshold, decide, evaluate_decisions, oracle_utility, top_source
from tle.seeds import rng

FEW_SHOT_M = [5, 10, 15, 20, 30]
N_SUBSETS = 200


def predict(gate: dict, df: pd.DataFrame) -> pd.DataFrame:
    X = df[gate["feature_columns"]].to_numpy(float)
    p = gate["utility_model"].predict(X)
    p["p_harm"] = gate["harm_model"].predict_proba(X)
    keep = ["pair_id", "target_id", "source_id", "split", "group", "U_mean", "harm"]
    return pd.concat([df[keep].reset_index(drop=True), p], axis=1)


def evaluate(top, lam, oracle, eps, alpha, label):
    r = evaluate_decisions(top.reset_index(drop=True), decide(top, lam), oracle, eps)
    lo, hi = clopper_pearson(r["n_harm_accepted"], r["n_accepted"])
    return {**label, "alpha": alpha, "lambda": lam, **r, "harm_CP_lo": lo, "harm_CP_hi": hi,
            "harm_le_alpha": r["harm_rate_among_accepted"] <= alpha}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    eps, delta = float(cfg["labels"]["epsilon"]), float(cfg["gate"]["delta"])
    alphas = [float(a) for a in cfg["gate"]["alpha"]]
    exp = cfg["experiment"]
    out = repo_path(cfg["results_dir"]) / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    gate = joblib.load(repo_path(cfg["frozen_gate"]))
    pairs, feats, probe = read_table(cfg, "pairs"), read_table(cfg, "features"), read_table(cfg, "probe")
    if "group" not in pairs:
        pairs["group"] = "all"
    df = design(feats, pairs, probe)
    pred = predict(gate, df)
    pred.to_csv(out / "predictions_frozen.csv", index=False)
    oracle = oracle_utility(pairs)

    acc_rows, gate_rows, few = [], [], []
    for grp, g in pred.groupby("group"):
        auc = roc_auc_score(g.harm, g.p_harm) if g.harm.nunique() > 1 else float("nan")
        acc_rows.append({"experiment": exp, "group": grp, "n_pairs": len(g), "harm_share": float(g.harm.mean()),
                         "U_mean": float(g.U_mean.mean()), "spearman": float(spearmanr(g.u_mean, g.U_mean).statistic),
                         "auroc": float(auc), "mae": float((g.u_mean - g.U_mean).abs().mean())})
        top = top_source(g)
        top_test = top[top.split == "test"]
        top_cal = top[top.split == "calibration"]
        for a in alphas:
            gate_rows.append(evaluate(top_test, gate["lambda"][a], oracle, eps, a,
                                      {"experiment": exp, "group": grp, "lambda_source": "frozen E1"}))
            if len(top_cal):
                lam = choose_threshold(top_cal.p_harm.to_numpy(), top_cal.harm.to_numpy(), a, delta)
                gate_rows.append(evaluate(top_test, lam, oracle, eps, a,
                                          {"experiment": exp, "group": grp, "lambda_source": "recalibrated"}))
                r = rng(int(cfg["seed"]), "resplit", 70)
                for m in [m for m in FEW_SHOT_M if m <= len(top_cal)]:
                    for s in range(N_SUBSETS):
                        sub = top_cal.iloc[r.choice(len(top_cal), m, replace=False)]
                        lam_m = choose_threshold(sub.p_harm.to_numpy(), sub.harm.to_numpy(), a, delta)
                        res = evaluate_decisions(top_test.reset_index(drop=True), decide(top_test, lam_m), oracle, eps)
                        few.append({"experiment": exp, "group": grp, "alpha": a, "m": m, "subset": s,
                                    "never_transfer": lam_m is None, **res})
    acc = pd.DataFrame(acc_rows)
    gates = pd.DataFrame(gate_rows)
    acc.to_csv(out / "predictor_accuracy.csv", index=False)
    gates.to_csv(out / "gate_results.csv", index=False)
    print(acc.round(3).to_string())
    print(gates[["group", "lambda_source", "alpha", "n_targets", "n_accepted", "harm_rate_among_accepted",
                 "harm_CP_hi", "abstention_rate", "mean_utility", "share_oracle_captured"]].round(3).to_string())
    summary = {"experiment": exp, "accuracy": acc.to_dict(orient="records"),
               "gate": gates.to_dict(orient="records"), "provenance": provenance(cfg)}
    if few:
        few = pd.DataFrame(few)
        few.to_csv(out / "few_shot_recalibration.csv", index=False)
        fs = few.groupby(["group", "alpha", "m"]).agg(
            mean_harm=("harm_rate_among_accepted", "mean"),
            mean_utility=("mean_utility", "mean"), never_transfer=("never_transfer", "mean")).reset_index()
        fs["share_le_alpha"] = [float((few[(few.group == r.group) & (few.alpha == r.alpha) & (few.m == r.m)]
                                       .harm_rate_among_accepted <= r.alpha).mean()) for r in fs.itertuples()]
        fs.to_csv(out / "few_shot_summary.csv", index=False)
        summary["few_shot"] = fs.to_dict(orient="records")
        fig, ax = plt.subplots(figsize=(5, 4))
        for (grp, a), g in fs.groupby(["group", "alpha"]):
            ax.plot(g.m, g.mean_harm, "o-", label=f"{grp}, α={a}")
            ax.axhline(a, color="grey", lw=0.6, ls=":")
        ax.set_xlabel("calibration targets from the new setting (m)")
        ax.set_ylabel("mean realised harm rate among accepted (test)")
        ax.set_title(f"F8b: few-shot recalibration ({exp})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(repo_path("figures") / f"f8b_recalibration_{exp.lower()}.png", dpi=150)
    (out / "phase7_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
