"""Risk-controlled gate on L-Town (where ~16% of transfers harm) with pooled training data.

Predictor trained on all Net3 (E1) pairs plus L-Town training targets; lambda by Learn-then-Test on L-Town
calibration targets; evaluated once on the L-Town test targets and on 200 random re-splits of the pooled L-Town
calibration + test targets. Compared with the ungated ranking (same predictor), always-transfer-nearest,
similarity top-1, fresh and the oracle.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tle.analysis.stats import clopper_pearson
from tle.analysis.tables import read_table
from tle.config import load_config, provenance, repo_path
from tle.models.ablation import design, model_specs
from tle.models.fit import fit_predict
from tle.models.gate import choose_threshold, decide, evaluate_decisions, oracle_utility, resplit_experiment, top_source
from tle.seeds import rng

ALPHAS = [0.10, 0.15, 0.20, 0.25, 0.30]


def load(cfg_path):
    cfg = load_config(cfg_path)
    return cfg, design(read_table(cfg, "features"), read_table(cfg, "pairs"), read_table(cfg, "probe")), read_table(cfg, "pairs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e4y.yaml")
    args = ap.parse_args()
    _, d1, _ = load("configs/e1.yaml")
    cfg, d4, pairs4 = load(args.config)
    seed, eps, delta = int(cfg["seed"]), float(cfg["labels"]["epsilon"]), float(cfg["gate"]["delta"])
    out = repo_path(cfg["results_dir"]) / "gate"
    out.mkdir(parents=True, exist_ok=True)
    orc = oracle_utility(pairs4)
    cal_ids = sorted(d4[d4.split == "calibration"].target_id.unique())
    rows, rs_all = [], []
    for model in ("M5", "M6"):
        cols = model_specs(d4)[model]
        for tname, train in (("L-Town train only", d4[d4.split == "train"]),
                             ("Net3 all + L-Town train", pd.concat([d1, d4[d4.split == "train"]]))):
            pred, _, _ = fit_predict(train, d4[d4.split != "train"], cols, seed)
            pred = pred.merge(d4[["pair_id", "a_scenario_distance", "c_state_cosine"]], on="pair_id")
            top = top_source(pred)
            tc, tt = top[top.split == "calibration"], top[top.split == "test"].reset_index(drop=True)
            auc = {s: float(roc_auc_score(pred[pred.split == s].harm, pred[pred.split == s].p_harm))
                   for s in ("calibration", "test")}
            base = {"model": model, "training": tname, "auroc_cal": auc["calibration"], "auroc_test": auc["test"]}
            rows.append({**base, "policy": "ungated ranking", **evaluate_decisions(tt, np.ones(len(tt), bool), orc, eps)})
            for a in ALPHAS:
                lam = choose_threshold(tc.p_harm.to_numpy(), tc.harm.to_numpy(), a, delta)
                r = evaluate_decisions(tt, decide(tt, lam), orc, eps)
                _, hi = clopper_pearson(r["n_harm_accepted"], r["n_accepted"])
                rows.append({**base, "policy": f"gate a={a:.2f}", "lambda": lam, **r, "harm_CP_hi": hi})
            rs = resplit_experiment(top.reset_index(drop=True), orc, ALPHAS, delta, len(cal_ids), 200,
                                    rng(seed, "resplit", 300), eps)
            rs_all.append(rs.assign(model=model, training=tname))
    t = pd.DataFrame(rows)
    rs = pd.concat(rs_all)
    summ = rs.groupby(["model", "training", "alpha"]).agg(
        mean_harm=("harm_rate_among_accepted", "mean"), mean_abstention=("abstention_rate", "mean"),
        mean_utility=("mean_utility", "mean"), never_transfer=("lambda", lambda s: float(s.isna().mean())),
        oracle_share=("share_oracle_captured", "mean")).reset_index()
    summ["share_splits_harm_le_alpha"] = [float((rs[(rs.model == m) & (rs.training == tr) & (rs.alpha == a)]
                                                 .harm_rate_among_accepted <= a + 1e-12).mean())
                                          for m, tr, a in zip(summ.model, summ.training, summ.alpha)]
    t.to_csv(out / "ltown_gate_test.csv", index=False)
    summ.to_csv(out / "ltown_gate_resplits.csv", index=False)
    pd.set_option("display.width", 250)
    print(t[["model", "training", "auroc_cal", "auroc_test", "policy", "lambda", "n_accepted",
             "harm_rate_among_accepted", "abstention_rate", "mean_utility", "share_oracle_captured"]].round(3).to_string())
    print(summ.round(3).to_string())
    (out / "ltown_gate_summary.json").write_text(json.dumps({"test": t.to_dict(orient="records"),
                                                             "resplits": summ.to_dict(orient="records"),
                                                             "provenance": provenance(cfg)}, indent=2, default=float),
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
