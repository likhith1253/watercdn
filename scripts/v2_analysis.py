"""TLE-WDN v2 analyses (docs/v2_plan.md). All outcomes are exact look-ups of logged runs.

  monitor --config C : V2-A trajectory-monitored transfer on an experiment with train/calibration/test targets
                       (final calibration -> test evaluation and 200 pooled re-splits per checkpoint)
  shift              : V2-B shift-aware gate on E2, E3 (frozen E1 model, E1 calibration) and E4 (L-Town)
  pooled             : V2-C predictor and gate trained/calibrated on pooled Net3 + L-Town data, tested on L-Town
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from tle.analysis.tables import read_table
from tle.config import load_config, provenance, repo_path
from tle.models.ablation import design, model_specs
from tle.models.fit import fit_predict, predict_with
from tle.models.gate import choose_threshold, decide, evaluate_decisions, oracle_utility, top_source
from tle.seeds import rng
from tle.v2.monitor import CHECKPOINTS, evaluate_monitor, fresh_progress, monitor_rows, score_monitor
from tle.v2.shift import domain_weights, weighted_threshold

N_SPLITS = 200


def load(cfg):
    pairs, feats, probe = read_table(cfg, "pairs"), read_table(cfg, "features"), read_table(cfg, "probe")
    return pairs, feats, design(feats, pairs, probe)


# --------------------------------------------------------------------------- V2-A
def run_monitor(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    seed, eps, delta = int(cfg["seed"]), float(cfg["labels"]["epsilon"]), float(cfg["gate"]["delta"])
    alpha = max(float(a) for a in cfg["gate"]["alpha"])
    out = repo_path(cfg["results_dir"]) / "v2"
    out.mkdir(parents=True, exist_ok=True)
    pairs, feats, df = load(cfg)
    runs, gens, curves = read_table(cfg, "runs"), read_table(cfg, "generations"), read_table(cfg, "curves")
    cols = model_specs(df)["M5"]
    train = df[df.split == "train"]
    pred, _, _ = fit_predict(train, df[df.split != "train"], cols, seed)
    m = int(runs[runs.method == "warm"].m_injected.iloc[0])
    prog = fresh_progress(runs, gens, feats, pairs, m, set(train.target_id))
    mon = monitor_rows(pairs, runs, gens, curves, feats, prog, eps)
    mon.to_parquet(out / "monitor_rows.parquet", index=False)
    cal_ids = sorted(df[df.split == "calibration"].target_id.unique())
    test_ids = sorted(df[df.split == "test"].target_id.unique())
    pool = np.array(cal_ids + test_ids)
    final, rs = [], []
    for g in CHECKPOINTS:
        for variant, use_pf in (("trajectory+pair", True), ("trajectory only", False)):
            sel = score_monitor(mon, df, pred, cols, g, seed, use_pair_features=use_pf)
            res, per_t = evaluate_monitor(sel, cal_ids, test_ids, alpha, delta)
            final.append({"experiment": cfg["experiment"], "variant": variant, "g": g, "alpha": alpha, **res})
            r = rng(seed, "resplit", 200 + g)
            for s in range(N_SPLITS):
                perm = r.permutation(pool)
                res_s, _ = evaluate_monitor(sel, perm[:len(cal_ids)], perm[len(cal_ids):], alpha, delta)
                rs.append({"variant": variant, "g": g, "split": s, **res_s})
    final, rs = pd.DataFrame(final), pd.DataFrame(rs)
    summ = rs.groupby(["variant", "g"]).agg(
        mean_U=("mean_U", "mean"), mean_U_no_monitor=("mean_U_no_monitor", "mean"),
        mean_abort=("abort_rate", "mean"), mean_harm_continued=("harm_rate_among_continued", "mean"),
        harm_no_monitor=("harm_rate_no_monitor", "mean"), oracle=("oracle_repeat_U", "mean")).reset_index()
    summ["share_splits_harm_le_alpha"] = [float((rs[(rs.variant == v) & (rs.g == g)].harm_rate_among_continued
                                                 <= alpha + 1e-12).mean()) for v, g in zip(summ.variant, summ.g)]
    final.to_csv(out / "monitor_final.csv", index=False)
    summ.to_csv(out / "monitor_resplits.csv", index=False)
    print(final.round(3).to_string())
    print(summ.round(3).to_string())
    (out / "monitor_summary.json").write_text(json.dumps({"final": final.to_dict(orient="records"),
                                                          "resplits": summ.to_dict(orient="records"),
                                                          "provenance": provenance(cfg)}, indent=2, default=float),
                                              encoding="utf-8")


# --------------------------------------------------------------------------- V2-B
def run_shift() -> None:
    cfg1 = load_config("configs/e1.yaml")
    seed, eps, delta = int(cfg1["seed"]), float(cfg1["labels"]["epsilon"]), float(cfg1["gate"]["delta"])
    gate = joblib.load(repo_path("results/e1/models/gate_M5.joblib"))
    cols = gate["feature_columns"]
    _, _, df1 = load(cfg1)
    cal1 = df1[df1.split == "calibration"]
    top_cal = top_source(predict_with(gate["utility_model"], gate["harm_model"], cols, cal1))
    Xcal = cal1.set_index("pair_id").loc[top_cal.pair_id, cols].to_numpy(float)
    rows = []
    for cfg_path in ("configs/e2.yaml", "configs/e3.yaml", "configs/e4x.yaml"):
        cfg = load_config(cfg_path)
        pairs, _, df = load(cfg)
        if "group" not in df:
            df["group"] = "all"
        pr = predict_with(gate["utility_model"], gate["harm_model"], cols, df)
        pr["group"] = df["group"].to_numpy()
        orc = oracle_utility(pairs)
        for grp, g in pr.groupby("group"):
            top = top_source(g)
            top = top[top.split == "test"].reset_index(drop=True)
            Xnew = df.set_index("pair_id").loc[top.pair_id, cols].to_numpy(float)
            w, auc = domain_weights(Xcal, Xnew, seed)
            for alpha in (0.10, 0.20):
                lam_plain = choose_threshold(top_cal.p_harm.to_numpy(), top_cal.harm.to_numpy(), alpha, delta)
                lam_w = weighted_threshold(top_cal.p_harm.to_numpy(), top_cal.harm.to_numpy(), w, alpha, delta)
                for name, lam in (("frozen (unweighted)", lam_plain), ("shift-aware (weighted)", lam_w)):
                    r = evaluate_decisions(top, decide(top, lam), orc, eps)
                    rows.append({"experiment": cfg["experiment"], "group": grp, "gate": name, "alpha": alpha,
                                 "domain_auc": auc, "weight_mass": float(w.sum()), "lambda": lam,
                                 **{k: r[k] for k in ("n_targets", "n_accepted", "harm_rate_among_accepted",
                                                      "abstention_rate", "mean_utility", "share_oracle_captured")}})
    # control: E1 test targets themselves (no shift) must keep transferring
    cfg = cfg1
    te = df1[df1.split == "test"]
    top = top_source(predict_with(gate["utility_model"], gate["harm_model"], cols, te)).reset_index(drop=True)
    w, auc = domain_weights(Xcal, te.set_index("pair_id").loc[top.pair_id, cols].to_numpy(float), seed)
    orc = oracle_utility(read_table(cfg1, "pairs"))
    for alpha in (0.10, 0.20):
        lam_w = weighted_threshold(top_cal.p_harm.to_numpy(), top_cal.harm.to_numpy(), w, alpha, delta)
        r = evaluate_decisions(top, decide(top, lam_w), orc, eps)
        rows.append({"experiment": "E1 (control)", "group": "test", "gate": "shift-aware (weighted)", "alpha": alpha,
                     "domain_auc": auc, "weight_mass": float(w.sum()), "lambda": lam_w,
                     **{k: r[k] for k in ("n_targets", "n_accepted", "harm_rate_among_accepted", "abstention_rate",
                                          "mean_utility", "share_oracle_captured")}})
    out = repo_path("results/v2")
    out.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "shift_aware_gate.csv", index=False)
    print(t.round(3).to_string())


# --------------------------------------------------------------------------- V2-C
def run_pooled() -> None:
    cfg1, cfg4 = load_config("configs/e1.yaml"), load_config("configs/e4x.yaml")
    seed, eps, delta = int(cfg4["seed"]), float(cfg4["labels"]["epsilon"]), float(cfg4["gate"]["delta"])
    _, _, d1 = load(cfg1)
    p4, _, d4 = load(cfg4)
    cols = [c for c in model_specs(d4)["M5"] if c in d1.columns]
    test4 = d4[d4.split == "test"]
    cal4 = d4[d4.split == "calibration"]
    orc = oracle_utility(p4)
    rows = []
    for train_name, train in (("L-Town only", d4[d4.split == "train"]),
                              ("Net3 + L-Town pooled", pd.concat([d1[d1.split == "train"], d4[d4.split == "train"]])),
                              ("Net3 only (zero-shot)", d1[d1.split == "train"])):
        pred, _, _ = fit_predict(train, pd.concat([cal4, test4, d1[d1.split == "calibration"]]), cols, seed)
        tt = top_source(pred[pred.target_id.isin(set(test4.target_id))]).reset_index(drop=True)
        tc4 = top_source(pred[pred.target_id.isin(set(cal4.target_id))])
        tc1 = top_source(pred[pred.target_id.isin(set(d1[d1.split == "calibration"].target_id))])
        rows.append({"train": train_name, "calibration": "none (ungated ranking)",
                     **evaluate_decisions(tt, np.ones(len(tt), bool), orc, eps)})
        Xt = d4.set_index("pair_id").loc[tt.pair_id, cols].to_numpy(float)
        for cal_name, tc in (("L-Town (70)", tc4), ("Net3 + L-Town pooled", pd.concat([tc1, tc4]))):
            for alpha in (0.10, 0.20):
                lam = choose_threshold(tc.p_harm.to_numpy(), tc.harm.to_numpy(), alpha, delta)
                rows.append({"train": train_name, "calibration": f"{cal_name}, a={alpha}",
                             **evaluate_decisions(tt, decide(tt, lam), orc, eps), "lambda": lam})
                if cal_name.startswith("Net3"):
                    Xc = pd.concat([d1, d4]).set_index("pair_id").loc[tc.pair_id, cols].to_numpy(float)
                    w, auc = domain_weights(Xc, Xt, seed)
                    lam_w = weighted_threshold(tc.p_harm.to_numpy(), tc.harm.to_numpy(), w, alpha, delta)
                    rows.append({"train": train_name, "calibration": f"pooled + shift weights, a={alpha}",
                                 **evaluate_decisions(tt, decide(tt, lam_w), orc, eps), "lambda": lam_w,
                                 "domain_auc": auc})
    t = pd.DataFrame(rows)
    out = repo_path("results/v2")
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "pooled_ltown.csv", index=False)
    print(t[["train", "calibration", "lambda", "n_accepted", "harm_rate_among_accepted", "abstention_rate",
             "mean_utility", "share_oracle_captured"]].round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["monitor", "shift", "pooled"])
    ap.add_argument("--config", default="configs/e4x.yaml")
    args = ap.parse_args()
    {"monitor": lambda: run_monitor(args.config), "shift": run_shift, "pooled": run_pooled}[args.what]()


if __name__ == "__main__":
    main()
