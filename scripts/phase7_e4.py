"""E4 (held-out L-Town): retrained gate and look-up baselines, next to the frozen / recalibrated
E1 gate produced by scripts/phase7_shift.py --config configs/e4.yaml.

(iii) retrained: M5 fitted on L-Town training pairs, lambda from L-Town calibration targets.
Baselines on L-Town test targets: B0, B1, B2 (similarity top-1), B5, ungated M5, oracle.
Library-size factor: candidates restricted to random library subsets of 10 and 15 runs (200 draws).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from tle.analysis.policies import choose_by, historical_bins, lookup_policy, water_outcomes
from tle.analysis.stats import clopper_pearson
from tle.analysis.tables import read_table
from tle.config import load_config, provenance, repo_path
from tle.models.ablation import design, model_specs
from tle.models.fit import fit_predict
from tle.models.gate import choose_threshold, decide, evaluate_decisions, oracle_utility, top_source
from tle.seeds import rng


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e4.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed, eps, delta = int(cfg["seed"]), float(cfg["labels"]["epsilon"]), float(cfg["gate"]["delta"])
    alphas = [float(a) for a in cfg["gate"]["alpha"]]
    out = repo_path(cfg["results_dir"]) / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    pairs, feats, probe, runs = (read_table(cfg, n) for n in ("pairs", "features", "probe", "runs"))
    df = design(feats, pairs, probe)
    train, cal, test = (df[df.split == s].reset_index(drop=True) for s in ("train", "calibration", "test"))
    cols = model_specs(train)["M5"]
    pred, _, _ = fit_predict(train, pd.concat([cal, test], ignore_index=True), cols, seed)
    pred = pred.merge(df[["pair_id", "a_scenario_distance", "c_state_cosine"]], on="pair_id")
    oracle = oracle_utility(pairs)
    top = top_source(pred)
    top_cal, top_test = top[top.split == "calibration"], top[top.split == "test"].reset_index(drop=True)

    rows = []
    for a in alphas:
        lam = choose_threshold(top_cal.p_harm.to_numpy(), top_cal.harm.to_numpy(), a, delta)
        r = evaluate_decisions(top_test, decide(top_test, lam), oracle, eps)
        lo, hi = clopper_pearson(r["n_harm_accepted"], r["n_accepted"])
        rows.append({"method": f"retrained gate a={a:.2f}", "alpha": a, "lambda": lam, **r,
                     "harm_CP_lo": lo, "harm_CP_hi": hi})
    gates = pd.DataFrame(rows)

    test_ids = sorted(top_test.target_id)
    tp = pairs[pairs.target_id.isin(test_ids)]
    pt = pred[pred.split == "test"]
    water = water_outcomes(runs)
    tdf = test.assign(u_hist=historical_bins(train)(test))
    pol = pd.concat([
        lookup_policy("B0 fresh", pd.DataFrame({"target_id": test_ids, "source_id": [None] * len(test_ids)}), tp, water),
        lookup_policy("B1 nearest", choose_by(pt.assign(s=-pt.a_scenario_distance), "s"), tp, water),
        lookup_policy("B2 similarity top-1", choose_by(pt, "c_state_cosine"), tp, water),
        lookup_policy("B5 historical", choose_by(tdf, "u_hist", lambda r: r.u_hist > 0), tp, water),
        lookup_policy("M5 retrained ungated", choose_by(pt, "u_q10"), tp, water),
    ])
    base = pol.groupby("method").agg(mean_U=("U", "mean"), harm_rate_all=("harm", "mean"),
                                     abstention=("accepted", lambda s: 1 - s.mean())).reset_index()
    base["oracle_mean_U"] = float(oracle.loc[test_ids].mean())
    pol.to_csv(out / "e4_lookup_policies.csv", index=False)
    gates.to_csv(out / "e4_retrained_gate.csv", index=False)

    # library-size factor (frozen candidates restricted to a random library subset)
    lib = sorted(pairs.source_id.unique())
    r = rng(seed, "resplit", 40)
    lam = choose_threshold(top_cal.p_harm.to_numpy(), top_cal.harm.to_numpy(), max(alphas), delta)
    lsz = []
    for size in (5, 10, 15, len(lib)):
        for s in range(1 if size == len(lib) else 200):
            keep = set(r.choice(lib, size, replace=False))
            p = pt[pt.source_id.isin(keep)]
            tt = top_source(p)
            res = evaluate_decisions(tt.reset_index(drop=True), decide(tt, lam), oracle, eps) if len(tt) else None
            orc = tp[tp.source_id.isin(keep)].groupby("target_id").U_mean.max().clip(lower=0)
            if res:
                lsz.append({"library_size": size, "draw": s, "mean_utility_gate": res["mean_utility"] * len(tt) / len(test_ids),
                            "oracle_mean_utility": float(orc.sum() / len(test_ids))})
    lsz = pd.DataFrame(lsz).groupby("library_size").mean(numeric_only=True).reset_index()
    lsz.to_csv(out / "e4_library_size.csv", index=False)
    print(gates.round(3).to_string())
    print(base.round(3).to_string())
    print(lsz.round(3).to_string())
    (out / "e4_summary.json").write_text(json.dumps({
        "retrained_gate": gates.to_dict(orient="records"), "lookup_baselines": base.to_dict(orient="records"),
        "library_size": lsz.to_dict(orient="records"), "provenance": provenance(cfg)}, indent=2, default=float))


if __name__ == "__main__":
    main()
