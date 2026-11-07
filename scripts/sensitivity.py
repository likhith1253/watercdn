"""Sensitivity analysis on E1 (Phase 7 step 5; table T4). No retuning anywhere.

(1) epsilon in {0.25, 0.5, 1.0}: harm labels recomputed from the measured utilities; harm model refitted on training
    targets, lambda re-chosen on calibration targets, evaluated once on test targets (alpha = 0.20).
(2) action and budget: k in {1, 3, 5}, rho in {0.1, 0.2, 0.4}, B in {200, 400, 600} varied one at a time around the
    frozen (3, 0.4, 400) on 20 training targets x 6 sources x 5 repeats; label distribution, mean utility and the
    rank correlation of pair utilities with the frozen setting are reported.
"""
from __future__ import annotations

import argparse
import copy
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tle.analysis.labels import pair_labels
from tle.analysis.tables import read_table
from tle.config import load_config, provenance, repo_path
from tle.de.jobs import library_elites, run_jobs, target_job
from tle.models.ablation import design, model_specs
from tle.models.fit import fit_predict
from tle.models.gate import choose_threshold, decide, evaluate_decisions, oracle_utility, top_source
from tle.scenarios.generator import read_pairs, read_scenarios
from tle.seeds import rng

SETTINGS = {"base": {}, "k1": {"k": 1}, "k5": {"k": 5}, "rho0.1": {"rho": 0.1}, "rho0.2": {"rho": 0.2},
            "B200": {"B": 200}, "B600": {"B": 600}}
N_TARGETS, N_REPEATS = 20, 5


def sens_job(cfg, row, rep, src):
    res = target_job(cfg, row, rep, src)
    return {"target_id": res["target_id"], "repeat": rep,
            "bsf": {m: r.bsf.astype(np.float32) for m, r in res["runs"].items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e1.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed, delta = int(cfg["seed"]), float(cfg["gate"]["delta"])
    res_dir = repo_path(cfg["results_dir"])
    out = res_dir / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- (1) epsilon
    pairs, feats, probe = read_table(cfg, "pairs"), read_table(cfg, "features"), read_table(cfg, "probe")
    df = design(feats, pairs, probe)
    cols = model_specs(df)["M5"]
    rows = []
    for e in (0.25, 0.5, 1.0):
        d = df.assign(harm=(df.U_mean < -e).astype(int))
        tr, other = d[d.split == "train"], d[d.split != "train"]
        pred, _, _ = fit_predict(tr, other, cols, seed)
        top = top_source(pred)
        tc, tt = top[top.split == "calibration"], top[top.split == "test"].reset_index(drop=True)
        lam = choose_threshold(tc.p_harm.to_numpy(), tc.harm.to_numpy(), 0.20, delta)
        r = evaluate_decisions(tt, decide(tt, lam), oracle_utility(pairs), e)
        rows.append({"factor": "epsilon", "value": e, "harm_share_all_pairs": float((df.U_mean < -e).mean()),
                     "lambda": lam, **{k: r[k] for k in ("n_accepted", "harm_rate_among_accepted",
                                                         "abstention_rate", "mean_utility")}})
    eps_tab = pd.DataFrame(rows)

    # ---------------- (2) action and budget on training targets
    scen = read_scenarios(cfg["scenarios"]["table"]).set_index("scenario_id", drop=False)
    mp = read_pairs(cfg["scenarios"]["pairs"])
    tids = sorted(rng(seed, "sensitivity", 0).choice(sorted(mp[mp.split == "train"].target_id.unique()),
                                                     N_TARGETS, replace=False))
    lib_root = res_dir / "jobs" / "library"
    recs = {s: joblib.load(lib_root / f"{s}.joblib") for s in mp.source_id.unique()}
    jobs = []
    for name, ch in SETTINGS.items():
        c = copy.deepcopy(cfg)
        c["transfer"]["k"] = ch.get("k", c["transfer"]["k"])
        c["transfer"]["rho"] = ch.get("rho", c["transfer"]["rho"])
        c["de"]["budget_B"] = ch.get("B", c["de"]["budget_B"])
        for tid in tids:
            src = {s: library_elites(c, recs[s])[0] for s in mp[mp.target_id == tid].source_id}
            for rep in range(N_REPEATS):
                jobs.append((res_dir / "jobs" / "sensitivity" / name / f"{tid}_r{rep:02d}.joblib", sens_job,
                             (c, scen.loc[tid], rep, src)))
    print("sensitivity jobs run:", run_jobs(jobs))
    lab = []
    for name in SETTINGS:
        recs_s = [joblib.load(p) for p, *_ in jobs if p.parent.name == name]
        for tid in tids:
            rr = sorted([r for r in recs_s if r["target_id"] == tid], key=lambda r: r["repeat"])
            bf = np.vstack([r["bsf"]["fresh"] for r in rr])
            for s in mp[mp.target_id == tid].source_id:
                bw = np.vstack([r["bsf"][f"warm:{s}"] for r in rr])
                L = pair_labels(bf, bw, cfg["labels"]["epsilon"], cfg["labels"]["sigma_floor_frac"],
                                cfg["labels"]["target_fraction_of_budget"])
                lab.append({"setting": name, "pair_id": f"{tid}__{s}", "U": L["U_mean"], "harm": L["harm"],
                            "TG": L["TG"]})
    lab = pd.DataFrame(lab)
    lab.to_csv(out / "sensitivity_pair_labels.csv", index=False)
    base = lab[lab.setting == "base"].set_index("pair_id").U
    act = []
    for name, g in lab.groupby("setting", sort=False):
        u = g.set_index("pair_id").U.loc[base.index]
        act.append({"factor": "action/budget", "value": name, "mean_U": float(g.U.mean()),
                    "share_helpful": float((g.U > cfg["labels"]["epsilon"]).mean()), "share_harmful": float(g.harm.mean()),
                    "mean_TG": float(g.TG.mean()), "spearman_vs_base": float(spearmanr(u, base).statistic)})
    act_tab = pd.DataFrame(act)
    t4 = pd.concat([eps_tab, act_tab], ignore_index=True)
    t4.to_csv(out / "t4_sensitivity.csv", index=False)
    print(t4.round(3).to_string())
    (out / "sensitivity_summary.json").write_text(json.dumps({"targets": tids, "repeats": N_REPEATS,
                                                              "table": t4.to_dict(orient="records"),
                                                              "provenance": provenance(cfg)}, indent=2, default=float))


if __name__ == "__main__":
    main()
