"""Optimiser-robustness check (build guide P7.1 step 6): reduced E1 with a real-coded GA instead of DE.

GA library runs for the 40 library scenarios and GA paired runs (fresh + warm, same transfer action k = 3,
rho = 0.4, budget B = 400, common random numbers) for the 20 training targets x 6 sources x 5 repeats of the
sensitivity design. Compared with the DE labels of the same pairs (sensitivity 'base' setting).
"""
from __future__ import annotations

import copy
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tle.analysis.labels import pair_labels
from tle.config import load_config, provenance, repo_path
from tle.de.jobs import library_elites, library_job, run_jobs, target_job
from tle.scenarios.generator import read_pairs, read_scenarios


def ga_target_job(cfg, row, rep, src):
    res = target_job(cfg, row, rep, src)
    return {"target_id": res["target_id"], "repeat": rep,
            "bsf": {m: r.bsf.astype(np.float32) for m, r in res["runs"].items()}}


def main() -> None:
    cfg = load_config("configs/e1.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["optimizer"] = "ga"
    res_dir = repo_path(cfg["results_dir"])
    root = res_dir / "jobs" / "ga"
    out = res_dir / "ga_check"
    out.mkdir(parents=True, exist_ok=True)
    scen = read_scenarios(cfg["scenarios"]["table"]).set_index("scenario_id", drop=False)
    mp = read_pairs(cfg["scenarios"]["pairs"])
    sens = json.loads((res_dir / "sensitivity" / "sensitivity_summary.json").read_text(encoding="utf-8"))
    tids, R = sens["targets"], int(sens["repeats"])
    lib = scen[scen.split == "library"]
    print("GA library jobs:", run_jobs([(root / "library" / f"{s}.joblib", library_job, (cfg, r))
                                        for s, r in lib.iterrows()]))
    recs = {s: joblib.load(root / "library" / f"{s}.joblib") for s in lib.index}
    jobs = []
    for tid in tids:
        src = {s: library_elites(cfg, recs[s])[0] for s in mp[mp.target_id == tid].source_id}
        for rep in range(R):
            jobs.append((root / "targets" / f"{tid}_r{rep:02d}.joblib", ga_target_job, (cfg, scen.loc[tid], rep, src)))
    print("GA target jobs:", run_jobs(jobs))
    rows = []
    for tid in tids:
        rr = sorted([joblib.load(p) for p, *_ in jobs if p.name.startswith(tid + "_")], key=lambda r: r["repeat"])
        bf = np.vstack([r["bsf"]["fresh"] for r in rr])
        for s in mp[mp.target_id == tid].source_id:
            bw = np.vstack([r["bsf"][f"warm:{s}"] for r in rr])
            L = pair_labels(bf, bw, cfg["labels"]["epsilon"], cfg["labels"]["sigma_floor_frac"],
                            cfg["labels"]["target_fraction_of_budget"])
            rows.append({"pair_id": f"{tid}__{s}", "U_ga": L["U_mean"], "harm_ga": L["harm"], "TG_ga": L["TG"],
                         "Q_fresh_ga": L["Q_fresh_mean"]})
    ga = pd.DataFrame(rows)
    de = pd.read_csv(res_dir / "sensitivity" / "sensitivity_pair_labels.csv")
    de = de[de.setting == "base"].rename(columns={"U": "U_de", "harm": "harm_de", "TG": "TG_de"})
    m = ga.merge(de[["pair_id", "U_de", "harm_de", "TG_de"]], on="pair_id")
    m.to_csv(out / "ga_vs_de_pairs.csv", index=False)
    eps = cfg["labels"]["epsilon"]
    summary = {
        "n_pairs": int(len(m)),
        "GA": {"mean_U": float(m.U_ga.mean()), "share_helpful": float((m.U_ga > eps).mean()),
               "share_harmful": float(m.harm_ga.mean()), "mean_TG": float(m.TG_ga.mean())},
        "DE": {"mean_U": float(m.U_de.mean()), "share_helpful": float((m.U_de > eps).mean()),
               "share_harmful": float(m.harm_de.mean()), "mean_TG": float(m.TG_de.mean())},
        "spearman_U_ga_vs_de": float(spearmanr(m.U_ga, m.U_de).statistic),
        "provenance": provenance(cfg),
    }
    (out / "ga_check_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
