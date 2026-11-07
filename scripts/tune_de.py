"""Phase 1 step 10-11: tune DE on training scenarios only, then choose B by the non-triviality rule.

Tuning: grid F x CR x N, fresh DE at a fixed tuning budget on training scenarios
and seeds. Score of a setting = median over (scenario, seed) of Q / Q_best(scenario),
Q_best being the best value any setting reached on that scenario.

Budget rule: for five training scenarios run fresh DE (frozen setting) with
budget 5 * max(B'). DE has no budget-dependent schedule, so the best-so-far at
B' and at 5B' of one long run equal those of separate runs. Remaining regret
r(B') = (Q_B' - Q_5B') / (Q_init - Q_5B'). Choose the B' whose median r lies in
[3%, 20%], closest to the middle of that band on the log scale.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import yaml

from tle.config import load_config, provenance, repo_path
from tle.de.runner import parallel_map, run_fresh
from tle.scenarios.generator import read_scenarios, to_scenario
from tle.seeds import rng, seed_sequence

GRID = {"F": [0.4, 0.6, 0.8], "CR": [0.5, 0.9], "N": [20, 40]}
TUNING_B = 1000
B_CANDIDATES = [300, 400, 600, 800, 1000, 1200, 1500]
REGRET_BAND = (0.03, 0.20)


def pick_train(scen: pd.DataFrame, per_zone: dict, master: int, purpose_idx: int) -> pd.DataFrame:
    tr = scen[scen.split == "train"]
    r = rng(master, "tuning", purpose_idx)
    parts = []
    for z, n in per_zone.items():
        ids = tr[tr.zone == z].scenario_id.to_numpy()
        parts.extend(r.choice(ids, size=n, replace=False))
    return tr.set_index("scenario_id").loc[parts].reset_index()


def _tune_job(args):
    cfg, row, si, seed, F, CR, N = args
    ss = seed_sequence(cfg["seed"], "tuning", 100 + si, seed)
    ev, _ = run_fresh(cfg, to_scenario(row), ss, N, F, CR, TUNING_B)
    return {"scenario_id": row.scenario_id, "seed": seed, "F": F, "CR": CR, "N": N,
            "Q": float(ev.best_so_far[-1]), "feasible": bool(ev.best_result["feasible"]),
            "n_failed": ev.n_failed}


def _budget_job(args):
    cfg, row, si, seed, N, F, CR = args
    ss = seed_sequence(cfg["seed"], "budget", si, seed)
    Bmax = 5 * max(B_CANDIDATES)
    ev, _ = run_fresh(cfg, to_scenario(row), ss, N, F, CR, Bmax)
    bsf = ev.best_so_far
    out = {"scenario_id": row.scenario_id, "seed": seed, "Q_init": float(bsf[N - 1])}
    for b in B_CANDIDATES:
        out[f"Q_{b}"] = float(bsf[b - 1])
        out[f"Q_{5 * b}"] = float(bsf[5 * b - 1])
    out["feasible_at_max"] = bool(ev.best_result["feasible"])
    return out


def tune_grid(cfg, scen, seeds, outdir):
    tune_sc = pick_train(scen, {"main": 3, "A": 2, "B": 1}, cfg["seed"], 0)
    jobs = [(cfg, row, si, s, F, CR, N)
            for si, (_, row) in enumerate(tune_sc.iterrows())
            for s in range(seeds)
            for F, CR, N in itertools.product(GRID["F"], GRID["CR"], GRID["N"])]
    tune = pd.DataFrame(parallel_map(_tune_job, jobs))
    tune["Q_rel"] = tune.Q / tune.groupby("scenario_id").Q.transform("min")
    tune.to_csv(outdir / "de_grid.csv", index=False)
    score = tune.groupby(["F", "CR", "N"]).agg(median_Q_rel=("Q_rel", "median"), mean_Q_rel=("Q_rel", "mean"),
                                               feasible=("feasible", "mean")).reset_index()
    score = score.sort_values(["median_Q_rel", "mean_Q_rel"]).reset_index(drop=True)
    print(score.to_string())
    best = score.iloc[0]
    return float(best.F), float(best.CR), int(best.N), tune_sc, score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e1.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget-only", action="store_true",
                    help="keep F, CR, N from the config and only apply the budget rule (E4)")
    ap.add_argument("--budget-zones", default="main:2,A:2,B:1",
                    help="training scenarios per leak zone for the budget rule")
    ap.add_argument("--out", default="configs/generated/de_tuning.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scen = read_scenarios(cfg["scenarios"]["table"])
    outdir = repo_path(cfg["results_dir"]) / "tuning"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.budget_only:
        F, CR, N = float(cfg["de"]["F"]), float(cfg["de"]["CR"]), int(cfg["de"]["N"])
        tune_sc, score = None, pd.DataFrame()
    else:
        F, CR, N, tune_sc, score = tune_grid(cfg, scen, args.seeds, outdir)

    per_zone = {z: int(n) for z, n in (kv.split(":") for kv in args.budget_zones.split(","))}
    bud_sc = pick_train(scen, per_zone, cfg["seed"], 1)
    jobs = [(cfg, row, si, s, N, F, CR) for si, (_, row) in enumerate(bud_sc.iterrows()) for s in range(3)]
    bud = pd.DataFrame(parallel_map(_budget_job, jobs))
    bud.to_csv(outdir / "budget_runs.csv", index=False)
    regret = {}
    for b in B_CANDIDATES:
        r = (bud[f"Q_{b}"] - bud[f"Q_{5 * b}"]) / (bud.Q_init - bud[f"Q_{5 * b}"])
        regret[b] = float(np.median(r))
    in_band = [b for b in B_CANDIDATES if REGRET_BAND[0] <= regret[b] <= REGRET_BAND[1]]
    mid = np.sqrt(REGRET_BAND[0] * REGRET_BAND[1])
    chosen = min(in_band, key=lambda b: abs(np.log(regret[b]) - np.log(mid))) if in_band else None
    print("median remaining regret:", json.dumps(regret, indent=1), "chosen B:", chosen)

    out = {
        "de_grid": None if args.budget_only else GRID, "tuning_budget": None if args.budget_only else TUNING_B,
        "tuning_scenarios": [] if tune_sc is None else list(tune_sc.scenario_id),
        "seeds_per_scenario": args.seeds,
        "scores": score.to_dict(orient="records"),
        "chosen": {"F": F, "CR": CR, "N": N},
        "budget_rule": {"candidates": B_CANDIDATES, "band": list(REGRET_BAND),
                        "scenarios": list(bud_sc.scenario_id), "median_regret": regret,
                        "chosen_B": chosen,
                        "feasible_share_at_5Bmax": float(bud.feasible_at_max.mean())},
        "provenance": provenance(cfg),
    }
    repo_path(args.out).write_text(yaml.safe_dump(out, sort_keys=False))


if __name__ == "__main__":
    main()
