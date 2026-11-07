"""Run the paired-run campaign of one experiment (resumable; re-run to continue).

Stages: library (source runs), reference (state vectors + attainability runs),
pairs (fresh + warm runs for every target and repeat). Results are job files
under <results_dir>/jobs/; build tables with scripts/build_tables.py.
"""
from __future__ import annotations

import argparse
import time

import joblib

from tle.config import load_config, repo_path
from tle.de.jobs import library_elites, library_job, reference_job, run_jobs, target_job
from tle.scenarios.generator import read_pairs, read_scenarios


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e1.yaml")
    ap.add_argument("--stages", nargs="+", default=["library", "reference", "pairs"])
    ap.add_argument("--targets", nargs="*", default=None, help="restrict to these target ids (dry runs)")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = repo_path(cfg["results_dir"]) / "jobs"
    scen = read_scenarios(cfg["scenarios"]["table"])
    pairs = read_pairs(cfg["scenarios"]["pairs"])
    if args.targets:
        pairs = pairs[pairs.target_id.isin(args.targets)]

    if "library" in args.stages:
        t0 = time.time()
        lib = scen[scen.split == "library"]
        jobs = [(root / "library" / f"{r.scenario_id}.joblib", library_job, (cfg, r)) for _, r in lib.iterrows()]
        print("library jobs run:", run_jobs(jobs, args.jobs), f"({time.time() - t0:.0f}s)")

    if "reference" in args.stages:
        t0 = time.time()
        sel = scen if not args.targets else scen[scen.scenario_id.isin(set(args.targets) | set(pairs.source_id))]
        jobs = [(root / "reference" / f"{r.scenario_id}.joblib", reference_job, (cfg, r)) for _, r in sel.iterrows()]
        print("reference jobs run:", run_jobs(jobs, args.jobs), f"({time.time() - t0:.0f}s)")

    if "pairs" in args.stages:
        t0 = time.time()
        elites = {}
        for sid in sorted(pairs.source_id.unique()):
            rec = joblib.load(root / "library" / f"{sid}.joblib")
            elites[sid] = library_elites(cfg, rec)[0]
        jobs = []
        by_id = scen.set_index("scenario_id", drop=False)
        for tid, g in pairs.groupby("target_id"):
            row = by_id.loc[tid]
            R = int(cfg["pairs"]["test_repeats"] if row.split == "test" else cfg["pairs"]["repeats"])
            src = {s: elites[s] for s in g.source_id}
            for rep in range(R):
                jobs.append((root / "targets" / f"{tid}_r{rep:02d}.joblib", target_job, (cfg, row, rep, src)))
        print("target jobs:", len(jobs), "run now:", run_jobs(jobs, args.jobs), f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
