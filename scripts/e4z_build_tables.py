"""Merge the E4z test-size extension with the frozen E4y aggregate tables.

E4y's raw job archives are intentionally not part of the repository. This
builder keeps its committed aggregates intact and appends the new E4z jobs,
then regenerates pair labels, pre-decision features, probes, audit and quality
reports for the combined scenario manifest.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from tle.analysis.dictionary import dictionary, to_markdown
from tle.analysis.quality import audit_splits, quality_report
from tle.analysis.tables import build_pairs, collect, write_tables
from tle.config import load_config, provenance, repo_path
from tle.features.build import build_features, build_probe
from tle.scenarios.generator import read_pairs, read_scenarios


TABLES = ("library_runs", "reference", "runs", "curves")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e4z.yaml")
    ap.add_argument("--base-config", default="configs/e4y.yaml")
    args = ap.parse_args()
    cfg, base_cfg = load_config(args.config), load_config(args.base_config)
    old_dir = repo_path(base_cfg["results_dir"]) / "tables"
    old = {name: pd.read_parquet(old_dir / f"{name}.parquet") for name in TABLES}
    new = collect(cfg)
    valid_ids = set(read_scenarios(cfg["scenarios"]["table"]).scenario_id)
    new["reference"] = new["reference"][new["reference"].scenario_id.isin(valid_ids)].reset_index(drop=True)

    merged = {name: pd.concat([old[name], new[name]], ignore_index=True) for name in TABLES}
    for name, frame in merged.items():
        key = "run_id" if name in ("runs", "curves", "library_runs") else "scenario_id"
        # Library/reference records are reproducibly regenerated to provide the
        # raw source archives needed for features, but their frozen E4y table
        # rows remain authoritative in the combined aggregate.
        if name in ("curves", "library_runs", "reference"):
            frame = frame.drop_duplicates(key, keep="first").reset_index(drop=True)
            merged[name] = frame
        if frame[key].duplicated().any():
            raise ValueError(f"duplicate {key} values while merging {name}")

    scenarios = read_scenarios(cfg["scenarios"]["table"])
    manifest = read_pairs(cfg["scenarios"]["pairs"])
    expected_targets = set(scenarios.loc[scenarios.split != "library", "scenario_id"])
    expected_pairs = len(manifest)
    if not expected_targets.issubset(set(merged["reference"].scenario_id)):
        raise ValueError("reference table is missing one or more target scenarios")
    if not expected_targets.issubset(set(merged["runs"].scenario_id)):
        raise ValueError("run table is missing one or more target scenarios")

    pairs = build_pairs(cfg, merged["runs"], merged["curves"])
    if pairs.pair_id.duplicated().any() or len(pairs) != expected_pairs:
        raise ValueError(f"incomplete/duplicate pairs: built {len(pairs)}, expected {expected_pairs}")
    merged["pairs"] = pairs
    merged["features"] = build_features(cfg, pairs, merged["library_runs"], new["generations"],
                                        merged["curves"], merged["reference"])
    merged["probe"] = build_probe(pairs, merged["runs"])

    report = {
        "audit": audit_splits(scenarios, manifest, merged["runs"]),
        "quality": quality_report(pairs, merged["runs"], merged["reference"], cfg),
        "provenance": provenance(cfg),
        "base_config": args.base_config,
        "base_results": base_cfg["results_dir"],
        "scenario_count": int(len(scenarios)),
        "manifest_pair_count": int(expected_pairs),
    }
    if not report["audit"]["all_passed"]:
        raise ValueError(f"split audit failed: {report['audit']}")

    out = repo_path(cfg["results_dir"])
    write_tables(merged, out / "tables")
    rep_dir = out / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "phase3_quality.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    dictionary_frame = dictionary({k: merged[k] for k in ("pairs", "features", "probe")})
    (rep_dir / "data_dictionary.md").write_text(to_markdown(dictionary_frame), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
