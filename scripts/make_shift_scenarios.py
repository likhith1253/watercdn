"""Scenario tables and pair manifests for E2 (goal change) and E3 (layout change).

The library rows are copied unchanged from E1 (same ids, so the library source
runs are re-created bit-identically by the same master seed); targets are new
scenarios with shifted goals or layouts.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from tle.config import config_hash, load_config, repo_path
from tle.scenarios.generator import ScenarioDistance, pair_manifest, read_scenarios, write_manifest
from tle.scenarios.shift import shifted_targets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    sh = cfg["shift"]
    e1 = read_scenarios(sh["library_table"])
    lib = e1[e1.split == "library"].assign(group="library")
    tg = shifted_targets(cfg, sh["kind"], int(sh["targets_per_group"]), int(cfg["seed"]), int(sh["id_start"]),
                         float(sh["calibration_share"]))
    scen = pd.concat([lib, tg], ignore_index=True)
    pairs = pair_manifest(scen, ScenarioDistance(cfg), int(cfg["pairs"]["per_distance_tertile"]), int(cfg["seed"]),
                          target_splits=("calibration", "test"))
    pairs = pairs.merge(scen[["scenario_id", "group"]].rename(columns={"scenario_id": "target_id"}), on="target_id")
    manifest = {"experiment": cfg["experiment"], "kind": sh["kind"],
                "scenario_table_sha256": write_manifest(scen, cfg["scenarios"]["table"]),
                "pair_manifest_sha256": write_manifest(pairs, cfg["scenarios"]["pairs"]),
                "targets_per_group": tg.groupby(["group", "split"]).size().unstack().to_dict(),
                "config_hash": config_hash(cfg)}
    repo_path(cfg["scenarios"]["manifest"]).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
