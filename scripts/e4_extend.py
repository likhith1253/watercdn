"""E4 extension: add calibration targets on L-Town (all existing scenarios and runs are kept).

Tests whether certification becomes possible where harm is common once more calibration targets exist
(the build guide's sizing table: alpha = 0.20 needs >= 14 accepted transfers with zero harm).
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from tle.config import load_config, repo_path
from tle.scenarios.generator import (ScenarioDistance, pair_manifest, read_pairs, read_scenarios, sample_scenarios,
                                     write_manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e4x.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ext = cfg["extension"]
    base = read_scenarios(ext["base_table"])
    split = ext.get("split", "calibration")
    n_new = int(ext.get("n_extra", ext.get("n_extra_calibration")))
    new = sample_scenarios(cfg, n_new, int(cfg["seed"]) + int(ext.get("seed_offset", 7)), prefix="e4_",
                           start=int(ext["id_start"])).assign(split=split)
    scen = pd.concat([base, new], ignore_index=True)
    pairs_new = pair_manifest(scen, ScenarioDistance(cfg), int(cfg["pairs"]["per_distance_tertile"]),
                              int(cfg["seed"]) + int(ext.get("seed_offset", 7)),
                              target_splits=("train", "calibration", "test"))
    pairs_new = pairs_new[pairs_new.target_id.isin(set(new.scenario_id))]
    pairs = pd.concat([read_pairs(ext["base_pairs"]), pairs_new], ignore_index=True)
    m = {"scenario_table_sha256": write_manifest(scen, cfg["scenarios"]["table"]),
         "pair_manifest_sha256": write_manifest(pairs, cfg["scenarios"]["pairs"]),
         "targets_per_split": scen[scen.split != "library"].split.value_counts().to_dict()}
    repo_path(cfg["scenarios"]["manifest"]).write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(m)


if __name__ == "__main__":
    main()
