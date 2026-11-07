"""Generate the scenario table, the split manifest and the pair manifest of one experiment."""
from __future__ import annotations

import argparse
import json

from tle.config import config_hash, load_config, repo_path
from tle.scenarios.generator import (ScenarioDistance, assign_splits, pair_manifest, sample_scenarios,
                                     write_manifest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e1.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    sizes = {k: int(v) for k, v in cfg["splits"].items()}
    scen = sample_scenarios(cfg, sum(sizes.values()), seed, prefix=cfg["experiment"].lower() + "_")
    scen = assign_splits(scen, sizes, seed)
    dist = ScenarioDistance(cfg)
    pairs = pair_manifest(scen, dist, int(cfg["pairs"]["per_distance_tertile"]), seed)
    h_scen = write_manifest(scen, cfg["scenarios"]["table"])
    h_pairs = write_manifest(pairs, cfg["scenarios"]["pairs"])
    manifest = {
        "experiment": cfg["experiment"],
        "master_seed": seed,
        "sampler": "scrambled Sobol (continuous factors) + zone-stratified leak node",
        "split_sizes": sizes,
        "scenario_table_sha256": h_scen,
        "pair_manifest_sha256": h_pairs,
        "config_hash": config_hash(cfg),
        "zone_counts_by_split": scen.groupby(["split", "zone"]).size().unstack(fill_value=0).to_dict(),
        "pairs_per_split": pairs.split.value_counts().to_dict(),
    }
    repo_path(cfg["scenarios"]["manifest"]).write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
