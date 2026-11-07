"""Compute tier-e features (source elites evaluated in the target scenario) for an experiment's pairs."""
from __future__ import annotations

import argparse

from tle.analysis.tables import read_table
from tle.config import load_config, repo_path
from tle.features.elite_eval import elite_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    lib = repo_path(cfg["reuse_library_from"]) / "library" if cfg.get("reuse_library_from") else None
    f = elite_features(cfg, read_table(cfg, "pairs"), lib)
    f.to_parquet(repo_path(cfg["results_dir"]) / "tables" / "elite_features.parquet", index=False)
    print(f.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
