"""Phase 3 steps 7-10: tables, labels, features, audit, quality report and data dictionary."""
from __future__ import annotations

import argparse
import json

from tle.analysis.dictionary import dictionary, to_markdown
from tle.analysis.quality import audit_splits, quality_report
from tle.analysis.tables import build_pairs, collect, write_tables
from tle.config import load_config, provenance, repo_path
from tle.features.build import build_features, build_probe
from tle.scenarios.generator import read_pairs, read_scenarios


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e1.yaml")
    ap.add_argument("--dictionary", default="docs/data_dictionary.md")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = repo_path(cfg["results_dir"]) / "tables"
    t = collect(cfg)
    pairs = build_pairs(cfg, t["runs"], t["curves"])
    t["pairs"] = pairs
    t["features"] = build_features(cfg, pairs, t["library_runs"], t["generations"], t["curves"], t["reference"])
    t["probe"] = build_probe(pairs, t["runs"])
    write_tables(t, out)

    scen = read_scenarios(cfg["scenarios"]["table"])
    report = {
        "audit": audit_splits(scen, read_pairs(cfg["scenarios"]["pairs"]), t["runs"]),
        "quality": quality_report(pairs, t["runs"], t["reference"], cfg),
        "provenance": provenance(cfg),
    }
    rep_dir = repo_path(cfg["results_dir"]) / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "phase3_quality.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))
    d = dictionary({k: t[k] for k in ("pairs", "features", "probe")})
    repo_path(args.dictionary).write_text(to_markdown(d))


if __name__ == "__main__":
    main()
