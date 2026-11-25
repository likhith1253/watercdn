"""Generate and run the E4z L-Town held-out test extension (resumable)."""
from __future__ import annotations

import argparse
import subprocess
import sys

from tle.config import load_config
from tle.scenarios.generator import read_scenarios


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e4z.yaml")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    subprocess.run([sys.executable, "scripts/e4_extend.py", "--config", args.config], check=True)
    cfg = load_config(args.config)
    ext = cfg["extension"]
    first, stop = int(ext["id_start"]), int(ext["id_start"]) + int(ext["n_extra"])
    scenarios = read_scenarios(cfg["scenarios"]["table"])
    ids = [sid for sid in scenarios.scenario_id
           if first <= int(sid.rsplit("_", 1)[-1]) < stop]
    if len(ids) != int(ext["n_extra"]):
        raise ValueError(f"expected {ext['n_extra']} extension scenarios, found {len(ids)}")

    for stage in ("library", "reference", "pairs"):
        cmd = [sys.executable, "scripts/run_campaign.py", "--config", args.config,
               "--stages", stage, "--targets", *ids]
        if args.jobs is not None:
            cmd.extend(("--jobs", str(args.jobs)))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
