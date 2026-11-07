"""One-command reproduction of the whole study (also available as `make reproduce`).

Stages run in order; every stage is resumable (finished jobs are skipped).
  python scripts/reproduce.py                 # everything
  python scripts/reproduce.py --only e1       # E1 only (headline results)
  python scripts/reproduce.py --only report   # tables/figures from existing results
Approximate wall-clock on a 16-thread laptop: E1 ~1.5 h, E2+E3 ~1 h, E4 ~2 h.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

PY = sys.executable

STAGES = {
    "networks": [["-c", "from tle.hydraulics.network_prep import build_net3_prv; build_net3_prv('data/networks')"]],
    "e1": [["scripts/calibrate_weights.py"], ["scripts/make_scenarios.py"], ["scripts/tune_de.py"],
           ["scripts/pilot.py"], ["scripts/run_campaign.py"], ["scripts/build_tables.py"],
           ["scripts/phase4_models.py"], ["scripts/phase5_gate.py"], ["scripts/phase6_baselines.py"]],
    "e2": [["scripts/make_shift_scenarios.py", "--config", "configs/e2.yaml"],
           ["scripts/run_campaign.py", "--config", "configs/e2.yaml"],
           ["scripts/build_tables.py", "--config", "configs/e2.yaml", "--dictionary", "results/e2/data_dictionary.md"],
           ["scripts/phase7_shift.py", "--config", "configs/e2.yaml"]],
    "e3": [["scripts/make_shift_scenarios.py", "--config", "configs/e3.yaml"],
           ["scripts/run_campaign.py", "--config", "configs/e3.yaml"],
           ["scripts/build_tables.py", "--config", "configs/e3.yaml", "--dictionary", "results/e3/data_dictionary.md"],
           ["scripts/phase7_shift.py", "--config", "configs/e3.yaml"]],
    "e4": [["scripts/e4_prepare.py"],
           ["scripts/run_campaign.py", "--config", "configs/e4.yaml"],
           ["scripts/build_tables.py", "--config", "configs/e4.yaml", "--dictionary", "results/e4/data_dictionary.md"],
           ["scripts/phase7_shift.py", "--config", "configs/e4.yaml"], ["scripts/phase7_e4.py"]],
    "report": [["scripts/sensitivity.py"], ["scripts/phase8_report.py"]],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(STAGES), default=None)
    args = ap.parse_args()
    for stage in args.only or list(STAGES):
        for cmd in STAGES[stage]:
            print(f"[{stage}] python {' '.join(cmd)}", flush=True)
            subprocess.run([PY, *cmd], check=True)


if __name__ == "__main__":
    main()
