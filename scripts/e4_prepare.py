"""E4 preparation: L-Town download (if missing), edited network, weights, scenarios, budget rule."""
from __future__ import annotations

import subprocess
import sys
import urllib.request

from tle.config import repo_path
from tle.hydraulics.network_prep import LTOWN_FILE, LTOWN_URL, build_ltown_prv

PY = sys.executable


def main() -> None:
    src = repo_path("data/networks") / LTOWN_FILE
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LTOWN_URL, src)
    build_ltown_prv(repo_path("data/networks"))
    for cmd in (["scripts/calibrate_weights.py", "--config", "configs/base_ltown.yaml"],
                ["scripts/make_scenarios.py", "--config", "configs/e4.yaml"],
                ["scripts/tune_de.py", "--config", "configs/e4.yaml", "--budget-only",
                 "--budget-zones", "main:3,PRV-3:2", "--out", "configs/generated/e4_budget.yaml"]):
        subprocess.run([PY, *cmd], check=True)


if __name__ == "__main__":
    main()
