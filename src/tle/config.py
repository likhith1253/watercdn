"""Configuration loading, hashing and provenance."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import yaml

from .hydraulics.simulator import ObjectiveWeights, ProblemSpec

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(p: str | Path) -> Path:
    """Resolve a repository-relative path (configs store relative paths only)."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config. ``extends:`` (a list of repo-relative files) is merged first."""
    path = repo_path(path)
    cfg = yaml.safe_load(path.read_text())
    merged: dict = {}
    for parent in cfg.pop("extends", []) or []:
        merged = _deep_update(merged, load_config(parent))
    merged = _deep_update(merged, cfg)
    # objective weights are frozen in a generated file (normalisation rule, Phase 1)
    wfile = merged.get("objective", {}).get("weights_file")
    if wfile and repo_path(wfile).exists():
        merged["objective"]["weights"] = yaml.safe_load(repo_path(wfile).read_text())["weights"]
    # budget chosen by the remaining-regret rule (E4) lives in a generated file
    de = merged.get("de", {})
    bfile = de.get("budget_file")
    if de.get("budget_B") is None and bfile and repo_path(bfile).exists():
        de["budget_B"] = yaml.safe_load(repo_path(bfile).read_text())["budget_rule"]["chosen_B"]
    return merged


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def problem_spec(cfg: dict) -> ProblemSpec:
    net = cfg["network"]
    dec = cfg["decision"]
    w = cfg.get("objective", {}).get("weights") or {}
    return ProblemSpec(
        inp_file=str(repo_path(net["inp"])),
        meta_file=str(repo_path(net["meta"])),
        windows=int(dec["windows"]),
        horizon_h=int(dec["horizon_hours"]),
        pump_speed=tuple(dec["pump_speed"]),
        prv_setting_m=tuple(dec["prv_setting_m"]),
        tariff=tuple(cfg["objective"]["tariff"]),
        weights=ObjectiveWeights(**w) if w else ObjectiveWeights(),
        shortfall_tol_mh=float(cfg["objective"].get("shortfall_tol_mh", 1e-3)),
        tank_tol_m3=float(cfg["objective"].get("tank_tol_m3", 1.0)),
        tank_balance=str(cfg["objective"].get("tank_balance", "system")),
    )


@lru_cache(maxsize=1)
def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        commit = out.stdout.strip() or "uncommitted"
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def provenance(cfg: dict) -> dict:
    import wntr

    return {
        "wntr_version": wntr.__version__,
        "epanet_version": "2.2 (bundled with WNTR)",
        "git_commit": git_commit(),
        "config_hash": config_hash(cfg),
    }
