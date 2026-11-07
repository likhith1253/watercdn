"""Collect job files into Parquet result tables (runs, generations, curves, pairs).

Every row carries git_commit, config_hash and the seed spawn key of its run.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import repo_path
from ..scenarios.generator import read_pairs, read_scenarios
from .labels import pair_labels


def _prov(rec: dict) -> dict:
    p = rec["provenance"]
    return {"git_commit": p["git_commit"], "config_hash": p["config_hash"], "wntr_version": p["wntr_version"],
            "epanet_version": p["epanet_version"]}


def collect(cfg: dict) -> dict[str, pd.DataFrame]:
    root = repo_path(cfg["results_dir"]) / "jobs"
    de, tr = cfg["de"], cfg["transfer"]
    run_rows, gen_rows, curve_rows, lib_rows, ref_rows = [], [], [], [], []

    for f in sorted((root / "library").glob("*.joblib")):
        rec = joblib.load(f)
        sid = rec["scenario"]["scenario_id"]
        rid = f"L:{sid}"
        lib_rows.append({"run_id": rid, "scenario_id": sid, "role": "library", "method": "fresh",
                         "source_run_id": None, "seed_key": str(rec["seed_key"]), **rec["de"],
                         **rec["summary"], **rec["signature"], **_prov(rec)})
        for g in rec["generations"]:
            gen_rows.append({"run_id": rid, **g})
        curve_rows.append({"run_id": rid, "bsf": rec["bsf"].astype(np.float32)})

    for f in sorted((root / "reference").glob("*.joblib")):
        rec = joblib.load(f)
        ref_rows.append({"scenario_id": rec["scenario_id"], "ref_J": rec["ref_J"],
                         **{f"attain_{k}": v for k, v in rec["attain"].items()},
                         **{f"ref_{k}": v for k, v in rec["ref_signature"].items()},
                         "state_vector": rec["state_vector"]})

    for f in sorted((root / "targets").glob("*.joblib")):
        rec = joblib.load(f)
        tid, rep = rec["target_id"], rec["repeat"]
        m = rec["m_injected"]
        for method, run in rec["runs"].items():
            src = method.split(":", 1)[1] if method.startswith("warm:") else None
            rid = f"T:{tid}:r{rep:02d}:{method}"
            run_rows.append({
                "run_id": rid, "scenario_id": tid, "role": "target", "repeat": rep,
                "method": "warm" if src else method, "source_run_id": f"L:{src}" if src else None,
                "source_id": src, "seed_key": str(rec["seed_key"]),
                "k": tr["k"] if src else None, "rho": tr["rho"] if src else None,
                "sigma_frac": tr["sigma_frac"] if src else None,
                "N": de["N"], "F": de["F"], "CR": de["CR"], "B": de["budget_B"], "m_injected": m,
                **run.summary,
                "gen0_J": run.gen0_J.astype(np.float64), "gen0_feasible": run.gen0_feasible.astype(bool),
                **_prov(rec),
            })
            for g in run.generations:
                gen_rows.append({"run_id": rid, **g})
            curve_rows.append({"run_id": rid, "bsf": run.bsf.astype(np.float32)})

    return {
        "library_runs": pd.DataFrame(lib_rows),
        "reference": pd.DataFrame(ref_rows),
        "runs": pd.DataFrame(run_rows),
        "generations": pd.DataFrame(gen_rows),
        "curves": pd.DataFrame(curve_rows),
    }


def curves_matrix(curves: pd.DataFrame, run_ids: list[str]) -> np.ndarray:
    c = curves.set_index("run_id").bsf
    return np.vstack([np.asarray(c.loc[r], float) for r in run_ids])


def build_pairs(cfg: dict, runs: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """One row per (target, source) pair with all labels, computed after all repeats are done."""
    lab = cfg["labels"]
    manifest = read_pairs(cfg["scenarios"]["pairs"])
    t = runs[runs.role == "target"]
    out = []
    for _, p in manifest.iterrows():
        fr = t[(t.scenario_id == p.target_id) & (t.method == "fresh")].sort_values("repeat")
        wr = t[(t.scenario_id == p.target_id) & (t.source_id == p.source_id)].sort_values("repeat")
        if len(fr) == 0 or len(wr) != len(fr):
            continue
        assert list(fr["repeat"]) == list(wr["repeat"]), "repeats must be matched"
        bf = curves_matrix(curves, list(fr.run_id))
        bw = curves_matrix(curves, list(wr.run_id))
        labels = pair_labels(bf, bw, lab["epsilon"], lab["sigma_floor_frac"], lab["target_fraction_of_budget"],
                             n_gen0=int(cfg["de"]["N"]))
        out.append({**p.to_dict(), **labels,
                    "warm_feasible_share": float(wr.feasible.mean()),
                    "fresh_feasible_share": float(fr.feasible.mean())})
    return pd.DataFrame(out)


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)


def read_table(cfg: dict, name: str) -> pd.DataFrame:
    return pd.read_parquet(repo_path(cfg["results_dir"]) / "tables" / f"{name}.parquet")


def scenario_table(cfg: dict) -> pd.DataFrame:
    return read_scenarios(cfg["scenarios"]["table"])
