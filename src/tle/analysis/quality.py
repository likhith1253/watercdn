"""Phase 3 quality report and the split / leakage audit."""
from __future__ import annotations

import numpy as np
import pandas as pd


def audit_splits(scen: pd.DataFrame, pairs: pd.DataFrame, runs: pd.DataFrame) -> dict:
    """Checks that no information can flow from a target into its own prediction."""
    split = scen.set_index("scenario_id").split
    checks = {
        "sources_in_library": bool(pairs.source_id.map(split).eq("library").all()),
        "targets_not_in_library": bool(~pairs.target_id.map(split).eq("library").any()),
        "pair_split_matches_target": bool((pairs.split == pairs.target_id.map(split)).all()),
        "no_target_used_as_source": bool(not set(pairs.target_id) & set(pairs.source_id)),
        "warm_runs_use_library_sources": bool(
            runs[runs.method == "warm"].source_id.map(split).eq("library").all()),
        "splits_disjoint": bool(scen.scenario_id.is_unique),
    }
    checks["all_passed"] = all(checks.values())
    return checks


def quality_report(pairs: pd.DataFrame, runs: pd.DataFrame, ref: pd.DataFrame, cfg: dict) -> dict:
    eps = cfg["labels"]["epsilon"]
    U = pairs.U_mean
    helpful = (U > eps).mean()
    harmful = (U < -eps).mean()
    rep = {
        "n_pairs": int(len(pairs)),
        "n_targets": int(pairs.target_id.nunique()),
        "share_helpful_U_gt_eps": float(helpful),
        "share_neutral": float(1 - helpful - harmful),
        "share_harmful_U_lt_minus_eps": float(harmful),
        "U_mean": float(U.mean()), "U_median": float(U.median()), "U_std_across_pairs": float(U.std()),
        "U_se_median": float(pairs.U_se.median()),
        "noise_floor_ratio_se_over_spread": float(pairs.U_se.median() / U.std()),
        "harm_by_split": pairs.groupby("split").harm.mean().to_dict(),
        "harm_by_distance_band": pairs.groupby("distance_band").harm.mean().to_dict(),
        "U_by_distance_band": pairs.groupby("distance_band").U_mean.mean().to_dict(),
        "infeasible_run_share": float(1 - runs[runs.role == "target"].feasible.mean()),
        "infeasible_fresh_run_share": float(1 - runs[(runs.role == "target") & (runs.method == "fresh")].feasible.mean()),
        "failed_simulations": int(runs.n_failed_sims.sum()),
        "wall_s_per_run_median": float(runs.wall_s.median()),
        "wall_s_total_runs": float(runs.wall_s.sum()),
    }
    att = ref.set_index("scenario_id")
    targets = pairs.target_id.unique()
    attainable = att.loc[targets, "attain_feasible_long"]
    fresh = runs[(runs.role == "target") & (runs.method == "fresh")]
    fa = fresh[fresh.scenario_id.map(att.attain_feasible_long)]
    rep["unattainable_target_share"] = float(1 - attainable.mean())
    rep["fresh_feasible_share_on_attainable_targets"] = float(fa.feasible.mean())
    regret = (att.attain_Q_B - att.attain_Q_long) / (att.attain_Q_init - att.attain_Q_long)
    rep["median_remaining_regret_B_vs_5B_all_scenarios"] = float(np.median(regret))
    return rep
