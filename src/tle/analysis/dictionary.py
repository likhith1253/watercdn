"""Data dictionary: every column of the released tables, with unit, source and availability."""
from __future__ import annotations

import re

import pandas as pd

# (regex, description, unit, source, available_before_decision)
RULES = [
    (r"^pair_id$", "target id + '__' + source id", "-", "pair manifest", True),
    (r"^target_id$", "target scenario id", "-", "scenario table", True),
    (r"^source_id$", "library scenario whose source run provides the elites", "-", "pair manifest", True),
    (r"^split$", "train / calibration / test (of the target)", "-", "split manifest", True),
    (r"^distance_band$", "near / mid / far tertile of library scenario distance", "-", "pair manifest", True),
    (r"^scenario_distance$", "RMS of normalised scenario-factor differences", "-", "pair manifest", True),
    (r"^n_repeats$", "number of matched repeats R", "-", "paired runs", False),
    (r"^U_mean$", "utility: mean(Q_fresh - Q_warm) / sigma_fresh over matched repeats", "sigma_fresh", "paired runs", False),
    (r"^U_se$", "standard error of U", "sigma_fresh", "paired runs", False),
    (r"^U_raw$", "mean(Q_fresh - Q_warm), not normalised", "J", "paired runs", False),
    (r"^harm$", "1 if U < -epsilon", "-", "paired runs", False),
    (r"^anytime_U$", "area between mean best-so-far curves (fresh - warm) / (B sigma_fresh)", "-", "paired runs", False),
    (r"^E_fresh$|^E_warm$", "mean evaluations to reach Q* (censored at B)", "evaluations", "paired runs", False),
    (r"^TG$", "transfer gain (E_fresh - E_warm) / E_fresh", "-", "paired runs", False),
    (r"^Q_star$", "median fresh best-so-far at 0.8 B", "J", "fresh runs", False),
    (r"^sigma_fresh$", "seed-to-seed std of fresh final J, floored at 1% |mean|", "J", "fresh runs", False),
    (r"^Q_fresh_mean$|^Q_warm_mean$", "mean final J over repeats", "J", "paired runs", False),
    (r"^warm_feasible_share$|^fresh_feasible_share$", "share of repeats whose best solution is feasible", "-", "paired runs", False),
    (r"^a_d_log_area$", "log leak area: (target - source) / log range", "-", "scenario table", True),
    (r"^a_d_demand_mult$|^a_d_peak_shift$|^a_d_pattern_amp$|^a_d_tank_level$",
     "scenario factor difference (target - source) / factor range", "-", "scenario table", True),
    (r"^a_abs_", "absolute value of the corresponding difference", "-", "scenario table", True),
    (r"^a_hyd_dist$", "shortest-path pipe length between leak nodes / network diameter", "-", "network", True),
    (r"^a_zone_mismatch$", "1 if leak nodes are in different pressure zones", "-", "network", True),
    (r"^a_same_region$", "1 - zone mismatch", "-", "network", True),
    (r"^a_scenario_distance$", "RMS scenario distance (same as pair manifest)", "-", "scenario table", True),
    (r"^a_t_", "target scenario parameter (log area, demand multiplier, pattern amplitude, peak shift, tank level)",
     "-", "scenario table", True),
    (r"^a_[ts]_node_elev_z$", "leak-node elevation z-score within the network", "-", "network", True),
    (r"^a_[ts]_node_demand_rel$", "leak-node base demand / mean positive base demand", "-", "network", True),
    (r"^a_[ts]_node_dist_feeder_rel$", "leak-node distance to nearest pump outlet or tank / max", "-", "network", True),
    (r"^a_[ts]_node_diam_rel$", "mean diameter of pipes at the leak node / network mean", "-", "network", True),
    (r"^a_[ts]_node_in_prv_zone$", "1 if the leak node lies in a PRV zone", "-", "network", True),
    (r"^a_[ts]_ref_leak_frac$", "leak volume / delivered demand under the reference decision", "-", "reference evaluation", True),
    (r"^b_impr_", "source run: share of total improvement reached at 25/50/75% of B", "-", "source run", True),
    (r"^b_auc$", "source run: mean normalised gap of best-so-far to its final value", "-", "source run", True),
    (r"^b_div_decay$", "source run: slope of log population diversity per generation", "1/gen", "source run", True),
    (r"^b_div_final_rel$", "source run: final / initial diversity", "-", "source run", True),
    (r"^b_feasible_share_", "source run: share of feasible population members (mean over generations / final)", "-", "source run", True),
    (r"^b_plateau_gen_rel$", "source run: first generation within 0.1% of final best / number of generations", "-", "source run", True),
    (r"^b_final_rel_init$", "source run: final best / best of initial population", "-", "source run", True),
    (r"^b_elite_spread$", "mean normalised distance between the k elites", "-", "source run", True),
    (r"^b_elite_J_spread$", "relative objective spread of the k elites", "-", "source run", True),
    (r"^b_source_feasible$", "1 if the source run's best solution is feasible", "-", "source run", True),
    (r"^c_src_", "hydraulic signature of the source best solution in its own scenario (normalised)", "-", "source run", True),
    (r"^c_tref_", "hydraulic signature of the target under the reference decision (1 evaluation)", "-", "reference evaluation", True),
    (r"^c_dref_", "target minus source signature, both under the reference decision", "-", "reference evaluation", True),
    (r"^c_state_cosine$", "cosine similarity of hourly service-pressure state vectors (SEETO-style)", "-", "reference evaluation", True),
    (r"^c_state_rmsd$", "RMS difference of the state vectors", "p/p_req", "reference evaluation", True),
    (r"^probe_q$", "number of injected elites evaluated before the probe decision", "-", "design", True),
    (r"^repeat$", "matched repeat index", "-", "design", True),
    (r"^d_rank_best_inj$", "share of shared random gen-0 members better than the best probed elite", "-", "warm gen 0 (probe)", True),
    (r"^d_gap_best$|^d_gap_median$", "relative objective gap probed elites vs shared random members (clipped to +-5)", "-", "warm gen 0 (probe)", True),
    (r"^d_inj_feasible_share$|^d_rest_feasible_share$", "feasible share of probed elites / shared random members", "-", "warm gen 0 (probe)", True),
]


def describe(col: str) -> dict:
    for pat, desc, unit, src, avail in RULES:
        if re.search(pat, col):
            return {"column": col, "description": desc, "unit": unit, "source": src, "available_before_decision": avail}
    return {"column": col, "description": "", "unit": "", "source": "", "available_before_decision": None}


def dictionary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        for c in df.columns:
            rows.append({"table": name, **describe(c)})
    return pd.DataFrame(rows)


def to_markdown(d: pd.DataFrame) -> str:
    lines = ["# Data dictionary", "", "Generated by `scripts/build_tables.py`; do not edit by hand.", ""]
    for name, g in d.groupby("table", sort=False):
        lines += [f"## {name}", "", "| column | description | unit | source | available before decision |",
                  "|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| `{r.column}` | {r.description} | {r.unit} | {r.source} | {r.available_before_decision} |")
        lines.append("")
    return "\n".join(lines)
