"""Phase 2 pilot: k/rho grid, pilot transfer map (F2), elite feasibility and diversity checks.

Targets are two training scenarios (one main-zone, one zone-A leak). For each
target seven source scenarios are constructed by controlled shifts; every
source gets its own source run (pilot seeds). For each (k, rho) in the grid
every (target, repeat) runs one fresh and seven warm runs with common random
numbers. The (k, rho) with the highest mean utility over the 14 pilot pairs is
frozen. Outputs: results/e1/pilot/*.csv, figures/f2_pilot_transfer_map.png.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from tle.analysis.bootstrap import utility_ci
from tle.analysis.labels import harm_label, utility
from tle.config import load_config, provenance, repo_path
from tle.de.jobs import _execute, paired_runs
from tle.de.optimizer import Evaluator, initial_population
from tle.de.runner import de_settings, get_simulator, parallel_map
from tle.de.transfer import select_elites
from tle.scenarios.generator import ScenarioDistance, read_scenarios, to_scenario
from tle.scenarios.space import leak_candidates, network_info
from tle.seeds import rng, run_streams, seed_sequence

K_GRID = [1, 3, 5]
RHO_GRID = [0.1, 0.2, 0.4]
SHIFTS = ["identical", "small_shift", "leak_size_large", "demand_large", "location_same_zone",
          "location_other_zone", "mismatched"]


def clip(v, lo, hi):
    return float(min(max(v, lo), hi))


def make_sources(cfg, target, dist: ScenarioDistance):
    sp = cfg["scenario_space"]
    info = network_info(cfg)
    cands = leak_candidates(cfg)
    zone = info["junctions"][target.leak_node]["zone"]
    same = [j for j in cands if info["junctions"][j]["zone"] == zone and j != target.leak_node]
    other = [j for j in cands if info["junctions"][j]["zone"] != zone]
    far_same = max(same, key=lambda j: dist.hydraulic_distance(target.leak_node, j))
    near_other = min(other, key=lambda j: dist.hydraulic_distance(target.leak_node, j))
    a_lo, a_hi = sp["leak_area_m2"]
    mid_area = np.sqrt(a_lo * a_hi)
    big_area = clip(target.leak_area_m2 * 6, a_lo, a_hi) if target.leak_area_m2 < mid_area else clip(
        target.leak_area_m2 / 6, a_lo, a_hi)
    far_dem = sp["demand_mult"][1] if target.demand_mult < np.mean(sp["demand_mult"]) else sp["demand_mult"][0]
    far_amp = sp["pattern_amp"][1] if target.pattern_amp < np.mean(sp["pattern_amp"]) else sp["pattern_amp"][0]
    s = {
        "identical": target,
        "small_shift": replace(target, leak_area_m2=clip(target.leak_area_m2 * 1.25, a_lo, a_hi),
                               demand_mult=clip(target.demand_mult + 0.04, *sp["demand_mult"]),
                               pattern_amp=clip(target.pattern_amp + 0.04, *sp["pattern_amp"]),
                               tank_level_frac=clip(target.tank_level_frac + 0.05, *sp["tank_level_frac"])),
        "leak_size_large": replace(target, leak_area_m2=big_area),
        "demand_large": replace(target, demand_mult=far_dem, pattern_amp=far_amp),
        "location_same_zone": replace(target, leak_node=far_same),
        "location_other_zone": replace(target, leak_node=near_other),
        "mismatched": replace(target, leak_node=near_other, leak_area_m2=big_area, demand_mult=far_dem,
                              pattern_amp=far_amp, peak_shift_h=-target.peak_shift_h or 3,
                              tank_level_frac=sp["tank_level_frac"][1] if target.tank_level_frac < 0.55
                              else sp["tank_level_frac"][0]),
    }
    return {k: replace(v, scenario_id=f"pilot_{target.scenario_id}_{k}") for k, v in s.items()}


def source_job(args):
    cfg, ti, si, sc = args
    sim = get_simulator(cfg)
    N, F, CR, B = de_settings(cfg)
    r_init, r_de, _ = run_streams(seed_sequence(cfg["seed"], "pilot", 1, ti, si))
    ev = Evaluator(sim, sc, B)
    _execute(ev, initial_population(sim.lo, sim.hi, N, r_init), r_de, N, F, CR)
    return (ti, SHIFTS[si]), (ev.X.copy(), ev.J.copy(), ev.feasible.copy())


def pair_job(args):
    cfg, ti, target, repeat, k, rho, archives = args
    sim = get_simulator(cfg)
    t = dict(cfg["transfer"], k=k, rho=rho)
    elites = {name: select_elites(X, J, f, k, t["min_elite_distance_frac"], sim.lo, sim.hi)[0]
              for name, (X, J, f) in archives.items()}
    ss = seed_sequence(cfg["seed"], "pilot", 2, ti, repeat)
    res = paired_runs(cfg, target, ss, elites, transfer=t)
    m = res["m_injected"]
    rows = []
    fresh = res["runs"]["fresh"]
    for name in elites:
        w = res["runs"][f"warm:{name}"]
        rows.append({
            "target_id": target.scenario_id, "shift": name, "k": k, "rho": rho, "repeat": repeat,
            "Q_fresh": fresh.bsf[-1], "Q_warm": w.bsf[-1],
            "inj_feasible_share": float(w.gen0_feasible[:m].mean()),
            "inj_best_J": float(w.gen0_J[:m].min()), "fresh_gen0_best_J": float(fresh.gen0_J.min()),
            **{f"div_fresh_g{g}": fresh.generations[g]["diversity"] for g in range(4)},
            **{f"div_warm_g{g}": w.generations[g]["diversity"] for g in range(4)},
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e1.yaml")
    ap.add_argument("--repeats", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = repo_path(cfg["results_dir"]) / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    scen = read_scenarios(cfg["scenarios"]["table"])
    tr = scen[scen.split == "train"]
    r = rng(cfg["seed"], "pilot", 0)
    targets = [to_scenario(tr[tr.zone == z].iloc[int(r.integers((tr.zone == z).sum()))]) for z in ("main", "A")]
    dist = ScenarioDistance(cfg)
    sources = {ti: make_sources(cfg, t, dist) for ti, t in enumerate(targets)}
    src_rows = [{"target_id": targets[ti].scenario_id, "shift": k, **v.to_dict()}
                for ti, d in sources.items() for k, v in d.items()]
    pd.DataFrame(src_rows).to_csv(out / "pilot_sources.csv", index=False)

    arch = dict(parallel_map(source_job, [(cfg, ti, si, sources[ti][s]) for ti in sources
                                          for si, s in enumerate(SHIFTS)]))
    jobs = [(cfg, ti, targets[ti], rep, k, rho, {s: arch[(ti, s)] for s in SHIFTS})
            for ti in sources for rep in range(args.repeats) for k, rho in itertools.product(K_GRID, RHO_GRID)]
    rows = [row for rs in parallel_map(pair_job, jobs) for row in rs]
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "pilot_runs.csv", index=False)

    eps = cfg["labels"]["epsilon"]
    floor = cfg["labels"]["sigma_floor_frac"]
    recs = []
    for (tid, shift, k, rho), g in raw.groupby(["target_id", "shift", "k", "rho"]):
        U, se = utility(g.Q_fresh, g.Q_warm, floor)
        recs.append({"target_id": tid, "shift": shift, "k": k, "rho": rho, "U": U, "U_se": se,
                     "harm": harm_label(U, eps), "inj_feasible_share": g.inj_feasible_share.mean()})
    pairs = pd.DataFrame(recs)
    grid = pairs.groupby(["k", "rho"]).agg(mean_U=("U", "mean"), median_U=("U", "median"),
                                           n_harm=("harm", "sum"), n_pairs=("U", "size")).reset_index()
    grid = grid.sort_values(["mean_U", "n_harm"], ascending=[False, True]).reset_index(drop=True)
    grid.to_csv(out / "kr_grid.csv", index=False)
    k_star, rho_star = int(grid.iloc[0].k), float(grid.iloc[0].rho)
    print(grid.to_string())

    sel = raw[(raw.k == k_star) & (raw.rho == rho_star)]
    tmap = []
    for (tid, shift), g in sel.groupby(["target_id", "shift"], sort=False):
        U, se = utility(g.Q_fresh, g.Q_warm, floor)
        lo, hi = utility_ci(g.Q_fresh.to_numpy(), g.Q_warm.to_numpy(), floor, rng=np.random.default_rng(1))
        tmap.append({"target_id": tid, "shift": shift, "U": U, "U_se": se, "ci_lo": lo, "ci_hi": hi,
                     "clearly_helpful": lo > 0, "clearly_harmful": hi < 0,
                     "inj_feasible_share": g.inj_feasible_share.mean(),
                     "div_fresh_g1": g.div_fresh_g1.mean(), "div_warm_g1": g.div_warm_g1.mean(),
                     "div_fresh_g3": g.div_fresh_g3.mean(), "div_warm_g3": g.div_warm_g3.mean()})
    tmap = pd.DataFrame(tmap)
    tmap.to_csv(out / "pilot_transfer_map.csv", index=False)
    print(tmap.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (tid, g) in enumerate(tmap.groupby("target_id", sort=False)):
        g = g.set_index("shift").loc[SHIFTS]
        xpos = np.arange(len(SHIFTS)) + (i - 0.5) * 0.3
        ax.errorbar(xpos, g.U, yerr=[g.U - g.ci_lo, g.ci_hi - g.U], fmt="o", capsize=3,
                    label=f"target {tid} ({'main' if i == 0 else 'zone A'} leak)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(-eps, color="r", lw=0.8, ls="--", label=f"harm threshold -ε (ε={eps})")
    ax.set_xticks(range(len(SHIFTS)))
    ax.set_xticklabels([s.replace("_", "\n") for s in SHIFTS], fontsize=8)
    ax.set_ylabel("utility U (σ_fresh units), 95% bootstrap CI")
    ax.set_title(f"Pilot transfer map (k={k_star}, ρ={rho_star}, R={args.repeats})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_dir = repo_path("figures")
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "f2_pilot_transfer_map.png", dpi=150)

    summary = {"targets": [t.scenario_id for t in targets], "repeats": args.repeats,
               "grid": grid.to_dict(orient="records"), "chosen": {"k": k_star, "rho": rho_star},
               "n_clearly_helpful": int(tmap.clearly_helpful.sum()),
               "n_clearly_harmful": int(tmap.clearly_harmful.sum()),
               "provenance": provenance(cfg)}
    (out / "pilot_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
