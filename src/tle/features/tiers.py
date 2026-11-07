"""Pre-decision features, one builder per tier (Phase 4, P4.1).

No builder receives any column of the target's own runs except tier d, which
reads only generation-0 objective values that exist at probe time. All
features are normalised by network constants (required pressure, delivered
demand, network diameter, elevation spread) so a model can be applied to a
different network (E4).

tier a  scenario facts (target, source, differences, leak-node physics)
tier b  source search trajectory
tier c  hydraulic signatures (source best solution, target and source under the reference decision)
tier d  probe: generation-0 objective values of the injected elites vs the shared random members
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..de.transfer import normalised_distance

# ----------------------------------------------------------------------------- tier a
SCEN_COLS = ["leak_area_m2", "demand_mult", "peak_shift_h", "pattern_amp", "tank_level_frac"]


def node_features(net: dict) -> pd.DataFrame:
    """Network-normalised physical features of every junction."""
    j = pd.DataFrame(net["junctions"]).T
    for c in ("elevation", "base_demand", "dist_to_feeder_m", "local_diameter_m"):
        j[c] = j[c].astype(float)
    out = pd.DataFrame(index=j.index)
    out["elev_z"] = (j.elevation - j.elevation.mean()) / j.elevation.std()
    out["demand_rel"] = j.base_demand / j.base_demand[j.base_demand > 0].mean()
    out["dist_feeder_rel"] = j.dist_to_feeder_m / j.dist_to_feeder_m.max()
    out["diam_rel"] = j.local_diameter_m / j.local_diameter_m.mean()
    out["in_prv_zone"] = (j.zone != "main").astype(float)
    return out


def tier_a(pairs: pd.DataFrame, scen: pd.DataFrame, dist, net: dict, ref: pd.DataFrame) -> pd.DataFrame:
    s = scen.set_index("scenario_id")
    nf = node_features(net)
    rp = ref.set_index("scenario_id")
    rows = []
    for _, p in pairs.iterrows():
        t, src = s.loc[p.target_id], s.loc[p.source_id]
        comp = dist.components(t, src)
        r = {"pair_id": p.pair_id}
        r.update({f"a_{k}": v for k, v in comp.items()})
        r["a_abs_d_log_area"] = abs(comp["d_log_area"])
        r["a_abs_d_demand_mult"] = abs(comp["d_demand_mult"])
        r["a_scenario_distance"] = dist(t, src)
        r["a_same_region"] = 1.0 - comp["zone_mismatch"]
        r["a_t_log_area"] = np.log(t.leak_area_m2)
        r["a_t_demand_mult"] = t.demand_mult
        r["a_t_pattern_amp"] = t.pattern_amp
        r["a_t_peak_shift"] = t.peak_shift_h
        r["a_t_tank_level"] = t.tank_level_frac
        for who, sid, sc in (("t", p.target_id, t), ("s", p.source_id, src)):
            for c, v in nf.loc[sc.leak_node].items():
                r[f"a_{who}_node_{c}"] = float(v)
            # leak flow share under the reference decision ("leak flow in the baseline solution")
            r[f"a_{who}_ref_leak_frac"] = float(rp.loc[sid, "ref_sig_leak_frac"])
        rows.append(r)
    return pd.DataFrame(rows).set_index("pair_id")


# ----------------------------------------------------------------------------- tier b
def trajectory_summary(bsf: np.ndarray, gens: pd.DataFrame, N: int) -> dict:
    """Search-history summary of one source run."""
    bsf = np.asarray(bsf, float)
    B = len(bsf)
    q0, qf = bsf[N - 1], bsf[-1]
    span = max(q0 - qf, 1e-12)
    out = {}
    for frac in (0.25, 0.5, 0.75):
        out[f"b_impr_{int(frac * 100)}"] = float((q0 - bsf[int(frac * B) - 1]) / span)
    out["b_auc"] = float(np.mean((bsf[N - 1:] - qf) / span))
    g = gens.sort_values("gen")
    div = g.diversity.to_numpy()
    gidx = g.gen.to_numpy()
    out["b_div_decay"] = float(np.polyfit(gidx, np.log(np.maximum(div, 1e-9)), 1)[0])
    out["b_div_final_rel"] = float(div[-1] / max(div[0], 1e-12))
    out["b_feasible_share_mean"] = float(g.share_feasible.mean())
    out["b_feasible_share_final"] = float(g.share_feasible.iloc[-1])
    best = g.best_J.to_numpy()
    rel = (best - best[-1]) / max(best[0] - best[-1], 1e-12)
    plateau = int(np.argmax(rel <= 0.001)) if np.any(rel <= 0.001) else len(best) - 1
    out["b_plateau_gen_rel"] = float(plateau / max(len(best) - 1, 1))
    out["b_final_rel_init"] = float(qf / max(abs(q0), 1e-12))
    return out


def tier_b(pairs: pd.DataFrame, lib_runs: pd.DataFrame, gens: pd.DataFrame, curves: pd.DataFrame,
           elites: dict[str, np.ndarray], elite_J: dict[str, np.ndarray], lo: np.ndarray, hi: np.ndarray,
           N: int, shuffle_order: np.random.Generator | None = None) -> pd.DataFrame:
    """Tier b per source run. ``shuffle_order`` permutes generation order inside time-based summaries
    (order test, P4.3 step 6)."""
    cur = curves.set_index("run_id").bsf
    feats = {}
    for _, lr in lib_runs.iterrows():
        sid = lr.scenario_id
        bsf = np.asarray(cur.loc[lr.run_id], float)
        g = gens[gens.run_id == lr.run_id].copy()
        if shuffle_order is not None:
            # destroy temporal order: shuffle per-evaluation improvements and per-generation rows
            inc = np.diff(np.r_[bsf[N - 1], bsf[N - 1:]])
            perm = shuffle_order.permutation(len(inc))
            bsf = np.r_[bsf[: N - 1], bsf[N - 1] + np.cumsum(inc[perm])]
            g = g.assign(gen=shuffle_order.permutation(g.gen.to_numpy()))
        f = trajectory_summary(bsf, g, N)
        E = elites[sid]
        d = [normalised_distance(E[i], E[j], lo, hi) for i in range(len(E)) for j in range(i + 1, len(E))]
        f["b_elite_spread"] = float(np.mean(d)) if d else 0.0
        f["b_elite_J_spread"] = float((elite_J[sid].max() - elite_J[sid].min()) / max(abs(elite_J[sid].min()), 1e-12))
        f["b_source_feasible"] = float(lr.feasible)
        feats[sid] = f
    rows = [{"pair_id": p.pair_id, **feats[p.source_id]} for _, p in pairs.iterrows()]
    return pd.DataFrame(rows).set_index("pair_id")


# ----------------------------------------------------------------------------- tier c
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def tier_c(pairs: pd.DataFrame, lib_runs: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    sig_cols = [c for c in lib_runs.columns if c.startswith("sig_") and c not in ("sig_failed",)]
    ref_cols = [c for c in ref.columns if c.startswith("ref_sig_") and c != "ref_sig_failed"]
    L = lib_runs.set_index("scenario_id")
    R = ref.set_index("scenario_id")
    rows = []
    for _, p in pairs.iterrows():
        r = {"pair_id": p.pair_id}
        for c in sig_cols:
            r[f"c_src_{c[4:]}"] = float(L.loc[p.source_id, c])
        for c in ref_cols:
            tv, sv = float(R.loc[p.target_id, c]), float(R.loc[p.source_id, c])
            r[f"c_tref_{c[8:]}"] = tv
            r[f"c_dref_{c[8:]}"] = tv - sv
        r["c_state_cosine"] = cosine(R.loc[p.target_id, "state_vector"], R.loc[p.source_id, "state_vector"])
        r["c_state_rmsd"] = float(np.sqrt(np.mean((R.loc[p.target_id, "state_vector"]
                                                   - R.loc[p.source_id, "state_vector"]) ** 2)))
        rows.append(r)
    return pd.DataFrame(rows).set_index("pair_id")


# ----------------------------------------------------------------------------- tier d
def probe_features(warm_gen0: np.ndarray, warm_feas0: np.ndarray, m: int, q: int) -> dict:
    """Probe after evaluating the N-m shared random members and the first q injected elites."""
    inj, rest = warm_gen0[:q], warm_gen0[m:]
    best_inj, best_rest = inj.min(), rest.min()
    return {
        "d_rank_best_inj": float((rest < best_inj).mean()),          # 0 = best injected beats all random
        "d_gap_best": float(np.clip((best_inj - best_rest) / max(abs(best_rest), 1e-12), -5, 5)),
        "d_gap_median": float(np.clip((np.median(inj) - np.median(rest)) / max(abs(np.median(rest)), 1e-12), -5, 5)),
        "d_inj_feasible_share": float(warm_feas0[:q].mean()),
        "d_rest_feasible_share": float(warm_feas0[m:].mean()),
    }


def tier_d(pairs: pd.DataFrame, runs: pd.DataFrame, q: int) -> pd.DataFrame:
    """Per (pair, repeat) probe features with probe size q (q <= m injected)."""
    w = runs[(runs.role == "target") & (runs.method == "warm")]
    pid = dict(zip(zip(pairs.target_id, pairs.source_id), pairs.pair_id))
    rows = []
    for _, r in w.iterrows():
        key = (r.scenario_id, r.source_id)
        if key not in pid:
            continue
        m = int(r.m_injected)
        f = probe_features(np.asarray(r.gen0_J), np.asarray(r.gen0_feasible), m, min(q, m))
        rows.append({"pair_id": pid[key], "repeat": int(r.repeat), **f})
    return pd.DataFrame(rows)


TIER_PREFIX = {"a": "a_", "b": "b_", "c": "c_", "d": "d_"}


def tier_columns(df: pd.DataFrame, tiers: str) -> list[str]:
    return [c for c in df.columns if any(c.startswith(TIER_PREFIX[t]) for t in tiers)]
