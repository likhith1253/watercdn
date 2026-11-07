"""Feature-tier ablation M0-M6 with identical grouped folds, plus audits (Phase 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..features.tiers import tier_columns
from .predictors import HarmModel, UtilityModel, grouped_folds

PROBE_Q_PAIR_LEVEL = 8  # tier d at pair level: mean probe features with all injected elites probed


def model_specs(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "M0": [],
        "M1": ["c_state_cosine"],              # SEETO-style: similar means useful
        "M1_dist": ["a_scenario_distance"],    # similarity in scenario space
        "M2": tier_columns(df, "a"),
        "M3": tier_columns(df, "ab"),
        "M4": tier_columns(df, "ac"),
        "M5": tier_columns(df, "abc"),
        "M6": tier_columns(df, "abcd"),
    }


def pair_level_probe(probe: pd.DataFrame, q: int = PROBE_Q_PAIR_LEVEL) -> pd.DataFrame:
    d = probe[probe.probe_q == q].drop(columns=["probe_q", "repeat"])
    return d.groupby("pair_id").mean()


def design(features: pd.DataFrame, pairs: pd.DataFrame, probe: pd.DataFrame) -> pd.DataFrame:
    df = pairs.merge(features, on="pair_id", how="inner")
    return df.merge(pair_level_probe(probe).reset_index(), on="pair_id", how="left")


def oof_predictions(df: pd.DataFrame, cols: list[str], kind_u: str = "lgbm", kind_h: str = "lgbm",
                    seed: int = 0, y_col: str = "U_mean", h_col: str = "harm") -> pd.DataFrame:
    """Out-of-fold predictions over training targets (outer grouped CV; inner CV inside the models)."""
    groups = df.target_id.to_numpy()
    X = df[cols].to_numpy(float) if cols else np.zeros((len(df), 0))
    y, h = df[y_col].to_numpy(float), df[h_col].to_numpy(int)
    out = pd.DataFrame(index=df.index, columns=["u_mean", "u_q10", "u_q90", "p_harm"], dtype=float)
    for tr, va in grouped_folds(groups):
        um = UtilityModel(kind_u if cols else "constant", seed).fit(X[tr], y[tr], groups[tr])
        hm = HarmModel(kind_h if cols else "constant", seed).fit(X[tr], h[tr], groups[tr])
        out.iloc[va, :3] = um.predict(X[va]).to_numpy()
        out.iloc[va, 3] = hm.predict_proba(X[va])
    out["pair_id"] = df.pair_id.values
    out["target_id"] = df.target_id.values
    out["U"] = y
    out["harm"] = h
    return out


def metrics(o: pd.DataFrame) -> dict:
    sp = spearmanr(o.u_mean, o.U).statistic if o.u_mean.nunique() > 1 else 0.0
    auc = roc_auc_score(o.harm, o.p_harm) if o.harm.nunique() > 1 and o.p_harm.nunique() > 1 else 0.5
    return {"spearman": float(sp), "mae": float(np.mean(np.abs(o.u_mean - o.U))), "auroc": float(auc),
            "brier": float(brier_score_loss(o.harm, np.clip(o.p_harm, 0, 1))),
            "q10_coverage": float(np.mean(o.U >= o.u_q10))}


def bootstrap_metrics(o: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Bootstrap over targets (all rows of a resampled target come together)."""
    by_t = {t: g for t, g in o.groupby("target_id")}
    tids = np.array(list(by_t))
    rows = []
    for _ in range(n):
        pick = rng.choice(tids, size=len(tids), replace=True)
        rows.append(metrics(pd.concat([by_t[t] for t in pick], ignore_index=True)))
    return pd.DataFrame(rows)


def summarise(name: str, o: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict:
    m = metrics(o)
    b = bootstrap_metrics(o, n_boot, rng)
    out = {"model": name, **m}
    for k in ("spearman", "mae", "auroc", "brier"):
        out[f"{k}_lo"], out[f"{k}_hi"] = float(b[k].quantile(0.025)), float(b[k].quantile(0.975))
    return out


def paired_difference(o1: pd.DataFrame, o2: pd.DataFrame, metric: str, n: int,
                      rng: np.random.Generator) -> tuple[float, float, float]:
    """Bootstrap CI over targets of metric(o1) - metric(o2) (same targets in both)."""
    t1 = {t: g for t, g in o1.groupby("target_id")}
    t2 = {t: g for t, g in o2.groupby("target_id")}
    tids = np.array(list(t1))
    d = []
    for _ in range(n):
        pick = rng.choice(tids, size=len(tids), replace=True)
        a = metrics(pd.concat([t1[t] for t in pick]))[metric]
        b = metrics(pd.concat([t2[t] for t in pick]))[metric]
        d.append(a - b)
    full = metrics(o1)[metric] - metrics(o2)[metric]
    return float(full), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def permute_targets(df: pd.DataFrame, rng: np.random.Generator, cols=("U_mean", "harm")) -> pd.DataFrame:
    """Leakage audit: give each target the label block of another target (features unchanged)."""
    out = df.copy()
    tids = out.target_id.unique()
    perm = dict(zip(tids, rng.permutation(tids)))
    lab = {t: g[list(cols)].to_numpy() for t, g in out.groupby("target_id")}
    for t, g in out.groupby("target_id"):
        src = lab[perm[t]]
        n = min(len(g), len(src))
        out.loc[g.index[:n], list(cols)] = src[:n]
    return out


def group_permutation_importance(df: pd.DataFrame, cols: list[str], groups: dict[str, list[str]],
                                 seed: int, rng: np.random.Generator) -> pd.DataFrame:
    """Drop in OOF Spearman / AUROC when all columns of one tier are permuted jointly in the validation fold."""
    tg = df.target_id.to_numpy()
    X = df[cols].to_numpy(float)
    y, h = df.U_mean.to_numpy(float), df.harm.to_numpy(int)
    rows = []
    for tr, va in grouped_folds(tg):
        um = UtilityModel("lgbm", seed).fit(X[tr], y[tr], tg[tr])
        hm = HarmModel("lgbm", seed).fit(X[tr], h[tr], tg[tr])
        base_u = spearmanr(um.predict(X[va]).u_mean, y[va]).statistic
        base_h = roc_auc_score(h[va], hm.predict_proba(X[va])) if len(set(h[va])) > 1 else np.nan
        for gname, gcols in groups.items():
            idx = [cols.index(c) for c in gcols if c in cols]
            if not idx:
                continue
            Xp = X[va].copy()
            Xp[:, idx] = Xp[rng.permutation(len(va))][:, idx]
            pu = spearmanr(um.predict(Xp).u_mean, y[va]).statistic
            ph = roc_auc_score(h[va], hm.predict_proba(Xp)) if len(set(h[va])) > 1 else np.nan
            rows.append({"group": gname, "drop_spearman": base_u - pu, "drop_auroc": base_h - ph})
    return pd.DataFrame(rows).groupby("group").mean().reset_index()
