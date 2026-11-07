"""Statistics fixed in the pre-registration (P8.1). Unit of analysis = target scenario."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta, friedmanchisquare, wilcoxon


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).mean()
    lt = (a[:, None] < b[None, :]).mean()
    return float(gt - lt)


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation of paired differences (zeros dropped)."""
    d = np.asarray(diff, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = pd.Series(np.abs(d)).rank().to_numpy()
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def bootstrap_mean_ci(x: np.ndarray, n: int = 10_000, rng: np.random.Generator | None = None) -> tuple[float, float]:
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    a = (1 - level) / 2
    lo = float(beta.ppf(a, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(1 - a, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def compare_to_reference(wide: pd.DataFrame, reference: str, n_boot: int = 10_000,
                         rng: np.random.Generator | None = None) -> pd.DataFrame:
    """wide: rows = targets, columns = methods (realised utility). Paired Wilcoxon vs reference, Holm."""
    rng = rng or np.random.default_rng(0)
    rows, p = [], {}
    for m in wide.columns:
        lo, hi = bootstrap_mean_ci(wide[m].to_numpy(), n_boot, rng)
        row = {"method": m, "n_targets": int(wide[m].notna().sum()), "mean_U": float(wide[m].mean()),
               "median_U": float(wide[m].median()), "mean_U_ci_lo": lo, "mean_U_ci_hi": hi}
        if m != reference:
            d = (wide[reference] - wide[m]).to_numpy()
            if np.allclose(d, 0):
                p[m] = 1.0
            else:
                p[m] = float(wilcoxon(wide[reference], wide[m], zero_method="wilcox").pvalue)
            row["wilcoxon_p_vs_ref"] = p[m]
            row["rank_biserial_ref_minus_m"] = rank_biserial(d)
            row["cliffs_delta_ref_vs_m"] = cliffs_delta(wide[reference], wide[m])
        rows.append(row)
    adj = holm(p)
    out = pd.DataFrame(rows)
    out["holm_p_vs_ref"] = out.method.map(adj)
    return out


def friedman(wide: pd.DataFrame) -> dict:
    stat = friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    return {"statistic": float(stat.statistic), "pvalue": float(stat.pvalue), "k": wide.shape[1], "n": wide.shape[0]}
