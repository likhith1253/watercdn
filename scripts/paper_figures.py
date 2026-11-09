"""Figures for the arXiv paper (vector PDF, LaTeX-like fonts, colour-blind-safe palette).

Every number is read from the result files; nothing is typed in by hand.
Output: paper/arxiv/figures/*.pdf
"""
from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tle.config import repo_path

OUT = repo_path("paper/arxiv/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito palette
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#7f7f7f", "black": "#000000"}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W1, W2 = 3.35, 6.6  # single / full text width in inches (one-column article: use W2 or 0.48*W2)
EPS = 0.5


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    import os
    if os.environ.get("TLE_FIG_PREVIEW"):
        fig.savefig(os.path.join(os.environ["TLE_FIG_PREVIEW"], f"{name}.png"), dpi=170)
    plt.close(fig)
    print("wrote", name)


def pairs(exp):
    return pd.read_parquet(repo_path(f"results/{exp}/tables/pairs.parquet"))


# ---------------------------------------------------------------- Fig: example convergence curves
def fig_curves():
    p = pairs("e1")
    runs = pd.read_parquet(repo_path("results/e1/tables/runs.parquet"), columns=["run_id"])
    curves = pd.read_parquet(repo_path("results/e1/tables/curves.parquet")).set_index("run_id")
    ids = set(runs.run_id)
    good = p.sort_values("U_mean", ascending=False).iloc[0]
    bad = p.sort_values("U_mean").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.0), sharey=False)
    for ax, row, title in ((axes[0], good, "(a) helpful warm start"), (axes[1], bad, "(b) harmful warm start")):
        for kind, col, lab in (("fresh", C["grey"], "fresh"), ("warm", C["blue"] if row is good else C["red"], "warm start")):
            mats = []
            for r in range(int(row.n_repeats)):
                rid = f"T:{row.target_id}:r{r:02d}:fresh" if kind == "fresh" else f"T:{row.target_id}:r{r:02d}:warm:{row.source_id}"
                if rid in ids and rid in curves.index:
                    mats.append(np.asarray(curves.loc[rid, "bsf"], float))
            m = np.vstack([x[:min(map(len, mats))] for x in mats])
            x = np.arange(1, m.shape[1] + 1)
            for y in m:
                ax.plot(x, y, color=col, alpha=0.18, lw=0.6)
            ax.plot(x, np.median(m, 0), color=col, lw=1.6, label=f"{lab} (median of {len(m)})")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlabel("objective evaluations")
        ax.set_title(f"{title}: $U={row.U_mean:+.2f}$", loc="left")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("best-so-far penalised objective $J$")
    save(fig, "fig_curves")


# ---------------------------------------------------------------- Fig: utility distributions
def fig_utility():
    p1, p4 = pairs("e1"), pairs("e4y")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.1), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    bins = np.linspace(-6, 7, 53)
    for d, col, lab in ((p1, C["blue"], "Net3 (1080 pairs)"), (p4, C["orange"], "L-Town (540 pairs)")):
        u = np.clip(d.U_mean, -6, 7)
        ax.hist(u, bins=bins, density=True, alpha=0.55, color=col, label=lab, edgecolor="none")
    ax.axvspan(-6, -EPS, color=C["red"], alpha=0.07)
    ax.axvline(-EPS, color=C["red"], lw=0.8, ls="--")
    ax.axvline(EPS, color=C["green"], lw=0.8, ls="--")
    ax.text(-5.8, ax.get_ylim()[1] * 0.85, "harmful\n$U<-\\varepsilon$", color=C["red"], fontsize=7)
    ax.set_xlabel("utility $U$ of a named warm start ($\\sigma_{\\rm fresh}$ units; clipped to $[-6,7]$)")
    ax.set_ylabel("density")
    ax.set_ylim(0, 0.52)
    ax.legend(loc="upper right")
    ax.set_title("(a) distribution of measured utility", loc="left")
    ax = axes[1]
    bands = ["near", "mid", "far"]
    x = np.arange(3)
    for i, (d, col, lab) in enumerate(((p1, C["blue"], "Net3"), (p4, C["orange"], "L-Town"))):
        h = [d[d.distance_band == b].harm.mean() * 100 for b in bands]
        ax.bar(x + (i - 0.5) * 0.36, h, 0.34, color=col, label=lab)
        for xi, hi in zip(x + (i - 0.5) * 0.36, h):
            ax.text(xi, hi + 0.4, f"{hi:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, ["near", "mid", "far"])
    ax.set_xlabel("source–target distance tertile")
    ax.set_ylabel("harmful pairs (%)")
    ax.legend(loc="upper left")
    ax.set_title("(b) harm grows with distance", loc="left")
    save(fig, "fig_utility")


# ---------------------------------------------------------------- Fig: predictability
def fig_predict():
    t = pd.read_csv(repo_path("results/e1/phase4/t3_tier_ablation.csv")).set_index("model")
    order = [("M1", "state similarity (SEETO-style)"), ("M1_dist", "scenario distance"),
             ("M2", "a: scenario facts"), ("M3", "a+b: + source trajectory"), ("M4", "a+c: + hydraulic signatures"),
             ("M5", "a+b+c (zero-shot)"), ("M6", "a+b+c+d (with probe)")]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.45})
    ax = axes[0]
    y = np.arange(len(order))[::-1]
    for yi, (m, lab) in zip(y, order):
        r = t.loc[m]
        col = C["grey"] if m.startswith("M1") else (C["blue"] if m != "M6" else C["green"])
        ax.errorbar(r.spearman, yi, xerr=[[r.spearman - r.spearman_lo], [r.spearman_hi - r.spearman]],
                    fmt="o", color=col, ms=4, capsize=2, lw=1)
        ax.text(0.80, yi, f"{r.spearman:.2f}", va="center", fontsize=7)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks(y, [lab for _, lab in order])
    ax.set_xlim(-0.25, 0.88)
    ax.set_xlabel("Spearman $\\rho$ (predicted vs measured $U$), 95% CI")
    ax.set_title("(a) feature-tier ablation, out-of-fold", loc="left")
    ax = axes[1]
    o = pd.read_csv(repo_path("results/e1/phase4/oof_predictions.csv"))
    o = o[o.model == "M5"]
    ax.scatter(o.u_mean, o.U, s=5, alpha=0.45, color=C["blue"], edgecolor="none")
    hz = o[o.harm == 1]
    ax.scatter(hz.u_mean, hz.U, s=14, color=C["red"], marker="x", lw=0.9, label="harmful pairs")
    lo, hi = min(o.U.min(), o.u_mean.min()) - 0.3, np.percentile(np.r_[o.u_mean, o.U], 99.5)
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.5, ls=":")
    ax.axhline(-EPS, color=C["red"], lw=0.6, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("predicted mean utility $\\hat u$ (M5)")
    ax.set_ylabel("measured utility $U$")
    ax.legend(loc="upper left")
    ax.set_title("(b) M5 predictions, 360 training pairs", loc="left")
    save(fig, "fig_predict")


# ---------------------------------------------------------------- Fig: E1 gate vs alpha
def fig_gate():
    s = pd.read_csv(repo_path("results/e1/phase5/resplit_summary.csv"))
    rs = pd.read_csv(repo_path("results/e1/phase5/resplits.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.1))
    ax = axes[0]
    al = s.alpha.to_numpy()
    q = rs.groupby("alpha").harm_rate_among_accepted.quantile([0.5, 0.9]).unstack()
    ax.plot([0, 0.52], [0, 0.52], color="k", lw=0.6, ls=":", label="$y=\\alpha$ (bound)")
    ax.plot(al, s.mean_harm_rate, "o-", color=C["blue"], ms=3.5, label="mean realised harm")
    ax.plot(q.index, q[0.9], "s--", color=C["sky"], ms=3, label="90th percentile")
    sh = dict(zip(al, s.share_splits_harm_le_alpha))
    ax.text(0.5, 0.11, f"harm $\\leq\\alpha$ in {sh[0.05]*100:.1f}% of re-splits at $\\alpha=0.05$\n"
            f"and in {min(v for a, v in sh.items() if a >= 0.1)*100:.0f}% for every $\\alpha\\geq0.10$",
            ha="right", fontsize=6.5, color=C["grey"])
    ax.set_xlim(0, 0.52)
    ax.set_ylim(0, 0.52)
    ax.set_xlabel("target harm level $\\alpha$")
    ax.set_ylabel("harm rate among accepted")
    ax.legend(loc="upper left")
    ax.set_title("(a) harm control, 200 re-splits", loc="left")
    ax = axes[1]
    ax.plot(al, s.mean_utility, "o-", color=C["green"], ms=3.5, label="mean utility")
    ax.set_ylabel("mean realised utility", color=C["green"])
    ax.set_xlabel("target harm level $\\alpha$")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(al, s.mean_abstention, "s--", color=C["orange"], ms=3, label="abstention rate")
    ax2.set_ylabel("abstention rate", color=C["orange"])
    ax2.set_ylim(-0.03, 1.0)
    ax.set_ylim(-0.05, 2.0)
    ax.set_title("(b) the price of the guarantee", loc="left")
    save(fig, "fig_gate")


# ---------------------------------------------------------------- Fig: policy comparison
def fig_policies():
    t = pd.read_csv(repo_path("results/e1/phase6/t2_main_results.csv"))
    names = {"Oracle": "oracle (hindsight)", "M5 ungated (argmax u_q10)": "learned ranking, no gate",
             "B2g SEETO-style (top-5 softmax)": "SEETO-style multi-source", "B6 gate a=0.20": "risk-controlled gate ($\\alpha=0.2$)",
             "B2 similarity top-1": "similarity top-1", "B5 historical": "historical best source",
             "B1 nearest": "nearest scenario", "B4 PMTO-style": "PMTO-style task model",
             "B3 BCKT-style": "BCKT-style competition", "B0 fresh": "always fresh"}
    t = t[t.method.isin(names)].copy()
    t["lab"] = t.method.map(names)
    t = t.sort_values("mean_U")
    fig, ax = plt.subplots(figsize=(W2 * 0.62, 2.5))
    y = np.arange(len(t))
    for yi, (_, r) in zip(y, t.iterrows()):
        ours = r.method.startswith("B6") or r.method.startswith("M5")
        col = C["blue"] if ours else (C["grey"] if r.method in ("Oracle", "B0 fresh") else C["orange"])
        ax.errorbar(r.mean_U, yi, xerr=[[r.mean_U - r.mean_U_ci_lo], [r.mean_U_ci_hi - r.mean_U]], fmt="o",
                    color=col, ms=4, capsize=2, lw=1)
        p = r.holm_p_vs_ref
        star = "" if pd.isna(p) or r.method in ("Oracle", "B0 fresh") or ours else ("*" if p < 0.05 else " n.s.")
        ax.text(2.42, yi, f"{r.mean_U:.2f}{star}", va="center", fontsize=7)
    ax.set_yticks(y, t.lab)
    ax.set_xlim(-0.1, 2.75)
    ax.set_xlabel("mean utility on 40 test targets (95% bootstrap CI)")
    save(fig, "fig_policies")


# ---------------------------------------------------------------- Fig: layout shift
def fig_shift():
    acc = pd.read_csv(repo_path("results/e3/phase7/predictor_accuracy.csv"))
    g = pd.read_csv(repo_path("results/e3/phase7/gate_results.csv"))
    g = g[(g.alpha == 0.2) & (g.lambda_source == "frozen E1")]
    sa = pd.read_csv(repo_path("results/v2/shift_aware_gate.csv"))
    sa = sa[(sa.experiment == "E3") & (sa.alpha == 0.2)]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.1))
    k = acc.group.astype(int).to_numpy()
    ax = axes[0]
    ax.plot(k, acc.spearman, "o-", color=C["blue"], ms=3.5, label="predictor Spearman")
    ax.plot(k, acc.harm_share, "s--", color=C["red"], ms=3.5, label="share of harmful pairs")
    da = sa[sa.gate.str.startswith("shift")].sort_values("group").domain_auc.to_numpy()
    ax.plot(k, da, "^:", color=C["purple"], ms=3.5, label="domain-classifier AUC")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("number of closed pipes")
    ax.set_ylim(0, 1)
    ax.legend(loc="center right", fontsize=6.8)
    ax.set_title("(a) the predictor degrades", loc="left")
    ax = axes[1]
    fr = sa[sa.gate.str.startswith("frozen")].sort_values("group")
    sw = sa[sa.gate.str.startswith("shift")].sort_values("group")
    w = 0.36
    ax.bar(k - w / 2, fr.harm_rate_among_accepted * 100, w, color=C["red"], label="frozen gate")
    ax.bar(k + w / 2, sw.harm_rate_among_accepted * 100, w, color=C["green"], label="shift-aware gate")
    for xi, (a, ab) in zip(k + w / 2, zip(sw.n_accepted, sw.abstention_rate)):
        ax.text(xi, 1.0, f"{int(a)}/30\nacc.", ha="center", fontsize=6)
    ax.axhline(20, color="k", lw=0.7, ls="--")
    ax.text(0.55, 21, "$\\alpha=0.20$", fontsize=7)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("number of closed pipes")
    ax.set_ylabel("harm among accepted (%)")
    ax.set_ylim(0, 30)
    ax.legend(loc="upper left")
    ax.set_title("(b) the guarantee breaks; re-weighting abstains", loc="left")
    save(fig, "fig_shift")


# ---------------------------------------------------------------- Fig: L-Town gate
def fig_ltown():
    s = pd.read_csv(repo_path("results/e4y/gate/ltown_gate_resplits.csv"))
    t = pd.read_csv(repo_path("results/e4y/gate/ltown_gate_test.csv"))
    s = s[s.model == "M5"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.1))
    cols = {"L-Town train only": C["blue"], "Net3 all + L-Town train": C["orange"],
            "hybrid: rank L-Town, harm pooled": C["purple"]}
    labs = {"L-Town train only": "L-Town data only", "Net3 all + L-Town train": "Net3 + L-Town pooled",
            "hybrid: rank L-Town, harm pooled": "hybrid"}
    for tr, d in s.groupby("training"):
        axes[0].plot(d.alpha, d.mean_utility, "o-", color=cols[tr], ms=3, label=labs[tr])
        axes[1].plot(d.alpha, d.mean_abstention, "o-", color=cols[tr], ms=3, label=labs[tr])
    ung = t[(t.model == "M5") & (t.training == "L-Town train only") & (t.policy == "ungated ranking")].mean_utility.iloc[0]
    axes[0].axhline(ung, color=C["grey"], ls="--", lw=0.8)
    axes[0].text(0.101, ung + 0.04, f"ungated ranking (test targets): {ung:.2f}", fontsize=6.5, color=C["grey"])
    axes[0].set_ylim(-0.05, 2.0)
    axes[0].set_ylabel("mean realised utility")
    axes[1].set_ylabel("abstention rate")
    for ax in axes:
        ax.set_xlabel("target harm level $\\alpha$")
    axes[1].legend(loc="lower left")
    axes[0].set_title("(a) utility under the certified gate", loc="left")
    axes[1].set_title("(b) how often it refuses", loc="left")
    save(fig, "fig_ltown")


# ---------------------------------------------------------------- Fig: GA vs DE
def fig_ga():
    d = pd.read_csv(repo_path("results/e1/ga_check/ga_vs_de_pairs.csv"))
    js = json.loads(repo_path("results/e1/ga_check/ga_check_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    ax.scatter(d.U_de, d.U_ga, s=9, color=C["blue"], alpha=0.6, edgecolor="none")
    ax.axhline(-EPS, color=C["red"], lw=0.6, ls="--")
    ax.axvline(-EPS, color=C["red"], lw=0.6, ls="--")
    lo, hi = np.percentile(np.r_[d.U_de, d.U_ga], [0.5, 99.5])
    ax.plot([lo, hi], [lo, hi], color="k", lw=0.5, ls=":")
    out = d[d.U_ga < -3.5]
    ax.set_ylim(-3.5, 4.5)
    for _, r in out.iterrows():
        ax.annotate(f"$U_{{GA}}={r.U_ga:.1f}$", (r.U_de, -3.5), xytext=(r.U_de + 0.4, -2.9), fontsize=6.5,
                    arrowprops=dict(arrowstyle="->", lw=0.5))
    ax.set_xlabel("utility under DE")
    ax.set_ylabel("utility under GA")
    ax.text(0.03, 0.97, f"Spearman $\\rho={js['spearman_U_ga_vs_de']:.2f}$\n"
            f"harmful: DE {js['DE']['share_harmful']*100:.1f}%, GA {js['GA']['share_harmful']*100:.1f}%",
            transform=ax.transAxes, va="top", fontsize=7)
    save(fig, "fig_ga")


if __name__ == "__main__":
    for f in (fig_curves, fig_utility, fig_predict, fig_gate, fig_policies, fig_shift, fig_ltown, fig_ga):
        f()
