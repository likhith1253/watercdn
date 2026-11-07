"""Phase 4: tier ablation M0-M6, linear guards, order test, leakage audit, importance, F3/T3.

Uses training targets only. Calibration and test targets are not read here.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tle.analysis.tables import read_table
from tle.config import load_config, provenance, repo_path
from tle.de.runner import parallel_map
from tle.features.build import build_features
from tle.features.tiers import tier_columns
from tle.models.ablation import (design, group_permutation_importance, model_specs, oof_predictions,
                                 paired_difference, permute_targets, summarise)
from tle.seeds import rng

N_BOOT = 2000


def _fit(args):
    name, df, cols, ku, kh, seed = args
    return name, oof_predictions(df, cols, ku, kh, seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/e1.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    out = repo_path(cfg["results_dir"]) / "phase4"
    out.mkdir(parents=True, exist_ok=True)
    pairs, feats, probe = read_table(cfg, "pairs"), read_table(cfg, "features"), read_table(cfg, "probe")
    df = design(feats, pairs, probe)
    train = df[df.split == "train"].reset_index(drop=True)
    specs = model_specs(train)

    # shuffled-order variant of tier b for the order test (M3/M5)
    lib, gens, curves, ref = (read_table(cfg, n) for n in ("library_runs", "generations", "curves", "reference"))
    tp = pairs[pairs.split == "train"]
    feats_sh = build_features(cfg, tp, lib, gens, curves, ref, shuffle_rng=rng(seed, "models", 99))
    train_sh = design(feats_sh, tp, probe).set_index("pair_id").loc[train.pair_id].reset_index()

    jobs = [(n, train, c, "lgbm", "lgbm", seed) for n, c in specs.items()]
    jobs += [("M5_linear", train, specs["M5"], "linear", "logistic", seed),
             ("M2_linear", train, specs["M2"], "linear", "logistic", seed),
             ("M3_shuffled", train_sh, specs["M3"], "lgbm", "lgbm", seed),
             ("M5_shuffled", train_sh, specs["M5"], "lgbm", "lgbm", seed),
             ("M5_label_permuted", permute_targets(train, rng(seed, "models", 7)), specs["M5"], "lgbm", "lgbm", seed)]
    oof = dict(parallel_map(_fit, jobs))
    r = rng(seed, "bootstrap", 4)
    table = pd.DataFrame([summarise(n, o, N_BOOT, r) for n, o in oof.items()])
    table["n_features"] = [len(specs.get(n.split("_")[0], [])) if n != "M1_dist" else 1 for n in table.model]
    table.to_csv(out / "t3_tier_ablation.csv", index=False)
    pd.concat([o.assign(model=n) for n, o in oof.items()]).to_csv(out / "oof_predictions.csv", index=False)
    print(table.round(3).to_string())

    comps = [("M3", "M2"), ("M4", "M2"), ("M5", "M2"), ("M5", "M1"), ("M2", "M0"), ("M6", "M5"),
             ("M5", "M5_linear"), ("M3", "M3_shuffled"), ("M5", "M5_shuffled")]
    diffs = []
    for a, b in comps:
        for met in ("spearman", "auroc"):
            d, lo, hi = paired_difference(oof[a], oof[b], met, N_BOOT, r)
            diffs.append({"comparison": f"{a} - {b}", "metric": met, "diff": d, "ci_lo": lo, "ci_hi": hi})
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(out / "tier_differences.csv", index=False)
    print(diffs.round(3).to_string())

    groups = {"tier_a": tier_columns(train, "a"), "tier_b": tier_columns(train, "b"),
              "tier_c": tier_columns(train, "c")}
    imp = group_permutation_importance(train, specs["M5"], groups, seed, rng(seed, "models", 11))
    imp.to_csv(out / "tier_group_importance.csv", index=False)
    print(imp.round(3).to_string())

    o5 = oof["M5"].merge(train[["pair_id", "source_id", "distance_band"]], on="pair_id")
    o5.assign(abs_err=(o5.u_mean - o5.U).abs()).sort_values("abs_err", ascending=False).head(15).to_csv(
        out / "largest_errors_M5.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True, sharey=True)
    for ax, name in zip(axes, ["M1", "M2", "M3", "M5"]):
        o = oof[name]
        ax.scatter(o.u_mean, o.U, s=8, alpha=0.5)
        lim = [min(o.U.min(), o.u_mean.min()), max(o.U.max(), o.u_mean.max())]
        ax.plot(lim, lim, "k--", lw=0.8)
        row = table.set_index("model").loc[name]
        ax.set_title(f"{name}: ρ_s={row.spearman:.2f} [{row.spearman_lo:.2f},{row.spearman_hi:.2f}]", fontsize=9)
        ax.set_xlabel("predicted U (out-of-fold)")
    axes[0].set_ylabel("measured U")
    fig.tight_layout()
    fig.savefig(repo_path("figures") / "f3_predicted_vs_measured.png", dpi=150)

    summary = {"n_train_pairs": int(len(train)), "n_train_targets": int(train.target_id.nunique()),
               "harm_share_train": float(train.harm.mean()),
               "leakage_audit_label_permuted": table.set_index("model").loc["M5_label_permuted"][
                   ["spearman", "auroc"]].to_dict(),
               "provenance": provenance(cfg)}
    (out / "phase4_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
