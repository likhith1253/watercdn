"""Harm-prediction diagnostics on training targets only (grouped out-of-fold AUROC with bootstrap CIs).

Feature sets: M5 (tiers a-c), M6 (+ probe d), E (tier e: source elites evaluated in the target), M5+E, M6+E,
each trained on the experiment's own training targets and on Net3 + L-Town training targets pooled.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tle.analysis.tables import read_table
from tle.config import load_config, repo_path
from tle.features.tiers import tier_columns
from tle.models.ablation import bootstrap_metrics, design, metrics, oof_predictions
from tle.seeds import rng


def load(cfg_path):
    cfg = load_config(cfg_path)
    df = design(read_table(cfg, "features"), read_table(cfg, "pairs"), read_table(cfg, "probe"))
    e = pd.read_parquet(repo_path(cfg["results_dir"]) / "tables" / "elite_features.parquet")
    df = df.merge(e, on="pair_id", how="left")
    df["network"] = cfg["network"]["name"]
    return cfg, df


def main() -> None:
    _, d1 = load("configs/e1.yaml")
    cfg4, d4 = load("configs/e4x.yaml")
    sets = {"M5": tier_columns(d4, "abc"), "M6": tier_columns(d4, "abcd"),
            "E": [c for c in d4.columns if c.startswith("e_")]}
    sets["M5+E"] = sets["M5"] + sets["E"]
    sets["M6+E"] = sets["M6"] + sets["E"]
    rows = []
    r = rng(int(cfg4["seed"]), "bootstrap", 90)
    for data_name, train in (("L-Town train only", d4[d4.split == "train"]),
                             ("Net3 + L-Town train pooled", pd.concat([d1[d1.split == "train"], d4[d4.split == "train"]]))):
        train = train.reset_index(drop=True)
        for name, cols in sets.items():
            cols = [c for c in cols if c in train.columns]
            o = oof_predictions(train, cols, "lgbm", "lgbm", int(cfg4["seed"]))
            o_l = o[train.network.to_numpy() == "ltown_prv"]  # evaluate on L-Town rows
            m = metrics(o_l)
            b = bootstrap_metrics(o_l, 2000, r)
            rows.append({"training_data": data_name, "features": name, "n_train_rows": len(train),
                         "auroc_ltown": m["auroc"], "auroc_lo": b.auroc.quantile(0.025), "auroc_hi": b.auroc.quantile(0.975),
                         "spearman_ltown": m["spearman"], "brier": m["brier"]})
    t = pd.DataFrame(rows)
    out = repo_path("results/v2")
    t.to_csv(out / "harm_diagnostics.csv", index=False)
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
