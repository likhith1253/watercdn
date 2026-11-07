"""Fit the final utility / harm models on training pairs and predict other pairs."""
from __future__ import annotations

import pandas as pd

from .predictors import HarmModel, UtilityModel


def fit_predict(train: pd.DataFrame, other: pd.DataFrame, cols: list[str], seed: int,
                kind_u="lgbm", kind_h="lgbm", h_col="harm") -> tuple[pd.DataFrame, UtilityModel, HarmModel]:
    g = train.target_id.to_numpy()
    um = UtilityModel(kind_u, seed).fit(train[cols].to_numpy(float), train.U_mean.to_numpy(float), g)
    hm = HarmModel(kind_h, seed).fit(train[cols].to_numpy(float), train[h_col].to_numpy(int), g)
    return predict_with(um, hm, cols, other), um, hm


def predict_with(um, hm, cols: list[str], other: pd.DataFrame) -> pd.DataFrame:
    pred = um.predict(other[cols].to_numpy(float))
    pred["p_harm"] = hm.predict_proba(other[cols].to_numpy(float))
    keep = ["pair_id", "target_id", "source_id", "split", "U_mean", "harm", "TG", "anytime_U", "E_fresh", "E_warm"]
    return pd.concat([other[keep].reset_index(drop=True), pred], axis=1)
