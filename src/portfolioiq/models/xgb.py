"""XGBoost regressor helpers and the monthly information coefficient."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

BASE_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def fit_xgb(
    X: pd.DataFrame, y: pd.Series, params: dict, n_estimators: int, seed: int, n_jobs: int = 4,
    X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None, early_stopping_rounds: int | None = None,
) -> xgb.XGBRegressor:
    """Fit an ``XGBRegressor`` (optionally with early stopping on a validation set)."""
    kw = {**BASE_PARAMS, **params}
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds if X_val is not None else None, **kw,
    )
    if X_val is not None:
        model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X, y, verbose=False)
    return model


def monthly_ic(dates: pd.Series, pred: np.ndarray, realized: pd.Series, min_names: int = 5) -> pd.Series:
    """Spearman rank IC between predictions and realised values, per month."""
    df = pd.DataFrame({"d": np.asarray(dates), "p": np.asarray(pred), "r": np.asarray(realized)}).dropna()
    out = {}
    for d, g in df.groupby("d"):
        if len(g) >= min_names and g["p"].nunique() > 1:
            out[d] = spearmanr(g["p"], g["r"]).statistic
    return pd.Series(out, dtype=float).sort_index()
