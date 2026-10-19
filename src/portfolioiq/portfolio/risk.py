"""Covariance estimation: Ledoit-Wolf shrinkage on a rolling window of daily returns."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def ledoit_wolf_cov(returns: pd.DataFrame, min_obs: int = 126) -> tuple[np.ndarray, list[str]]:
    """Ledoit-Wolf covariance of daily returns (T x N window).

    Columns with fewer than ``min_obs`` observations are dropped; remaining gaps are filled
    with the column mean (i.e. zero after demeaning). Returns ``(cov, kept_columns)``.
    """
    counts = returns.notna().sum()
    keep = counts[counts >= min_obs].index.tolist()
    if not keep:
        return np.zeros((0, 0)), []
    r = returns[keep]
    r = r - r.mean()
    r = r.fillna(0.0)
    cov = LedoitWolf(assume_centered=True).fit(r.values).covariance_
    cov = (cov + cov.T) / 2
    return cov, keep
