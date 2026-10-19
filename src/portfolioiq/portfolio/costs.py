"""Transaction and borrow costs."""

from __future__ import annotations

import numpy as np


def turnover(w_new: np.ndarray, w_old: np.ndarray) -> float:
    """Two-sided traded notional sum |w_new - w_old| (fraction of NAV)."""
    return float(np.abs(np.asarray(w_new) - np.asarray(w_old)).sum())


def transaction_cost(w_new: np.ndarray, w_old: np.ndarray, bps_per_side: float) -> float:
    """Cost of rebalancing = bps per side x traded notional (each buy or sell pays once)."""
    return turnover(w_new, w_old) * bps_per_side / 1e4


def borrow_cost_daily(w: np.ndarray, bps_annual: float, days_per_year: int = 252) -> float:
    """Daily borrow fee on the short leg: bps/yr x sum of |short weights| / 252."""
    short = -np.minimum(np.asarray(w), 0.0).sum()
    return float(short * bps_annual / 1e4 / days_per_year)
