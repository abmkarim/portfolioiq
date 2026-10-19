"""Performance metrics on monthly returns.

Sharpe convention: the portfolio is dollar-neutral (self-financing), so its return is
already an excess return. Sharpe = mean(monthly net return) / std(monthly net return) x sqrt(12),
with no risk-free subtraction and sample std (ddof=1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(r: pd.Series) -> float:
    """Geometric annualised return from monthly returns."""
    r = pd.Series(r).dropna()
    if r.empty:
        return float("nan")
    return float((1 + r).prod() ** (12 / len(r)) - 1)


def annualized_vol(r: pd.Series) -> float:
    return float(pd.Series(r).dropna().std(ddof=1) * np.sqrt(12))


def sharpe_ratio(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(12)) if sd > 0 else float("nan")


def sortino_ratio(r: pd.Series) -> float:
    """mean / downside deviation (target 0, computed over all months) x sqrt(12)."""
    r = pd.Series(r).dropna()
    dd = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    return float(r.mean() / dd * np.sqrt(12)) if dd > 0 else float("nan")


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + pd.Series(r).fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    """Maximum peak-to-trough decline of compounded wealth (negative number)."""
    dd = drawdown_series(r)
    return float(min(dd.min(), 0.0)) if len(dd) else float("nan")


def hit_rate(r: pd.Series) -> float:
    r = pd.Series(r).dropna()
    return float((r > 0).mean()) if len(r) else float("nan")


def tstat(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(len(x))) if len(x) > 1 and sd > 0 else float("nan")


def summary(monthly: pd.DataFrame) -> dict:
    """Headline statistics for one strategy's monthly frame."""
    r = monthly["ret_net"]
    return {
        "n_months": int(r.notna().sum()),
        "annualized_return": annualized_return(r),
        "annualized_vol": annualized_vol(r),
        "sharpe": sharpe_ratio(r),
        "sortino": sortino_ratio(r),
        "max_drawdown": max_drawdown(r),
        "hit_rate": hit_rate(r),
        "annualized_return_gross": annualized_return(monthly["ret_gross"]),
        "sharpe_gross": sharpe_ratio(monthly["ret_gross"]),
        "avg_monthly_turnover": float(monthly["turnover"].mean()),
        "annual_turnover": float(monthly["turnover"].mean() * 12),
        "avg_gross_exposure": float(monthly["gross_exposure"].mean()),
        "avg_names_long": float(monthly.get("n_long", pd.Series(dtype=float)).mean()),
        "avg_names_short": float(monthly.get("n_short", pd.Series(dtype=float)).mean()),
    }
