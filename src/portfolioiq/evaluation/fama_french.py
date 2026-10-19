"""Fama-French 5 factors + Momentum time-series regression with Newey-West standard errors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

FACTORS = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]


def ff_regression(monthly: pd.DataFrame, factors: pd.DataFrame, lags: int = 6) -> dict:
    """Regress monthly net returns (already excess, dollar-neutral) on FF5 + Mom.

    ``monthly`` has ``month_end, ret_net``; ``factors`` has ``date`` (month end) and factor columns.
    """
    m = monthly[["month_end", "ret_net"]].copy()
    m["ym"] = pd.to_datetime(m["month_end"]).dt.to_period("M")
    f = factors.copy()
    f["ym"] = pd.to_datetime(f["date"]).dt.to_period("M")
    df = m.merge(f, on="ym", how="inner").dropna(subset=["ret_net"] + FACTORS)
    if len(df) < len(FACTORS) + 3:
        return {"n_obs": int(len(df))}
    X = sm.add_constant(df[FACTORS])
    res = sm.OLS(df["ret_net"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "n_obs": int(res.nobs),
        "alpha_monthly": float(res.params["const"]),
        "alpha_annualized": float(res.params["const"] * 12),
        "alpha_tstat_nw": float(res.tvalues["const"]),
        "alpha_pvalue": float(res.pvalues["const"]),
        "betas": {k: float(res.params[k]) for k in FACTORS},
        "beta_tstats": {k: float(res.tvalues[k]) for k in FACTORS},
        "r_squared": float(res.rsquared),
        "newey_west_lags": lags,
        "residual_vol_annualized": float(np.std(res.resid, ddof=len(FACTORS) + 1) * np.sqrt(12)),
    }
