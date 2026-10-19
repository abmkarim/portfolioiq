"""Daily backtest of monthly-rebalanced long-short portfolios (net of costs).

* Rebalance at each month-end formation date with weights from :func:`optimize_weights`.
* Between rebalances positions drift with daily returns (buy and hold).
* Costs: ``tc_bps`` per side on traded notional (charged on the first day of the holding
  period) and ``borrow_bps_annual`` on the short leg (charged daily).
* Delisting: on the first day after a series ends (before the end of the data), the
  configured delisting return is applied once; afterwards the position earns 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolioiq.config import Config
from portfolioiq.portfolio.costs import borrow_cost_daily, turnover
from portfolioiq.portfolio.optimizer import optimize_weights
from portfolioiq.portfolio.risk import ledoit_wolf_cov
from portfolioiq.utils import get_logger

log = get_logger(__name__)


def daily_returns_matrix(prices: pd.DataFrame, delist_returns: dict[str, float]) -> pd.DataFrame:
    """Dates x tickers matrix of daily returns, with delisting returns inserted after series ends."""
    px = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
    rets = px.pct_change(fill_method=None)
    data_end = px.index.max()
    for t, dl in delist_returns.items():
        if t not in px.columns:
            continue
        last = px[t].last_valid_index()
        if last is None or last >= data_end - pd.Timedelta(days=7):
            continue
        after = rets.index[rets.index > last]
        if len(after):
            rets.loc[after[0], t] = dl
    return rets


def scale_alpha(score: np.ndarray, sigma_monthly: np.ndarray, ic: float) -> np.ndarray:
    """Grinold scaling: alpha_i = IC x sigma_i x z_i (z = cross-sectional z-score of the score)."""
    s = np.asarray(score, dtype=float)
    sd = s.std()
    z = (s - s.mean()) / sd if sd > 0 else np.zeros_like(s)
    return ic * sigma_monthly * z


def hold_period(w: np.ndarray, rets: pd.DataFrame, borrow_bps: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Daily gross P&L of drifting positions; returns (daily frame, end-of-period drifted weights)."""
    r = rets.fillna(0.0).to_numpy()
    growth = np.cumprod(1 + r, axis=0)
    pos = w[None, :] * growth                     # position values (initial NAV = 1)
    nav = 1.0 + (pos - w[None, :]).sum(axis=1)    # long-short: NAV = 1 + cumulative P&L
    prev_nav = np.concatenate([[1.0], nav[:-1]])
    prev_pos = np.vstack([w[None, :], pos[:-1]])
    gross = nav / prev_nav - 1
    borrow = np.array([borrow_cost_daily(p / pn, borrow_bps) for p, pn in zip(prev_pos, prev_nav, strict=True)])
    daily = pd.DataFrame({"ret_gross": gross, "borrow": borrow}, index=rets.index)
    w_end = pos[-1] / nav[-1] if len(nav) else w
    return daily, w_end


def run_backtest(scores: pd.DataFrame, rets: pd.DataFrame, sectors: pd.Series, cfg: Config,
                 tc_bps: float | None = None, name: str = "strategy") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest one signal.

    ``scores``: columns ``month_end, ticker, score`` (formation dates).
    Returns (daily, monthly, weights) frames.
    """
    pc = cfg.portfolio
    tc = pc.tc_bps if tc_bps is None else tc_bps
    kappa = tc / 1e4 * pc.turnover_penalty_multiplier
    dates = sorted(pd.to_datetime(scores["month_end"]).unique())
    idx = rets.index
    w_prev = pd.Series(dtype=float)
    daily_frames, monthly_rows, weight_rows = [], [], []
    for k, t in enumerate(dates):
        t = pd.Timestamp(t)
        nxt = pd.Timestamp(dates[k + 1]) if k + 1 < len(dates) else t + pd.offsets.MonthEnd(1)
        s = scores[pd.to_datetime(scores["month_end"]) == t].dropna(subset=["score"])
        s = s[s["ticker"].isin(rets.columns)]
        window = rets.loc[(idx <= t)].tail(pc.cov_window)[s["ticker"].tolist()]
        # must be trading at formation date
        alive = window.columns[window.iloc[-pc.min_cov_obs // 6:].notna().any()] if len(window) else []
        cov_d, names = ledoit_wolf_cov(window[list(alive)], pc.min_cov_obs)
        s = s.set_index("ticker").loc[names]
        cov_m = cov_d * 21.0
        sigma = np.sqrt(np.diag(cov_m)) if len(names) else np.zeros(0)
        alpha = scale_alpha(s["score"].to_numpy(), sigma, pc.alpha_ic)
        sec = sectors.reindex(names).fillna(12).to_numpy()
        wp = w_prev.reindex(names).fillna(0.0).to_numpy()
        w = optimize_weights(alpha, cov_m, sec, wp, pc.risk_aversion, kappa, pc.max_weight, pc.max_gross,
                             pc.solver)
        w_new = pd.Series(w, index=names)
        all_names = w_new.index.union(w_prev.index)
        to = turnover(w_new.reindex(all_names).fillna(0).to_numpy(), w_prev.reindex(all_names).fillna(0).to_numpy())
        cost = to * tc / 1e4
        held = w_new[w_new != 0]
        period = rets.loc[(idx > t) & (idx <= nxt), held.index]
        if period.empty:
            break
        daily, w_end = hold_period(held.to_numpy(), period, pc.borrow_bps_annual)
        daily["tc"] = 0.0
        daily.iloc[0, daily.columns.get_loc("tc")] = cost
        daily["ret_net"] = daily["ret_gross"] - daily["borrow"] - daily["tc"]
        daily_frames.append(daily)
        w_prev = pd.Series(w_end, index=held.index)
        m_gross = float((1 + daily["ret_gross"]).prod() - 1)
        m_net = float((1 + daily["ret_net"]).prod() - 1)
        monthly_rows.append({
            "formation_date": t, "month_end": period.index.max(), "strategy": name, "ret_gross": m_gross,
            "ret_net": m_net, "turnover": to, "cost": cost + daily["borrow"].sum(),
            "gross_exposure": float(np.abs(w).sum()), "n_long": int((w > 0).sum()), "n_short": int((w < 0).sum()),
            "n_universe": len(names),
        })
        weight_rows.append(pd.DataFrame({"formation_date": t, "ticker": names, "weight": w, "strategy": name}))
    daily = pd.concat(daily_frames) if daily_frames else pd.DataFrame()
    daily["strategy"] = name
    monthly = pd.DataFrame(monthly_rows)
    weights = pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame()
    log.info("backtest done", strategy=name, tc_bps=tc, months=len(monthly),
             mean_net=float(monthly["ret_net"].mean()) if len(monthly) else None)
    return daily.reset_index(names="date"), monthly, weights
