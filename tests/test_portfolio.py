import numpy as np
import pandas as pd
import pytest

from portfolioiq.portfolio.backtest import daily_returns_matrix, hold_period, scale_alpha
from portfolioiq.portfolio.costs import borrow_cost_daily, transaction_cost, turnover
from portfolioiq.portfolio.optimizer import optimize_weights
from portfolioiq.portfolio.risk import ledoit_wolf_cov


@pytest.fixture()
def problem():
    rng = np.random.default_rng(0)
    n = 80
    rets = pd.DataFrame(rng.normal(0, 0.02, (300, n)), columns=[f"S{i}" for i in range(n)])
    cov, names = ledoit_wolf_cov(rets, 126)
    sectors = np.array([i % 5 for i in range(n)])
    alpha = rng.normal(0, 0.01, n)
    return alpha, cov * 21, sectors


def test_ledoit_wolf_psd_and_min_obs():
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, 0.01, (200, 5)), columns=list("abcde"))
    r.loc[:150, "e"] = np.nan  # only 49 observations
    cov, keep = ledoit_wolf_cov(r, 126)
    assert keep == list("abcd")
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_optimizer_constraints(problem):
    alpha, cov, sectors = problem
    w = optimize_weights(alpha, cov, sectors, np.zeros(len(alpha)), risk_aversion=10, kappa=0.001,
                         max_weight=0.02, max_gross=2.0)
    assert abs(w.sum()) < 1e-6                                    # dollar neutral
    for s in np.unique(sectors):
        assert abs(w[sectors == s].sum()) < 1e-6                  # sector neutral
    assert np.abs(w).max() <= 0.02 + 1e-9                         # position bounds
    assert np.abs(w).sum() <= 2.0 + 1e-6                          # gross leverage
    assert np.abs(w).sum() > 0.5                                  # not a trivial solution
    assert alpha @ w > 0


def test_turnover_penalty_reduces_trading(problem):
    alpha, cov, sectors = problem
    w0 = optimize_weights(alpha, cov, sectors, np.zeros(len(alpha)), 10, 0.0, 0.02, 2.0)
    alpha2 = alpha + np.random.default_rng(5).normal(0, 0.01, len(alpha))
    w_free = optimize_weights(alpha2, cov, sectors, w0, 10, 0.0, 0.02, 2.0)
    w_pen = optimize_weights(alpha2, cov, sectors, w0, 10, 0.01, 0.02, 2.0)
    assert turnover(w_pen, w0) < turnover(w_free, w0)


def test_costs():
    w_old = np.array([0.02, -0.02, 0.0])
    w_new = np.array([0.0, -0.02, 0.02])
    assert turnover(w_new, w_old) == pytest.approx(0.04)
    assert transaction_cost(w_new, w_old, 10) == pytest.approx(0.04 * 0.001)
    # 50 bps/yr on a 1.0 short leg -> 0.005/252 per day
    assert borrow_cost_daily(np.array([0.5, -0.6, -0.4]), 50) == pytest.approx(1.0 * 0.005 / 252)


def test_hold_period_drift_and_pnl():
    w = np.array([0.5, -0.5])
    rets = pd.DataFrame({"A": [0.10, 0.0], "B": [0.0, 0.10]}, index=pd.date_range("2020-01-01", periods=2))
    daily, w_end = hold_period(w, rets, borrow_bps=0.0)
    # day 1: +0.05 ; day 2: short B (-0.5) loses 0.05 on NAV 1.05
    assert daily["ret_gross"].iloc[0] == pytest.approx(0.05)
    assert daily["ret_gross"].iloc[1] == pytest.approx(-0.05 / 1.05)
    assert w_end[0] == pytest.approx(0.55 / 1.0)


def test_delisting_return_inserted():
    dates = pd.bdate_range("2020-01-01", periods=30)
    prices = pd.concat([
        pd.DataFrame({"date": dates, "ticker": "A", "adj_close": np.linspace(10, 11, 30)}),
        pd.DataFrame({"date": dates[:10], "ticker": "B", "adj_close": 20.0}),
    ])
    rets = daily_returns_matrix(prices, {"B": -0.30})
    assert rets.loc[dates[10], "B"] == pytest.approx(-0.30)
    assert rets.loc[dates[11]:, "B"].isna().all()


def test_scale_alpha_zero_mean():
    a = scale_alpha(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.1, 0.1]), 0.05)
    assert a.mean() == pytest.approx(0.0)
    assert a[2] > a[0]
