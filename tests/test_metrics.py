import numpy as np
import pandas as pd
import pytest

from portfolioiq.evaluation import metrics as M
from portfolioiq.evaluation.fama_french import ff_regression


def test_sharpe_known_case():
    r = pd.Series([0.01, 0.03] * 12)
    # mean 0.02, std (ddof=1) = 0.01 * sqrt(24/23)
    expected = 0.02 / (0.01 * np.sqrt(24 / 23)) * np.sqrt(12)
    assert M.sharpe_ratio(r) == pytest.approx(expected)


def test_max_drawdown_known_case():
    r = pd.Series([0.10, -0.50, 0.20, 0.50])
    # wealth 1.1, 0.55, 0.66, 0.99 -> peak 1.1, trough 0.55 -> -50 %
    assert M.max_drawdown(r) == pytest.approx(-0.5)
    assert M.max_drawdown(pd.Series([0.01, 0.02])) == 0.0


def test_annualized_return_and_hit_rate():
    r = pd.Series([0.01] * 12)
    assert M.annualized_return(r) == pytest.approx(1.01 ** 12 - 1)
    assert M.hit_rate(pd.Series([0.1, -0.1, 0.2, 0.0])) == pytest.approx(0.5)


def test_sortino_positive_when_mean_positive():
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    assert M.sortino_ratio(r) > 0


def test_ff_regression_recovers_alpha():
    rng = np.random.default_rng(0)
    n = 120
    months = pd.date_range("2010-01-31", periods=n, freq="ME")
    f = pd.DataFrame({"date": months, **{k: rng.normal(0, 0.03, n) for k in
                                         ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]}})
    ret = 0.004 + 0.5 * f["mkt_rf"] - 0.3 * f["mom"] + rng.normal(0, 0.002, n)
    res = ff_regression(pd.DataFrame({"month_end": months, "ret_net": ret}), f, lags=6)
    assert res["alpha_annualized"] == pytest.approx(0.048, abs=0.006)
    assert res["betas"]["mkt_rf"] == pytest.approx(0.5, abs=0.05)
    assert res["alpha_tstat_nw"] > 3
