"""Anti look-ahead checks on synthetic data: filing availability, panel attachment, fold splits."""

import duckdb
import numpy as np
import pandas as pd

from portfolioiq.models.walkforward import fold_masks, inner_split
from portfolioiq.nlp.features import fold_cutoffs, previous_filing


def test_available_date_strictly_after_acceptance_sql():
    """Same SQL logic as dbt's int_filings_available + asof join used by fct_monthly_panel."""
    con = duckdb.connect()
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    con.register("td", pd.DataFrame({"trade_date": days.date}))
    filings = pd.DataFrame({
        "cik": [1, 1, 2],
        "accession": ["a1", "a2", "b1"],
        # accepted after the close on a Friday, during a trading day, and on the last day of a month
        "acceptance_datetime": pd.to_datetime(["2020-01-31 18:30", "2020-02-12 09:15", "2020-02-28 16:05"]),
    })
    con.register("f", filings)
    avail = con.sql("""
        select f.*, (select min(trade_date) from td where trade_date > cast(f.acceptance_datetime as date))
               as feature_available_date from f
    """).df()
    avail["feature_available_date"] = pd.to_datetime(avail["feature_available_date"])
    assert (avail["feature_available_date"] > avail["acceptance_datetime"]).all()
    assert avail.set_index("accession").loc["a1", "feature_available_date"] == pd.Timestamp("2020-02-03")
    # month-end panel: a filing accepted ON the month end must not be attached to that month end
    con.register("av", avail)
    con.register("panel", pd.DataFrame({"cik": [1, 2, 2], "month_end": pd.to_datetime(
        ["2020-01-31", "2020-02-28", "2020-03-31"]).date}))
    res = con.sql("""
        select p.*, a.accession from panel p
        asof left join av a on p.cik = a.cik and p.month_end >= a.feature_available_date
        order by p.cik, p.month_end
    """).df()
    assert res["accession"].isna().iloc[0]           # a1 accepted 2020-01-31 -> not usable on 2020-01-31
    assert pd.isna(res.iloc[1]["accession"])          # b1 accepted 2020-02-28 -> not usable on 2020-02-28
    assert res.iloc[2]["accession"] == "b1"


def _panel():
    months = pd.date_range("2008-01-01", "2024-11-01", freq="MS")
    rows = [{"month": m, "month_end": m + pd.offsets.MonthEnd(0), "target": 0.1, "ticker": t}
            for m in months for t in "ABC"]
    return pd.DataFrame(rows)


def test_fold_masks_embargo_and_no_overlap():
    df = _panel()
    train, test = fold_masks(df, 2018, embargo_months=1)
    hold = df["month"] + pd.DateOffset(months=1)
    assert hold[test].dt.year.unique().tolist() == [2018]
    # last training holding month = Nov 2017; Dec 2017 is the embargo month
    assert hold[train].max() == pd.Timestamp("2017-11-01")
    assert df.loc[test, "month"].min() == pd.Timestamp("2017-12-01")
    # every training target is realised before the first test formation date
    assert (hold[train] + pd.offsets.MonthEnd(0)).max() < df.loc[test, "month_end"].min()


def test_inner_validation_inside_training_window():
    df = _panel()
    train, _ = fold_masks(df, 2020, embargo_months=1)
    inner, val = inner_split(df, train, validation_years=2, embargo_months=1)
    assert not (inner & val).any()
    assert val[~train].sum() == 0 and inner[~train].sum() == 0
    hold = df["month"] + pd.DateOffset(months=1)
    assert hold[inner].max() < hold[val].min() - pd.DateOffset(months=1)


def test_fold_cutoffs_and_previous_filing():
    me = pd.Series(pd.date_range("2016-01-31", "2019-12-31", freq="ME"))
    cut = fold_cutoffs(me, [2018, 2019])
    assert cut[2018] == pd.Timestamp("2017-12-31")
    f = pd.DataFrame({"cik": [1, 1, 1, 2], "accession": ["x", "y", "z", "w"],
                      "acceptance_datetime": pd.to_datetime(["2015-02-01", "2016-02-03", "2019-02-01", "2016-01-01"])})
    prev = previous_filing(f)
    assert prev.tolist()[:2] == [np.nan, "x"] or (pd.isna(prev.iloc[0]) and prev.iloc[1] == "x")
    assert pd.isna(prev.iloc[2])  # 3-year gap -> no "previous year" filing
    assert pd.isna(prev.iloc[3])
