import pandas as pd

from portfolioiq.ingestion.universe import (
    build_intervals,
    classify_removal,
    clip_intervals,
    members_on,
    name_similarity,
    select_sample,
)


def _snapshots():
    return pd.DataFrame({
        "date": ["2010-01-04", "2012-06-01", "2015-03-02", "2020-01-02"],
        "tickers": ["AAA,BBB,CCC", "AAA,BBB,DDD", "AAA,DDD,EEE", "AAA,DDD,EEE,BBB"],
    })


def test_intervals_point_in_time():
    iv = build_intervals(_snapshots())
    # CCC removed on 2012-06-01 -> member until the day before
    ccc = iv[iv.ticker == "CCC"].iloc[0]
    assert ccc.start_date == pd.Timestamp("2010-01-04")
    assert ccc.end_date == pd.Timestamp("2012-05-31")
    # BBB leaves then re-enters: two intervals
    assert (iv.ticker == "BBB").sum() == 2
    # still-member tickers are open-ended
    assert pd.isna(iv[iv.ticker == "AAA"].end_date.iloc[0])


def test_members_on_includes_later_removed_names():
    iv = build_intervals(_snapshots())
    assert members_on(iv, "2011-01-01") == {"AAA", "BBB", "CCC"}      # CCC later removed: still a member here
    assert members_on(iv, "2012-06-01") == {"AAA", "BBB", "DDD"}
    assert members_on(iv, "2016-01-01") == {"AAA", "DDD", "EEE"}      # no look-ahead: BBB returns only in 2020
    assert "BBB" in members_on(iv, "2021-01-01")
    assert members_on(iv, "2009-01-01") == set()


def test_clip_and_sample_deterministic():
    iv = clip_intervals(build_intervals(_snapshots()), "2011-01-01", "2019-12-31")
    assert iv.start_date.min() >= pd.Timestamp("2011-01-01")
    assert iv.end_date.max() <= pd.Timestamp("2019-12-31")
    s1 = select_sample(iv, 3, "2016-01-01", 0.67, seed=1)
    s2 = select_sample(iv, 3, "2016-01-01", 0.67, seed=1)
    assert s1 == s2 and len(s1) == 3


def test_name_matching_and_causes():
    assert name_similarity("Altera Corp.", "ALTERA CORP") > 0.95
    assert name_similarity("Apple Inc.", "Pfizer Inc") < 0.6
    assert classify_removal("Acquired by Microsoft") == "acquisition"
    assert classify_removal("Market capitalization change") == "unknown"
    assert classify_removal("Filed for Chapter 11 bankruptcy") == "bankruptcy"
