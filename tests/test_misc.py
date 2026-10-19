import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolioiq.config import load_config
from portfolioiq.ingestion.factors import parse_french_csv
from portfolioiq.ingestion.prices import unadjust_close, yahoo_symbol
from portfolioiq.ingestion.sectors import sic_to_ff12
from portfolioiq.models.xgb import monthly_ic

ROOT = Path(__file__).resolve().parents[1]


def test_configs_load():
    s = load_config(ROOT / "configs" / "sample.yaml")
    d = load_config(ROOT / "configs" / "default.yaml")
    assert s.name == "sample" and s.universe.sample_size == 50
    assert d.universe.sample_size is None
    assert s.portfolio.max_weight == d.portfolio.max_weight == 0.02   # inherited
    assert s.model.oos_start_year == 2018 and s.model.oos_end_year == 2024


def test_ff12_mapping():
    assert sic_to_ff12(3571) == 6      # computers -> BusEq
    assert sic_to_ff12(6021) == 11     # banks -> Money
    assert sic_to_ff12(2834) == 10     # pharma -> Hlth
    assert sic_to_ff12(1311) == 4      # oil & gas -> Enrgy
    assert sic_to_ff12(4911) == 8      # electric -> Utils
    assert sic_to_ff12(None) == 12 and sic_to_ff12(9999) == 12


def test_french_csv_parser():
    text = ("This file was created...\n\n,Mkt-RF,SMB,HML,RMW,CMA,RF\n196307,  -0.39, -0.44, -0.89,  0.68, -1.23,  0.27\n"
            "196308,   5.07, -0.75,  1.68,  0.36, -0.34,  0.25\n\n Annual Factors\n,Mkt-RF\n1964, 1.0\n")
    df = parse_french_csv(text, "monthly")
    assert len(df) == 2
    assert df["date"].iloc[0] == pd.Timestamp("1963-07-31")
    assert df["mkt_rf"].iloc[1] == pytest.approx(0.0507)


def test_unadjust_close_and_symbols():
    close = pd.Series([50.0, 51.0, 52.0])          # split-adjusted
    splits = pd.Series([0.0, 0.0, 2.0])             # 2:1 split on day 3
    raw = unadjust_close(close, splits)
    assert raw.tolist() == [100.0, 102.0, 52.0]
    assert yahoo_symbol("BRK.B") == "BRK-B"


def test_monthly_ic():
    d = ["m1"] * 6 + ["m2"] * 6
    p = np.r_[np.arange(6), np.arange(6)]
    r = np.r_[np.arange(6), -np.arange(6)]
    ic = monthly_ic(pd.Series(d), p, pd.Series(r))
    assert ic["m1"] == pytest.approx(1.0) and ic["m2"] == pytest.approx(-1.0)


def test_dag_structure():
    """Validate the DAG file without Airflow: task ids and dependency chain."""
    src = (ROOT / "dags" / "portfolioiq_dag.py").read_text(encoding="utf-8")
    ast.parse(src)
    expected = ["ingest_universe", "ingest_filings", "ingest_prices", "ingest_factors", "dbt_run", "dbt_test",
                "build_features", "train_walkforward", "backtest", "evaluate", "export_powerbi"]
    assert 'dag_id="portfolioiq_pipeline"' in src
    for t in expected:
        assert f'"{t}"' in src


def test_dagbag_imports_without_errors():
    pytest.importorskip("airflow.models")
    from airflow.models import DagBag

    bag = DagBag(dag_folder=str(ROOT / "dags"), include_examples=False)
    assert not bag.import_errors
    dag = bag.get_dag("portfolioiq_pipeline")
    assert dag.get_task("export_powerbi").upstream_task_ids == {"evaluate"}
