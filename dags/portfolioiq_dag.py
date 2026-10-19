"""Airflow DAG ``portfolioiq_pipeline``: same step functions as the CLI (``portfolioiq.pipeline``).

Heavy imports happen inside the task callables so that DAG parsing stays fast and only needs Airflow.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

SRC = os.environ.get("PIQ_SRC", "/opt/airflow/src")
CONFIG = os.environ.get("PIQ_CONFIG", "/opt/airflow/configs/sample.yaml")

STEPS = [
    "ingest_universe", "ingest_filings", "ingest_prices", "ingest_factors", "dbt_run", "dbt_test",
    "build_features", "train_walkforward", "backtest", "evaluate", "export_powerbi",
]


def _run(step: str) -> None:
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    from portfolioiq.config import load_config
    from portfolioiq.pipeline import run_step
    from portfolioiq.utils import setup_logging

    setup_logging()
    run_step(step, load_config(CONFIG))


default_args = {
    "owner": "portfolioiq",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
}

with DAG(
    dag_id="portfolioiq_pipeline",
    description="10-K NLP long-short S&P 500 pipeline (idempotent, cached steps)",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["portfolioiq"],
) as dag:
    tasks = {
        s: PythonOperator(task_id=s, python_callable=_run, op_kwargs={"step": s}) for s in STEPS
    }
    for up, down in zip(STEPS[:-1], STEPS[1:], strict=True):
        tasks[up] >> tasks[down]
