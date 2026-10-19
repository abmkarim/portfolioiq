"""Pipeline steps shared by the CLI and the Airflow DAG (each step is idempotent)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable

import duckdb

from portfolioiq.config import Config, set_seeds
from portfolioiq.utils import get_logger

log = get_logger(__name__)


def ingest_universe(cfg: Config) -> None:
    from portfolioiq.ingestion.universe import ingest_universe as f

    f(cfg)


def ingest_filings(cfg: Config) -> None:
    from portfolioiq.ingestion.edgar import ingest_filings as f
    from portfolioiq.ingestion.sectors import build_sectors

    f(cfg)
    build_sectors(cfg)


def ingest_prices(cfg: Config) -> None:
    from portfolioiq.ingestion.edgar import ingest_shares
    from portfolioiq.ingestion.prices import ingest_prices as f

    f(cfg)
    ingest_shares(cfg)


def ingest_factors(cfg: Config) -> None:
    from portfolioiq.ingestion.factors import ingest_factors as f

    f(cfg)


def _dbt(cfg: Config, args: list[str]) -> None:
    from dbt.cli.main import dbtRunner

    dbt_dir = cfg.path("dbt_dir")
    os.environ["PIQ_WAREHOUSE"] = str(cfg.warehouse_path)
    variables = {
        "data_dir": cfg.processed_dir.as_posix(), "interim_dir": cfg.interim_dir.as_posix(),
        "reports_dir": cfg.reports_dir.as_posix(), "run_name": cfg.name,
        "delist_return_acquisition": cfg.prices.delisting_return_acquisition,
        "delist_return_unknown": cfg.prices.delisting_return_unknown,
        "max_filing_age_months": cfg.nlp.max_filing_age_months,
    }
    full = args + ["--project-dir", str(dbt_dir), "--profiles-dir", str(dbt_dir), "--vars", json.dumps(variables)]
    res = dbtRunner().invoke(full)
    if not res.success:
        raise RuntimeError(f"dbt {' '.join(args)} failed: {res.exception}")


def dbt_run(cfg: Config) -> None:
    _dbt(cfg, ["run", "--exclude", "tag:post_model"])


def dbt_test(cfg: Config) -> None:
    _dbt(cfg, ["test", "--exclude", "tag:post_model"])


def build_features(cfg: Config) -> None:
    from portfolioiq.nlp.features import build_features as f

    f(cfg)


def train_walkforward(cfg: Config) -> None:
    from portfolioiq.evaluation.reports import ABLATIONS
    from portfolioiq.models.walkforward import train_walkforward as f

    f(cfg, "main")
    if cfg.evaluation.run_ablations:
        for name, groups in ABLATIONS.items():
            f(cfg, name, drop_groups=groups, log_mlflow=True, do_shap=False)


def backtest(cfg: Config) -> None:
    from portfolioiq.evaluation.reports import run_backtests

    run_backtests(cfg)


def evaluate(cfg: Config) -> None:
    from portfolioiq.evaluation.reports import evaluate as f

    f(cfg)


def export_powerbi(cfg: Config) -> None:
    """Build post-model marts, test them, then export all marts to ``powerbi/data/*.csv``."""
    _dbt(cfg, ["build", "--select", "tag:post_model"])
    out = cfg.path("powerbi_dir")
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(cfg.warehouse_path), read_only=True)
    try:
        for t in ("dim_company", "fct_monthly_panel", "fct_backtest_returns", "fct_shap_importance"):
            con.sql(f"select * from {t}").df().to_csv(out / f"{t}.csv", index=False)
    finally:
        con.close()
    log.info("powerbi export", dir=str(out))


STEPS: dict[str, Callable[[Config], None]] = {
    "ingest_universe": ingest_universe,
    "ingest_filings": ingest_filings,
    "ingest_prices": ingest_prices,
    "ingest_factors": ingest_factors,
    "dbt_run": dbt_run,
    "dbt_test": dbt_test,
    "build_features": build_features,
    "train_walkforward": train_walkforward,
    "backtest": backtest,
    "evaluate": evaluate,
    "export_powerbi": export_powerbi,
}


def run_step(name: str, cfg: Config) -> float:
    """Run one step with seeds fixed; returns elapsed seconds."""
    set_seeds(cfg.seed)
    t0 = time.time()
    log.info("step start", step=name, config=cfg.name)
    STEPS[name](cfg)
    elapsed = time.time() - t0
    log.info("step done", step=name, seconds=round(elapsed, 1))
    return elapsed
