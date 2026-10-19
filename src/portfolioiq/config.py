"""Configuration loading: a single YAML file plus environment variables (``.env``)."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PathsConfig(BaseModel):
    data_dir: str = "data"
    reports_dir: str = "reports"
    mlruns_dir: str = "mlruns"
    logs_dir: str = "logs"
    powerbi_dir: str = "powerbi/data"
    dbt_dir: str = "dbt"


class UniverseConfig(BaseModel):
    components_url: str
    wiki_title: str = "List of S&P 500 companies"
    wiki_revid: int = 1292523673
    start_date: str = "2007-01-01"
    end_date: str = "2024-12-31"
    sample_size: int | None = None
    sample_oos_share: float = 0.8
    sample_anchor_date: str = "2018-01-02"
    name_match_threshold: float = 0.82


class EdgarConfig(BaseModel):
    forms: list[str] = Field(default_factory=lambda: ["10-K", "10-K405"])
    filing_start: str = "2007-01-01"
    filing_end: str = "2024-12-31"
    max_requests_per_second: float = 8.0
    download_workers: int = 4
    lookback_days_before_entry: int = 450
    min_section_chars: int = 2000


class PricesConfig(BaseModel):
    source: str = "yfinance"
    start: str = "2006-01-01"
    end: str = "2025-01-31"
    batch_size: int = 40
    delisting_return_acquisition: float = 0.0
    delisting_return_unknown: float = -0.30
    delisting_match_days: int = 120


class FinbertConfig(BaseModel):
    model_name: str = "ProsusAI/finbert"
    max_sentences_per_item: int = 64
    batch_size: int = 32
    max_length: int = 128
    quantize: bool = True
    num_threads: int = 0


class Word2VecConfig(BaseModel):
    vector_size: int = 100
    window: int = 5
    min_count: int = 5
    epochs: int = 3
    max_tokens_per_doc: int = 6000
    workers: int = 4


class NlpConfig(BaseModel):
    finbert: FinbertConfig = Field(default_factory=FinbertConfig)
    word2vec: Word2VecConfig = Field(default_factory=Word2VecConfig)
    max_filing_age_months: int = 15
    tfidf_max_features: int = 20000


class ModelConfig(BaseModel):
    oos_start_year: int = 2018
    oos_end_year: int = 2024
    embargo_months: int = 1
    validation_years: int = 2
    param_grid: list[dict[str, Any]] = Field(default_factory=list)
    n_estimators_max: int = 600
    early_stopping_rounds: int = 50
    shap_sample: int = 2000
    n_jobs: int = 4


class PortfolioConfig(BaseModel):
    risk_aversion: float = 10.0
    turnover_penalty_multiplier: float = 1.0
    max_weight: float = 0.02
    max_gross: float = 2.0
    cov_window: int = 252
    min_cov_obs: int = 126
    tc_bps: float = 10.0
    borrow_bps_annual: float = 50.0
    alpha_ic: float = 0.05
    solver: str = "CLARABEL"


class EvaluationConfig(BaseModel):
    newey_west_lags: int = 6
    cost_sensitivity_bps: list[float] = Field(default_factory=lambda: [5.0, 20.0])
    subperiods: list[list[str]] = Field(
        default_factory=lambda: [["2018-01", "2020-12"], ["2021-01", "2024-12"]]
    )
    run_ablations: bool = True


class Config(BaseModel):
    """Top-level configuration (one YAML file per run scale)."""

    name: str = "default"
    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    universe: UniverseConfig
    edgar: EdgarConfig = Field(default_factory=EdgarConfig)
    prices: PricesConfig = Field(default_factory=PricesConfig)
    nlp: NlpConfig = Field(default_factory=NlpConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # ---- resolved paths -------------------------------------------------
    def path(self, attr: str) -> Path:
        """Return an absolute path for one of the ``paths`` entries."""
        p = Path(getattr(self.paths, attr))
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def data_dir(self) -> Path:
        return self.path("data_dir")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        """Run-specific outputs (features, predictions) — separated by config name."""
        return self.data_dir / "processed" / self.name

    @property
    def reports_dir(self) -> Path:
        return self.path("reports_dir")

    @property
    def warehouse_path(self) -> Path:
        return self.data_dir / f"warehouse_{self.name}.duckdb"

    @property
    def sec_user_agent(self) -> str:
        """SEC User-Agent from the environment (never logged)."""
        ua = os.environ.get("SEC_USER_AGENT", "").strip()
        if not ua:
            raise RuntimeError("SEC_USER_AGENT is not set (see .env.example)")
        return ua


def load_config(path: str | Path | None = None) -> Config:
    """Load ``.env`` then a YAML config file into a validated :class:`Config`."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    base = raw.pop("extends", None)
    if base:
        with open(path.parent / base, encoding="utf-8") as fh:
            parent = yaml.safe_load(fh) or {}
        raw = _deep_merge(parent, raw)
    return Config(**raw)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def set_seeds(seed: int) -> None:
    """Fix seeds for python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass
