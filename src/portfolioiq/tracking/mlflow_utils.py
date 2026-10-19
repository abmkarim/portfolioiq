"""MLflow helpers (local file store under ``mlruns/``)."""

from __future__ import annotations

import contextlib
import os

from portfolioiq.config import Config


def setup_mlflow(cfg: Config, experiment: str = "portfolioiq"):
    """Point MLflow at the local ``mlruns/`` directory and select the experiment."""
    import mlflow

    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    uri = cfg.path("mlruns_dir").resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(f"{experiment}-{cfg.name}")
    return mlflow


@contextlib.contextmanager
def maybe_run(mlflow, enabled: bool, **kwargs):
    """``mlflow.start_run`` when enabled, no-op context otherwise (used by ablations)."""
    if enabled:
        with mlflow.start_run(**kwargs) as run:
            yield run
    else:
        yield None
