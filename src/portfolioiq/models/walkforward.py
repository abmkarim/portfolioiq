"""Expanding-window walk-forward training (2018-2024) with a one-month embargo.

Conventions
-----------
* A panel row is formed at ``month_end`` (month *m*); its target is the return of the
  *holding month* m+1.
* Fold *Y* predicts every row whose holding month falls in calendar year *Y*
  (formation dates Dec Y-1 ... Nov Y), so out-of-sample returns span Jan Y ... Dec Y.
* Training rows: holding month <= December Y-1 minus ``embargo_months`` (default: holding month
  <= November Y-1). All their targets are realised before the first test formation date, plus a
  one-month gap.
* Hyper-parameters are chosen ONLY by an internal temporal validation inside the training
  window (last ``validation_years`` of holding months, with the same embargo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolioiq.config import Config
from portfolioiq.models import shap_analysis
from portfolioiq.models.xgb import fit_xgb, monthly_ic
from portfolioiq.nlp.features import ALL_FEATURES, FEATURE_GROUPS, W2V_FEATURES, cs_rank
from portfolioiq.tracking.mlflow_utils import setup_mlflow
from portfolioiq.utils import get_logger, write_parquet

log = get_logger(__name__)

DEFAULT_GRID = [
    {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 50},
    {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 50},
    {"max_depth": 3, "learning_rate": 0.02, "min_child_weight": 200, "reg_lambda": 5.0},
    {"max_depth": 4, "learning_rate": 0.03, "min_child_weight": 100},
]


def holding_month(month: pd.Series) -> pd.Series:
    return pd.to_datetime(month) + pd.DateOffset(months=1)


def fold_masks(df: pd.DataFrame, year: int, embargo_months: int) -> tuple[pd.Series, pd.Series]:
    """Boolean masks (train, test) for fold ``year``."""
    hold = holding_month(df["month"])
    test = hold.dt.year == year
    last_train_hold = pd.Timestamp(year=year - 1, month=12, day=1) - pd.DateOffset(months=embargo_months)
    train = (hold <= last_train_hold) & df["target"].notna()
    return train, test


def inner_split(df: pd.DataFrame, train: pd.Series, validation_years: int, embargo_months: int):
    """Split the training window into inner-train / validation by holding month (with embargo)."""
    hold = holding_month(df["month"])
    last = hold[train].max()
    val_start = last - pd.DateOffset(months=12 * validation_years - 1)
    val = train & (hold >= val_start)
    inner = train & (hold <= val_start - pd.DateOffset(months=embargo_months + 1))
    return inner, val


def select_params(df: pd.DataFrame, feats: list[str], inner: pd.Series, val: pd.Series, cfg: Config):
    """Grid search on the internal validation window; criterion = mean monthly Spearman IC."""
    grid = cfg.model.param_grid or DEFAULT_GRID
    best = None
    for params in grid:
        m = fit_xgb(df.loc[inner, feats], df.loc[inner, "target"], params, cfg.model.n_estimators_max,
                    cfg.seed, cfg.model.n_jobs, df.loc[val, feats], df.loc[val, "target"],
                    cfg.model.early_stopping_rounds)
        ic = monthly_ic(df.loc[val, "month_end"], m.predict(df.loc[val, feats]), df.loc[val, "ret_fwd_1m"])
        score = float(ic.mean()) if len(ic) else -np.inf
        n_trees = int(getattr(m, "best_iteration", cfg.model.n_estimators_max - 1) or 0) + 1
        if best is None or score > best[1]:
            best = (params, score, max(n_trees, 20))
    return best


def _fold_frame(panel: pd.DataFrame, w2v: pd.DataFrame, year: int) -> pd.DataFrame:
    """Attach fold-specific Word2Vec similarities and rank them cross-sectionally."""
    w = w2v[w2v["fold_year"] == year].drop(columns="fold_year")
    df = panel.drop(columns=[c for c in W2V_FEATURES if c in panel.columns]).merge(w, on="accession", how="left")
    for c in W2V_FEATURES:
        if c not in df:
            df[c] = np.nan
    return cs_rank(df, W2V_FEATURES)


def train_walkforward(cfg: Config, variant: str = "main", drop_groups: tuple[str, ...] = (),
                      log_mlflow: bool = True, do_shap: bool = True) -> pd.DataFrame:
    """Run all folds for one feature-set variant; writes ``predictions_<variant>.parquet``."""
    panel = pd.read_parquet(cfg.processed_dir / "panel_features.parquet")
    w2v = pd.read_parquet(cfg.processed_dir / "w2v_features.parquet")
    feats = [f for f in ALL_FEATURES if FEATURE_GROUPS[f] not in drop_groups]
    years = list(range(cfg.model.oos_start_year, cfg.model.oos_end_year + 1))
    mlflow = setup_mlflow(cfg) if log_mlflow else None
    fig_dir = cfg.reports_dir / "figures"
    preds, imps, shap_last = [], [], None

    parent = mlflow.start_run(run_name=f"walkforward_{variant}") if mlflow else None
    try:
        if mlflow:
            mlflow.log_params({"variant": variant, "features": ",".join(feats), "n_features": len(feats),
                               "embargo_months": cfg.model.embargo_months, "seed": cfg.seed,
                               "config": cfg.name, "dropped_groups": ",".join(drop_groups) or "none"})
        for year in years:
            df = _fold_frame(panel, w2v, year)
            train, test = fold_masks(df, year, cfg.model.embargo_months)
            if train.sum() < 200 or test.sum() == 0:
                log.warning("fold skipped", fold=year, n_train=int(train.sum()), n_test=int(test.sum()))
                continue
            inner, val = inner_split(df, train, cfg.model.validation_years, cfg.model.embargo_months)
            params, val_ic, n_trees = select_params(df, feats, inner, val, cfg)
            model = fit_xgb(df.loc[train, feats], df.loc[train, "target"], params, n_trees, cfg.seed, cfg.model.n_jobs)
            p = model.predict(df.loc[test, feats])
            out = df.loc[test, ["month", "month_end", "ticker", "cik", "ff12", "ret_fwd_1m", "target",
                                "mom_12_1"]].copy()
            out["pred"], out["fold_year"] = p, year
            preds.append(out)
            ic = monthly_ic(out["month_end"], p, out["ret_fwd_1m"])
            mse = float(np.nanmean((out["target"] - p) ** 2))
            log.info("fold done", variant=variant, fold=year, n_train=int(train.sum()), n_test=int(test.sum()),
                     val_ic=round(val_ic, 4), test_ic=round(float(ic.mean()), 4), params=params, n_trees=n_trees)
            if mlflow:
                with mlflow.start_run(run_name=f"fold_{year}", nested=True):
                    mlflow.log_params({**{f"xgb_{k}": v for k, v in params.items()}, "n_estimators": n_trees,
                                       "fold_year": year, "n_train": int(train.sum()), "n_test": int(test.sum())})
                    mlflow.log_metrics({"val_ic": val_ic, "test_ic_mean": float(ic.mean()), "test_mse": mse})
                    if do_shap:
                        vals, Xs = shap_analysis.shap_values(model, df.loc[test, feats], cfg.model.shap_sample, cfg.seed)
                        imp = shap_analysis.importance_table(vals, feats, year, FEATURE_GROUPS)
                        imps.append(imp)
                        bp = shap_analysis.beeswarm(vals, Xs, fig_dir / "folds" / f"shap_beeswarm_{year}.png",
                                                    f"SHAP — fold {year}")
                        mlflow.log_artifact(str(bp), "shap")
                        shap_last = (vals, Xs)
                    try:
                        import mlflow.xgboost

                        mlflow.xgboost.log_model(model, artifact_path="model")
                    except Exception as exc:  # noqa: BLE001
                        log.warning("model logging failed", error=str(exc)[:200])
        if not preds:
            raise RuntimeError("no fold could be trained")
        pred_df = pd.concat(preds, ignore_index=True)
        write_parquet(pred_df, cfg.processed_dir / f"predictions_{variant}.parquet")
        ic_all = monthly_ic(pred_df["month_end"], pred_df["pred"], pred_df["ret_fwd_1m"])
        if mlflow:
            mlflow.log_metrics({"oos_ic_mean": float(ic_all.mean()),
                                "oos_ic_tstat": float(ic_all.mean() / ic_all.std(ddof=1) * np.sqrt(len(ic_all)))})
        if imps:
            imp_df = pd.concat(imps, ignore_index=True)
            write_parquet(imp_df, cfg.reports_dir / f"shap_importance_{cfg.name}.parquet")
            sp, stab = shap_analysis.stability_figure(imp_df, fig_dir / "shap_stability.png")
            gp = shap_analysis.global_importance_figure(imp_df, fig_dir / "shap_importance.png")
            if shap_last is not None:
                bp = shap_analysis.beeswarm(shap_last[0], shap_last[1], fig_dir / "shap_beeswarm.png",
                                            "SHAP — dernier fold")
            if mlflow:
                mlflow.log_metric("shap_rank_stability", stab)
                for f in (sp, gp, bp):
                    mlflow.log_artifact(str(f), "shap")
    finally:
        if parent is not None:
            mlflow.end_run()
    return pred_df
