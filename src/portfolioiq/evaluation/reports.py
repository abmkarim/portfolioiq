"""Backtests of all signals, evaluation, ``reports/metrics.json`` and figures."""

from __future__ import annotations

import datetime as dt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from portfolioiq.config import Config  # noqa: E402
from portfolioiq.evaluation import metrics as M  # noqa: E402
from portfolioiq.evaluation.fama_french import ff_regression  # noqa: E402
from portfolioiq.models.xgb import monthly_ic  # noqa: E402
from portfolioiq.portfolio.backtest import daily_returns_matrix, run_backtest  # noqa: E402
from portfolioiq.utils import get_logger, read_json, write_json, write_parquet  # noqa: E402

log = get_logger(__name__)

ABLATIONS = {"no_word2vec": ("word2vec",), "no_finbert": ("finbert",)}


def _signals(cfg: Config) -> dict[str, pd.DataFrame]:
    pred = pd.read_parquet(cfg.processed_dir / "predictions_main.parquet")
    base = pred[["month_end", "ticker"]].copy()
    rng = np.random.default_rng(cfg.seed)
    sig = {
        "model": base.assign(score=pred["pred"].to_numpy()),
        "random": base.assign(score=rng.standard_normal(len(base))),
        "momentum": base.assign(score=pred["mom_12_1"].to_numpy()),
    }
    for name in ABLATIONS:
        p = cfg.processed_dir / f"predictions_{name}.parquet"
        if p.exists():
            a = pd.read_parquet(p)
            sig[f"model_{name}"] = a[["month_end", "ticker"]].assign(score=a["pred"].to_numpy())
    return sig


def run_backtests(cfg: Config) -> pd.DataFrame:
    """Backtest the model, the benchmarks, cost sensitivities and ablations."""
    prices = pd.read_parquet(cfg.processed_dir / "prices.parquet")
    universe = pd.read_parquet(cfg.processed_dir / "universe.parquet")
    pc = cfg.prices
    causes = universe.groupby("ticker")["removal_cause"].first()
    delist = {t: pc.delisting_return_acquisition if c == "acquisition" else pc.delisting_return_unknown
              for t, c in causes.items()}
    rets = daily_returns_matrix(prices, delist)
    panel = pd.read_parquet(cfg.processed_dir / "panel_features.parquet", columns=["ticker", "ff12"])
    sectors = panel.drop_duplicates("ticker", keep="last").set_index("ticker")["ff12"]

    monthly_all, daily_all, weights_all = [], [], []
    sig = _signals(cfg)
    runs = [(name, s, None) for name, s in sig.items()]
    runs += [(f"model_tc{int(b)}bps", sig["model"], b) for b in cfg.evaluation.cost_sensitivity_bps]
    for name, s, tc in runs:
        d, m, w = run_backtest(s, rets, sectors, cfg, tc_bps=tc, name=name)
        monthly_all.append(m)
        daily_all.append(d)
        weights_all.append(w)
    monthly = pd.concat(monthly_all, ignore_index=True)
    write_parquet(monthly, cfg.reports_dir / f"backtest_monthly_{cfg.name}.parquet")
    write_parquet(pd.concat(daily_all, ignore_index=True), cfg.reports_dir / f"backtest_daily_{cfg.name}.parquet")
    write_parquet(pd.concat(weights_all, ignore_index=True), cfg.reports_dir / f"weights_{cfg.name}.parquet")
    return monthly


def _ic_block(pred: pd.DataFrame) -> dict:
    ic = monthly_ic(pred["month_end"], pred["pred"], pred["ret_fwd_1m"])
    by_year = ic.groupby(pd.to_datetime(ic.index).year).mean()
    hold_year = ic.groupby((pd.to_datetime(ic.index) + pd.offsets.MonthEnd(1)).year).mean()
    return {"mean": float(ic.mean()), "tstat": M.tstat(ic), "std": float(ic.std(ddof=1)),
            "n_months": int(len(ic)), "share_positive": float((ic > 0).mean()),
            "by_holding_year": {int(k): float(v) for k, v in hold_year.items()},
            "_series": ic, "_by_formation_year": by_year}


def evaluate(cfg: Config) -> dict:
    """Compute every metric and figure; writes ``reports/metrics.json`` (or ``metrics_<run>.json``)."""
    monthly = pd.read_parquet(cfg.reports_dir / f"backtest_monthly_{cfg.name}.parquet")
    factors = pd.read_parquet(cfg.interim_dir / "factors_monthly.parquet")
    lags = cfg.evaluation.newey_west_lags
    start, end = f"{cfg.model.oos_start_year}-01", f"{cfg.model.oos_end_year}-12"
    ym = pd.to_datetime(monthly["month_end"]).dt.to_period("M")
    monthly = monthly[(ym >= start) & (ym <= end)]

    def block(name: str) -> dict:
        m = monthly[monthly["strategy"] == name].sort_values("month_end")
        if m.empty:
            return {}
        return {**M.summary(m), "fama_french": ff_regression(m, factors, lags)}

    out: dict = {
        "run": {
            "config": cfg.name, "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "oos_period": [start, end], "price_source": cfg.prices.source,
            "universe_sample_size": cfg.universe.sample_size,
            "finbert_max_sentences_per_item": cfg.nlp.finbert.max_sentences_per_item,
            "tc_bps_per_side": cfg.portfolio.tc_bps, "borrow_bps_annual": cfg.portfolio.borrow_bps_annual,
            "sharpe_convention": "mean/std of monthly net returns x sqrt(12); dollar-neutral => no RF subtraction",
        },
        "strategy": block("model"),
        "benchmarks": {"random_signal": block("random"), "momentum_12_1": block("momentum")},
    }
    pred = pd.read_parquet(cfg.processed_dir / "predictions_main.parquet")
    ic = _ic_block(pred)
    out["ic"] = {k: v for k, v in ic.items() if not k.startswith("_")}

    rob: dict = {"costs": {}, "subperiods": {}, "ablations": {}}
    for b in cfg.evaluation.cost_sensitivity_bps:
        rob["costs"][f"{int(b)}bps"] = block(f"model_tc{int(b)}bps")
    main = monthly[monthly["strategy"] == "model"].sort_values("month_end")
    for a, b in cfg.evaluation.subperiods:
        p = pd.to_datetime(main["month_end"]).dt.to_period("M")
        sub = main[(p >= a) & (p <= b)]
        if len(sub):
            rob["subperiods"][f"{a}_{b}"] = {**M.summary(sub), "fama_french": ff_regression(sub, factors, lags)}
    for name in ABLATIONS:
        b = block(f"model_{name}")
        p = cfg.processed_dir / f"predictions_{name}.parquet"
        if b and p.exists():
            a_ic = _ic_block(pd.read_parquet(p))
            b["ic_mean"], b["ic_tstat"] = a_ic["mean"], a_ic["tstat"]
            rob["ablations"][name] = b
    out["robustness"] = rob
    out["data_quality"] = read_json(cfg.reports_dir / f"data_quality_{cfg.name}.json", {})
    _figures(cfg, monthly, ic)
    # The README reads reports/metrics.json: written by the run defined by RUN_SCALE.
    write_json(_clean(out), cfg.reports_dir / f"metrics_{cfg.name}.json")
    write_json(_clean(out), cfg.reports_dir / "metrics.json")
    log.info("evaluation written", sharpe=out["strategy"].get("sharpe"), ic=out["ic"]["mean"])
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, float | np.floating):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _figures(cfg: Config, monthly: pd.DataFrame, ic: dict) -> None:
    fig_dir = cfg.reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = {"model": "Modèle NLP (net)", "random": "Signal aléatoire", "momentum": "Momentum 12-1"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, lab in labels.items():
        m = monthly[monthly["strategy"] == k].sort_values("month_end")
        if len(m):
            ax.plot(pd.to_datetime(m["month_end"]), (1 + m["ret_net"]).cumprod(), label=lab,
                    lw=2 if k == "model" else 1.2)
    ax.axhline(1, color="grey", lw=0.6)
    ax.set_title(f"Capital cumulé, net de coûts ({cfg.name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "equity_curve.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3))
    for k, lab in labels.items():
        m = monthly[monthly["strategy"] == k].sort_values("month_end")
        if len(m):
            ax.fill_between(pd.to_datetime(m["month_end"]), M.drawdown_series(m["ret_net"].to_numpy()), 0,
                            alpha=0.5 if k == "model" else 0.2, label=lab)
    ax.set_title("Drawdown (net)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "drawdown.png", dpi=110)
    plt.close(fig)

    by_year = pd.Series(ic["by_holding_year"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(by_year.index.astype(str), by_year.values, color=["#4C72B0" if v >= 0 else "#C44E52" for v in by_year])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title(f"IC de Spearman moyen par année (moyenne globale {ic['mean']:.3f}, t = {ic['tstat']:.2f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "ic_by_year.png", dpi=110)
    plt.close(fig)
