"""Daily prices (adjusted) from ``yfinance`` (default), ``tiingo`` or a ``crsp_csv`` export.

Output (``prices.parquet``): ``date, ticker, adj_close, close_raw, volume`` where
``adj_close`` is split+dividend adjusted (for returns) and ``close_raw`` is the
unadjusted close (for market capitalisation = close_raw x shares outstanding).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from portfolioiq.config import Config
from portfolioiq.utils import get_logger, read_json, retry_call, update_quality, write_json, write_parquet

log = get_logger(__name__)

PRICE_COLS = ["date", "ticker", "adj_close", "close_raw", "volume"]


def yahoo_symbol(ticker: str) -> str:
    """Index ticker -> Yahoo symbol (``BRK.B`` -> ``BRK-B``)."""
    return ticker.replace(".", "-").upper()


def unadjust_close(close_split_adj: pd.Series, splits: pd.Series) -> pd.Series:
    """Undo split adjustment: raw_t = close_t x prod(split ratios strictly after t)."""
    s = splits.fillna(0.0).replace(0.0, 1.0)
    # factor applying to date t = product of ratios at dates > t
    factor = s[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return close_split_adj * factor


def _yf_batch(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    data = yf.download(
        symbols, start=start, end=end, auto_adjust=False, actions=True, group_by="ticker",
        progress=False, threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if data is None or data.empty:
        return out
    for sym in symbols:
        try:
            d = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
        except KeyError:
            continue
        d = d.dropna(subset=["Adj Close"])
        if d.empty:
            continue
        splits = d["Stock Splits"] if "Stock Splits" in d else pd.Series(0.0, index=d.index)
        out[sym] = pd.DataFrame({
            "date": pd.to_datetime(d.index).tz_localize(None),
            "adj_close": d["Adj Close"].to_numpy(),
            "close_raw": unadjust_close(d["Close"], splits).to_numpy(),
            "volume": d["Volume"].to_numpy(),
        })
    return out


def _tiingo(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    import requests

    token = os.environ.get("TIINGO_API_KEY")
    if not token:
        raise RuntimeError("TIINGO_API_KEY not set")
    url = f"https://api.tiingo.com/tiingo/daily/{ticker.replace('.', '-')}/prices"
    resp = requests.get(url, params={"startDate": start, "endDate": end, "token": token}, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    d = pd.DataFrame(resp.json())
    if d.empty:
        return None
    return pd.DataFrame({
        "date": pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize(),
        "adj_close": d["adjClose"], "close_raw": d["close"], "volume": d["volume"],
    })


def _crsp_csv(cfg: Config, tickers: list[str]) -> pd.DataFrame:
    """Load a WRDS CRSP daily export (``date, TICKER, PRC, RET, VOL``) from ``data/external/crsp.csv``."""
    path = cfg.data_dir / "external" / "crsp.csv"
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df = df[df["ticker"].isin(tickers)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0)
    df = df.sort_values(["ticker", "date"])
    df["adj_close"] = df.groupby("ticker")["ret"].transform(lambda r: (1 + r).cumprod())
    df["close_raw"] = df["prc"].abs()
    df["volume"] = df.get("vol", np.nan)
    return df[PRICE_COLS]


def ingest_prices(cfg: Config) -> pd.DataFrame:
    """Download prices for all universe tickers with a per-ticker disk cache."""
    universe = pd.read_parquet(cfg.processed_dir / "universe.parquet")
    tickers = sorted(universe["ticker"].unique())
    pc = cfg.prices
    if pc.source == "crsp_csv":
        prices = _crsp_csv(cfg, tickers)
    else:
        cache = cfg.raw_dir / "prices" / pc.source
        cache.mkdir(parents=True, exist_ok=True)
        missing_path = cache / "_missing.json"
        missing = set(read_json(missing_path, []) or [])
        todo = [t for t in tickers if not (cache / f"{t}.parquet").exists() and t not in missing]
        log.info("prices to download", n=len(todo), source=pc.source)
        if pc.source == "yfinance":
            for i in range(0, len(todo), pc.batch_size):
                batch = todo[i:i + pc.batch_size]
                syms = {yahoo_symbol(t): t for t in batch}
                got = retry_call(_yf_batch, list(syms), pc.start, pc.end, what="yfinance batch") or {}
                for sym, t in syms.items():
                    if sym in got:
                        write_parquet(got[sym].assign(ticker=t)[PRICE_COLS], cache / f"{t}.parquet")
                    else:
                        missing.add(t)
                write_json(sorted(missing), missing_path)
                log.info("prices progress", done=min(i + pc.batch_size, len(todo)), total=len(todo))
        elif pc.source == "tiingo":
            for t in todo:
                d = retry_call(_tiingo, t, pc.start, pc.end, what="tiingo")
                if d is None:
                    missing.add(t)
                else:
                    write_parquet(d.assign(ticker=t)[PRICE_COLS], cache / f"{t}.parquet")
            write_json(sorted(missing), missing_path)
        else:
            raise ValueError(f"unknown price source {pc.source}")
        frames = [pd.read_parquet(cache / f"{t}.parquet") for t in tickers if (cache / f"{t}.parquet").exists()]
        prices = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLS)
    prices = prices[(prices["date"] >= pc.start) & (prices["date"] <= pc.end)]
    prices = prices.dropna(subset=["adj_close"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    write_parquet(prices, cfg.processed_dir / "prices.parquet")
    update_quality(cfg.reports_dir, cfg.name, "prices", price_coverage(universe, prices))
    return prices


def price_coverage(universe: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Share of historical members (and member-months) with price history."""
    have = set(prices["ticker"].unique())
    first_last = prices.groupby("ticker")["date"].agg(["min", "max"])
    covered_days = 0
    total_days = 0
    for r in universe.itertuples(index=False):
        days = (r.end_date - r.start_date).days + 1
        total_days += days
        if r.ticker in first_last.index:
            lo, hi = first_last.loc[r.ticker, "min"], first_last.loc[r.ticker, "max"]
            ov = (min(hi, r.end_date) - max(lo, r.start_date)).days + 1
            covered_days += max(ov, 0)
    left = universe.groupby("ticker")["end_date"].max() < universe["end_date"].max()
    left_tickers = set(left[left].index)
    return {
        "source": "see config prices.source",
        "n_tickers": int(universe["ticker"].nunique()),
        "tickers_with_prices": int(len(have & set(universe["ticker"]))),
        "coverage_tickers": float(len(have & set(universe["ticker"])) / max(universe["ticker"].nunique(), 1)),
        "coverage_member_days": float(covered_days / max(total_days, 1)),
        "n_tickers_left_index": int(len(left_tickers)),
        "coverage_tickers_left_index": float(len(have & left_tickers) / max(len(left_tickers), 1)),
    }


def load_prices(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
