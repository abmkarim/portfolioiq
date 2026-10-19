"""Fama-French 5 factors (2x3) + Momentum from the Kenneth R. French Data Library."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from portfolioiq.config import Config
from portfolioiq.utils import get_logger, retry_call, write_parquet

log = get_logger(__name__)

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    ("ff5", "monthly"): "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    ("ff5", "daily"): "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    ("mom", "monthly"): "F-F_Momentum_Factor_CSV.zip",
    ("mom", "daily"): "F-F_Momentum_Factor_daily_CSV.zip",
}


def parse_french_csv(text: str, freq: str) -> pd.DataFrame:
    """Parse the first data block of a French-library CSV (values in %, converted to decimals)."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith(","))
    header = [c.strip() for c in lines[start].split(",")]
    rows = []
    for ln in lines[start + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if not parts or not parts[0].isdigit():
            break
        if freq == "monthly" and len(parts[0]) != 6:
            break
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date"] + header[1:])
    fmt = "%Y%m" if freq == "monthly" else "%Y%m%d"
    df["date"] = pd.to_datetime(df["date"], format=fmt)
    if freq == "monthly":
        df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    df.columns = ["date"] + [c.strip().replace("-", "_").lower() for c in df.columns[1:]]
    return df


def _fetch(name: str) -> bytes:
    resp = requests.get(BASE + name, timeout=60, headers={"User-Agent": "Mozilla/5.0 portfolioiq"})
    resp.raise_for_status()
    return resp.content


def ingest_factors(cfg: Config) -> dict[str, pd.DataFrame]:
    """Download (cached) and write ``factors_monthly.parquet`` / ``factors_daily.parquet``."""
    cache = cfg.raw_dir / "factors"
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for freq in ("monthly", "daily"):
        frames = []
        for kind in ("ff5", "mom"):
            name = FILES[(kind, freq)]
            path = cache / name
            if not path.exists():
                path.write_bytes(retry_call(_fetch, name, what=f"french {name}"))
            with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as z:
                text = z.read(z.namelist()[0]).decode("latin-1")
            frames.append(parse_french_csv(text, freq).set_index("date"))
        df = frames[0].join(frames[1], how="inner").reset_index()
        write_parquet(df, cfg.interim_dir / f"factors_{freq}.parquet")
        out[freq] = df
        log.info("factors", freq=freq, n=len(df), start=str(df["date"].min().date()), end=str(df["date"].max().date()))
    return out
