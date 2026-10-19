"""Shared helpers: structured logging, rate-limited HTTP with retries, IO."""

from __future__ import annotations

import gzip
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "ctx", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure JSON-structured logging on stderr (and optionally a file)."""
    root = logging.getLogger()
    root.handlers.clear()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(_JsonFormatter())
        root.addHandler(h)
    root.setLevel(level)
    for noisy in ("urllib3", "yfinance", "peewee", "matplotlib", "gensim", "httpx", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.LoggerAdapter:
    """Return a logger whose ``extra`` kwargs are emitted as JSON fields."""

    class _Adapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            ctx = {k: kwargs.pop(k) for k in list(kwargs) if k not in ("exc_info", "stack_info", "extra")}
            kwargs["extra"] = {"ctx": ctx}
            return msg, kwargs

    return _Adapter(logging.getLogger(name), {})


log = get_logger(__name__)


class RateLimiter:
    """Thread-safe limiter guaranteeing at most ``rate`` calls per second."""

    def __init__(self, rate: float) -> None:
        self.min_interval = 1.0 / rate
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.min_interval
        if wait > 0:
            time.sleep(wait)


class HttpClient:
    """requests wrapper with rate limiting and exponential-backoff retries."""

    def __init__(
        self,
        user_agent: str,
        rate: float = 8.0,
        max_retries: int = 6,
        backoff: float = 1.0,
        timeout: float = 60.0,
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.limiter = RateLimiter(rate)
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        """GET with retries on network errors, 429 and 5xx. Raises on final failure."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                resp = self.session.get(url, timeout=self.timeout, **kwargs)
                if resp.status_code == 404:
                    resp.raise_for_status()
                if resp.status_code in (403, 429) or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    raise
                last_exc = exc
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
            sleep = self.backoff * (2**attempt)
            log.warning("http retry", url=url, attempt=attempt + 1, sleep=sleep, error=str(last_exc))
            time.sleep(sleep)
        raise RuntimeError(f"GET failed after {self.max_retries} attempts: {url}") from last_exc


def retry_call(fn, *args, retries: int = 5, backoff: float = 2.0, what: str = "call", **kwargs):
    """Call ``fn`` with exponential backoff on any exception."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            sleep = backoff * (2**attempt)
            log.warning("retry", what=what, attempt=attempt + 1, sleep=sleep, error=str(exc)[:200])
            time.sleep(sleep)
    return None


def write_gz(path: Path, data: bytes) -> None:
    """Atomically write gzip-compressed bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        fh.write(data)
    tmp.replace(path)


def read_gz(path: Path) -> bytes:
    with gzip.open(path, "rb") as fh:
        return fh.read()


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomically write a parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str, ensure_ascii=False)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def update_quality(cfg_reports_dir: Path, run_name: str, section: str, values: dict) -> None:
    """Merge data-quality statistics into ``reports/data_quality_<run>.json``."""
    path = cfg_reports_dir / f"data_quality_{run_name}.json"
    data = read_json(path, {}) or {}
    data[section] = values
    write_json(data, path)
