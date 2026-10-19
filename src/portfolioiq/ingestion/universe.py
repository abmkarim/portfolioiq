"""Point-in-time S&P 500 universe and ticker -> CIK mapping.

The universe at date *t* is the set of index members **at t** according to the daily
snapshots of the public ``fja05680/sp500`` dataset, including names that were later
removed, acquired or went bankrupt (no survivorship bias).
"""

from __future__ import annotations

import io
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from portfolioiq.config import Config
from portfolioiq.ingestion import edgar
from portfolioiq.utils import HttpClient, get_logger, retry_call, update_quality, write_parquet

log = get_logger(__name__)

WIKI_API = "https://en.wikipedia.org/w/index.php"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"


# --------------------------------------------------------------------------- intervals
def build_intervals(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Turn dated membership snapshots into ``(ticker, start_date, end_date)`` intervals.

    ``snapshots`` has columns ``date`` and ``tickers`` (comma separated). A ticker is a
    member from the first snapshot where it appears until the day before the first later
    snapshot where it is absent. ``end_date`` is NaT when still a member in the last snapshot.
    """
    snaps = snapshots.copy()
    snaps["date"] = pd.to_datetime(snaps["date"])
    snaps = snaps.sort_values("date").reset_index(drop=True)
    dates = snaps["date"].tolist()
    sets = [set(t.strip() for t in str(s).split(",") if t.strip()) for s in snaps["tickers"]]
    open_since: dict[str, pd.Timestamp] = {}
    rows = []
    for i, (d, members) in enumerate(zip(dates, sets, strict=True)):
        for t in members:
            open_since.setdefault(t, d)
        for t in list(open_since):
            if t not in members:
                rows.append((t, open_since.pop(t), d - pd.Timedelta(days=1)))
        if i == len(dates) - 1:
            for t, s in open_since.items():
                rows.append((t, s, pd.NaT))
    out = pd.DataFrame(rows, columns=["ticker", "start_date", "end_date"])
    return out.sort_values(["ticker", "start_date"]).reset_index(drop=True)


def members_on(intervals: pd.DataFrame, date: str | pd.Timestamp) -> set[str]:
    """Index members on ``date`` (point in time)."""
    d = pd.Timestamp(date)
    end = intervals["end_date"].fillna(pd.Timestamp.max)
    mask = (intervals["start_date"] <= d) & (end >= d)
    return set(intervals.loc[mask, "ticker"])


def clip_intervals(intervals: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Keep intervals overlapping ``[start, end]`` and clip them to it."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    iv = intervals.copy()
    iv["end_date"] = iv["end_date"].fillna(e).clip(upper=e)
    iv["start_date"] = iv["start_date"].clip(lower=s)
    return iv[iv["start_date"] <= iv["end_date"]].reset_index(drop=True)


def select_sample(intervals: pd.DataFrame, n: int, anchor: str, oos_share: float, seed: int) -> list[str]:
    """Deterministic sample: ``oos_share`` of names among members at ``anchor``, rest among the others."""
    rng = np.random.default_rng(seed)
    at_anchor = sorted(members_on(intervals, anchor))
    others = sorted(set(intervals["ticker"]) - set(at_anchor))
    n_a = min(len(at_anchor), int(round(n * oos_share)))
    n_o = min(len(others), n - n_a)
    pick = list(rng.choice(at_anchor, n_a, replace=False)) + list(rng.choice(others, n_o, replace=False))
    return sorted(pick)


# --------------------------------------------------------------------------- names
_SUFFIX = re.compile(
    r"\b(the|inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|lp|sa|nv|ag|"
    r"class [a-z]|cl [a-z]|new|del|de)\b"
)


def normalize_name(name: str) -> str:
    """Normalise a company name for fuzzy matching."""
    s = str(name).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = _SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


# --------------------------------------------------------------------------- downloads
def _download_components(cfg: Config, client: HttpClient) -> pd.DataFrame:
    path = cfg.raw_dir / "universe" / "sp500_components.csv"
    if not path.exists():
        resp = client.get(cfg.universe.components_url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    return pd.read_csv(path)


def _download_wiki(cfg: Config, client: HttpClient) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = cfg.raw_dir / "universe" / f"wiki_sp500_{cfg.universe.wiki_revid}.html"
    if not path.exists():
        resp = client.get(WIKI_API, params={"oldid": cfg.universe.wiki_revid})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    tables = pd.read_html(io.StringIO(path.read_text(encoding="utf-8")))
    current = tables[0].rename(columns={"Symbol": "ticker", "Security": "name", "CIK": "cik"})
    changes = tables[1]
    changes.columns = ["date", "added_ticker", "added_name", "removed_ticker", "removed_name", "reason"]
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
    return current[["ticker", "name", "cik"]], changes


def _download_sec_tickers(cfg: Config, client: HttpClient) -> pd.DataFrame:
    path = cfg.raw_dir / "universe" / "company_tickers.json"
    if not path.exists():
        resp = client.get(SEC_TICKERS_URL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    df = pd.read_json(path, orient="index")
    df = df.rename(columns={"cik_str": "cik", "title": "name"})
    df["ticker"] = df["ticker"].str.upper()
    return df[["ticker", "cik", "name"]]


def _efts_candidates(client: HttpClient, name: str) -> list[tuple[int, str]]:
    try:
        resp = client.get(EFTS_URL, params={"keysTyped": normalize_name(name)[:60]})
        hits = resp.json().get("hits", {}).get("hits", [])
        return [(int(h["_id"]), h["_source"].get("entity", "")) for h in hits]
    except Exception as exc:  # noqa: BLE001
        log.warning("efts search failed", name=name, error=str(exc)[:200])
        return []


def classify_removal(reason: str) -> str:
    """Map a free-text removal reason to ``acquisition`` / ``bankruptcy`` / ``unknown``."""
    r = str(reason).lower()
    if re.search(r"acqui|merg|buyout|bought|purchas|taken private|go private|spun|spin|combin", r):
        return "acquisition"
    if re.search(r"bankrupt|chapter 11|delist", r):
        return "bankruptcy"
    return "unknown"


# --------------------------------------------------------------------------- main step
def ingest_universe(cfg: Config) -> pd.DataFrame:
    """Build the point-in-time universe with CIK mapping. Writes ``universe.parquet``."""
    client = HttpClient(cfg.sec_user_agent, rate=cfg.edgar.max_requests_per_second)
    comps = retry_call(_download_components, cfg, client, what="components")
    intervals = clip_intervals(build_intervals(comps), cfg.universe.start_date, cfg.universe.end_date)
    log.info("universe intervals", n_intervals=len(intervals), n_tickers=intervals["ticker"].nunique())

    wiki_current, changes = retry_call(_download_wiki, cfg, client, what="wiki")
    sec_tickers = retry_call(_download_sec_tickers, cfg, client, what="sec tickers")

    tickers = sorted(intervals["ticker"].unique())
    if cfg.universe.sample_size:
        tickers = select_sample(
            intervals, cfg.universe.sample_size, cfg.universe.sample_anchor_date,
            cfg.universe.sample_oos_share, cfg.seed,
        )
        intervals = intervals[intervals["ticker"].isin(tickers)].reset_index(drop=True)
        log.info("sample selected", n=len(tickers))

    # names for removed tickers (most recent removal first)
    removed = changes.dropna(subset=["removed_ticker"]).sort_values("date", ascending=False)
    added = changes.dropna(subset=["added_ticker"]).sort_values("date", ascending=False)
    wiki_cik = {str(t).replace("-", "."): int(c) for t, c in zip(wiki_current["ticker"], wiki_current["cik"],
                                                                 strict=True)}
    wiki_name = dict(zip(wiki_current["ticker"].astype(str).str.replace("-", "."), wiki_current["name"],
                         strict=True))
    sec_cik = {t.replace("-", "."): int(c) for t, c in zip(sec_tickers["ticker"], sec_tickers["cik"], strict=True)}
    sec_name = {t.replace("-", "."): n for t, n in zip(sec_tickers["ticker"], sec_tickers["name"], strict=True)}

    rows = []
    for t in tickers:
        iv = intervals[intervals["ticker"] == t]
        lo, hi = iv["start_date"].min(), iv["end_date"].max()
        rem = removed[removed["removed_ticker"] == t]
        rem = rem[(rem["date"] >= lo - pd.Timedelta(days=30)) & (rem["date"] <= hi + pd.Timedelta(days=60))]
        add = added[added["added_ticker"] == t]
        name = (rem["removed_name"].iloc[0] if len(rem) else None) or (
            add["added_name"].iloc[0] if len(add) else None) or wiki_name.get(t) or sec_name.get(t)
        candidates: list[tuple[int, str]] = []
        if t in wiki_cik:
            candidates.append((wiki_cik[t], "wikipedia_constituents"))
        if t in sec_cik:
            candidates.append((sec_cik[t], "sec_company_tickers"))
        cik, source, cname = None, None, None
        for c, src in candidates:
            if edgar.has_10k_between(cfg, client, c, lo - pd.Timedelta(days=400), hi + pd.Timedelta(days=400)):
                cik, source = c, src
                break
        if cik is None and name:
            best = None
            for c, ent in _efts_candidates(client, name)[:5]:
                sim = name_similarity(name, ent)
                if sim >= cfg.universe.name_match_threshold and (best is None or sim > best[2]):
                    best = (c, ent, sim)
            if best and edgar.has_10k_between(cfg, client, best[0], lo - pd.Timedelta(days=400),
                                              hi + pd.Timedelta(days=400)):
                cik, source, cname = best[0], "edgar_name_search", best[1]
        reason = rem["reason"].iloc[0] if len(rem) else None
        rows.append({
            "ticker": t, "cik": cik, "map_source": source or "unmapped",
            "name": name or cname, "removal_reason": reason,
            "removal_date": rem["date"].iloc[0] if len(rem) else pd.NaT,
            "removal_cause": classify_removal(reason) if reason else "unknown",
        })
    mapping = pd.DataFrame(rows)
    universe = intervals.merge(mapping, on="ticker", how="left")
    universe["cik"] = universe["cik"].astype("Int64")
    write_parquet(universe, cfg.processed_dir / "universe.parquet")

    # mapping rate, weighted by membership days as well
    days = (universe["end_date"] - universe["start_date"]).dt.days + 1
    stats = {
        "n_tickers": int(mapping["ticker"].nunique()),
        "n_mapped": int(mapping["cik"].notna().sum()),
        "mapping_rate_tickers": float(mapping["cik"].notna().mean()),
        "mapping_rate_member_days": float(days[universe["cik"].notna()].sum() / days.sum()),
        "by_source": mapping["map_source"].value_counts().to_dict(),
        "n_currently_absent": int(mapping["ticker"].isin(set(intervals.loc[intervals["end_date"] <
                                                                              pd.Timestamp(cfg.universe.end_date),
                                                                              "ticker"])).sum()),
    }
    update_quality(cfg.reports_dir, cfg.name, "universe", stats)
    log.info("universe mapped", **stats)
    return universe
