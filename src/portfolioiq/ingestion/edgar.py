"""SEC EDGAR access: submissions, 10-K selection and download, XBRL shares outstanding.

All responses are cached on disk (gzip) so that re-running never downloads twice.
Requests go through :class:`~portfolioiq.utils.HttpClient` (<= 8 req/s, retries).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from portfolioiq.config import Config
from portfolioiq.utils import HttpClient, get_logger, read_gz, update_quality, write_gz, write_parquet

log = get_logger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{tax}/{tag}.json"

_FILING_COLS = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "primaryDocument"]


def _cached_json(client: HttpClient, url: str, path: Path) -> dict | None:
    if path.exists():
        return json.loads(read_gz(path))
    try:
        resp = client.get(url)
    except Exception as exc:  # noqa: BLE001  (404 = no data for this entity)
        log.warning("edgar json unavailable", url=url, error=str(exc)[:120])
        write_gz(path, b"null")
        return None
    write_gz(path, resp.content)
    return resp.json()


def submissions(cfg: Config, client: HttpClient, cik: int, need_before: pd.Timestamp | None = None) -> dict | None:
    """Return the submissions JSON for ``cik`` with older filing pages merged when needed."""
    base = cfg.raw_dir / "edgar" / "submissions"
    main = _cached_json(client, SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"), base / f"CIK{cik:010d}.json.gz")
    if not main:
        return None
    frames = [pd.DataFrame({c: main["filings"]["recent"].get(c, []) for c in _FILING_COLS})]
    for f in main["filings"].get("files", []):
        if need_before is not None and pd.Timestamp(f["filingTo"]) < need_before:
            continue  # page entirely before the window
        page = _cached_json(client, SUBMISSIONS_URL.format(name=f["name"]), base / f"{f['name']}.gz")
        if page:
            frames.append(pd.DataFrame({c: page.get(c, []) for c in _FILING_COLS}))
    main["_filings"] = pd.concat(frames, ignore_index=True)
    return main


def filings_frame(sub: dict, forms: list[str]) -> pd.DataFrame:
    """10-K-type filings from a submissions dict (amendments excluded)."""
    df = sub["_filings"].copy()
    df = df[df["form"].isin(forms)].copy()
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    df["acceptanceDateTime"] = pd.to_datetime(df["acceptanceDateTime"], utc=True).dt.tz_localize(None)
    df["reportDate"] = pd.to_datetime(df["reportDate"], errors="coerce")
    return df.drop_duplicates("accessionNumber").sort_values("acceptanceDateTime").reset_index(drop=True)


def has_10k_between(cfg: Config, client: HttpClient, cik: int, lo: pd.Timestamp, hi: pd.Timestamp) -> bool:
    """True if ``cik`` filed at least one 10-K between ``lo`` and ``hi`` (used to validate mappings)."""
    sub = submissions(cfg, client, int(cik), need_before=lo)
    if not sub:
        return False
    f = filings_frame(sub, cfg.edgar.forms)
    return bool(((f["filingDate"] >= lo) & (f["filingDate"] <= hi)).any())


def raw_filing_path(cfg: Config, cik: int, accession: str, doc: str) -> Path:
    ext = Path(doc).suffix.lower() or ".txt"
    return cfg.raw_dir / "edgar" / "filings" / f"{cik}" / f"{accession}{ext}.gz"


def download_filing(cfg: Config, client: HttpClient, cik: int, accession: str, doc: str) -> Path | None:
    """Download the primary document of a filing (idempotent)."""
    path = raw_filing_path(cfg, cik, accession, doc or f"{accession}.txt")
    if path.exists():
        return path
    acc = accession.replace("-", "")
    url = ARCHIVE_URL.format(cik=cik, acc=acc, doc=doc) if doc else ARCHIVE_URL.format(
        cik=cik, acc=acc, doc=f"{accession}.txt")
    try:
        resp = client.get(url)
    except Exception as exc:  # noqa: BLE001
        log.warning("filing download failed", cik=cik, accession=accession, error=str(exc)[:160])
        return None
    write_gz(path, resp.content)
    return path


def company_info(sub: dict, cik: int) -> dict:
    return {
        "cik": int(cik),
        "company_name": sub.get("name"),
        "sic": int(sub["sic"]) if str(sub.get("sic") or "").isdigit() else None,
        "sic_description": sub.get("sicDescription"),
        "edgar_tickers": ",".join(sub.get("tickers") or []),
    }


def ingest_filings(cfg: Config) -> pd.DataFrame:
    """Select and download all 10-K filings needed for the universe; write the filings index."""
    from portfolioiq.ingestion.parsing import parse_all

    universe = pd.read_parquet(cfg.processed_dir / "universe.parquet")
    client = HttpClient(cfg.sec_user_agent, rate=cfg.edgar.max_requests_per_second)
    fstart, fend = pd.Timestamp(cfg.edgar.filing_start), pd.Timestamp(cfg.edgar.filing_end)
    rows, companies = [], []
    for cik, grp in universe.dropna(subset=["cik"]).groupby("cik"):
        cik = int(cik)
        lo = max(fstart, grp["start_date"].min() - pd.Timedelta(days=cfg.edgar.lookback_days_before_entry))
        hi = min(fend, grp["end_date"].max())
        sub = submissions(cfg, client, cik, need_before=lo)
        if not sub:
            continue
        companies.append(company_info(sub, cik))
        f = filings_frame(sub, cfg.edgar.forms)
        f = f[(f["acceptanceDateTime"] >= lo) & (f["acceptanceDateTime"] <= hi + pd.Timedelta(days=1))]
        for r in f.itertuples(index=False):
            rows.append({
                "cik": cik, "accession": r.accessionNumber, "form": r.form,
                "filing_date": r.filingDate, "report_date": r.reportDate,
                "acceptance_datetime": r.acceptanceDateTime, "primary_document": r.primaryDocument,
            })
    index = pd.DataFrame(rows)
    log.info("filings selected", n=len(index), n_companies=index["cik"].nunique() if len(index) else 0)

    todo = [r for r in index.itertuples(index=False)
            if not raw_filing_path(cfg, r.cik, r.accession, r.primary_document or f"{r.accession}.txt").exists()]
    log.info("filings to download", n=len(todo), cached=len(index) - len(todo))
    done = 0
    with ThreadPoolExecutor(max_workers=cfg.edgar.download_workers) as pool:
        futs = [pool.submit(download_filing, cfg, client, r.cik, r.accession, r.primary_document) for r in todo]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 100 == 0:
                log.info("download progress", done=done, total=len(todo))
    index["raw_path"] = [
        str(raw_filing_path(cfg, r.cik, r.accession, r.primary_document or f"{r.accession}.txt"))
        for r in index.itertuples(index=False)
    ]
    index["downloaded"] = [Path(p).exists() for p in index["raw_path"]]
    write_parquet(pd.DataFrame(companies), cfg.processed_dir / "companies.parquet")

    parsed = parse_all(cfg, index)
    index = index.merge(parsed, on="accession", how="left")
    write_parquet(index.drop(columns=["raw_path"]), cfg.processed_dir / "filings_index.parquet")
    stats = {
        "n_filings": int(len(index)),
        "download_rate": float(index["downloaded"].mean()) if len(index) else 0.0,
        "item1a_extraction_rate": float(index["item1a_ok"].fillna(False).mean()) if len(index) else 0.0,
        "item7_extraction_rate": float(index["item7_ok"].fillna(False).mean()) if len(index) else 0.0,
        "both_items_rate": float((index["item1a_ok"].fillna(False) & index["item7_ok"].fillna(False)).mean())
        if len(index) else 0.0,
        "by_form": index["form"].value_counts().to_dict() if len(index) else {},
    }
    update_quality(cfg.reports_dir, cfg.name, "filings", stats)
    log.info("filings ingested", **stats)
    return index


# --------------------------------------------------------------------------- XBRL shares
def shares_outstanding(cfg: Config, client: HttpClient, cik: int) -> pd.DataFrame:
    """Point-in-time shares outstanding from XBRL (dei, fallback us-gaap), keyed by ``filed`` date."""
    out = []
    for tax, tag in (("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")):
        path = cfg.raw_dir / "edgar" / "xbrl" / f"{cik}_{tag}.json.gz"
        data = _cached_json(client, CONCEPT_URL.format(cik=cik, tax=tax, tag=tag), path)
        if not data:
            continue
        units = data.get("units", {}).get("shares", [])
        if not units:
            continue
        df = pd.DataFrame(units)
        df = df[df["form"].astype(str).str.startswith(("10-K", "10-Q"))]
        if df.empty:
            continue
        # multi-class issuers report one value per class within a filing: sum distinct values
        df = df.drop_duplicates(["accn", "end", "val"])
        g = df.groupby(["accn", "filed"], as_index=False)["val"].sum()
        g["cik"], g["source"] = cik, f"{tax}:{tag}"
        out.append(g.rename(columns={"filed": "filed_date", "val": "shares"}))
        break  # dei preferred; us-gaap only as fallback
    if not out:
        return pd.DataFrame(columns=["accn", "filed_date", "shares", "cik", "source"])
    res = pd.concat(out)
    res["filed_date"] = pd.to_datetime(res["filed_date"])
    return res


def ingest_shares(cfg: Config) -> pd.DataFrame:
    """Download XBRL shares outstanding for all mapped companies."""
    universe = pd.read_parquet(cfg.processed_dir / "universe.parquet")
    client = HttpClient(cfg.sec_user_agent, rate=cfg.edgar.max_requests_per_second)
    frames = [shares_outstanding(cfg, client, int(c)) for c in sorted(universe["cik"].dropna().unique())]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_parquet(df, cfg.processed_dir / "shares.parquet")
    n_ciks = universe["cik"].dropna().nunique()
    stats = {
        "n_companies": int(n_ciks),
        "companies_with_shares": int(df["cik"].nunique()) if len(df) else 0,
        "coverage": float(df["cik"].nunique() / n_ciks) if len(df) and n_ciks else 0.0,
        "by_source": df.groupby("source")["cik"].nunique().to_dict() if len(df) else {},
    }
    update_quality(cfg.reports_dir, cfg.name, "shares", stats)
    log.info("shares ingested", **stats)
    return df
