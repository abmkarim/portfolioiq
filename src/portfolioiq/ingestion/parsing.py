"""10-K parsing: HTML/TXT -> clean text -> Item 1A (Risk Factors) and Item 7 (MD&A)."""

from __future__ import annotations

import html as htmlmod
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from portfolioiq.config import Config
from portfolioiq.utils import get_logger, read_gz, write_parquet

log = get_logger(__name__)

_BLOCK_TAGS = ("p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "td")

# Section headers. Anchored versions require the header to start a line (avoids TOC cross references).
_ITEM_PATTERNS = {
    "item1a": (
        r"item\s*1a\s*[\.\:\-–—]?\s*risk\s+factors",
        r"item\s*(?:1b\s*[\.\:\-–—]?\s*unresolved|2\s*[\.\:\-–—]?\s*(?:description\s+of\s+)?properties)",
    ),
    "item7": (
        r"item\s*7\s*[\.\:\-–—]?\s*management.{0,3}s?\s+discussion",
        r"item\s*(?:7a\s*[\.\:\-–—]?\s*quantitative|8\s*[\.\:\-–—]?\s*(?:consolidated\s+)?financial\s+statements)",
    ),
}


def html_to_text(raw: bytes) -> str:
    """Convert an EDGAR HTML (or SGML/TXT) document to plain text with line breaks."""
    head = raw[:2000].lower()
    is_html = b"<html" in head or b"<div" in raw[:20000].lower() or b"<p" in raw[:20000].lower()
    if is_html:
        try:
            import lxml.html

            doc = lxml.html.fromstring(raw)
            for bad in doc.xpath("//script|//style|//*[name()='ix:header']"):
                bad.drop_tree()
            for el in doc.iter(*_BLOCK_TAGS):
                el.tail = "\n" + (el.tail or "")
            text = doc.text_content()
        except Exception:  # noqa: BLE001
            text = re.sub(r"<[^>]+>", " ", raw.decode("utf-8", errors="ignore"))
    else:
        text = re.sub(r"<[^>]+>", " ", raw.decode("utf-8", errors="ignore"))
    text = htmlmod.unescape(text).replace("\xa0", " ").replace("​", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text


def extract_section(text: str, start_pat: str, end_pat: str, min_chars: int = 2000) -> str:
    """Extract the longest ``start .. end`` span; line-anchored headers first, then any position."""
    for anchored in (True, False):
        pre = r"(?:^|\n)\s*" if anchored else ""
        starts = [m.start() for m in re.finditer(pre + start_pat, text, flags=re.I)]
        ends = [m.start() for m in re.finditer(pre + end_pat, text, flags=re.I)]
        best = ""
        for s in starts:
            nxt = [e for e in ends if e > s]
            if not nxt:
                continue
            chunk = text[s:nxt[0]]
            if len(chunk) > len(best):
                best = chunk
        if len(best) >= min_chars:
            return best.strip()
    return ""


def parse_document(raw: bytes, min_chars: int = 2000) -> dict[str, str]:
    """Return ``{"item1a": ..., "item7": ...}`` (empty strings when not found)."""
    text = html_to_text(raw)
    return {k: extract_section(text, s, e, min_chars) for k, (s, e) in _ITEM_PATTERNS.items()}


def _parse_cik(args: tuple[int, list[tuple[str, str]], str, int]) -> list[dict]:
    cik, items, out_path, min_chars = args
    out = Path(out_path)
    if out.exists():
        cached = pd.read_parquet(out)
        if set(a for a, _ in items) <= set(cached["accession"]):
            return cached.drop(columns=["item1a", "item7"]).to_dict("records")
    rows = []
    for acc, raw_path in items:
        p = Path(raw_path)
        if not p.exists():
            rows.append({"accession": acc, "item1a": "", "item7": ""})
            continue
        try:
            secs = parse_document(read_gz(p), min_chars)
        except Exception:  # noqa: BLE001
            secs = {"item1a": "", "item7": ""}
        rows.append({"accession": acc, **secs})
    df = pd.DataFrame(rows)
    df["cik"] = cik
    df["item1a_chars"] = df["item1a"].str.len()
    df["item7_chars"] = df["item7"].str.len()
    df["item1a_ok"] = df["item1a_chars"] > 0
    df["item7_ok"] = df["item7_chars"] > 0
    write_parquet(df, out)
    return df.drop(columns=["item1a", "item7"]).to_dict("records")


def text_path(cfg: Config, cik: int) -> Path:
    return cfg.interim_dir / "text" / f"{int(cik)}.parquet"


def parse_all(cfg: Config, index: pd.DataFrame, workers: int = 4) -> pd.DataFrame:
    """Parse every downloaded filing (cached per CIK in ``data/interim/text``)."""
    jobs = [
        (int(cik), list(zip(g["accession"], g["raw_path"], strict=True)), str(text_path(cfg, cik)),
         cfg.edgar.min_section_chars)
        for cik, g in index.groupby("cik")
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(_parse_cik, jobs)):
            rows.extend(res)
            if (i + 1) % 50 == 0:
                log.info("parse progress", companies=i + 1, total=len(jobs))
    df = pd.DataFrame(rows)
    return df[["accession", "item1a_ok", "item7_ok", "item1a_chars", "item7_chars"]]


def load_texts(cfg: Config, ciks: list[int] | None = None) -> pd.DataFrame:
    """Load cleaned texts (accession, cik, item1a, item7)."""
    files = sorted((cfg.interim_dir / "text").glob("*.parquet"))
    if ciks is not None:
        wanted = {int(c) for c in ciks}
        files = [f for f in files if int(f.stem) in wanted]
    cols = ["accession", "cik", "item1a", "item7"]
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True) if files else \
        pd.DataFrame(columns=cols)
