"""SIC code -> Fama-French 12 industries (a proxy for GICS sectors)."""

from __future__ import annotations

import pandas as pd

from portfolioiq.config import Config
from portfolioiq.utils import get_logger, write_parquet

log = get_logger(__name__)

FF12_NAMES = {
    1: "NoDur", 2: "Durbl", 3: "Manuf", 4: "Enrgy", 5: "Chems", 6: "BusEq",
    7: "Telcm", 8: "Utils", 9: "Shops", 10: "Hlth", 11: "Money", 12: "Other",
}

# Ranges from Kenneth French's Siccodes12 definition.
FF12_RANGES: dict[int, list[tuple[int, int]]] = {
    1: [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
    2: [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716),
        (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
    3: [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
        (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
        (3830, 3839), (3860, 3899)],
    4: [(1200, 1399), (2900, 2999)],
    5: [(2800, 2829), (2840, 2899)],
    6: [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
    7: [(4800, 4899)],
    8: [(4900, 4949)],
    9: [(5000, 5999), (7200, 7299), (7600, 7699)],
    10: [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
    11: [(6000, 6999)],
}


def sic_to_ff12(sic: int | float | None) -> int:
    """Map a 4-digit SIC code to its Fama-French 12 industry number (12 = Other / unknown)."""
    if sic is None or pd.isna(sic):
        return 12
    s = int(sic)
    for ind, ranges in FF12_RANGES.items():
        if any(lo <= s <= hi for lo, hi in ranges):
            return ind
    return 12


def build_sectors(cfg: Config) -> pd.DataFrame:
    """Write ``sectors.parquet`` (cik, sic, ff12, ff12_name) from EDGAR submissions data."""
    comp = pd.read_parquet(cfg.processed_dir / "companies.parquet")
    comp["ff12"] = comp["sic"].map(sic_to_ff12)
    comp["ff12_name"] = comp["ff12"].map(FF12_NAMES)
    write_parquet(comp, cfg.processed_dir / "sectors.parquet")
    log.info("sectors", counts=comp["ff12_name"].value_counts().to_dict())
    return comp
