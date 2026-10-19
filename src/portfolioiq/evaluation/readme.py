"""Render README / DATA.md tables straight from ``reports/metrics.json`` (no hand-typed numbers)."""

from __future__ import annotations

import re
from pathlib import Path

from portfolioiq.config import PROJECT_ROOT
from portfolioiq.utils import read_json


def _pct(x, d: int = 1) -> str:
    return "n/d" if x is None else f"{x * 100:.{d}f} %"


def _num(x, d: int = 2) -> str:
    return "n/d" if x is None else f"{x:.{d}f}"


def results_markdown(m: dict) -> str:
    """Main results table + robustness tables."""
    s, b = m.get("strategy", {}), m.get("benchmarks", {})
    cols = [("Modèle NLP", s), ("Signal aléatoire", b.get("random_signal", {})),
            ("Momentum 12-1", b.get("momentum_12_1", {}))]
    rows = [
        ("Rendement annualisé (net)", "annualized_return", _pct),
        ("Volatilité annualisée", "annualized_vol", _pct),
        ("Sharpe (net)", "sharpe", _num),
        ("Sortino", "sortino", _num),
        ("Max drawdown", "max_drawdown", _pct),
        ("Taux de mois positifs", "hit_rate", _pct),
        ("Turnover mensuel moyen (Σ|Δw|)", "avg_monthly_turnover", _pct),
        ("Exposition brute moyenne", "avg_gross_exposure", _num),
        ("Sharpe brut (avant coûts)", "sharpe_gross", _num),
    ]
    run = m.get("run", {})
    out = [f"*Run `{run.get('config')}` — période hors échantillon {run.get('oos_period', ['?', '?'])[0]} → "
           f"{run.get('oos_period', ['?', '?'])[1]}, nets de {run.get('tc_bps_per_side')} bps/côté et "
           f"{run.get('borrow_bps_annual')} bps/an d'emprunt. Généré le {run.get('generated_at')}.*", "",
           "| Métrique | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---:|" * len(cols)]
    for label, key, fmt in rows:
        out.append(f"| {label} | " + " | ".join(fmt(c.get(key)) for _, c in cols) + " |")
    ff = [c.get("fama_french", {}) for _, c in cols]
    out.append("| Alpha FF5+Mom annualisé | " + " | ".join(_pct(f.get("alpha_annualized"), 2) for f in ff) + " |")
    out.append("| t-stat Newey-West (6 lags) | " + " | ".join(_num(f.get("alpha_tstat_nw")) for f in ff) + " |")
    out.append("| R² FF5+Mom | " + " | ".join(_num(f.get("r_squared")) for f in ff) + " |")
    out.append("| Nombre de mois | " + " | ".join(str(c.get("n_months", "n/d")) for _, c in cols) + " |")
    ic = m.get("ic", {})
    out += ["", f"**IC de Spearman mensuel** (prédiction vs rendement du mois suivant) : moyenne "
                f"{_num(ic.get('mean'), 4)}, t-stat {_num(ic.get('tstat'))}, {ic.get('n_months')} mois, "
                f"{_pct(ic.get('share_positive'), 0)} de mois positifs.", ""]
    by = ic.get("by_holding_year", {})
    if by:
        out += ["| Année | " + " | ".join(by) + " |", "|---|" + "---:|" * len(by),
                "| IC moyen | " + " | ".join(_num(v, 3) for v in by.values()) + " |", ""]
    rob = m.get("robustness", {})
    out += ["**Robustesse**", "", "| Variante | Rendement ann. | Sharpe | Alpha FF ann. | t NW | IC moyen |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Référence (10 bps) | {_pct(s.get('annualized_return'))} | {_num(s.get('sharpe'))} | "
            f"{_pct(s.get('fama_french', {}).get('alpha_annualized'), 2)} | "
            f"{_num(s.get('fama_french', {}).get('alpha_tstat_nw'))} | {_num(ic.get('mean'), 4)} |"]
    names = {"5bps": "Coûts 5 bps", "20bps": "Coûts 20 bps", "no_word2vec": "Sans Word2Vec",
             "no_finbert": "Sans FinBERT"}
    for group in ("costs", "subperiods", "ablations"):
        for k, v in rob.get(group, {}).items():
            if not v:
                continue
            label = names.get(k, "Sous-période " + k.replace("_", " → "))
            out.append(f"| {label} | {_pct(v.get('annualized_return'))} | {_num(v.get('sharpe'))} | "
                       f"{_pct(v.get('fama_french', {}).get('alpha_annualized'), 2)} | "
                       f"{_num(v.get('fama_french', {}).get('alpha_tstat_nw'))} | "
                       f"{_num(v.get('ic_mean'), 4) if 'ic_mean' in v else '—'} |")
    return "\n".join(out)


def data_markdown(m: dict) -> str:
    q = m.get("data_quality", {})
    u, f, p, sh = q.get("universe", {}), q.get("filings", {}), q.get("prices", {}), q.get("shares", {})
    src = u.get("by_source", {})
    lines = [
        f"*Statistiques du run `{m.get('run', {}).get('config')}` (lues dans `reports/metrics.json`).*", "",
        "| Indicateur | Valeur |", "|---|---:|",
        f"| Tickers de l'univers historique traités | {u.get('n_tickers')} |",
        f"| Taux de mapping ticker → CIK (tickers) | {_pct(u.get('mapping_rate_tickers'))} |",
        f"| Taux de mapping pondéré par jours d'appartenance | {_pct(u.get('mapping_rate_member_days'))} |",
        f"| … via table Wikipédia / `company_tickers.json` / recherche EDGAR par nom | "
        f"{src.get('wikipedia_constituents', 0)} / {src.get('sec_company_tickers', 0)} / "
        f"{src.get('edgar_name_search', 0)} (non mappés : {src.get('unmapped', 0)}) |",
        f"| 10-K sélectionnés | {f.get('n_filings')} |",
        f"| Taux de téléchargement | {_pct(f.get('download_rate'))} |",
        f"| Extraction Item 1A réussie | {_pct(f.get('item1a_extraction_rate'))} |",
        f"| Extraction Item 7 réussie | {_pct(f.get('item7_extraction_rate'))} |",
        f"| Extraction des deux items | {_pct(f.get('both_items_rate'))} |",
        f"| Couverture prix — tickers avec historique ({m.get('run', {}).get('price_source')}) | "
        f"{_pct(p.get('coverage_tickers'))} ({p.get('tickers_with_prices')}/{p.get('n_tickers')}) |",
        f"| Couverture prix — jours d'appartenance couverts | {_pct(p.get('coverage_member_days'))} |",
        f"| Couverture prix — titres sortis de l'indice avant fin 2024 | "
        f"{_pct(p.get('coverage_tickers_left_index'))} (sur {p.get('n_tickers_left_index')}) |",
        f"| Sociétés avec actions en circulation XBRL | {_pct(sh.get('coverage'))} |",
    ]
    return "\n".join(lines)


def inject(path: Path, marker: str, content: str) -> bool:
    """Replace the text between ``<!-- marker:start -->`` and ``<!-- marker:end -->``."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    pat = re.compile(rf"<!-- {marker}:start -->.*?<!-- {marker}:end -->", re.S)
    if not pat.search(text):
        return False
    block = f"<!-- {marker}:start -->\n{content}\n<!-- {marker}:end -->"
    path.write_text(pat.sub(lambda _: block, text), encoding="utf-8")
    return True


def update_docs(metrics_path: Path | None = None) -> None:
    """Refresh the generated blocks of README.md and docs/DATA.md."""
    m = read_json(metrics_path or PROJECT_ROOT / "reports" / "metrics.json", {})
    if not m:
        return
    inject(PROJECT_ROOT / "README.md", "metrics", results_markdown(m))
    inject(PROJECT_ROOT / "README.md", "data", data_markdown(m))
    inject(PROJECT_ROOT / "docs" / "DATA.md", "data", data_markdown(m))
