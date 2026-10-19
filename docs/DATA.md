# Données

## Sources

| Donnée | Source | Accès | Cache |
|---|---|---|---|
| Composition historique du S&P 500 | GitHub `fja05680/sp500`, « S&P 500 Historical Components & Changes (Updated).csv » | HTTP, retry exponentiel | `data/raw/universe/` |
| Noms et raisons de sortie | Wikipédia *List of S&P 500 companies*, révision figée oldid 1292523673 | HTTP | `data/raw/universe/` |
| Ticker → CIK | `sec.gov/files/company_tickers.json`, table Wikipédia, recherche d'entité EDGAR | HTTP, ≤ 8 req/s | `data/raw/universe/`, `data/raw/edgar/submissions/` |
| 10-K / 10-K405 | `data.sec.gov/submissions` + archives EDGAR (document principal) | `User-Agent` = `SEC_USER_AGENT`, ≤ 8 req/s, retry | `data/raw/edgar/filings/<cik>/<accession>.htm.gz` |
| Texte nettoyé (Item 1A, Item 7) | parsing local | — | `data/interim/text/<cik>.parquet` |
| Prix journaliers | `yfinance` (option `tiingo`, `crsp_csv`) | retry exponentiel | `data/raw/prices/<source>/<ticker>.parquet` |
| Actions en circulation | XBRL `companyconcept` `dei:EntityCommonStockSharesOutstanding` (repli `us-gaap:CommonStockSharesOutstanding`) | ≤ 8 req/s | `data/raw/edgar/xbrl/` |
| Secteurs | SIC des submissions EDGAR → 12 industries Fama-French | — | `data/processed/<run>/sectors.parquet` |
| Facteurs | Kenneth R. French Data Library : FF5 (2×3) et Momentum, mensuels et journaliers | retry exponentiel | `data/raw/factors/` |

Tous les téléchargements sont idempotents : relancer une étape ne retélécharge rien de ce qui est déjà en cache (y compris les absences, mémorisées pour les prix).

## Qualité des données

<!-- data:start -->
Indicateurs calculés à chaque run et enregistrés dans `reports/metrics.json` (section `data_quality`) : nombre de tickers de l'univers historique traités, taux de mapping ticker → CIK (brut et pondéré par jours d'appartenance, par source), nombre de 10-K sélectionnés, taux de téléchargement, taux d'extraction des Items 1A et 7, couverture des prix (tickers, jours d'appartenance, titres sortis de l'indice) et part des sociétés avec actions en circulation XBRL.
<!-- data:end -->

## Remarques

- **Couverture des prix** : `yfinance` ne fournit en général plus d'historique pour les titres délistés (rachats, faillites) ni pour les tickers réattribués. C'est la principale limite du projet : une partie des titres sortis de l'indice n'a pas de prix et ne peut donc pas être investie, ce qui réintroduit un biais du survivant partiel. Les sources `tiingo` (clé API) ou `crsp_csv` (export WRDS) sont prévues dans `configs/*.yaml` (`prices.source`) pour corriger ce point.
- **Mapping CIK** : chaque CIK candidat est validé par la présence d'un 10-K pendant la période d'appartenance à l'indice, pour éviter les tickers réutilisés par une autre société.
- **Extraction** : les échecs viennent surtout des 10-K qui renvoient le MD&A vers l'exhibit 13 (rapport annuel) ou de mises en forme atypiques des en-têtes.
- **Secteurs** : les 12 industries Fama-French (à partir du code SIC courant de la société) sont un *proxy* de GICS ; le SIC n'est pas historisé.
- **Actions en circulation** : la donnée XBRL n'existe qu'à partir de 2009-2011 selon la taille de l'émetteur ; avant, la log-capitalisation est imputée à la médiane transversale.
