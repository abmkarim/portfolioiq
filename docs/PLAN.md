# Plan d'exécution — PortfolioIQ

Légende : [x] fait, [ ] à faire, [~] partiel / limite documentée.

| # | Étape | Critère de validation | État |
|---|---|---|---|
| 1 | Squelette du dépôt : `.gitignore`, `pyproject.toml`, `tasks.ps1`, docs | `git status` ne montre ni `.env` ni `data/` | [x] |
| 2 | `.\tasks.ps1 setup` (venv `.venv`, Python 3.11, torch CPU) | `import portfolioiq` OK | [x] |
| 3 | Config YAML + env (`config.py`), logs structurés, seeds | tests config | [x] |
| 4 | Ingestion : univers point-in-time, mapping CIK, 10-K EDGAR (cache, 8 req/s), parsing Item 1A / 7, prix yfinance, facteurs FF5+Mom, secteurs SIC→FF12, actions en circulation XBRL | taux de mapping / extraction / couverture écrits dans `reports/data_quality.json` puis `docs/DATA.md` | [x] |
| 5 | dbt-duckdb : staging → intermediate → marts + tests (dont point-in-time et appartenance à l'indice) | `dbt build` vert | [x] |
| 6 | NLP : FinBERT (phrases, plafond, cache), TF-IDF / Word2Vec par fold, deltas YoY, contrôles, rangs | tests anti-fuite Word2Vec | [x] |
| 7 | Modèle : walk-forward 2018–2024, embargo 1 mois, XGBoost (HP par validation interne), MLflow, SHAP | runs MLflow parent + folds | [x] |
| 8 | Portefeuille : Ledoit-Wolf 252 j, CVXPY (dollar-neutre, secteur-neutre, levier ≤ 2, \|w\| ≤ 2 %), coûts, backtest journalier, benchmarks | tests contraintes optimiseur et coûts | [x] |
| 9 | Évaluation : métriques, IC, FF5+Mom Newey-West, robustesse (5/20 bps, sous-périodes, ablations) | `reports/metrics.json` + figures | [x] |
| 10 | Airflow : DAG + `docker-compose.yml` | `airflow dags list-import-errors` vide dans le conteneur officiel 2.10.5, arbre des 11 tâches conforme | [x] |
| 11 | Power BI : export CSV + `powerbi/README.md` | CSV dans `powerbi/data/` | [x] |
| 12 | Tests pytest hors réseau + ruff | `.\tasks.ps1 test` vert | [x] |
| 13 | `.\tasks.ps1 sample` bout en bout (< 30 min CPU, hors premier téléchargement) | `reports/metrics.json` produit | [x] |
| 14 | Run `full` | reporté ; limite documentée dans le README | [~] |
| 15 | README (métriques lues dans `metrics.json`), scan de secrets, push `origin main` | `git status` propre | [x] |
