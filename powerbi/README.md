# Tableau de bord Power BI — guide

L'étape `export_powerbi` (dernière tâche du DAG, incluse dans `.\tasks.ps1 sample|full`) écrit les marts dbt en CSV dans `powerbi/data/` (dossier ignoré par git) :

| Fichier | Grain | Colonnes principales |
|---|---|---|
| `dim_company.csv` | 1 ligne / CIK | `cik`, `company_name`, `sic`, `ff12`, `ff12_name`, `tickers`, `first_member_date`, `last_member_date` |
| `fct_monthly_panel.csv` | 1 ligne / (fin de mois, ticker) membre de l'indice à cette date | `month_end`, `ticker`, `cik`, `ff12_name`, `ret_1m`, `ret_fwd_1m`, `mom_12_1`, `vol_3m`, `log_mcap`, `accession`, `feature_available_date` |
| `fct_backtest_returns.csv` | 1 ligne / (stratégie, mois) | `month_end`, `strategy`, `ret_net`, `ret_gross`, `turnover`, `cost`, `gross_exposure` |
| `fct_shap_importance.csv` | 1 ligne / (feature, fold) | `fold_year`, `feature`, `feature_group`, `mean_abs_shap`, `importance_rank` |

Aucun fichier `.pbix` n'est généré : le rapport se construit à la main en suivant ce guide.

## 1. Import et modèle de données

1. *Obtenir les données → Texte/CSV* pour les quatre fichiers. Types : dates en `Date`, rendements en `Nombre décimal`, `cik` en `Nombre entier`.
2. Créer une table calendrier : `Calendrier = CALENDAR(DATE(2008,1,1), DATE(2024,12,31))` avec `Annee = YEAR([Date])` et `Mois = FORMAT([Date], "yyyy-mm")`. La marquer comme table de dates.
3. Relations (cardinalité plusieurs-à-un, filtrage simple) :
   - `fct_monthly_panel[cik]` → `dim_company[cik]`
   - `fct_monthly_panel[month_end]` → `Calendrier[Date]`
   - `fct_backtest_returns[month_end]` → `Calendrier[Date]`
   - `fct_shap_importance` reste isolée (filtrée par ses propres segments `fold_year`, `feature_group`).

## 2. Mesures DAX

```DAX
Rendement mensuel moyen = AVERAGE ( fct_backtest_returns[ret_net] )

Capital cumulé =
VAR d = MAX ( fct_backtest_returns[month_end] )
RETURN
    PRODUCTX (
        FILTER ( ALLSELECTED ( fct_backtest_returns[month_end] ), fct_backtest_returns[month_end] <= d ),
        1 + CALCULATE ( SUM ( fct_backtest_returns[ret_net] ) )
    )

Rendement annualisé =
VAR n = COUNTROWS ( fct_backtest_returns )
RETURN IF ( n > 0, PRODUCTX ( fct_backtest_returns, 1 + fct_backtest_returns[ret_net] ) ^ ( 12 / n ) - 1 )

Volatilité annualisée = STDEV.S ( fct_backtest_returns[ret_net] ) * SQRT ( 12 )

Sharpe = DIVIDE ( [Rendement mensuel moyen], STDEV.S ( fct_backtest_returns[ret_net] ) ) * SQRT ( 12 )

Taux de réussite =
DIVIDE (
    COUNTROWS ( FILTER ( fct_backtest_returns, fct_backtest_returns[ret_net] > 0 ) ),
    COUNTROWS ( fct_backtest_returns )
)

Drawdown =
VAR d = MAX ( fct_backtest_returns[month_end] )
VAR pic =
    MAXX (
        FILTER ( ALLSELECTED ( fct_backtest_returns[month_end] ), fct_backtest_returns[month_end] <= d ),
        [Capital cumulé]
    )
RETURN DIVIDE ( [Capital cumulé], pic ) - 1

Turnover mensuel moyen = AVERAGE ( fct_backtest_returns[turnover] )
Coûts annuels = SUM ( fct_backtest_returns[cost] ) / DISTINCTCOUNT ( Calendrier[Annee] )
Exposition brute moyenne = AVERAGE ( fct_backtest_returns[gross_exposure] )

Nb titres membres = DISTINCTCOUNT ( fct_monthly_panel[ticker] )
Part avec 10-K = DIVIDE ( COUNT ( fct_monthly_panel[accession] ), COUNTROWS ( fct_monthly_panel ) )

Importance SHAP = AVERAGE ( fct_shap_importance[mean_abs_shap] )
Part importance groupe =
DIVIDE ( [Importance SHAP], CALCULATE ( [Importance SHAP], ALL ( fct_shap_importance[feature_group] ) ) )
```

## 3. Pages proposées

### Page 1 — Performance
- Segment `strategy` (par défaut : `model`, `random`, `momentum`).
- Courbe : `Capital cumulé` par `month_end`, légende `strategy`.
- Cartes : `Rendement annualisé`, `Volatilité annualisée`, `Sharpe`, `Taux de réussite`.
- Histogramme groupé : `Rendement mensuel moyen` par `Calendrier[Annee]` et `strategy`.
- Tableau de sensibilité : `Sharpe` pour `model`, `model_tc5bps`, `model_tc20bps`, `model_no_word2vec`, `model_no_finbert`.

### Page 2 — Risque et expositions
- Aire : `Drawdown` par `month_end` (stratégie `model`).
- Lignes : `Exposition brute moyenne` et `Turnover mensuel moyen` par mois.
- Barres : `Nb titres membres` par `dim_company[ff12_name]` (composition point-in-time de l'univers).
- Courbe : `Part avec 10-K` par mois (couverture des features textuelles).

### Page 3 — Explicabilité (SHAP)
- Barres horizontales : `Importance SHAP` par `feature`, couleur `feature_group`.
- Matrice `feature` × `fold_year` avec `Importance SHAP` (mise en forme conditionnelle) : stabilité des importances entre folds.
- Anneau : `Part importance groupe` par `feature_group` (FinBERT, Word2Vec, TF-IDF/longueur, contrôles).
- Les beeswarms par fold sont dans `reports/figures/folds/` et dans MLflow (`.\tasks.ps1 mlflow-ui`).
