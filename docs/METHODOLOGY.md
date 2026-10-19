# Méthodologie

## 1. Calendrier et conventions temporelles

- **Date de formation** *t* : dernier jour de bourse du mois *m* (`month_end`). Toutes les informations utilisées doivent être disponibles à *t*.
- **Mois de détention** : *m + 1*. La cible d'une ligne du panel est le rendement de ce mois.
- **Disponibilité d'un 10-K** : premier jour de bourse dont la date est strictement postérieure à la date d'`acceptanceDateTime` EDGAR. Un 10-K accepté le jour *t* n'est donc jamais utilisé à *t*. Test dbt `assert_point_in_time_features` + test pytest `test_available_date_strictly_after_acceptance_sql`.

## 2. Univers sans biais du survivant

À chaque date de formation, l'univers est l'ensemble des membres du S&P 500 **à cette date** (instantanés `fja05680/sp500`), y compris les titres sortis ensuite. Le test dbt `assert_panel_rows_in_index` vérifie que chaque ligne du panel appartient à l'indice à sa date. Les titres sans CIK (pas de 10-K) ou sans prix ne sont pas investissables : c'est la principale source de biais résiduel, quantifiée dans `docs/DATA.md`.

## 3. Features (rangs transversaux par mois)

| Groupe | Features | Calcul |
|---|---|---|
| FinBERT | `fb_tone_*`, `fb_neg_*`, `fb_unc_*`, `d_tone_*` | ton moyen p(pos) − p(neg), part de phrases négatives, entropie normalisée moyenne, variation du ton vs 10-K précédent ; Item 1A (`_1a`) et Item 7 (`_7`) |
| TF-IDF / longueur | `tfidf_sim_*`, `len_chg_*`, `log_len_*` | cosinus TF-IDF entre le 10-K et celui de l'année précédente (même société, 9-24 mois d'écart), log-variation de longueur, log-longueur |
| Word2Vec | `w2v_sim_*` | cosinus entre embeddings de document (moyenne pondérée TF-IDF des vecteurs de mots) ; un modèle **par fold**, entraîné sur les seuls documents disponibles avant la première date de formation du fold |
| Contrôles | `log_mcap`, `mom_12_1`, `vol_3m`, `rev_1m` | log-capitalisation point-in-time, momentum t-12 → t-1, volatilité 63 jours annualisée, rendement du mois *m* |

## 4. Modèle

- Cible : rang transversal du rendement du mois suivant, démeané par secteur FF12.
- XGBoost (`hist`, sous-échantillonnage 0,8). Walk-forward à fenêtre expansive, folds 2018 → 2024 (le fold *Y* contient les mois de détention de l'année *Y*), embargo d'un mois entre la dernière cible d'entraînement et la première date de test.
- Hyperparamètres : petite grille fixée a priori, choix par IC moyen sur les deux dernières années de la fenêtre d'entraînement (validation interne, même embargo), nombre d'arbres par early stopping.
- Suivi MLflow : un run parent par variante (`main`, `no_word2vec`, `no_finbert`), un run imbriqué par fold (paramètres, IC de validation, IC et MSE de test, modèle, beeswarm SHAP).

## 5. Portefeuille

Chaque mois : `max αᵀw − λ wᵀΣw − κ ||w − w_prev||₁` sous `Σw = 0`, neutralité par secteur FF12, `||w||₁ ≤ 2`, `|wᵢ| ≤ 2 %`.

- α = IC a priori (0,05) × σᵢ mensuelle × z-score du score du modèle (Grinold-Kahn).
- Σ = Ledoit-Wolf sur 252 jours de rendements journaliers (≥ 126 observations requises), × 21.
- κ = coût de transaction par côté ; `w_prev` = poids dérivés à la fin du mois précédent.
- Coûts : 10 bps par côté sur le notionnel échangé + 50 bps/an sur la jambe short, prélevés dans les rendements journaliers.

## 6. Évaluation

Période hors échantillon : rendements 2018-01 → 2024-12, nets de coûts. Sharpe = moyenne / écart-type mensuels × √12 sans taux sans risque (portefeuille dollar-neutre). Alpha : régression des rendements mensuels nets sur FF5 + Momentum, erreurs Newey-West 6 retards. IC : corrélation de Spearman mensuelle entre prédiction et rendement réalisé du mois suivant, t-stat = moyenne / (écart-type / √n). Robustesse : coûts 5/20 bps, sous-périodes 2018-2020 et 2021-2024, ablations sans Word2Vec et sans FinBERT (réentraînement complet).
