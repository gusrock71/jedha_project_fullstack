---
title: Jedha Project Fullstack
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Prédiction de tendance boursière (AF, TTE, RNO) — Projet Jedha
---

# Impact des variations d'indices macroéconomiques, financiers et géopolitiques sur les rendements de certaines valeurs boursières françaises

**Projet de fin d'études — Jedha Bootcamp**

Prédiction de la probabilité de hausse du rendement journalier d'**Air France (AF)**,
**TotalEnergies (TTE)** et **Renault (RNO)** à partir de variables géopolitiques,
financières et macroéconomiques (Brent, GPR, VIX, STOXX50, CLI), sur la période
2007-07-30 → 2026-04-01.

Une application **Streamlit** permet de visualiser la prédiction du jour pour
chaque actif et d'explorer les détails du modèle.

---

## Structure du projet

```
jedha_project/
│
├── main.py                       ← Pipeline complet (ingestion → EDA → Logit, sur AF)
├── multi_asset_experiment.py     ← Script principal : Logit / RF / XGBoost sur AF, TTE, RNO
├── xgboost_experiment.py         ← Comparaison Logit vs XGBoost avec recherche de seuil
├── rf_feature_importance.py      ← Importance des features (MDI + permutation)
├── model_performance.py          ← Tableau récapitulatif des performances par modèle/actif
├── app.py                         ← Application Streamlit (page publique + page paramètres)
│
├── src/
│   ├── config.py                 ← Paramètres centralisés (dates, tickers, lags, splits…)
│   ├── ingestion.py               ← Chargement et assemblage des données (yfinance, GPR, CLI)
│   ├── eda.py                     ← Analyse exploratoire et visualisations
│   ├── features.py                ← Feature engineering (lags, split train/valid/test)
│   └── modeling.py                ← Régression logistique (statsmodels + sklearn)
│
├── assets/
│   └── jedha_logo.png             ← Logo affiché dans l'application Streamlit
│
├── data/raw/                       ← Fichiers externes (à placer ici)
│   └── OECD_SDD_STES_DSD_STES_DF_CLI__all.csv   (CLI OCDE — fallback GPR)
│
├── outputs/
│   ├── df_combined.csv            ← Dataset final exporté
│   ├── figures/                   ← Graphiques générés automatiquement
│   └── models/                    ← Modèles entraînés (.pkl) + metadata.json
│
├── Dockerfile                      ← Build Hugging Face Spaces (SDK Docker + Streamlit)
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Utilisation

### 1. Génération du dataset + modélisation Air France (pipeline historique)
```bash
python main.py
```

| Option | Description |
|---|---|
| `--skip-eda` | Saute l'analyse exploratoire |
| `--skip-granger` | Saute le test de causalité de Granger (long) |
| `--no-save` | Affiche les figures au lieu de les sauvegarder |
| `--from-csv PATH` | Charge `df_combined` depuis un CSV existant (évite le re-téléchargement) |

```bash
python main.py --from-csv outputs/df_combined.csv --skip-eda
```

### 2. Entraînement multi-actifs (script principal)
```bash
python multi_asset_experiment.py --from-csv outputs/df_combined.csv
```

Pour chaque actif (AF, TTE, RNO) : optimisation des lags par corrélation de
Spearman, entraînement Logit / Random Forest / XGBoost, évaluation sur
validation puis test, sauvegarde des modèles dans `outputs/models/`.

### 3. Lancer l'application Streamlit
```bash
streamlit run app.py
```

- **Page publique** : tendance du jour (hausse/baisse) + graphique prix & probabilités
- **Page paramètres** : sélection du seuil de décision, détails des features et métriques

---

## Données

| Source | Variables | Mode | Disponibilité live (app) |
|---|---|---|---|
| Yahoo Finance | AF, TTE, RNO, STOXX50, CAC40, Brent, VIX | Log-returns | ✅ Temps réel |
| Yahoo Finance | EUR/USD | Variation absolue | ✅ Temps réel |
| Caldara & Iacoviello | GPRD, GPRD_ACT, GPRD_THREAT | Variation absolue | ✅ Téléchargement automatique (`data_gpr_daily_recent.xls`) |
| OCDE (G4E, série LI, IX, AA) | CLI (Composite Leading Indicator) | Variation absolue (mensuelle) | ✅ API SDMX OCDE |

**Période d'analyse :** 2007-07-30 → 2026-04-01 (inclut crise 2008, dette
européenne 2011, COVID-19 2020, guerre en Ukraine 2022, conflit Israël-Hamas 2023).

---

## Découpage temporel

| Ensemble | Période | Rôle |
|---|---|---|
| Train | 2007-07-30 → 2020-12-31 | Apprentissage (inclut crise 2008, COVID) |
| Validation | 2021-01-01 → 2022-12-31 | Ajustement (inclut guerre en Ukraine) |
| Test | 2023-01-01 → 2026-04-01 | Évaluation finale, hors-échantillon |

---

## Modèles

Trois modèles entraînés et comparés pour chaque actif (AF, TTE, RNO) :
**Régression Logistique**, **Random Forest**, **XGBoost**.

Features sélectionnées par optimisation des lags (corrélation de Spearman sur
le train set), spécifiques à chaque actif :

| Feature | Lag AF | Lag TTE | Lag RNO |
|---|---|---|---|
| GPRD_ACT_Diff | 3 | 13 | 7 |
| GPRD_THREAT_Diff | 2 | 27 | 11 |
| CLI_Diff | 9 | 8 | 14 |
| STOXX_Return | 0 | 0 | 0 |
| VIX_Return | 0 | 0 | 0 |
| Brent_Return | 0 | 0 | 0 |

---

## Résultats clés (test set, seuil = 0.40)

| Actif | Meilleur modèle | AUC | F1 |
|---|---|---|---|
| AF | Random Forest | 0.822 | 0.684 |
| TTE | Random Forest | 0.855 | 0.735 |
| RNO | Random Forest | 0.855 | 0.725 |

- Random Forest surpasse XGBoost et la régression logistique sur les trois actifs.
- STOXX50 (lag 0) est la feature dominante pour les trois actifs.
- Les variables géopolitiques (GPR) et macroéconomiques (CLI) apportent un
  signal complémentaire non redondant avec le marché global.

---

## Déploiement (Hugging Face Spaces)

Ce dépôt est configuré pour un déploiement via **SDK Docker** (le SDK Streamlit
natif étant déprécié sur Hugging Face Spaces). Le `Dockerfile` installe les
dépendances et lance `streamlit run app.py` sur le port 8501.
