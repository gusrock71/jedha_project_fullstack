# Impact des variations d'indices macroéconomiques, financiers et géopolitiques sur les rendements de certaines valeurs boursières françaises

**Projet de fin d'études — Jedha Bootcamp**

Analyse de l'influence des variables géopolitiques, financières et macroéconomiques
(Brent, GPR, VIX, CLI, TTF, EUR/USD) sur les rendements d'**Air France**,
**Renault** et **TotalEnergies** entre 2017 et 2024.

---

## Structure du projet

```
jedha_project/
│
├── main.py                  ← Point d'entrée unique
│
├── src/
│   ├── config.py            ← Paramètres centralisés (dates, tickers, lags…)
│   ├── ingestion.py         ← Chargement et assemblage des données
│   ├── eda.py               ← Analyse exploratoire et visualisations
│   ├── features.py          ← Feature engineering (lags, split train/test)
│   └── modeling.py          ← Modèles ML et évaluation
│
├── data/                    ← Fichiers CSV externes (à placer ici)
│   ├── Dutch TTF Natural Gas Futures Historical Data UK (1).csv
│   ├── data_gpr_daily_recent_.csv
│   └── export-2026-05-17T19_45_48.138Z.csv   (CLI OCDE)
│
├── outputs/
│   ├── df_combined.csv      ← Dataset final exporté
│   ├── figures/             ← Graphiques générés automatiquement
│   └── models/              ← Modèles sauvegardés (à venir)
│
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Utilisation

### Pipeline complet
```bash
python main.py
```

### Options disponibles

| Option | Description |
|---|---|
| `--skip-eda` | Saute l'analyse exploratoire |
| `--skip-granger` | Saute le test de Granger (long) |
| `--no-save` | Affiche les figures au lieu de les sauvegarder |
| `--from-csv PATH` | Charge df_combined depuis un CSV existant (évite le re-téléchargement) |

### Exemple : relancer uniquement la modélisation sur données déjà téléchargées
```bash
python main.py --from-csv outputs/df_combined.csv --skip-eda
```

---

## Données

| Source | Variables | Mode |
|---|---|---|
| Yahoo Finance | AF, TTE, RNO, STOXX50, CAC40, Brent, JetFuel, VIX | Log-returns |
| Yahoo Finance | EUR/USD | Variation absolue |
| Caldara & Iacoviello | GPR, GPRD_ACT, GPRD_THREAT | Variation absolue |
| OCDE | CLI (Composite Leading Indicator) | Variation absolue |

**Période d'analyse :** octobre 2017 → août 2024

---

## Modèle

Régression logistique binaire : exemple de prédiction du **sens du rendement journalier d'Air France** (hausse / baisse).

Features sélectionnées avec optimisation des lags par corrélation de Spearman sur le train set :

| Feature | Lag retenu |
|---|---|
| GPRD_ACT_Diff | 19 jours |
| GPRD_THREAT_Diff | 20 jours |
| CLI_Diff | 16 jours |
| STOXX_Return | 0 jour |
| VIX_Return | 0 jour |
| Brent_Return | 0 jour |

---

## Résultats clés

- Corrélation forte CAC40 ↔ STOXX50 (synchronisation marchés européens)
- VIX négativement corrélé aux indices boursiers (fuite vers la qualité)
- Distributions leptokurtiques (kurtosis 11–30) → modèles non linéaires justifiés
- Effets lag géopolitiques : 1–3 jours (Air France), 12+ jours (Renault)
