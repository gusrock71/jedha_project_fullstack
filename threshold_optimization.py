"""
threshold_optimization.py — Optimisation du seuil de décision
================================================================
Pour chaque actif (AF, TTE, RNO) et chaque modèle (Logistic Regression,
Random Forest, XGBoost), recherche le seuil de décision optimal sur le
VALIDATION set (jamais sur le test set) selon deux critères :

  1. Seuil F1-max     : maximise le F1-score de la classe 1 (Hausse)
  2. Seuil Youden's J : maximise (TPR - FPR), point optimal de la courbe ROC

Produit :
  - un tableau récapitulatif en console
  - un graphique F1 vs seuil par actif (3 modèles superposés),
    avec le seuil actuel (0.40 / 0.50) et le seuil F1-max repérés

Usage :
    python3 threshold_optimization.py
    python3 threshold_optimization.py --from-csv outputs/df_combined.csv
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, roc_curve, roc_auc_score

from src.config import COMBINED_CSV, FIGURES_DIR, MODELS_DIR, VALID_START, VALID_END


# =============================================================================
# CONFIG
# =============================================================================

ASSETS  = ["AF", "TTE", "RNO"]
MODELS  = {
    "Logistic Regression": "Logistic_Regression",
    "Random Forest":       "Random_Forest_s04",
    "XGBoost":             "XGBoost_s04",
}

# Seuil actuellement utilisé en production (cf. multi_asset_experiment.py /
# model_performance.py) — Logit = 0.5 toujours, RF/XGB = CURRENT_SEUIL
CURRENT_SEUIL = {
    "Logistic Regression": 0.50,
    "Random Forest":       0.40,
    "XGBoost":             0.40,
}

THRESHOLDS = np.arange(0.05, 0.96, 0.01)


# =============================================================================
# HELPERS
# =============================================================================

def _savefig(fig, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  → Sauvegardé : {path}")
    plt.close(fig)


def load_metadata() -> dict:
    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        return json.load(f)


def build_valid_data(df: pd.DataFrame, meta: dict, asset: str) -> tuple:
    """Reconstruit X_valid, y_valid à partir des best_lags stockés dans metadata."""
    target    = meta[asset]["target"]
    best_lags = meta[asset]["best_lags"]

    df_model     = df.copy()
    feature_cols = []
    for feat, lag in best_lags.items():
        col_name = f"{feat}_lag{lag}"
        df_model[col_name] = df_model[feat].shift(lag)
        feature_cols.append(col_name)

    df_model = df_model.dropna()
    df_model.index = pd.to_datetime(df_model.index)

    valid_df = df_model.loc[VALID_START:VALID_END]
    X_valid  = valid_df[feature_cols]
    y_valid  = (valid_df[target] > 0).astype(int)

    return X_valid, y_valid


def load_model(asset: str, model_key: str):
    for fname in os.listdir(MODELS_DIR):
        if fname.startswith(f"{asset}_{model_key}") and fname.endswith(".pkl"):
            return joblib.load(os.path.join(MODELS_DIR, fname))
    return None


# =============================================================================
# RECHERCHE DU SEUIL OPTIMAL
# =============================================================================

def f1_curve(y_true: pd.Series, y_proba: np.ndarray) -> pd.DataFrame:
    """F1-score pour chaque seuil testé."""
    f1s = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in THRESHOLDS]
    return pd.DataFrame({"seuil": THRESHOLDS, "f1": f1s})


def youden_optimal(y_true: pd.Series, y_proba: np.ndarray) -> tuple:
    """Seuil maximisant Youden's J = TPR - FPR (point optimal de la courbe ROC)."""
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    idx = np.argmax(j)
    # roc_curve renvoie parfois un seuil > 1 pour le premier point (artefact sklearn)
    seuil = min(thr[idx], 1.0)
    return seuil, j[idx]


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_f1_curves(asset: str, curves: dict, summary_rows: list, save: bool = True):
    """Courbes F1 vs seuil pour les 3 modèles d'un actif."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    colors = {"Logistic Regression": "steelblue",
              "Random Forest":       "forestgreen",
              "XGBoost":             "darkorange"}

    for model_name, df_curve in curves.items():
        color = colors.get(model_name, "grey")
        ax.plot(df_curve["seuil"], df_curve["f1"], color=color, linewidth=2, label=model_name)

    for row in summary_rows:
        color = colors.get(row["Modèle"], "grey")
        # Seuil actuel (croix)
        ax.scatter(row["Seuil actuel"], row["F1 @ seuil actuel"],
                   color=color, marker="x", s=70, zorder=5)
        # Seuil F1-max (point plein)
        ax.scatter(row["Seuil F1-max"], row["F1-max"],
                   color=color, marker="o", s=60, zorder=5,
                   edgecolors="black", linewidths=0.8)

    ax.axvline(0.5, color="grey", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_title(f"{asset} — F1-score (classe Hausse) en fonction du seuil de décision\n"
                  "(validation set 2021-2022)  —  ○ = seuil F1-max  |  × = seuil actuel",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("F1-score (classe 1 = Hausse)")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        _savefig(fig, f"threshold_optimization_{asset}.png")
    else:
        plt.show()


# =============================================================================
# RÉSUMÉ CONSOLE
# =============================================================================

def print_summary(df_summary: pd.DataFrame):
    print(f"\n{'=' * 100}")
    print("  SEUILS DE DÉCISION — Optimisation sur le validation set (2021-2022)")
    print(f"{'=' * 100}")

    header = (
        f"{'Actif':<6} {'Modèle':<22} {'AUC':>6}  "
        f"{'Seuil actuel':>13} {'F1 @ actuel':>12}  "
        f"{'Seuil F1-max':>13} {'F1-max':>8}  "
        f"{'Seuil Youden':>13} {'Youden J':>9}"
    )
    print(header)
    print("-" * len(header))

    prev_asset = None
    for _, row in df_summary.iterrows():
        if prev_asset and prev_asset != row["Actif"]:
            print()
        print(
            f"{row['Actif']:<6} {row['Modèle']:<22} {row['AUC']:>6.3f}  "
            f"{row['Seuil actuel']:>13.2f} {row['F1 @ seuil actuel']:>12.4f}  "
            f"{row['Seuil F1-max']:>13.2f} {row['F1-max']:>8.4f}  "
            f"{row['Seuil Youden']:>13.2f} {row['Youden J']:>9.4f}"
        )
        prev_asset = row["Actif"]

    print(f"{'=' * 100}\n")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Optimisation du seuil de décision")
    parser.add_argument("--from-csv", type=str, default=None, metavar="PATH")
    parser.add_argument("--no-save",  action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    save = not args.no_save

    csv_path = args.from_csv or COMBINED_CSV
    df       = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    metadata = load_metadata()
    print(f"Données : {df.shape[0]} lignes  |  Validation : {VALID_START} → {VALID_END}")

    rows = []

    for asset in ASSETS:
        X_valid, y_valid = build_valid_data(df, metadata, asset)
        curves = {}

        for model_name, model_key in MODELS.items():
            model = load_model(asset, model_key)
            if model is None:
                print(f"  ⚠ Modèle introuvable : {asset}_{model_key}")
                continue

            y_proba = model.predict_proba(X_valid)[:, 1]

            # Courbe F1 vs seuil
            df_curve = f1_curve(y_valid, y_proba)
            curves[model_name] = df_curve

            # Seuil F1-max
            idx_best   = df_curve["f1"].idxmax()
            seuil_f1   = df_curve.loc[idx_best, "seuil"]
            f1_max     = df_curve.loc[idx_best, "f1"]

            # Seuil Youden
            seuil_youden, youden_j = youden_optimal(y_valid, y_proba)

            # F1 au seuil actuel
            seuil_actuel = CURRENT_SEUIL[model_name]
            f1_actuel    = f1_score(y_valid, (y_proba >= seuil_actuel).astype(int), zero_division=0)

            auc = roc_auc_score(y_valid, y_proba)

            rows.append({
                "Actif":             asset,
                "Modèle":            model_name,
                "AUC":               auc,
                "Seuil actuel":      seuil_actuel,
                "F1 @ seuil actuel": f1_actuel,
                "Seuil F1-max":      round(seuil_f1, 2),
                "F1-max":            f1_max,
                "Seuil Youden":      round(seuil_youden, 2),
                "Youden J":          youden_j,
            })

        df_summary_asset = pd.DataFrame([r for r in rows if r["Actif"] == asset])
        plot_f1_curves(asset, curves, df_summary_asset.to_dict("records"), save)

    df_summary = pd.DataFrame(rows)
    print_summary(df_summary)

    out_csv = os.path.join("outputs", "threshold_optimization.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"Tableau sauvegardé → {out_csv}")
    print(f"Figures sauvegardées dans : {FIGURES_DIR}/")
    print("Script terminé.\n")


if __name__ == "__main__":
    main()
