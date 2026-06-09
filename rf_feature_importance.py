"""
rf_feature_importance.py — Importance des features Random Forest
=================================================================
Trois méthodes complémentaires pour chaque actif (AF, TTE, RNO) :
  1. Importance MDI (Mean Decrease Impurity) — natif sklearn
  2. Importance par permutation — plus robuste sur les données corrélées
  3. Graphique comparatif des trois actifs côte-à-côte

Usage :
    python3 rf_feature_importance.py
    python3 rf_feature_importance.py --from-csv outputs/df_combined.csv
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.inspection import permutation_importance

from src.config import COMBINED_CSV, FIGURES_DIR, MODELS_DIR


# =============================================================================
# HELPERS
# =============================================================================

def _savefig(fig, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  → Sauvegardé : {path}")
    plt.close(fig)


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# =============================================================================
# CHARGEMENT MODÈLES ET MÉTADONNÉES
# =============================================================================

def load_rf_model(asset: str):
    """Charge le modèle Random Forest pour un actif."""
    for fname in os.listdir(MODELS_DIR):
        if fname.startswith(f"{asset}_Random_Forest") and fname.endswith(".pkl"):
            model = joblib.load(os.path.join(MODELS_DIR, fname))
            print(f"  Modèle chargé : {fname}")
            return model
    raise FileNotFoundError(f"Modèle RF introuvable pour {asset} dans {MODELS_DIR}/")


def load_metadata() -> dict:
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(meta_path) as f:
        return json.load(f)


# =============================================================================
# RECONSTRUCTION DES DONNÉES DE TEST
# =============================================================================

def build_test_data(df: pd.DataFrame, meta: dict, asset: str) -> tuple:
    """
    Reconstruit X_test et y_test pour un actif à partir des lags stockés
    dans les métadonnées.
    """
    from src.config import TEST_START, TEST_END

    target    = meta[asset]["target"]
    best_lags = meta[asset]["best_lags"]

    df_model = df.copy()
    feature_cols = []
    for feat, lag in best_lags.items():
        col_name = f"{feat}_lag{lag}"
        df_model[col_name] = df_model[feat].shift(lag)
        feature_cols.append(col_name)

    df_model = df_model.dropna()
    df_model.index = pd.to_datetime(df_model.index)

    test_df  = df_model.loc[TEST_START:TEST_END]
    X_test   = test_df[feature_cols]
    y_test   = (test_df[target] > 0).astype(int)

    return X_test, y_test, feature_cols


# =============================================================================
# 1. IMPORTANCE MDI (Mean Decrease Impurity)
# =============================================================================

def plot_mdi_importance(asset: str, model, feature_cols: list, save: bool = True):
    """
    Importance MDI native du Random Forest.
    Rapide mais peut surestimer les features à forte cardinalité.
    """
    # Extrait le RandomForestClassifier depuis le Pipeline
    rf = model.named_steps["clf"]
    importances = rf.feature_importances_
    std         = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

    imp_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": importances,
        "Std":        std,
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(imp_df["Feature"], imp_df["Importance"],
                   xerr=imp_df["Std"], color="forestgreen", alpha=0.8,
                   error_kw={"ecolor": "grey", "capsize": 3})
    ax.set_title(f"{asset} — Importance MDI (Mean Decrease Impurity)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance moyenne (± std entre arbres)")

    # Annotations
    for bar, val in zip(bars, imp_df["Importance"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    if save:
        _savefig(fig, f"rf_mdi_{asset}.png")
    else:
        plt.show()

    print(f"\n  MDI Importance — {asset} :")
    print(imp_df[["Feature", "Importance", "Std"]].sort_values(
        "Importance", ascending=False).to_string(index=False))

    return imp_df


# =============================================================================
# 2. IMPORTANCE PAR PERMUTATION
# =============================================================================

def plot_permutation_importance(
    asset: str, model, X_test: pd.DataFrame, y_test: pd.Series,
    save: bool = True
):
    """
    Importance par permutation sur le test set.
    Plus robuste que MDI : mesure la dégradation réelle du score
    quand on mélange aléatoirement une feature.
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats  = 30,
        random_state = 42,
        scoring    = "roc_auc",
    )

    perm_df = pd.DataFrame({
        "Feature":    X_test.columns,
        "Importance": result.importances_mean,
        "Std":        result.importances_std,
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["forestgreen" if v >= 0 else "tomato" for v in perm_df["Importance"]]
    bars = ax.barh(perm_df["Feature"], perm_df["Importance"],
                   xerr=perm_df["Std"], color=colors, alpha=0.8,
                   error_kw={"ecolor": "grey", "capsize": 3})
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{asset} — Importance par permutation (AUC, test set)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Baisse d'AUC lors de la permutation (± std)")
    ax.set_caption = ""

    plt.tight_layout()
    if save:
        _savefig(fig, f"rf_permutation_{asset}.png")
    else:
        plt.show()

    print(f"\n  Permutation Importance — {asset} :")
    print(perm_df[["Feature", "Importance", "Std"]].sort_values(
        "Importance", ascending=False).to_string(index=False))

    return perm_df


# =============================================================================
# 3. COMPARATIF MDI — 3 ACTIFS CÔTE-À-CÔTE
# =============================================================================

def plot_comparative_importance(all_mdi: dict, save: bool = True):
    """
    Heatmap normalisée des importances MDI pour les 3 actifs.
    Permet de comparer quelles features dominent selon l'actif.
    """
    # Construire un DataFrame commun
    rows = {}
    for asset, imp_df in all_mdi.items():
        rows[asset] = imp_df.set_index("Feature")["Importance"]

    df_comp = pd.DataFrame(rows).fillna(0)

    # Normaliser par colonne (somme = 1) pour comparaison équitable
    df_comp_norm = df_comp / df_comp.sum()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Valeurs brutes
    sns.heatmap(
        df_comp.T, annot=True, fmt=".4f", cmap="YlOrRd",
        linewidths=0.5, ax=axes[0]
    )
    axes[0].set_title("Importance MDI — Valeurs brutes", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Feature")
    axes[0].set_ylabel("Actif")

    # Valeurs normalisées
    sns.heatmap(
        df_comp_norm.T, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, ax=axes[1]
    )
    axes[1].set_title("Importance MDI — Normalisée par actif", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("Actif")

    plt.suptitle("Comparaison des importances Random Forest — AF / TTE / RNO",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        _savefig(fig, "rf_importance_comparative.png")
    else:
        plt.show()


# =============================================================================
# RÉSUMÉ CONSOLE
# =============================================================================

def print_summary(all_mdi: dict, all_perm: dict):
    print_section("RÉSUMÉ — FEATURE LA PLUS IMPORTANTE PAR ACTIF")

    header = f"{'Actif':<6} {'Méthode':<15} {'Feature N°1':<30} {'Importance':>12}"
    print(header)
    print("-" * 65)

    for asset in all_mdi:
        top_mdi  = all_mdi[asset].sort_values("Importance", ascending=False).iloc[0]
        top_perm = all_perm[asset].sort_values("Importance", ascending=False).iloc[0]
        print(f"{asset:<6} {'MDI':<15} {top_mdi['Feature']:<30} {top_mdi['Importance']:>12.4f}")
        print(f"{'':<6} {'Permutation':<15} {top_perm['Feature']:<30} {top_perm['Importance']:>12.4f}")
        print()


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="RF Feature Importance")
    parser.add_argument("--from-csv", type=str, default=None, metavar="PATH")
    parser.add_argument("--no-save",  action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    save = not args.no_save

    # Chargement données
    print_section("Chargement")
    csv_path = args.from_csv or COMBINED_CSV
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"  → {df.shape[0]} lignes × {df.shape[1]} colonnes")

    metadata = load_metadata()

    all_mdi  = {}
    all_perm = {}

    for asset in ["AF", "TTE", "RNO"]:
        print_section(f"ACTIF : {asset}")

        # Chargement modèle
        model = load_rf_model(asset)

        # Données test
        X_test, y_test, feature_cols = build_test_data(df, metadata, asset)
        print(f"  Test set : {X_test.shape}")

        # MDI
        mdi_df  = plot_mdi_importance(asset, model, feature_cols, save)
        all_mdi[asset] = mdi_df

        # Permutation
        print(f"\n  Calcul permutation importance (30 répétitions)...")
        perm_df = plot_permutation_importance(asset, model, X_test, y_test, save)
        all_perm[asset] = perm_df

    # Comparatif
    print_section("Comparatif 3 actifs")
    plot_comparative_importance(all_mdi, save)

    # Résumé
    print_summary(all_mdi, all_perm)

    print(f"\nFigures sauvegardées dans : {FIGURES_DIR}/")
    print("Script terminé.\n")


if __name__ == "__main__":
    main()
