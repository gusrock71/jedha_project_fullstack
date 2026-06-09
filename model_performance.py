"""
model_performance.py — Tableau de performance de tous les modèles × actifs
===========================================================================
Affiche et sauvegarde un récapitulatif complet des métriques
(Accuracy, AUC, Precision, Recall, F1) pour chaque combinaison
modèle × actif sur le test set.

Usage :
    python3 model_performance.py
    python3 model_performance.py --from-csv outputs/df_combined.csv
    python3 model_performance.py --seuil 0.4
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

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    confusion_matrix,
)

from src.config import COMBINED_CSV, FIGURES_DIR, MODELS_DIR, TEST_START, TEST_END


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


# =============================================================================
# RECONSTRUCTION DES DONNÉES DE TEST PAR ACTIF
# =============================================================================

def build_test_data(df: pd.DataFrame, meta: dict, asset: str) -> tuple:
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

    test_df = df_model.loc[TEST_START:TEST_END]
    X_test  = test_df[feature_cols]
    y_test  = (test_df[target] > 0).astype(int)

    return X_test, y_test


# =============================================================================
# ÉVALUATION D'UN MODÈLE
# =============================================================================

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, seuil: float) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= seuil).astype(int)

    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred),               4),
        "AUC":       round(roc_auc_score(y_test, y_proba),               4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
        "y_proba":   y_proba,
        "y_pred":    y_pred,
    }


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_metrics_heatmap(df_scores: pd.DataFrame, save: bool = True):
    """Heatmap des métriques pour tous les modèles × actifs."""
    metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))

    for ax, metric in zip(axes, metrics):
        pivot = df_scores.pivot(index="Actif", columns="Modèle", values=metric)
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0.5, vmax=0.9, ax=ax, linewidths=0.5,
            annot_kws={"size": 9},
        )
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "Performance des modèles sur le test set (2023 → 2026)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    if save:
        _savefig(fig, "perf_heatmap.png")
    else:
        plt.show()


def plot_metrics_barplot(df_scores: pd.DataFrame, save: bool = True):
    """Barplot groupé par actif pour chaque métrique."""
    metrics  = ["Accuracy", "AUC", "F1"]
    assets   = df_scores["Actif"].unique()
    n_assets = len(assets)

    fig, axes = plt.subplots(1, n_assets, figsize=(6 * n_assets, 5), sharey=False)
    if n_assets == 1:
        axes = [axes]

    colors = {"Logistic Regression": "steelblue",
              "Random Forest":       "forestgreen",
              "XGBoost":             "darkorange"}

    for ax, asset in zip(axes, assets):
        sub = df_scores[df_scores["Actif"] == asset].set_index("Modèle")
        x   = np.arange(len(metrics))
        w   = 0.25

        for i, (model_name, row) in enumerate(sub.iterrows()):
            vals   = [row[m] for m in metrics]
            offset = (i - 1) * w
            bars   = ax.bar(x + offset, vals, w,
                            label  = model_name,
                            color  = colors.get(model_name, f"C{i}"),
                            alpha  = 0.85,
                            edgecolor = "white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(asset, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.legend(fontsize=8)
        ax.set_ylabel("Score")

    fig.suptitle("Comparaison des modèles par actif — Test set",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        _savefig(fig, "perf_barplot.png")
    else:
        plt.show()


def plot_roc_all(roc_data: dict, save: bool = True):
    """Courbes ROC pour tous les modèles × actifs sur une seule figure."""
    assets = list(roc_data.keys())
    n      = len(assets)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    styles = {"Logistic Regression": ("steelblue",   "-"),
              "Random Forest":       ("forestgreen",  "--"),
              "XGBoost":             ("darkorange",   "-.")}

    for ax, asset in zip(axes, assets):
        for model_name, (fpr, tpr, auc) in roc_data[asset].items():
            color, ls = styles.get(model_name, ("grey", "-"))
            ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2,
                    label=f"{model_name} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_title(asset, fontsize=13, fontweight="bold")
        ax.set_xlabel("Taux faux positifs")
        ax.set_ylabel("Taux vrais positifs")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Courbes ROC — Test set (2023 → 2026)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        _savefig(fig, "perf_roc_all.png")
    else:
        plt.show()


def plot_confusion_all(conf_data: dict, save: bool = True):
    """Matrices de confusion pour tous les modèles × actifs."""
    assets = list(conf_data.keys())
    models = list(conf_data[assets[0]].keys())
    n_a    = len(assets)
    n_m    = len(models)

    fig, axes = plt.subplots(n_m, n_a, figsize=(5 * n_a, 4 * n_m))

    for j, asset in enumerate(assets):
        for i, model_name in enumerate(models):
            ax = axes[i][j]
            cm = conf_data[asset][model_name]
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Baisse", "Hausse"],
                yticklabels=["Baisse", "Hausse"],
                ax=ax, cbar=False,
            )
            ax.set_title(f"{asset} — {model_name}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Prédit")
            ax.set_ylabel("Réel")

    fig.suptitle("Matrices de Confusion — Test set", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        _savefig(fig, "perf_confusion_all.png")
    else:
        plt.show()


# =============================================================================
# RÉSUMÉ CONSOLE
# =============================================================================

def print_full_table(df_scores: pd.DataFrame):
    print(f"\n{'=' * 80}")
    print(f"  PERFORMANCE SUR LE TEST SET ({TEST_START} → {TEST_END})")
    print(f"{'=' * 80}")

    header = f"{'Actif':<6} {'Modèle':<25} {'Accuracy':>10} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print(header)
    print("-" * 80)

    prev_asset = None
    for _, row in df_scores.iterrows():
        if prev_asset and prev_asset != row["Actif"]:
            print()
        print(
            f"{row['Actif']:<6} {row['Modèle']:<25} "
            f"{row['Accuracy']:>10.4f} {row['AUC']:>8.4f} "
            f"{row['Precision']:>10.4f} {row['Recall']:>8.4f} {row['F1']:>8.4f}"
        )
        prev_asset = row["Actif"]

    print(f"{'=' * 80}\n")

    # Meilleur modèle par actif (AUC)
    print("  MEILLEUR MODÈLE PAR ACTIF (AUC) :")
    for asset, group in df_scores.groupby("Actif"):
        best = group.loc[group["AUC"].idxmax()]
        print(f"  {asset} → {best['Modèle']:<25} AUC={best['AUC']:.4f}  F1={best['F1']:.4f}")
    print()


# =============================================================================
# MAIN
# =============================================================================

ASSETS  = ["AF", "TTE", "RNO"]
MODELS  = {
    "Logistic Regression": "Logistic_Regression",
    "Random Forest":       "Random_Forest_s04",
    "XGBoost":             "XGBoost_s04",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Model Performance Summary")
    parser.add_argument("--from-csv", type=str, default=None, metavar="PATH")
    parser.add_argument("--no-save",  action="store_true")
    parser.add_argument("--seuil",    type=float, default=0.4,
                        help="Seuil de décision pour RF et XGBoost (défaut : 0.4)")
    return parser.parse_args()


def main():
    args = parse_args()
    save  = not args.no_save
    seuil = args.seuil

    # Chargement données
    csv_path = args.from_csv or COMBINED_CSV
    df       = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    metadata = load_metadata()
    print(f"Données : {df.shape[0]} lignes  |  Test : {TEST_START} → {TEST_END}  |  Seuil : {seuil}")

    rows      = []
    roc_data  = {a: {} for a in ASSETS}
    conf_data = {a: {} for a in ASSETS}

    for asset in ASSETS:
        X_test, y_test = build_test_data(df, metadata, asset)

        for model_name, model_key in MODELS.items():

            # Chargement modèle
            pkl_found = False
            for fname in os.listdir(MODELS_DIR):
                if fname.startswith(f"{asset}_{model_key}") and fname.endswith(".pkl"):
                    model = joblib.load(os.path.join(MODELS_DIR, fname))
                    pkl_found = True
                    break

            if not pkl_found:
                print(f"  ⚠ Modèle introuvable : {asset}_{model_key}")
                continue

            # Seuil : 0.5 pour Logit, seuil paramétré pour RF et XGB
            s = 0.5 if model_name == "Logistic Regression" else seuil
            metrics = evaluate(model, X_test, y_test, s)

            rows.append({
                "Actif":  asset,
                "Modèle": model_name,
                **{k: v for k, v in metrics.items() if k not in ("y_proba", "y_pred")},
            })

            # ROC
            fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
            roc_data[asset][model_name]  = (fpr, tpr, metrics["AUC"])

            # Confusion matrix
            conf_data[asset][model_name] = confusion_matrix(y_test, metrics["y_pred"])

    df_scores = pd.DataFrame(rows)

    # Affichage console
    print_full_table(df_scores)

    # Figures
    print("Génération des figures...")
    plot_metrics_heatmap(df_scores, save)
    plot_metrics_barplot(df_scores, save)
    plot_roc_all(roc_data, save)
    plot_confusion_all(conf_data, save)

    print(f"Figures sauvegardées dans : {FIGURES_DIR}/")
    print("Script terminé.\n")


if __name__ == "__main__":
    main()
