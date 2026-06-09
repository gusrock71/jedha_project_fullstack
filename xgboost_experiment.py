"""
xgboost_experiment.py — Comparaison XGBoost vs Logistic Regression
====================================================================
Script autonome à lancer depuis la racine du projet :

    python3 xgboost_experiment.py
    python3 xgboost_experiment.py --from-csv outputs/df_combined.csv

Prérequis supplémentaire :
    pip install xgboost
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError(
        "\nXGBoost non installé. Lance : pip install xgboost\n"
    )

from src.ingestion import build_dataset
from src.features  import build_lag_features, split_train_test
from src.config    import COMBINED_CSV, FIGURES_DIR, SELECTED_LAGS, TARGET


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
# 1. ENTRAÎNEMENT DES MODÈLES
# =============================================================================

def train_logistic(X_train, y_train) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            random_state=42,
            class_weight="balanced",
            max_iter=5000,
        )),
    ])
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """
    XGBoost avec paramètres raisonnables pour des séries financières :
    - scale_pos_weight compense le déséquilibre de classes
    - subsample + colsample_bytree réduisent l'overfitting
    - early_stopping désactivé ici pour garder une API simple
    """
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = ratio,
        eval_metric       = "logloss",
        random_state      = 42,
        verbosity         = 0,
    )
    model.fit(X_train, y_train)
    return model


# =============================================================================
# 2. ÉVALUATION COMPLÈTE
# =============================================================================

def evaluate(name: str, model, X_test, y_test, seuil: float = 0.5) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= seuil).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n--- {name} ---")
    print(f"  Accuracy : {acc:.4f}  |  ROC-AUC : {auc:.4f}")
    print(f"  Precision (class 1) : {cr['1']['precision']:.4f}")
    print(f"  Recall    (class 1) : {cr['1']['recall']:.4f}")
    print(f"  F1-score  (class 1) : {cr['1']['f1-score']:.4f}")
    print("\n  Confusion Matrix :")
    print(f"  {cm}")

    return {
        "name":    name,
        "model":   model,
        "y_pred":  y_pred,
        "y_proba": y_proba,
        "acc":     acc,
        "auc":     auc,
        "cm":      cm,
        "cr":      cr,
    }


# =============================================================================
# 3. CROSS-VALIDATION TEMPORELLE
# =============================================================================

def cross_validate_models(models: dict, X_train, y_train, cv: int = 5):
    """
    TimeSeriesSplit-style cross-validation sur le train set.
    Affiche accuracy moyenne ± std pour chaque modèle.
    """
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=cv)

    print_section(f"Cross-Validation temporelle ({cv} folds)")

    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="accuracy")
        cv_results[name] = scores
        print(f"  {name:25s} → {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_results


# =============================================================================
# 4. VISUALISATIONS
# =============================================================================

def plot_roc_curves(results: list, save: bool = True):
    """Courbes ROC superposées pour tous les modèles."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["steelblue", "darkorange", "green", "red"]
    for i, res in enumerate(results):
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"])
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                label=f"{res['name']} (AUC = {res['auc']:.4f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_title("Courbes ROC — Comparaison des modèles", fontsize=13, fontweight="bold")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.legend()
    plt.tight_layout()

    if save:
        _savefig(fig, "xgb_roc_curves.png")
    else:
        plt.show()


def plot_confusion_matrices(results: list, save: bool = True):
    """Matrices de confusion côte-à-côte."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        sns.heatmap(
            res["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["Baisse (0)", "Hausse (1)"],
            yticklabels=["Baisse (0)", "Hausse (1)"],
            ax=ax,
        )
        ax.set_title(f"{res['name']}\nAcc={res['acc']:.4f}  AUC={res['auc']:.4f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")

    fig.suptitle("Matrices de Confusion", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        _savefig(fig, "xgb_confusion_matrices.png")
    else:
        plt.show()


def plot_feature_importance(xgb_model, feature_cols: list, save: bool = True):
    """Importance des features XGBoost (gain) vs coefficients Logit."""
    importances = xgb_model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(imp_df["Feature"], imp_df["Importance"], color="darkorange")
    ax.set_title("XGBoost — Importance des features (gain)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")

    # Annotations
    for bar, val in zip(bars, imp_df["Importance"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    if save:
        _savefig(fig, "xgb_feature_importance.png")
    else:
        plt.show()

    print("\nImportance des features (XGBoost) :")
    print(imp_df.sort_values("Importance", ascending=False).to_string(index=False))

    return imp_df


def plot_cv_comparison(cv_results: dict, save: bool = True):
    """Boxplot comparant la distribution des scores CV par modèle."""
    fig, ax = plt.subplots(figsize=(7, 5))

    data   = list(cv_results.values())
    labels = list(cv_results.keys())

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    boxprops=dict(facecolor="lightsteelblue"),
                    medianprops=dict(color="navy", linewidth=2))

    ax.set_title("Cross-Validation temporelle — Distribution des scores Accuracy",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, linestyle="--", color="red", alpha=0.5, label="Baseline aléatoire")
    ax.legend()
    plt.tight_layout()

    if save:
        _savefig(fig, "xgb_cv_comparison.png")
    else:
        plt.show()


def plot_predictions_timeline(results: list, X_test, y_test, save: bool = True):
    """
    Visualise les prédictions des modèles dans le temps
    avec le signal réel en fond.
    """
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 4 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        # Signal réel
        ax.fill_between(X_test.index,
                        y_test.values, alpha=0.15, color="green", label="Réel (1=hausse)")

        # Probabilité prédite
        ax.plot(X_test.index, res["y_proba"], color="darkorange",
                linewidth=1.2, label="Proba prédite (hausse)")

        # Seuil 0.5
        ax.axhline(0.5, linestyle="--", color="grey", alpha=0.7, linewidth=0.8)

        ax.set_title(f"{res['name']} — Probabilité de hausse sur le test set",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Probabilité")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()

    if save:
        _savefig(fig, "xgb_predictions_timeline.png")
    else:
        plt.show()


# =============================================================================
# RÉSUMÉ COMPARATIF
# =============================================================================

def print_summary(results: list):
    print_section("RÉSUMÉ COMPARATIF")
    header = f"{'Modèle':<25} {'Accuracy':>10} {'ROC-AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    print(header)
    print("-" * 75)
    for res in results:
        cr = res["cr"]
        print(
            f"{res['name']:<25} "
            f"{res['acc']:>10.4f} "
            f"{res['auc']:>10.4f} "
            f"{cr['1']['precision']:>10.4f} "
            f"{cr['1']['recall']:>10.4f} "
            f"{cr['1']['f1-score']:>10.4f}"
        )


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost experiment — Jedha project")
    parser.add_argument("--from-csv", type=str, default=None,
                        metavar="PATH", help="Charger df_combined depuis un CSV existant")
    parser.add_argument("--no-save", action="store_true",
                        help="Afficher les figures au lieu de les sauvegarder")
    return parser.parse_args()


def main():
    args = parse_args()
    save = not args.no_save

    # ------------------------------------------------------------------
    # Chargement des données
    # ------------------------------------------------------------------
    print_section("Chargement des données")

    csv_path = args.from_csv or COMBINED_CSV
    if os.path.exists(csv_path):
        print(f"Chargement depuis {csv_path}...")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        print("CSV non trouvé, lancement de l'ingestion complète...")
        df = build_dataset()
        df.to_csv(COMBINED_CSV)

    print(f"  → {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    print_section("Feature engineering")

    df_model, feature_cols = build_lag_features(df)
    X_train, X_test, y_train, y_test = split_train_test(df_model, feature_cols)

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------
    print_section("Entraînement des modèles")

    print("  Logistic Regression...")
    logit = train_logistic(X_train, y_train)
    print("  XGBoost...")
    xgb = train_xgboost(X_train, y_train)
    print("  Modèles entraînés.")

    # ------------------------------------------------------------------
    # Évaluation
    # ------------------------------------------------------------------
    print_section("Évaluation sur le test set")

    res_logit = evaluate("Logistic Regression", logit, X_test, y_test)
    res_logit["y_test"] = y_test

    all_results = [res_logit]

    for s in [0.30, 0.40, 0.50, 0.60]:
        res = evaluate(f"XGBoost (seuil={s})", xgb, X_test, y_test, seuil=s)
        res["y_test"] = y_test
        all_results.append(res)

    # ------------------------------------------------------------------
    # Cross-validation temporelle
    # ------------------------------------------------------------------
    models_for_cv = {
        "Logistic Regression": logit,
        "XGBoost":             xgb,
    }
    cv_results = cross_validate_models(models_for_cv, X_train, y_train, cv=5)

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    print_section("Génération des figures")

    plot_roc_curves(all_results, save)
    plot_confusion_matrices(all_results, save)
    plot_feature_importance(xgb, feature_cols, save)
    plot_cv_comparison(cv_results, save)
    plot_predictions_timeline(all_results, X_test, y_test, save)

    # ------------------------------------------------------------------
    # Résumé
    # ------------------------------------------------------------------
    print_summary(all_results)

    print(f"\nFigures sauvegardées dans : {FIGURES_DIR}/")
    print("Script terminé.\n")


if __name__ == "__main__":
    main()
