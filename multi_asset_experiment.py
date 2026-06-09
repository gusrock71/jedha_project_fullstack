"""
multi_asset_experiment.py — XGBoost sur AF, TTE, RNO avec lags optimisés par actif
=====================================================================================
Pour chaque actif cible (AF_Return, TTE_Return, RNO_Return) :
  1. Optimisation des lags par corrélation de Spearman sur le train set
  2. Construction des features laggées spécifiques à l'actif
  3. Entraînement XGBoost + Logistic Regression (baseline)
  4. Évaluation sur le test set (accuracy, AUC, F1)
  5. Visualisations comparatives

Usage :
    python3 multi_asset_experiment.py
    python3 multi_asset_experiment.py --from-csv outputs/df_combined.csv
    python3 multi_asset_experiment.py --seuil 0.4
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("XGBoost non installé. Lance : pip install xgboost")

from src.ingestion import build_dataset
from src.config    import (
    COMBINED_CSV, FIGURES_DIR, MODELS_DIR,
    TRAIN_START, TRAIN_END,
    VALID_START, VALID_END,
    TEST_START,  TEST_END,
)


# =============================================================================
# PARAMÈTRES
# =============================================================================

TARGETS = ["AF_Return", "TTE_Return", "RNO_Return"]

CANDIDATE_FEATURES = [
    "GPRD_ACT_Diff",
    "GPRD_THREAT_Diff",
    "CLI_Diff",
    "STOXX_Return",
    "VIX_Return",
    "Brent_Return",
]

MAX_LAG = 30


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
# 1. OPTIMISATION DES LAGS PAR ACTIF
# =============================================================================

def optimize_lags_for_target(
    train_df: pd.DataFrame,
    target: str,
    candidate_features: list,
    max_lag: int = MAX_LAG,
) -> dict:
    """
    Pour un actif cible donné, cherche le lag (0 → max_lag) qui maximise
    la corrélation de Spearman absolue entre la feature laggée et le target.

    Retourne {feature: lag_optimal}.
    """
    best_lags = {}

    for feature in candidate_features:
        best_corr = 0
        best_lag  = 0

        for lag in range(max_lag + 1):
            tmp = pd.DataFrame({
                "target":  train_df[target],
                "feature": train_df[feature].shift(lag),
            }).dropna()

            if len(tmp) < 30:
                continue

            corr, _ = spearmanr(tmp["target"], tmp["feature"])

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag  = lag

        best_lags[feature] = best_lag
        print(f"    {feature:30s} → lag={best_lag:2d}  (corr={best_corr:+.4f})")

    return best_lags


# =============================================================================
# 2. CONSTRUCTION DES FEATURES LAGGÉES
# =============================================================================

def build_features(df: pd.DataFrame, selected_lags: dict) -> tuple[pd.DataFrame, list]:
    """
    Construit les colonnes laggées et retourne (df_model, feature_cols).
    """
    df_model     = df.copy()
    feature_cols = []

    for feature, lag in selected_lags.items():
        col_name = f"{feature}_lag{lag}"
        df_model[col_name] = df_model[feature].shift(lag)
        feature_cols.append(col_name)

    return df_model.dropna(), feature_cols


# =============================================================================
# 3. SPLIT TRAIN / VALID / TEST
# =============================================================================

def split(df_model: pd.DataFrame, feature_cols: list, target: str) -> tuple:
    """
    Retourne (X_train, X_valid, X_test, y_train, y_valid, y_test).

    Train : 2007-07-30 → 2020-12-31  (apprentissage — inclut crise 2008, COVID)
    Valid : 2021-01-01 → 2022-12-31  (tuning hyperparamètres — inclut Ukraine)
    Test  : 2023-01-01 → 2026-04-01  (évaluation finale — ne toucher qu'une fois)
    """
    df_model.index = pd.to_datetime(df_model.index)

    train_df = df_model.loc[TRAIN_START:TRAIN_END]
    valid_df  = df_model.loc[VALID_START:VALID_END]
    test_df  = df_model.loc[TEST_START:TEST_END]

    X_train = train_df[feature_cols]
    X_valid  = valid_df[feature_cols]
    X_test  = test_df[feature_cols]

    y_train = (train_df[target] > 0).astype(int)
    y_valid  = (valid_df[target]  > 0).astype(int)
    y_test  = (test_df[target]  > 0).astype(int)

    print(f"  Train : {X_train.shape}  |  Valid : {X_valid.shape}  |  Test : {X_test.shape}")
    print(f"  Taux positif — train : {y_train.mean():.2%}  |  valid : {y_valid.mean():.2%}  |  test : {y_test.mean():.2%}")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


# =============================================================================
# 4. ENTRAÎNEMENT
# =============================================================================

def train_logistic(X_train, y_train) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            random_state=42, class_weight="balanced", max_iter=5000
        )),
    ])
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> XGBClassifier:
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = ratio,
        eval_metric      = "logloss",
        random_state     = 42,
        verbosity        = 0,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators = 300,
            max_depth    = 5,
            class_weight = "balanced",
            random_state = 42,
            n_jobs       = -1,
        )),
    ])
    model.fit(X_train, y_train)
    return model


# =============================================================================
# 5. ÉVALUATION
# =============================================================================

def evaluate(name: str, model, X_test, y_test, seuil: float = 0.5) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= seuil).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cr  = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm  = confusion_matrix(y_test, y_pred)

    # Gestion cas où classe 1 absente des prédictions
    prec = cr.get("1", {}).get("precision", 0)
    rec  = cr.get("1", {}).get("recall", 0)
    f1   = cr.get("1", {}).get("f1-score", 0)

    return {
        "name":    name,
        "model":   model,
        "y_proba": y_proba,
        "y_pred":  y_pred,
        "acc":     acc,
        "auc":     auc,
        "prec":    prec,
        "rec":     rec,
        "f1":      f1,
        "cm":      cm,
    }


# =============================================================================
# 5b. CROSS-VALIDATION TEMPORELLE
# =============================================================================

def cross_validate_models(models: dict, X_train, y_train, cv: int = 5):
    """
    TimeSeriesSplit cross-validation sur le train set.
    Affiche accuracy moyenne ± std pour chaque modèle.
    """
    tscv   = TimeSeriesSplit(n_splits=cv)
    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="accuracy")
        results[name] = scores
        print(f"    {name:25s} → {scores.mean():.4f} ± {scores.std():.4f}")

    return results


# =============================================================================
# 6. VISUALISATIONS PAR ACTIF
# =============================================================================

def plot_asset_roc(asset: str, results: list, y_test, save: bool = True):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["steelblue", "forestgreen", "darkorange"]

    for i, res in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                label=f"{res['name']} (AUC={res['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title(f"{asset} — Courbes ROC", fontsize=12, fontweight="bold")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.legend()
    plt.tight_layout()

    if save:
        _savefig(fig, f"multi_{asset}_roc.png")
    else:
        plt.show()


def plot_asset_confusion(asset: str, results: list, save: bool = True):
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
        ax.set_title(f"{res['name']}\nAcc={res['acc']:.3f}  AUC={res['auc']:.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")

    fig.suptitle(f"{asset} — Matrices de Confusion", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        _savefig(fig, f"multi_{asset}_confusion.png")
    else:
        plt.show()


def plot_asset_feature_importance(asset: str, xgb_model, rf_model, feature_cols: list, save: bool = True):
    """Importance des features XGBoost et Random Forest côte-à-côte."""
    xgb_imp = xgb_model.feature_importances_
    rf_imp  = rf_model.named_steps["clf"].feature_importances_

    imp_df = pd.DataFrame({
        "Feature":  feature_cols,
        "XGBoost":  xgb_imp,
        "RandomForest": rf_imp,
    }).sort_values("XGBoost", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, col, color in zip(axes, ["XGBoost", "RandomForest"], ["darkorange", "forestgreen"]):
        ax.barh(imp_df["Feature"], imp_df[col], color=color)
        ax.set_title(f"{asset} — {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Importance (gain)")

    plt.suptitle(f"{asset} — Importance des features", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        _savefig(fig, f"multi_{asset}_importance.png")
    else:
        plt.show()


# =============================================================================
# 7. VISUALISATION COMPARATIVE FINALE
# =============================================================================

def plot_comparative_summary(summary: dict, save: bool = True):
    """
    Barplots comparatifs Accuracy / AUC / F1 pour les 3 actifs × 2 modèles.
    """
    rows = []
    for asset, results in summary.items():
        for res in results:
            rows.append({
                "Actif":   asset,
                "Modèle":  res["name"],
                "Accuracy": res["acc"],
                "AUC":      res["auc"],
                "F1":       res["f1"],
            })
    df_plot = pd.DataFrame(rows)

    metrics = ["Accuracy", "AUC", "F1"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric in zip(axes, metrics):
        pivot = df_plot.pivot(index="Actif", columns="Modèle", values=metric)
        pivot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black", width=0.6)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, linestyle="--", color="red", alpha=0.4, linewidth=0.8)
        ax.tick_params(axis="x", rotation=0)
        ax.legend(fontsize=8)

    fig.suptitle("Comparaison des modèles — AF / TTE / RNO", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        _savefig(fig, "multi_comparative_summary.png")
    else:
        plt.show()


def plot_lags_heatmap(lags_by_asset: dict, save: bool = True):
    """
    Heatmap des lags optimaux retenus par actif et par feature.
    Permet de voir d'un coup d'œil si les dynamiques diffèrent entre actifs.
    """
    df_lags = pd.DataFrame(lags_by_asset).T  # actifs en lignes, features en colonnes
    df_lags = df_lags[CANDIDATE_FEATURES]    # ordre cohérent

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        df_lags.astype(int), annot=True, fmt="d",
        cmap="YlOrRd", linewidths=0.5, ax=ax,
    )
    ax.set_title("Lags optimaux par actif et par feature (jours)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Actif")
    plt.tight_layout()

    if save:
        _savefig(fig, "multi_lags_heatmap.png")
    else:
        plt.show()


# =============================================================================
# RÉSUMÉ CONSOLE
# =============================================================================

def print_full_summary(summary: dict):
    print_section("RÉSUMÉ COMPARATIF FINAL")
    header = f"{'Actif':<12} {'Modèle':<25} {'Accuracy':>10} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print(header)
    print("-" * 85)
    for asset, results in summary.items():
        for res in results:
            print(
                f"{asset:<12} {res['name']:<25} "
                f"{res['acc']:>10.4f} {res['auc']:>8.4f} "
                f"{res['prec']:>10.4f} {res['rec']:>8.4f} {res['f1']:>8.4f}"
            )
        print()


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-asset XGBoost experiment")
    parser.add_argument("--from-csv", type=str, default=None, metavar="PATH")
    parser.add_argument("--no-save",  action="store_true")
    parser.add_argument("--seuil",    type=float, default=0.4,
                        help="Seuil de décision XGBoost (défaut : 0.4)")
    return parser.parse_args()


def main():
    args = parse_args()
    save  = not args.no_save
    seuil = args.seuil

    # ------------------------------------------------------------------
    # Chargement
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

    # Split train/test sur le df brut (pour l'optimisation des lags)
    df.index = pd.to_datetime(df.index)
    train_df = df.loc[TRAIN_START:TRAIN_END]

    # ------------------------------------------------------------------
    # Boucle par actif
    # ------------------------------------------------------------------
    summary       = {}   # résultats évaluation
    lags_by_asset = {}   # lags retenus par actif

    for target in TARGETS:
        asset = target.replace("_Return", "")
        print_section(f"ACTIF : {asset}  ({target})")

        # 1. Optimisation des lags sur le train set uniquement
        print(f"\n  Optimisation des lags (max={MAX_LAG} jours)...")
        best_lags = optimize_lags_for_target(train_df, target, CANDIDATE_FEATURES, MAX_LAG)
        lags_by_asset[asset] = best_lags

        # 2. Construction des features
        df_model, feature_cols = build_features(df, best_lags)

        # 3. Split train / valid / test
        print()
        X_train, X_valid, X_test, y_train, y_valid, y_test = split(df_model, feature_cols, target)

        # 4. Entraînement sur train
        print(f"\n  Entraînement...")
        logit = train_logistic(X_train, y_train)
        rf    = train_random_forest(X_train, y_train)
        xgb   = train_xgboost(X_train, y_train)

        # 5a. Évaluation sur valid (tuning)
        print(f"\n  --- Validation set ---")
        res_logit_v = evaluate("Logistic Regression", logit, X_valid, y_valid, seuil=0.5)
        res_rf_v    = evaluate(f"Random Forest (s={seuil})", rf,  X_valid, y_valid, seuil=seuil)
        res_xgb_v   = evaluate(f"XGBoost (s={seuil})",      xgb,  X_valid, y_valid, seuil=seuil)
        print(f"  Logistic Regression  → Acc={res_logit_v['acc']:.4f}  AUC={res_logit_v['auc']:.4f}  F1={res_logit_v['f1']:.4f}")
        print(f"  Random Forest        → Acc={res_rf_v['acc']:.4f}  AUC={res_rf_v['auc']:.4f}  F1={res_rf_v['f1']:.4f}")
        print(f"  XGBoost              → Acc={res_xgb_v['acc']:.4f}  AUC={res_xgb_v['auc']:.4f}  F1={res_xgb_v['f1']:.4f}")

        # 5b. Évaluation finale sur test
        print(f"\n  --- Test set (évaluation finale) ---")
        res_logit = evaluate("Logistic Regression", logit, X_test, y_test, seuil=0.5)
        res_rf    = evaluate(f"Random Forest (s={seuil})", rf,   X_test, y_test, seuil=seuil)
        res_xgb   = evaluate(f"XGBoost (s={seuil})",      xgb,   X_test, y_test, seuil=seuil)
        print(f"  Logistic Regression  → Acc={res_logit['acc']:.4f}  AUC={res_logit['auc']:.4f}  F1={res_logit['f1']:.4f}")
        print(f"  Random Forest        → Acc={res_rf['acc']:.4f}  AUC={res_rf['auc']:.4f}  F1={res_rf['f1']:.4f}")
        print(f"  XGBoost              → Acc={res_xgb['acc']:.4f}  AUC={res_xgb['auc']:.4f}  F1={res_xgb['f1']:.4f}")

        summary[asset] = [res_logit, res_rf, res_xgb]

        # 6. Visualisations par actif (sur test set)
        plot_asset_roc(asset, [res_logit, res_rf, res_xgb], y_test, save)
        plot_asset_confusion(asset, [res_logit, res_rf, res_xgb], save)
        plot_asset_feature_importance(asset, xgb, rf, feature_cols, save)

        # 7. Cross-validation temporelle par actif (sur train uniquement)
        print(f"\n  Cross-validation temporelle ({asset})...")
        models_cv = {
            "Logistic Regression": logit,
            "Random Forest":       rf,
            "XGBoost":             xgb,
        }
        cross_validate_models(models_cv, X_train, y_train, cv=5)

    # ------------------------------------------------------------------
    # Visualisations comparatives
    # ------------------------------------------------------------------
    print_section("Figures comparatives")
    plot_comparative_summary(summary, save)
    plot_lags_heatmap(lags_by_asset, save)

    # ------------------------------------------------------------------
    # Résumé console
    # ------------------------------------------------------------------
    print_full_summary(summary)

    # Tableau des lags retenus
    print_section("LAGS OPTIMAUX PAR ACTIF")
    df_lags = pd.DataFrame(lags_by_asset).T
    print(df_lags[CANDIDATE_FEATURES].to_string())

    # ------------------------------------------------------------------
    # Sauvegarde des modèles
    # ------------------------------------------------------------------
    import joblib
    import json

    print_section("Sauvegarde des modèles")
    os.makedirs(MODELS_DIR, exist_ok=True)

    metadata = {}

    for target in TARGETS:
        asset      = target.replace("_Return", "")
        res_list   = summary[asset]           # [logit, rf, xgb]
        best_lags  = lags_by_asset[asset]

        # On sauvegarde les 3 modèles
        for res in res_list:
            safe_name = res["name"].replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(".", "")
            path = os.path.join(MODELS_DIR, f"{asset}_{safe_name}.pkl")
            joblib.dump(res["model"], path)
            print(f"  → {path}")

        # Métadonnées : lags + métriques du meilleur modèle (XGBoost)
        res_xgb = res_list[2]
        metadata[asset] = {
            "target":      target,
            "best_lags":   best_lags,
            "features":    [f"{feat}_lag{lag}" for feat, lag in best_lags.items()],
            "metrics": {
                "acc":  round(res_xgb["acc"],  4),
                "auc":  round(res_xgb["auc"],  4),
                "f1":   round(res_xgb["f1"],   4),
                "prec": round(res_xgb["prec"], 4),
                "rec":  round(res_xgb["rec"],  4),
            }
        }

    # Sauvegarde des métadonnées en JSON (lues par l'app Streamlit)
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  → Métadonnées : {meta_path}")

    print(f"\nFigures sauvegardées dans : {FIGURES_DIR}/")
    print("Script terminé.\n")


if __name__ == "__main__":
    main()
