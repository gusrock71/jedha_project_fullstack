# =============================================================================
# MODELING.PY — Entraînement et évaluation des modèles
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from src.config import FIGURES_DIR, MODELS_DIR


def _savefig(fig, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  → Figure sauvegardée : {path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# RÉGRESSION LOGISTIQUE (sklearn Pipeline)
# -----------------------------------------------------------------------------

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Entraîne une régression logistique avec StandardScaler.
    Retourne le pipeline sklearn ajusté.
    """
    logit = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            random_state=42,
            class_weight="balanced",
            max_iter=5000,
        )),
    ])
    logit.fit(X_train, y_train)
    print("Modèle entraîné (LogisticRegression).")
    return logit


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Évalue le modèle sur l'ensemble de test.
    Affiche accuracy, matrice de confusion et rapport de classification.
    Retourne un dict avec les métriques.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred)

    print(f"\nAccuracy : {acc:.4f}")
    print("\nConfusion Matrix :")
    print(cm)
    print("\nClassification Report :")
    print(cr)

    return {"accuracy": acc, "confusion_matrix": cm, "classification_report": cr}


# -----------------------------------------------------------------------------
# COEFFICIENTS — Tableau + graphique
# -----------------------------------------------------------------------------

def plot_coefficients(
    model: Pipeline,
    feature_cols: list,
    save: bool = True,
) -> pd.DataFrame:
    """
    Extrait les coefficients du modèle logistique, les trie par valeur absolue
    et les visualise sous forme de barh chart.
    """
    coef_df = pd.DataFrame({
        "Feature":     feature_cols,
        "Coefficient": model.named_steps["clf"].coef_[0],
    })
    coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs_Coefficient", ascending=False)

    print("\nCoefficients :")
    print(coef_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(coef_df["Feature"], coef_df["Coefficient"])
    ax.axvline(x=0, linestyle="--")
    ax.set_title("Logistic Regression — Coefficients")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    if save:
        _savefig(fig, "logit_coefficients.png")
    else:
        plt.show()

    return coef_df


# -----------------------------------------------------------------------------
# STATSMODELS — Régression logistique avec p-values
# -----------------------------------------------------------------------------

def run_statsmodels_logit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """
    Ajuste une régression logistique statsmodels sur X_train (standardisé)
    et affiche le summary complet avec p-values.
    Ignore les colonnes constantes qui rendraient la matrice singulière.
    """
    # Supprime les colonnes constantes (variance nulle)
    X_valid = X_train.loc[:, X_train.std() > 0]
    dropped = set(X_train.columns) - set(X_valid.columns)
    if dropped:
        print(f"  ⚠ Colonnes constantes ignorées par statsmodels : {dropped}")

    # Standardisation en conservant les noms de colonnes
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_valid),
        columns = X_valid.columns,
        index   = X_valid.index,
    )
    X_sm = sm.add_constant(X_scaled)

    try:
        logit_sm = sm.Logit(y_train, X_sm)
        results  = logit_sm.fit(disp=False)
        print("\n=== Statsmodels Logit Summary ===")
        print(results.summary())
        return results
    except Exception as e:
        print(f"  ⚠ Statsmodels Logit échoué : {e}")
        return None


# -----------------------------------------------------------------------------
# PIPELINE MODÉLISATION COMPLET
# -----------------------------------------------------------------------------

def run_modeling(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    y_test:  pd.Series,
    feature_cols: list,
    save: bool = True,
) -> dict:
    """
    Orchestre entraînement, évaluation, visualisation des coefficients
    et summary statsmodels.
    """
    print("\n=== MODÉLISATION ===")

    model   = train_logistic_regression(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    coef_df = plot_coefficients(model, feature_cols, save)
    sm_results = run_statsmodels_logit(X_train, y_train)

    print("=== Modélisation terminée ===\n")

    return {
        "model":      model,
        "metrics":    metrics,
        "coef_df":    coef_df,
        "sm_results": sm_results,
    }