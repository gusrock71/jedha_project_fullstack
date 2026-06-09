# =============================================================================
# FEATURES.PY — Feature engineering : sélection et construction des lags
# =============================================================================

import pandas as pd
from scipy.stats import spearmanr

from src.config import (
    TARGET,
    SELECTED_LAGS,
    TRAIN_START, TRAIN_END,
    VALID_START, VALID_END,
    TEST_START,  TEST_END,
)


def optimize_lags(
    train_df: pd.DataFrame,
    target: str,
    candidate_features: list,
    max_lag: int = 30,
) -> dict:
    """
    Recherche le lag optimal (corrélation de Spearman maximale en valeur absolue)
    pour chaque feature candidate sur l'ensemble d'entraînement.

    Retourne un dict {feature: {"lag": int, "corr": float}}.
    """
    best_lags = {}

    for feature in candidate_features:
        best_corr = 0
        best_lag  = 0

        for lag in range(max_lag + 1):
            corr_df = pd.DataFrame({
                "target":  train_df[target],
                "feature": train_df[feature].shift(lag),
            }).dropna()

            corr, _ = spearmanr(corr_df["target"], corr_df["feature"])

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag  = lag

        best_lags[feature] = {"lag": best_lag, "corr": best_corr}
        print(f"  {feature:30s} → lag={best_lag:2d}  corr={best_corr:+.4f}")

    return best_lags


def build_lag_features(
    df: pd.DataFrame,
    selected_lags: dict | None = None,
) -> tuple[pd.DataFrame, list]:
    """
    Construit les colonnes laggées dans df_model et retourne
    (df_model, feature_cols).

    selected_lags : dict {feature_name: lag_int}
                    Par défaut, utilise SELECTED_LAGS depuis config.
    """
    if selected_lags is None:
        selected_lags = SELECTED_LAGS

    df_model     = df.copy()
    feature_cols = []

    for feature, lag in selected_lags.items():
        new_col = f"{feature}_lag{lag}"
        df_model[new_col] = df_model[feature].shift(lag)
        feature_cols.append(new_col)

    df_model = df_model.dropna()
    return df_model, feature_cols


def split_train_valid_test(
    df_model: pd.DataFrame,
    feature_cols: list,
    target: str = TARGET,
) -> tuple:
    """
    Découpe df_model en (X_train, X_valid, X_test, y_train, y_valid, y_test)
    selon les dates définies dans config.

    Train : 2007-07-30 → 2020-12-31  (crise 2008, dette euro, COVID)
    Valid : 2021-01-01 → 2022-12-31  (rebond post-COVID, Ukraine)
    Test  : 2023-01-01 → 2026-04-01  (évaluation finale — ne toucher qu'une fois)

    La cible est binaire : 1 si le return est positif, 0 sinon.
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

    print(f"Train : {X_train.shape}  |  Valid : {X_valid.shape}  |  Test : {X_test.shape}")
    print(f"Taux positif — train : {y_train.mean():.2%}  |  valid : {y_valid.mean():.2%}  |  test : {y_test.mean():.2%}")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def split_train_test(
    df_model: pd.DataFrame,
    feature_cols: list,
    target: str = TARGET,
) -> tuple:
    """
    Découpe simplifiée train / test (train+valid fusionnés).
    Utilisée par main.py et modeling.py.
    """
    df_model.index = pd.to_datetime(df_model.index)

    train_df = df_model.loc[TRAIN_START:VALID_END]   # train + valid fusionnés
    test_df  = df_model.loc[TEST_START:TEST_END]

    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]

    y_train = (train_df[target] > 0).astype(int)
    y_test  = (test_df[target]  > 0).astype(int)

    print(f"Train : {X_train.shape}  |  Test : {X_test.shape}")
    print(f"Taux positif — train : {y_train.mean():.2%}  |  test : {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test
