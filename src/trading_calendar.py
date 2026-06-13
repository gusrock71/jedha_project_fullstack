"""
src/trading_calendar.py — Calendrier de cotation Euronext Paris
=================================================================
Fournit le calendrier des jours de cotation réels d'Euronext Paris
(basé sur AF.PA, Yahoo Finance) et une fonction pour filtrer un
DataFrame sur ces jours.

Pourquoi ce module ?
---------------------
`df_combined` est construit en assemblant plusieurs sources cotées sur
des places différentes (Paris pour AF/TTE/RNO, Eurostoxx, VIX US,
Brent...). L'index résultant peut donc contenir des dates où Euronext
Paris est fermée (jours fériés français) mais où un autre marché a
coté — ces lignes contiennent alors des rendements AF/TTE/RNO nuls ou
ré-échantillonnés (ffill) qui ne reflètent aucune réalité de marché et
introduisent du bruit dans l'entraînement.

Ce module centralise le filtrage afin qu'il soit appliqué de manière
identique par tous les scripts du pipeline (main.py,
multi_asset_experiment.py, xgboost_experiment.py, etc.), garantissant
la cohérence entre les données d'entraînement et les données utilisées
pour l'évaluation / l'importance des features.
"""

import pandas as pd
import yfinance as yf

REF_TICKER = "AF.PA"


def get_trading_days(start: str, end: str, ref_ticker: str = REF_TICKER) -> pd.DatetimeIndex:
    """
    Retourne l'index des jours de cotation réels (Euronext Paris) en se
    basant sur le calendrier de cotation Yahoo Finance d'un ticker de
    référence cotée à Paris (AF.PA par défaut).
    """
    raw = yf.download(ref_ticker, start=start, end=end, auto_adjust=True, progress=False)
    return pd.to_datetime(raw.index).normalize()


def filter_to_trading_days(
    df: pd.DataFrame,
    ref_ticker: str = REF_TICKER,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filtre `df` aux dates où Euronext Paris était ouverte (calendrier
    de référence : `ref_ticker`).

    Retire :
      - les week-ends (si présents dans df)
      - les jours fériés français pendant lesquels d'autres marchés
        (STOXX, VIX, Brent...) ont coté

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame indexé par date (tout format convertible en datetime).
    ref_ticker : str
        Ticker de référence pour le calendrier de cotation.
    verbose : bool
        Affiche un résumé du nombre de lignes retirées.

    Retour
    ------
    pd.DataFrame filtré (copie), index normalisé en datetime sans heure.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()

    trading_days = get_trading_days(
        start=df.index.min().strftime("%Y-%m-%d"),
        end=(df.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        ref_ticker=ref_ticker,
    )

    df_filtered = df.loc[df.index.isin(trading_days)]

    if verbose:
        n_removed = len(df) - len(df_filtered)
        print(
            f"  [trading_calendar] Filtrage Euronext Paris ({ref_ticker}) : "
            f"{len(df)} → {len(df_filtered)} lignes ({n_removed:+d})"
        )

    return df_filtered
