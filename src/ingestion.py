# =============================================================================
# INGESTION.PY — Chargement et assemblage des données brutes
# =============================================================================

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    TICKERS, EURUSD_TICKER,
    GPR_CSV, CLI_CSV,
    START_DATE, END_DATE,
)


# -----------------------------------------------------------------------------
# 1. YAHOO FINANCE — Prix + Log-returns (AF, TTE, RNO, STOXX, Brent, JetFuel, CAC40, VIX)
# -----------------------------------------------------------------------------

def load_yfinance_assets() -> pd.DataFrame:
    """
    Télécharge les prix de clôture via yfinance, construit un calendrier
    journalier complet (ffill/bfill), calcule les log-returns et retourne
    un DataFrame avec colonnes {ASSET}_Price et {ASSET}_Return.
    """
    print("Téléchargement des données Yahoo Finance...")

    df_raw = yf.download(
        list(TICKERS.values()),
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )["Close"]

    reverse_tickers = {v: k for k, v in TICKERS.items()}
    df_raw.rename(columns=reverse_tickers, inplace=True)

    # Calendrier quotidien complet
    all_days = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    df_prices = df_raw.reindex(all_days).ffill().bfill()

    # Log-returns
    df_returns = np.log(df_prices / df_prices.shift(1)).dropna()

    # Assemblage côte-à-côte
    combined_cols = {}
    for asset in df_prices.columns:
        combined_cols[f"{asset}_Price"]  = df_prices[asset]
        combined_cols[f"{asset}_Return"] = df_returns[asset]

    df_combined = pd.DataFrame(combined_cols)
    print(f"  → {len(df_combined)} lignes, {len(df_combined.columns)} colonnes")
    return df_combined


# -----------------------------------------------------------------------------
# 2. EURUSD — Prix + Variation absolue
# -----------------------------------------------------------------------------

def load_eurusd(df_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Télécharge l'EUR/USD, aligne sur l'index de df_combined,
    calcule la variation absolue et ajoute EURUSD_Price / EURUSD_Abs_Change.
    """
    print("Téléchargement EUR/USD...")

    raw = yf.download(
        EURUSD_TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )

    df_eurusd = raw[["Close"]].copy()
    df_eurusd.columns = ["EURUSD_Price"]

    # Alignement sur l'index principal
    df_eurusd = df_eurusd.reindex(df_combined.index).ffill().bfill()
    df_eurusd["EURUSD_Abs_Change"] = df_eurusd["EURUSD_Price"].diff().fillna(0)

    df_combined["EURUSD_Price"]      = df_eurusd["EURUSD_Price"]
    df_combined["EURUSD_Abs_Change"] = df_eurusd["EURUSD_Abs_Change"]

    print("  → EUR/USD intégré")
    return df_combined



# -----------------------------------------------------------------------------
# 4. GPR — Indice de risque géopolitique (XLS Caldara & Iacoviello)
# -----------------------------------------------------------------------------

def load_gpr(df_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Télécharge le fichier XLS GPR journalier depuis matteoiacoviello.com.
    Source : data_gpr_daily_recent.xls
    Fallback : GPR_CSV local si le téléchargement échoue.

    Colonnes source : GPRA → GPRD_ACT / GPRT → GPRD_THREAT
    Calcule GPRD_Diff, GPRD_ACT_Diff, GPRD_THREAT_Diff et aligne sur df_combined.
    """
    import requests
    import io as _io

    URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

    def _parse_xls(content: bytes) -> pd.DataFrame:
        df = pd.read_excel(_io.BytesIO(content), engine="xlrd")
        df.columns = df.columns.str.strip()

        # La colonne date s'appelle DAY, format entier YYYYMMDD (ex: 19850101)
        # Les colonnes GPRD, GPRD_ACT, GPRD_THREAT sont déjà bien nommées
        df["DAY"] = pd.to_datetime(df["DAY"].astype(str), format="%Y%m%d", errors="coerce")

        # Supprime les lignes de métadonnées (DAY non parseable → NaT)
        df = df.dropna(subset=["DAY"]).set_index("DAY").sort_index()

        # Garde uniquement les colonnes utiles
        cols_to_keep = [c for c in ["GPRD", "GPRD_ACT", "GPRD_THREAT"] if c in df.columns]
        return df[cols_to_keep]

    # --- Téléchargement ---
    df_gpr = None
    try:
        print(f"  Téléchargement GPR depuis data_gpr_daily_recent.xls...")
        resp = requests.get(URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        df_gpr = _parse_xls(resp.content)
        print(f"  → GPR téléchargé ({len(df_gpr)} lignes, dernière date : {df_gpr.index[-1].date()})")
    except Exception as e:
        print(f"  ⚠ Téléchargement échoué : {e} — fallback local {GPR_CSV}")
        df_gpr = pd.read_csv(GPR_CSV)
        df_gpr["date"] = pd.to_datetime(df_gpr["date"])
        df_gpr = df_gpr.sort_values("date").set_index("date")
        if "GPRA" in df_gpr.columns:
            df_gpr = df_gpr.rename(columns={"GPRA": "GPRD_ACT", "GPRT": "GPRD_THREAT"})

    # --- Calcul des diffs ---
    df_gpr = df_gpr[df_gpr.index >= pd.Timestamp(START_DATE)]

    df_gpr_final = pd.DataFrame(index=df_gpr.index)
    for src, name in [("GPRD", "GPRD"), ("GPRD_ACT", "GPRD_ACT"), ("GPRD_THREAT", "GPRD_THREAT")]:
        if src in df_gpr.columns:
            series = pd.to_numeric(df_gpr[src], errors="coerce")
            df_gpr_final[f"{name}_Value"] = series
            df_gpr_final[f"{name}_Diff"]  = series.diff()

    df_gpr_final = df_gpr_final.fillna(0)

    # --- Alignement ---
    df_gpr_final = df_gpr_final.reindex(df_combined.index).ffill().bfill()
    df_combined  = df_combined.join(df_gpr_final, how="left")

    print("  → GPR intégré")
    return df_combined


# -----------------------------------------------------------------------------
# 5. CLI — Composite Leading Indicator OCDE (CSV OCDE)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 5. CLI — Composite Leading Indicator OCDE (G4E — Major 4 European)
# -----------------------------------------------------------------------------

def load_cli(df_combined: pd.DataFrame) -> pd.DataFrame:
    """
    Charge le fichier CLI OCDE (format SDMX/CSV complet).
    Série retenue : G4E / LI / Mensuel / Indice (IX) / Corrigé amplitude (AA).
    Construit une série journalière par forward-fill et aligne sur df_combined.
    """
    print(f"Chargement CLI depuis {CLI_CSV}...")

    df_raw = pd.read_csv(CLI_CSV)

    mask = (
        (df_raw["REF_AREA"]       == "G4E") &
        (df_raw["MEASURE"]        == "LI")  &
        (df_raw["FREQ"]           == "M")   &
        (df_raw["TRANSFORMATION"] == "IX")  &
        (df_raw["ADJUSTMENT"]     == "AA")
    )

    df_cli = df_raw[mask][["TIME_PERIOD", "OBS_VALUE"]].copy()
    df_cli.columns = ["date", "CLI"]
    df_cli["date"] = pd.to_datetime(df_cli["date"])
    df_cli["CLI"]  = pd.to_numeric(df_cli["CLI"], errors="coerce")
    df_cli = (
        df_cli
        .dropna()
        .sort_values("date")
        .drop_duplicates(subset="date")
        .set_index("date")
    )
    df_cli["CLI_Diff"] = df_cli["CLI"].diff().fillna(0)

    print(f"  → CLI chargé ({len(df_cli)} mois, dernière date : {df_cli.index[-1].date()})")

    # Expansion en fréquence journalière
    daily_index = pd.date_range(
        start = df_cli.index.min(),
        end   = df_cli.index.max(),
        freq  = "D",
    )
    df_daily = pd.DataFrame(index=daily_index)
    df_daily["CLI"]      = df_cli["CLI"].reindex(df_daily.index).ffill()
    df_daily["CLI_Diff"] = 0.0
    df_daily.loc[df_cli.index, "CLI_Diff"] = df_cli["CLI_Diff"].values

    # Alignement sur df_combined
    df_cli_daily = df_daily.loc[START_DATE:].reindex(df_combined.index)
    df_cli_daily["CLI"]      = df_cli_daily["CLI"].ffill()
    df_cli_daily["CLI_Diff"] = df_cli_daily["CLI_Diff"].fillna(0)

    df_combined["CLI"]      = df_cli_daily["CLI"]
    df_combined["CLI_Diff"] = df_cli_daily["CLI_Diff"]

    print("  → CLI intégré")
    return df_combined


# -----------------------------------------------------------------------------
# PIPELINE COMPLET
# -----------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    """
    Orchestre l'ensemble des chargements et retourne df_combined prêt à l'emploi,
    filtré sur la fenêtre d'analyse et sans valeurs manquantes.
    """
    from src.config import ANALYSIS_START, ANALYSIS_END

    df = load_yfinance_assets()
    df = load_eurusd(df)
    df = load_gpr(df)
    df = load_cli(df)

    # Fenêtre retenue
    df = df.loc[ANALYSIS_START:ANALYSIS_END]
    df = df.fillna(0)

    print(f"\nDataset final : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df