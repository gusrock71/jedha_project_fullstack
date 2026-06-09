# =============================================================================
# CONFIG.PY — Paramètres centralisés du projet
# =============================================================================

# ------------------------------------------------------------------------------
# PERIODE DE DONNEES
# ------------------------------------------------------------------------------
START_DATE = "2007-07-30"
END_DATE   = "2026-04-01"

# Fenêtre d'analyse retenue après nettoyage
ANALYSIS_START = "2007-07-30"
ANALYSIS_END   = "2026-04-01"

# ------------------------------------------------------------------------------
# SPLIT TRAIN / VALID / TEST
# ------------------------------------------------------------------------------
TRAIN_START = "2007-07-30"
TRAIN_END   = "2020-12-31"   # inclut : crise 2008, dette euro 2011, COVID 2020

VALID_START = "2021-01-01"
VALID_END   = "2022-12-31"   # inclut : rebond post-COVID, guerre Ukraine

TEST_START  = "2023-01-01"
TEST_END    = "2026-04-01"   # inclut : Israël-Hamas, régime taux élevés

# ------------------------------------------------------------------------------
# TICKERS YAHOO FINANCE
# ------------------------------------------------------------------------------
TICKERS = {
    "AF":      "AF.PA",
    "TTE":     "TTE.PA",
    "RNO":     "RNO.PA",
    "STOXX":   "^STOXX50E",
    "Brent":   "BZ=F",
    "JetFuel": "HO=F",
    "CAC40":   "^FCHI",
    "VIX":     "^VIX",
}

EURUSD_TICKER = "EURUSD=X"

# ------------------------------------------------------------------------------
# FICHIERS CSV EXTERNES (à placer dans data/)
# ------------------------------------------------------------------------------
GPR_CSV  = "data/raw/data_gpr_daily_recent_.csv"
CLI_CSV  = "data/raw/OECD_SDD_STES_DSD_STES_DF_CLI__all.csv"

# ------------------------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------------------------
COMBINED_CSV   = "outputs/df_combined.csv"
FIGURES_DIR    = "outputs/figures"
MODELS_DIR     = "outputs/models"

# ------------------------------------------------------------------------------
# MODELE — Features et lags sélectionnés
# ------------------------------------------------------------------------------
TARGET = "AF_Return"

SELECTED_LAGS = {
    "GPRD_ACT_Diff":    19,
    "GPRD_THREAT_Diff": 20,
    "CLI_Diff":         16,
    "STOXX_Return":      0,
    "VIX_Return":        0,
    "Brent_Return":      0,
}

# Predicteurs pour le test de Granger
GRANGER_PREDICTORS = [
    "GPRD_Diff",
    "GPRD_ACT_Diff",
    "GPRD_THREAT_Diff",
    "CLI_Diff",
    "STOXX_Return",
    "Brent_Return",
    "VIX_Return",
]

GRANGER_MAXLAG = 30

# Colonnes de corrélation Spearman
CORR_COLS = [
    "AF_Return",
    "RNO_Return",
    "TTE_Return",
    "CAC40_Return",
    "STOXX_Return",
    "Brent_Return",
    "JetFuel_Return",
    "EURUSD_Abs_Change",
    "VIX_Return",
    "GPRD_Diff",
    "GPRD_ACT_Diff",
    "GPRD_THREAT_Diff",
    "CLI_Diff",
]
