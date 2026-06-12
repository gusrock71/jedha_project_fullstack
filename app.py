"""
app.py — Interface Streamlit : Prédiction de direction de marché
=================================================================
Lance en local :
    streamlit run app.py

Déploiement Hugging Face Spaces :
    Uploader app.py + outputs/models/ + requirements_streamlit.txt
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# =============================================================================
# CONFIG
# =============================================================================

MODELS_DIR   = "outputs/models"
META_PATH    = os.path.join(MODELS_DIR, "metadata.json")
LOGO_PATH    = "assets/jedha_logo.png"

ASSETS = {
    "Air France (AF)":        "AF",
    "TotalEnergies (TTE)":    "TTE",
    "Renault (RNO)":          "RNO",
}

# Meilleur modèle retenu par actif selon les résultats d'évaluation
# AF  → XGBoost  (AUC 0.822, F1 0.701)
# TTE → XGBoost  (AUC 0.869, F1 0.749)
# RNO → Random Forest (AUC 0.884, F1 0.761)
BEST_MODEL_BY_ASSET = {
    "AF":  {"label": "Random Forest", "key": "Random_Forest_s04", "auc": 0.822, "f1": 0.684},
    "TTE": {"label": "Random Forest", "key": "Random_Forest_s04", "auc": 0.855, "f1": 0.735},
    "RNO": {"label": "Random Forest", "key": "Random_Forest_s04", "auc": 0.855, "f1": 0.725},
}

# Mapping features → tickers Yahoo Finance pour données live
FEATURE_TICKERS = {
    "STOXX_Return":     "^STOXX50E",
    "VIX_Return":       "^VIX",
    "Brent_Return":     "BZ=F",
    "GPRD_ACT_Diff":    None,   # pas disponible live
    "GPRD_THREAT_Diff": None,
    "CLI_Diff":         None,
}

TICKERS_LIVE = {
    "AF.PA":    "AF",
    "TTE.PA":   "TTE",
    "RNO.PA":   "RNO",
    "^STOXX50E":"STOXX",
    "BZ=F":     "Brent",
    "^VIX":     "VIX",
}

# =============================================================================
# CHARGEMENT
# =============================================================================

def render_logo():
    """Affiche le logo Jedha en haut de page."""
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=200)


@st.cache_resource
def load_metadata():
    if not os.path.exists(META_PATH):
        return None
    with open(META_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_model(asset: str, model_type_key: str):
    """Charge un modèle .pkl depuis outputs/models/."""
    pattern = f"{asset}_{model_type_key}"
    for fname in os.listdir(MODELS_DIR):
        if fname.startswith(pattern) and fname.endswith(".pkl"):
            return joblib.load(os.path.join(MODELS_DIR, fname))
    return None


@st.cache_data(ttl=3600)
def load_live_prices(days: int = 60):
    """Télécharge les prix récents via yfinance."""
    end   = datetime.today()
    start = end - timedelta(days=days)

    tickers = list(TICKERS_LIVE.keys())
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    raw.rename(columns=TICKERS_LIVE, inplace=True)

    # Comble les trous ponctuels (ex: jour férié sur un seul marché)
    raw = raw.ffill().bfill()

    # Log-returns — dropna uniquement sur les lignes entièrement vides
    returns = np.log(raw / raw.shift(1)).dropna(how="all")
    return raw, returns


@st.cache_data(ttl=86400)   # cache 24h — données mises à jour mensuellement
def load_gpr_live() -> pd.DataFrame | None:
    """
    Charge le GPR journalier depuis matteoiacoviello.com :
      1. data_gpr_daily_recent.xls  (journalier — référence)
      2. data_gpr_export.xls        (mensuel — fallback distant)
    Colonnes source : GPRA → GPRD_ACT / GPRT → GPRD_THREAT.
    Retourne None si les deux téléchargements échouent (features GPR fixées à 0).
    """
    import requests, io

    URLS = [
        "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls",
        "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls",
    ]

    def parse_xls(content: bytes) -> pd.DataFrame:
        df = pd.read_excel(io.BytesIO(content), engine="xlrd")
        df.columns = df.columns.str.strip()
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        rename = {}
        if "GPRA" in df.columns:
            rename["GPRA"] = "GPRD_ACT"
        if "GPRT" in df.columns:
            rename["GPRT"] = "GPRD_THREAT"
        df = df.rename(columns=rename)
        df["GPRD_ACT"]    = pd.to_numeric(df.get("GPRD_ACT"),    errors="coerce")
        df["GPRD_THREAT"] = pd.to_numeric(df.get("GPRD_THREAT"), errors="coerce")
        df["GPRD_ACT_Diff"]    = df["GPRD_ACT"].diff().fillna(0)
        df["GPRD_THREAT_Diff"] = df["GPRD_THREAT"].diff().fillna(0)
        return df[["GPRD_ACT_Diff", "GPRD_THREAT_Diff"]]

    for url in URLS:
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            return parse_xls(resp.content)
        except Exception:
            continue

    return None


@st.cache_data(ttl=86400)   # cache 24h — données mensuelles
def load_cli_live() -> dict | None:
    """
    Récupère automatiquement les deux dernières valeurs CLI (G4E, mensuel,
    indice IX, ajustement AA) via l'API SDMX de l'OCDE.

    Retourne {"prev": float, "curr": float, "diff": float,
              "prev_date": str, "curr_date": str} ou None si échec.
    """
    URL = (
        "https://sdmx.oecd.org/public/rest/data/"
        "OECD.SDD.STES,DSD_STES@DF_CLI/G4E.M.LI.._Z.AA.IX._Z.H"
        "?format=csvfilewithlabels"
    )

    try:
        import requests, io
        resp = requests.get(URL, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        df = pd.read_csv(io.StringIO(resp.text))
        df = df[["TIME_PERIOD", "OBS_VALUE"]].dropna()
        df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
        df = df.sort_values("TIME_PERIOD")

        if len(df) < 2:
            return None

        prev_row = df.iloc[-2]
        curr_row = df.iloc[-1]

        return {
            "prev":      float(prev_row["OBS_VALUE"]),
            "curr":      float(curr_row["OBS_VALUE"]),
            "diff":      round(float(curr_row["OBS_VALUE"]) - float(prev_row["OBS_VALUE"]), 6),
            "prev_date": prev_row["TIME_PERIOD"].strftime("%Y-%m"),
            "curr_date": curr_row["TIME_PERIOD"].strftime("%Y-%m"),
        }
    except Exception:
        return None


# =============================================================================
# FEATURE BUILDING
# =============================================================================

def build_live_features(
    returns: pd.DataFrame,
    best_lags: dict,
    df_gpr: pd.DataFrame | None = None,
    cli_diff: float = 0.0,
) -> pd.DataFrame:
    """
    Construit les features laggées à partir des données live.
    - STOXX, VIX, Brent  → Yahoo Finance (temps réel)
    - GPRD_ACT, GPRD_THREAT → matteoiacoviello.com si disponible, sinon 0
    - CLI_Diff → valeur saisie manuellement par l'administrateur (défaut 0)
    """
    feature_map = {
        "STOXX_Return":     "STOXX",
        "VIX_Return":       "VIX",
        "Brent_Return":     "Brent",
        "GPRD_ACT_Diff":    None,
        "GPRD_THREAT_Diff": None,
        "CLI_Diff":         None,
    }

    df = pd.DataFrame(index=returns.index)

    for feat, lag in best_lags.items():
        col_name = f"{feat}_lag{lag}"
        src_col  = feature_map.get(feat)

        if src_col and src_col in returns.columns:
            # Feature Yahoo Finance
            df[col_name] = returns[src_col].shift(lag)

        elif feat in ("GPRD_ACT_Diff", "GPRD_THREAT_Diff") and df_gpr is not None:
            # Feature GPR live
            gpr_series = df_gpr[feat].reindex(returns.index, method="ffill").fillna(0)
            df[col_name] = gpr_series.shift(lag)

        elif feat == "CLI_Diff":
            # CLI saisi manuellement — valeur propagée sur tout l'index avec lag
            df[col_name] = cli_diff

        else:
            df[col_name] = 0.0

    return df.dropna()


# =============================================================================
# CHARGEMENT COMMUN (utilisé par les deux pages)
# =============================================================================

def load_common_data(asset: str, seuil: float, cli_diff: float = 0.0):
    """
    Charge modèle, métadonnées, données live et construit les features.
    Retourne un dict avec tout le nécessaire pour l'affichage, ou None si erreur.
    """
    metadata = load_metadata()
    if metadata is None:
        st.error(
            "⚠️ Fichier `outputs/models/metadata.json` introuvable. "
            "Lance d'abord `python3 multi_asset_experiment.py` pour générer les modèles."
        )
        return None

    best_model_info = BEST_MODEL_BY_ASSET[asset]
    model_label     = best_model_info["label"]
    model_key       = best_model_info["key"]

    model = load_model(asset, model_key)
    if model is None:
        st.error(f"⚠️ Modèle introuvable pour {asset} / {model_label}. Vérifiez `outputs/models/`.")
        return None

    meta      = metadata[asset]
    best_lags = meta["best_lags"]

    with st.spinner("Chargement des données de marché..."):
        try:
            prices, returns = load_live_prices(days=90)
            data_ok = True
        except Exception:
            data_ok = False
            prices, returns = None, None

    with st.spinner("Chargement GPR (Caldara & Iacoviello)..."):
        df_gpr = load_gpr_live()
    gpr_ok = df_gpr is not None

    if not data_ok:
        st.warning("Impossible de télécharger les données de marché en direct.")
        return None

    features_df = build_live_features(returns, best_lags, df_gpr=df_gpr, cli_diff=cli_diff)
    if len(features_df) == 0:
        st.error("Pas assez de données pour calculer les features.")

        with st.expander("🔍 Diagnostic"):
            st.write("**Shape `returns`** :", returns.shape if returns is not None else None)
            st.write("**Colonnes `returns`** :", list(returns.columns) if returns is not None else None)
            if returns is not None:
                st.write("**Valeurs manquantes par colonne** :")
                st.write(returns.isna().sum())
                st.write("**Dernières lignes de `returns`** :")
                st.dataframe(returns.tail(10))
            st.write("**best_lags** :", best_lags)
            st.write("**GPR disponible** :", gpr_ok)

        return None

    feature_cols = list(features_df.columns)
    last_row     = features_df.iloc[[-1]][feature_cols]
    proba        = model.predict_proba(last_row)[0][1]
    signal       = "HAUSSE" if proba >= seuil else "BAISSE"
    last_date    = features_df.index[-1].strftime("%d/%m/%Y")
    next_date    = (features_df.index[-1] + timedelta(days=1)).strftime("%d/%m/%Y")

    return {
        "metadata":     metadata,
        "meta":         meta,
        "best_lags":    best_lags,
        "model":        model,
        "model_label":  model_label,
        "prices":       prices,
        "returns":      returns,
        "features_df":  features_df,
        "feature_cols": feature_cols,
        "proba":        proba,
        "signal":       signal,
        "last_date":    last_date,
        "next_date":    next_date,
        "gpr_ok":       gpr_ok,
    }


def render_price_probability_chart(asset: str, asset_label: str, data: dict, seuil: float):
    """Graphique combiné prix + probabilités sur 30 jours (axe double)."""
    model        = data["model"]
    features_df  = data["features_df"]
    feature_cols = data["feature_cols"]
    prices       = data["prices"]

    asset_ticker_map = {"AF": "AF", "TTE": "TTE", "RNO": "RNO"}
    yf_col = asset_ticker_map.get(asset, asset)

    # Calcul des probabilités sur les 30 derniers jours
    n_days = min(30, len(features_df))
    probas = []
    for i in range(n_days):
        idx = -(n_days - i)
        row = features_df.iloc[idx][feature_cols]
        p   = model.predict_proba(row.values.reshape(1, -1))[0][1]
        probas.append({"date": features_df.index[idx], "proba": p})
    df_probas = pd.DataFrame(probas).set_index("date")

    if yf_col not in prices.columns:
        st.warning("Données de prix indisponibles pour cet actif.")
        return df_probas

    prix_30 = prices[yf_col].reindex(df_probas.index, method="ffill").dropna()
    dates   = prix_30.index

    fig, ax1 = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#1e2530")
    ax1.set_facecolor("#1e2530")

    # Axe gauche — Prix
    ax1.plot(dates, prix_30.values, color="#4fc3f7",
             linewidth=2, label=f"Prix {asset}", zorder=3)
    ax1.fill_between(dates, prix_30.values, prix_30.min(),
                     alpha=0.08, color="#4fc3f7")
    ax1.set_ylabel("Prix (€)", color="#4fc3f7", fontsize=11)
    ax1.tick_params(axis="y", colors="#4fc3f7")
    ax1.tick_params(axis="x", colors="white")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=30)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333")

    # Axe droit — Probabilités
    ax2 = ax1.twinx()
    probas_aligned = df_probas["proba"].reindex(dates, method="ffill")
    colors_bar = ["#4caf50" if p >= seuil else "#ef5350"
                  for p in probas_aligned.values]

    ax2.bar(dates, probas_aligned.values, color=colors_bar,
            alpha=0.45, width=0.8, label="Proba hausse", zorder=2)
    ax2.axhline(seuil, color="white", linestyle="--",
                linewidth=1.2, label=f"Seuil ({seuil:.0%})", zorder=4)
    ax2.set_ylabel("Probabilité de hausse", color="white", fontsize=11)
    ax2.tick_params(axis="y", colors="white")
    ax2.set_ylim(0, 1.4)

    # Légende unifiée
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#1e2530", labelcolor="white",
               loc="upper left", fontsize=9)

    ax1.set_title(
        f"{asset_label} — Prix de clôture & Probabilité de hausse (30 derniers jours)",
        fontsize=12, color="white", pad=12,
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    return df_probas


# =============================================================================
# PAGE 1 — GRAND PUBLIC
# =============================================================================

def render_public_page():
    render_logo()
    st.title("📈 Tendance du marché")
    st.caption("Prédiction de la tendance du lendemain pour Air France, TotalEnergies et Renault")

    # Sélecteur d'actif simple, pas de sidebar
    asset_label = st.selectbox("Choisissez un actif", list(ASSETS.keys()))
    asset = ASSETS[asset_label]

    seuil = 0.40  # seuil par défaut, non modifiable côté public

    # CLI récupéré automatiquement (silencieux côté grand public)
    cli_live = load_cli_live()
    cli_diff = cli_live["diff"] if cli_live is not None else 0.0

    data = load_common_data(asset, seuil, cli_diff=cli_diff)
    if data is None:
        return

    st.divider()

    # ------------------------------------------------------------------
    # SIGNAL DU JOUR — gros affichage central
    # ------------------------------------------------------------------
    signal    = data["signal"]
    proba     = data["proba"]
    last_date = data["last_date"]
    next_date = data["next_date"]

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown(f"<p style='text-align:center; color:#999; margin-bottom:0;'>Tendance prévue pour le {next_date} (basée sur les données du {last_date})</p>", unsafe_allow_html=True)

        if signal == "HAUSSE":
            st.markdown(
                f"<p style='text-align:center; font-size:4rem; font-weight:bold; "
                f"color:#4caf50; margin:0.5rem 0;'>▲ HAUSSE</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='text-align:center; font-size:4rem; font-weight:bold; "
                f"color:#ef5350; margin:0.5rem 0;'>▼ BAISSE</p>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<p style='text-align:center; color:#ccc; font-size:1.1rem;'>"
            f"Probabilité de hausse estimée : <b>{proba:.0%}</b></p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ------------------------------------------------------------------
    # GRAPHIQUE PRIX + PROBABILITÉS
    # ------------------------------------------------------------------
    render_price_probability_chart(asset, asset_label, data, seuil)

    df_probas = data["features_df"]  # juste pour compatibilité, non utilisé ici

    st.divider()
    st.caption(
        "⚠️ **Avertissement** : Ces prédictions sont issues d'un modèle académique "
        "et ne constituent pas des conseils en investissement."
    )


# =============================================================================
# PAGE 2 — PARAMÈTRES (toutes les fonctions avancées)
# =============================================================================

def render_settings_page():
    render_logo()
    st.title("⚙️ Paramètres avancés")

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    st.sidebar.title("⚙️ Paramètres")

    asset_label = st.sidebar.selectbox("Actif", list(ASSETS.keys()))
    asset       = ASSETS[asset_label]

    best_model_info = BEST_MODEL_BY_ASSET[asset]
    model_label     = best_model_info["label"]

    st.sidebar.markdown(f"**Modèle retenu** : {model_label}")
    st.sidebar.caption("Sélection basée sur AUC et F1 out-of-sample")

    seuil = st.sidebar.slider(
        "Seuil de décision",
        min_value = 0.30,
        max_value = 0.70,
        value     = 0.40,
        step      = 0.05,
        help      = "Probabilité minimale pour déclencher un signal HAUSSE"
    )

    st.sidebar.markdown("---")

    # CLI — récupération automatique OCDE (silencieux, non affiché)
    cli_live = load_cli_live()
    cli_diff = cli_live["diff"] if cli_live is not None else 0.0

    st.sidebar.markdown("---")

    # Tableau récapitulatif des meilleurs modèles
    st.sidebar.markdown("**Meilleurs modèles par actif**")
    df_best = pd.DataFrame([
        {"Actif": k, "Modèle": v["label"], "AUC": v["auc"], "F1": v["f1"]}
        for k, v in BEST_MODEL_BY_ASSET.items()
    ])
    st.sidebar.dataframe(df_best, hide_index=True, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**À propos**")
    st.sidebar.markdown(
        "Projet Jedha — Prédiction des probabilité de hausse de certaines "
        "valeurs boursières du CAC40 (AF, TTE, RNO) en fonction des "
        "variations d'indices macroéconomiques, financiers et géopolitiques."
    )

    # ------------------------------------------------------------------
    # Chargement des données
    # ------------------------------------------------------------------
    data = load_common_data(asset, seuil, cli_diff=cli_diff)
    if data is None:
        return

    meta         = data["meta"]
    best_lags    = data["best_lags"]
    features_df  = data["features_df"]
    feature_cols = data["feature_cols"]
    proba        = data["proba"]
    signal       = data["signal"]
    last_date    = data["last_date"]
    gpr_ok       = data["gpr_ok"]

    # ------------------------------------------------------------------
    # Statut des sources de données
    # ------------------------------------------------------------------
    sources = []
    sources.append("✅ Yahoo Finance (STOXX, VIX, Brent)")
    sources.append("✅ GPR live — data_gpr_daily_recent.xls (Caldara & Iacoviello)" if gpr_ok else "⚠️ GPR indisponible → fixé à 0")
    sources.append(f"✅ CLI (diff = {cli_diff:+.4f})" if cli_diff != 0.0 else "⚠️ CLI_Diff = 0")
    st.caption(f"Actif : **{asset_label}**  |  Modèle : **{model_label}**  |  Seuil : **{seuil}**  |  " + "  |  ".join(sources))
    st.divider()

    # ------------------------------------------------------------------
    # SIGNAL DU JOUR + MÉTRIQUES + FEATURES
    # ------------------------------------------------------------------
    col_signal, col_metrics, col_model_info = st.columns([1.5, 1.5, 2])

    with col_signal:
        st.subheader("Signal du jour")
        st.caption(f"Basé sur les données du {last_date}")

        if signal == "HAUSSE":
            st.markdown(f'<p class="signal-up">▲ {signal}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="signal-down">▼ {signal}</p>', unsafe_allow_html=True)

        st.metric("Probabilité de hausse", f"{proba:.1%}")
        st.metric("Seuil appliqué",        f"{seuil:.0%}")

    with col_metrics:
        st.subheader("Performance modèle")
        m = meta["metrics"]
        st.metric("Accuracy", f"{m['acc']:.1%}")
        st.metric("AUC",      f"{m['auc']:.3f}")
        st.metric("F1-score", f"{m['f1']:.3f}")

    with col_model_info:
        st.subheader("Features utilisées")
        lag_data = [
            {"Feature": feat, "Lag (jours)": lag, "Col. modèle": f"{feat}_lag{lag}"}
            for feat, lag in best_lags.items()
        ]
        st.dataframe(pd.DataFrame(lag_data), hide_index=True, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # GRAPHIQUES
    # ------------------------------------------------------------------
    tab1, tab2 = st.tabs(["📊 Prix & Probabilités", "📋 Données brutes"])

    with tab1:
        df_probas = render_price_probability_chart(asset, asset_label, data, seuil)
        n_hausse = (df_probas["proba"] >= seuil).sum()
        st.caption(
            f"🟢 **{n_hausse} signaux HAUSSE** sur {len(df_probas)} jours  |  "
            f"Barres vertes = proba ≥ seuil ({seuil:.0%})  |  "
            f"Barres rouges = proba < seuil"
        )

    with tab2:
        st.caption("Dernières valeurs des features (avec lags appliqués)")
        st.dataframe(
            features_df.tail(20).style.format("{:.6f}"),
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # FOOTER
    # ------------------------------------------------------------------
    st.divider()
    st.caption(
        "⚠️ **Avertissement** : Ces prédictions sont issues d'un modèle académique "
        "et ne constituent pas des conseils en investissement."
    )


# =============================================================================
# ROUTAGE PRINCIPAL
# =============================================================================

def main():
    st.set_page_config(
        page_title  = "Prédiction Marchés — Jedha",
        page_icon   = "📈",
        layout      = "wide",
        initial_sidebar_state = "collapsed",
    )

    # CSS minimal
    st.markdown("""
    <style>
        .metric-card {
            background: #1e2530;
            border-radius: 10px;
            padding: 1rem 1.5rem;
            border-left: 4px solid #4fc3f7;
            margin-bottom: 0.5rem;
        }
        .signal-up   { color: #4caf50; font-size: 2rem; font-weight: bold; }
        .signal-down { color: #ef5350; font-size: 2rem; font-weight: bold; }
        .signal-neutral { color: #ffa726; font-size: 2rem; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    # Initialisation de la page courante
    if "page" not in st.session_state:
        st.session_state.page = "public"

    # Bouton de navigation en haut à droite
    col_nav1, col_nav2 = st.columns([6, 1])
    with col_nav2:
        if st.session_state.page == "public":
            if st.button("⚙️ Paramètres", use_container_width=True):
                st.session_state.page = "settings"
                st.rerun()
        else:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.page = "public"
                st.rerun()

    # Affichage de la page sélectionnée
    if st.session_state.page == "public":
        st.sidebar.empty()  # masque la sidebar sur la page publique
        render_public_page()
    else:
        render_settings_page()


if __name__ == "__main__":
    main()
