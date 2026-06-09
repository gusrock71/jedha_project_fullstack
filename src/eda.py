# =============================================================================
# EDA.PY — Analyse exploratoire et visualisations
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from statsmodels.tsa.stattools import grangercausalitytests

from src.config import (
    FIGURES_DIR,
    CORR_COLS,
    GRANGER_PREDICTORS,
    GRANGER_MAXLAG,
)


def _savefig(fig, name: str):
    """Sauvegarde une figure dans FIGURES_DIR."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  → Figure sauvegardée : {path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 1. PRIX & RETURNS — Vue d'ensemble par binôme
# -----------------------------------------------------------------------------

def plot_price_vs_return(df: pd.DataFrame, save: bool = True):
    """
    Pour chaque paire (Prix, Return), trace deux graphiques côte-à-côte.
    """
    pairs = [
        (col.replace("_Price", "_Price"), col.replace("_Price", "_Return"))
        for col in df.columns if "_Price" in col
        and col.replace("_Price", "_Return") in df.columns
    ]
    # Cas particuliers non couverts par la règle générale
    extras = [
        ("EURUSD_Price", "EURUSD_Abs_Change"),
        ("GPRD_Value",   "GPRD_Diff"),
        ("GPRD_ACT_Value", "GPRD_ACT_Diff"),
        ("GPRD_THREAT_Value", "GPRD_THREAT_Diff"),
        ("CLI",          "CLI_Diff"),
    ]
    for p in extras:
        if p[0] in df.columns and p not in pairs:
            pairs.append(p)

    print(f"Tracé de {len(pairs)} binômes Prix/Return...")

    for price_col, return_col in pairs:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        axes[0].plot(df.index, df[price_col])
        axes[0].set_title(price_col)
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Prix")

        axes[1].plot(df.index, df[return_col])
        axes[1].set_title(return_col)
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Return / Variation")

        plt.tight_layout()
        if save:
            _savefig(fig, f"price_return_{price_col}.png")
        else:
            plt.show()


# -----------------------------------------------------------------------------
# 2. CHOCS GÉOPOLITIQUES — Base 100 sur toute la période
# -----------------------------------------------------------------------------

def plot_geopolitical_shocks(df: pd.DataFrame, save: bool = True):
    """
    Visualise l'évolution des actifs en base 100 avec superposition
    des pics de chocs géopolitiques (top 5% GPRD_Diff).
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    seuil_gpr  = df["GPRD_Diff"].quantile(0.95)
    dates_crises = df[df["GPRD_Diff"] >= seuil_gpr].index

    price_cols = [c for c in df.columns if "_Price" in c]
    df_base100 = (df[price_cols] / df[price_cols].iloc[0]) * 100

    fig, ax = plt.subplots(figsize=(16, 8))

    label_ajoute = False
    for date in dates_crises:
        kwargs = {"color": "red", "alpha": 0.15, "linestyle": "--"}
        if not label_ajoute:
            kwargs["label"] = "Choc Géopolitique (Top 5% GPRD)"
            label_ajoute = True
        ax.axvline(date, **kwargs)

    for col in df_base100.columns:
        lw = 2.5 if any(x in col for x in ["AF", "Brent"]) else 1.2
        ax.plot(df_base100.index, df_base100[col], label=col, linewidth=lw)

    ax.set_title(
        "Variations relatives des Actifs (Base 100 — oct. 2017) et Chocs Géopolitiques Majeurs",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Valeur Relative (Échelle Log — Base 100)", fontsize=12)
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    if save:
        _savefig(fig, "geopolitical_shocks_base100.png")
    else:
        plt.show()

    print(f"  → {len(dates_crises)} pics géopolitiques représentés")


# -----------------------------------------------------------------------------
# 3. FOCUS ÉVÉNEMENTS — COVID / Ukraine / Proche-Orient
# -----------------------------------------------------------------------------

def plot_event_period(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    title_event: str,
    save: bool = True,
):
    """
    Zoom sur une période de crise spécifique avec base 100 locale.
    """
    df_evt = df.loc[start_date:end_date].copy()

    seuil_gpr    = df_evt["GPRD_Diff"].quantile(0.95)
    dates_crises = df_evt[df_evt["GPRD_Diff"] >= seuil_gpr].index

    price_cols = [c for c in df_evt.columns if "_Price" in c]
    df_base100 = (df_evt[price_cols] / df_evt[price_cols].iloc[0]) * 100

    fig, ax = plt.subplots(figsize=(16, 8))

    label_ajoute = False
    for date in dates_crises:
        kwargs = {"color": "red", "alpha": 0.20, "linestyle": "--"}
        if not label_ajoute:
            kwargs["label"] = "Top 5% chocs GPR"
            label_ajoute = True
        ax.axvline(date, **kwargs)

    for col in df_base100.columns:
        lw = 2.5 if any(x in col for x in ["AF", "CAC40", "Brent"]) else 1.2
        ax.plot(df_base100.index, df_base100[col], label=col, linewidth=lw)

    ax.set_title(f"{title_event}\n({start_date} → {end_date})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Base 100")
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    if save:
        safe_name = title_event.lower().replace(" ", "_").replace("-", "_")
        _savefig(fig, f"event_{safe_name}.png")
    else:
        plt.show()

    print(f"  {title_event} : {len(dates_crises)} pics géopolitiques détectés")


def plot_all_crisis_events(df: pd.DataFrame, save: bool = True):
    """Lance les trois zooms de crise."""
    plot_event_period(df, "2020-02-01", "2020-05-01", "COVID-19", save)
    plot_event_period(df, "2022-02-01", "2022-05-01", "Guerre en Ukraine", save)
    plot_event_period(df, "2023-10-01", "2024-02-01", "Conflit Israël-Hamas", save)


# -----------------------------------------------------------------------------
# 4. DISTRIBUTIONS — Histogramme + KDE + Boxplot
# -----------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame, save: bool = True):
    """
    Pour chaque colonne de returns/variations, trace un histogramme KDE
    et un boxplot avec cartouches statistiques.
    """
    sns.set_theme(style="whitegrid")

    cols_to_plot = [
        c for c in df.columns
        if ("_Return" in c or "_Diff" in c or c == "EURUSD_Abs_Change")
        and df[c].notna().sum() > 0
    ]
    print(f"Analyse des distributions : {len(cols_to_plot)} variables")

    n = len(cols_to_plot)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(16, 4 * n))
    if n == 1:
        axes = np.array([axes])

    for i, col in enumerate(cols_to_plot):
        data = df[col].dropna()

        # Histogramme + KDE
        sns.histplot(data, kde=True, stat="density", color="steelblue", ax=axes[i, 0])
        axes[i, 0].set_title(f"Distribution — {col}", fontsize=12, fontweight="bold")
        axes[i, 0].axvline(data.mean(), color="red", linestyle="--", linewidth=2, label="Moyenne")

        xmin, xmax = axes[i, 0].get_xlim()
        x = np.linspace(xmin, xmax, 300)
        axes[i, 0].plot(x, norm.pdf(x, data.mean(), data.std()), "k--", linewidth=2, label="Normale théorique")

        stats_text = (
            f"N = {len(data):,.0f}\nMean = {data.mean():.4f}\nStd = {data.std():.4f}\n"
            f"Skew = {data.skew():.2f}\nKurt = {data.kurtosis():.2f}\n"
            f"Q05 = {data.quantile(0.05):.4f}\nQ95 = {data.quantile(0.95):.4f}\n"
            f"Min = {data.min():.4f}\nMax = {data.max():.4f}"
        )
        axes[i, 0].text(0.98, 0.98, stats_text, transform=axes[i, 0].transAxes,
                        fontsize=9, va="top", ha="right",
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="grey", alpha=0.90))
        axes[i, 0].legend()

        # Boxplot
        sns.boxplot(x=data, color="lightgreen", ax=axes[i, 1])
        axes[i, 1].set_title(f"Boxplot — {col}", fontsize=12, fontweight="bold")
        box_text = (
            f"Q1 = {data.quantile(0.25):.4f}\nMedian = {data.median():.4f}\nQ3 = {data.quantile(0.75):.4f}"
        )
        axes[i, 1].text(0.98, 0.95, box_text, transform=axes[i, 1].transAxes,
                        fontsize=9, va="top", ha="right",
                        bbox=dict(boxstyle="round", facecolor="white", edgecolor="grey", alpha=0.90))

    fig.suptitle("Analyse des Distributions, Normalité et Outliers", fontsize=18, fontweight="bold", y=1.005)
    plt.tight_layout()

    if save:
        _savefig(fig, "distributions.png")
    else:
        plt.show()


# -----------------------------------------------------------------------------
# 5. MATRICE DE CORRÉLATION SPEARMAN
# -----------------------------------------------------------------------------

def plot_correlation_matrix(df: pd.DataFrame, save: bool = True):
    """Heatmap de corrélation de Spearman sur les colonnes de returns."""
    available = [c for c in CORR_COLS if c in df.columns]
    corr_matrix = df[available].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Matrice de corrélation de Spearman", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        _savefig(fig, "correlation_spearman.png")
    else:
        plt.show()


# -----------------------------------------------------------------------------
# 6. SENSIBILITÉ AU VIX
# -----------------------------------------------------------------------------

def plot_vix_sensitivity(df: pd.DataFrame, save: bool = True):
    """
    Scatter plots des rendements AF / RNO / TTE en fonction du VIX,
    colorés par le rendement STOXX50.
    """
    actifs = {"AF": "AF_Return", "RNO": "RNO_Return", "TTE": "TTE_Return"}

    for ticker, col_return in actifs.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(
            df["VIX_Return"], df[col_return],
            c=df["STOXX_Return"], cmap="RdYlGn",
            alpha=0.6, edgecolors="none", s=15,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Rendement Journalier du STOXX50 (Marché Global)")

        x = df["VIX_Return"]
        y = df[col_return]
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m * x + b, color="black", linestyle="--", label=f"Tendance (β ≈ {m:.2f})")

        ax.set_title(f"Rendements de {ticker} en fonction du VIX et du marché")
        ax.set_xlabel("Returns — VIX")
        ax.set_ylabel(f"Returns — {ticker}")
        ax.legend()
        plt.tight_layout()

        if save:
            _savefig(fig, f"vix_sensitivity_{ticker}.png")
        else:
            plt.show()


# -----------------------------------------------------------------------------
# 7. CAUSALITÉ DE GRANGER
# -----------------------------------------------------------------------------

def granger_pvalues(df: pd.DataFrame, target: str, predictors: list, maxlag: int = 30) -> pd.DataFrame:
    """Calcule les p-values du test de Granger pour chaque prédicteur × lag."""
    pval_matrix = pd.DataFrame(index=predictors, columns=range(1, maxlag + 1))

    for predictor in predictors:
        results = grangercausalitytests(
            df[[target, predictor]].dropna(), maxlag=maxlag, verbose=False
        )
        for lag in range(1, maxlag + 1):
            pval_matrix.loc[predictor, lag] = results[lag][0]["ssr_ftest"][1]

    return pval_matrix.astype(float)


def plot_granger_significance(df: pd.DataFrame, save: bool = True):
    """
    Calcule et visualise les matrices de significativité des lags
    pour AF, RNO, TTE (p < 5%).
    """
    print("Calcul des tests de causalité de Granger (peut prendre quelques minutes)...")

    targets = {"AF": "AF_Return", "RNO": "RNO_Return", "TTE": "TTE_Return"}
    sig_matrices = {}

    for label, target in targets.items():
        matrix = granger_pvalues(df, target, GRANGER_PREDICTORS, GRANGER_MAXLAG)
        sig_matrices[label] = (matrix < 0.05).astype(int)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for ax, (title, matrix) in zip(axes, sig_matrices.items()):
        sns.heatmap(matrix, ax=ax, annot=True, cmap="RdYlGn", cbar=False, vmin=0, vmax=1)
        ax.set_title(title)

    fig.suptitle(
        "Détermination des lags (en jours) — Relations de causalité significatives (p < 5%)",
        fontsize=14,
    )
    plt.tight_layout()

    if save:
        _savefig(fig, "granger_significance.png")
    else:
        plt.show()

    return sig_matrices


# -----------------------------------------------------------------------------
# PIPELINE EDA COMPLET
# -----------------------------------------------------------------------------

def run_eda(df: pd.DataFrame, save: bool = True):
    """Lance l'ensemble de l'analyse exploratoire."""
    print("\n=== EDA ===")
    plot_price_vs_return(df, save)
    plot_geopolitical_shocks(df, save)
    plot_all_crisis_events(df, save)
    plot_distributions(df, save)
    plot_correlation_matrix(df, save)
    plot_vix_sensitivity(df, save)
    plot_granger_significance(df, save)
    print("=== EDA terminée ===\n")
