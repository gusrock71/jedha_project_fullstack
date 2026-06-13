"""
filter_trading_days.py — Filtre df_combined sur les jours de bourse Euronext Paris
====================================================================================
Retire les jours de fermeture (week-ends + jours fériés français) en se basant
sur le calendrier de cotation réel d'AF.PA (Yahoo Finance), qui ne fournit des
cotations que pour les jours d'ouverture d'Euronext Paris.

Usage :
    python filter_trading_days.py
    python filter_trading_days.py --csv outputs/df_combined.csv --out outputs/df_combined_trading.csv
"""

import argparse
import pandas as pd
import yfinance as yf


def get_trading_days(start: str, end: str, ref_ticker: str = "AF.PA") -> pd.DatetimeIndex:
    """
    Retourne l'index des jours de cotation réels (Euronext Paris) en se basant
    sur un téléchargement Yahoo Finance d'un ticker de référence cotée à Paris.
    """
    print(f"Téléchargement du calendrier de cotation via {ref_ticker} ...")
    raw = yf.download(ref_ticker, start=start, end=end, auto_adjust=True, progress=False)
    trading_days = pd.to_datetime(raw.index).normalize()
    print(f"  → {len(trading_days)} jours de bourse trouvés "
          f"({trading_days.min().date()} → {trading_days.max().date()})")
    return trading_days


def filter_to_trading_days(df: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Filtre df aux dates présentes dans trading_days."""
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    return df.loc[df.index.isin(trading_days)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filtre df_combined.csv sur les jours de bourse Euronext Paris"
    )
    parser.add_argument("--csv", type=str, default="outputs/df_combined.csv",
                        help="Chemin vers df_combined.csv (entrée)")
    parser.add_argument("--out", type=str, default="outputs/df_combined_trading.csv",
                        help="Chemin de sortie pour le fichier filtré")
    parser.add_argument("--ref-ticker", type=str, default="AF.PA",
                        help="Ticker de référence pour le calendrier de cotation (défaut: AF.PA)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Chargement de {args.csv} ...")
    df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    print(f"  → {df.shape[0]} lignes avant filtrage "
          f"({df.index.min().date()} → {df.index.max().date()})")

    trading_days = get_trading_days(
        start=df.index.min().strftime("%Y-%m-%d"),
        end=(df.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        ref_ticker=args.ref_ticker,
    )

    df_trading = filter_to_trading_days(df, trading_days)
    print(f"  → {df_trading.shape[0]} lignes après filtrage "
          f"({df_trading.shape[0] - df.shape[0]:+d} lignes)")

    # --- Vérification de l'équilibre 0/1 après filtrage ---
    print("\nÉquilibre 0/1 après filtrage (sur l'ensemble du dataset) :")
    for target in ["AF_Return", "TTE_Return", "RNO_Return"]:
        if target in df_trading.columns:
            y = (df_trading[target] > 0).astype(int)
            props = y.value_counts(normalize=True).reindex([0, 1], fill_value=0)
            print(f"  {target:12s} → Baisse (0): {props[0]:.1%}  |  Hausse (1): {props[1]:.1%}")

    df_trading.to_csv(args.out, index=True)
    print(f"\nFichier filtré sauvegardé → {args.out}")


if __name__ == "__main__":
    main()
