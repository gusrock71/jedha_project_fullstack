"""
check_class_balance.py — Vérification de l'équilibre 0/1 par split
====================================================================
Calcule le nombre et la proportion de jours de baisse (0) et de hausse (1)
pour la target sélectionnée, séparément sur train / validation / test.

Usage :
    python check_class_balance.py
    python check_class_balance.py --target TTE_Return
    python check_class_balance.py --target RNO_Return --csv outputs/df_combined.csv
"""

import argparse
import pandas as pd


# =============================================================================
# PARAMÈTRES PAR DÉFAUT — adapter si besoin
# =============================================================================

DEFAULT_CSV = "outputs/df_combined.csv"

# Bornes de split (cohérentes avec le projet : train 2007-2020,
# validation 2021-2022, test 2023-2026)
TRAIN_START, TRAIN_END = "2007-07-30", "2020-12-31"
VALID_START, VALID_END = "2021-01-01", "2022-12-31"
TEST_START,  TEST_END  = "2023-01-01", "2026-04-01"


# =============================================================================
# FONCTIONS
# =============================================================================

def load_data(csv_path: str) -> pd.DataFrame:
    """Charge df_combined depuis le CSV et s'assure que l'index est bien daté."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


def get_splits(df: pd.DataFrame) -> dict:
    """Découpe le DataFrame en train / validation / test selon les bornes définies."""
    return {
        "Train":      df.loc[TRAIN_START:TRAIN_END],
        "Validation": df.loc[VALID_START:VALID_END],
        "Test":       df.loc[TEST_START:TEST_END],
    }


def print_balance(splits: dict, target: str):
    """Affiche le nombre et la proportion de 0/1 pour chaque split."""
    print(f"\nCible : {target}  (1 = rendement > 0, 0 = rendement <= 0)")
    print("=" * 60)

    header = f"{'Split':<12} {'N':>6} {'Baisse (0)':>14} {'Hausse (1)':>14}"
    print(header)
    print("-" * len(header))

    for name, sub_df in splits.items():
        if sub_df.empty:
            print(f"{name:<12} {'(vide)':>6}")
            continue

        y = (sub_df[target] > 0).astype(int)
        counts = y.value_counts().reindex([0, 1], fill_value=0)
        props  = y.value_counts(normalize=True).reindex([0, 1], fill_value=0)

        col0 = f"{counts[0]:>5} ({props[0]:.1%})"
        col1 = f"{counts[1]:>5} ({props[1]:.1%})"

        print(f"{name:<12} {len(y):>6} {col0:>14} {col1:>14}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Vérifie l'équilibre des classes 0/1 sur train/validation/test"
    )
    parser.add_argument(
        "--csv", type=str, default=DEFAULT_CSV,
        help=f"Chemin vers df_combined.csv (défaut: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Colonne cible (ex: AF_Return). Si non précisé, traite AF/TTE/RNO.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Chargement de {args.csv} ...")
    df = load_data(args.csv)
    print(f"  → {df.shape[0]} lignes du {df.index.min().date()} au {df.index.max().date()}")

    splits = get_splits(df)

    targets = [args.target] if args.target else ["AF_Return", "TTE_Return", "RNO_Return"]

    for target in targets:
        if target not in df.columns:
            print(f"\n[!] Colonne '{target}' absente du DataFrame — ignorée.")
            continue
        print_balance(splits, target)

    print()


if __name__ == "__main__":
    main()
