"""
main.py — Point d'entrée du projet Jedha
=========================================
Impact des variations de variables macroéconomiques, financières et géopolitiques sur certaines actions françaises
(Air France, Renault, TotalEnergies)

Usage
-----
    python main.py                   # pipeline complet (ingestion → EDA → modèle)
    python main.py --skip-eda        # saute l'EDA (rapide)
    python main.py --no-save         # affiche les figures sans les sauvegarder
    python main.py --skip-granger    # saute le test de Granger (long)
"""

import argparse
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Import des modules du projet
# ---------------------------------------------------------------------------
from src.ingestion import build_dataset
from src.eda       import plot_granger_significance
from src.features  import build_lag_features, split_train_test
from src.modeling  import run_modeling
from src.config    import COMBINED_CSV, FIGURES_DIR, MODELS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline Jedha — Pétrole & Marchés financiers")
    parser.add_argument("--skip-eda",      action="store_true", help="Ne pas lancer l'EDA")
    parser.add_argument("--skip-granger",  action="store_true", help="Ne pas lancer le test de Granger")
    parser.add_argument("--no-save",       action="store_true", help="Afficher les figures (plt.show) plutôt que les sauvegarder")
    parser.add_argument("--from-csv",      type=str, default=None,
                        metavar="PATH",    help="Charger df_combined depuis un CSV existant (bypass ingestion)")
    return parser.parse_args()


def ensure_dirs():
    for d in [FIGURES_DIR, MODELS_DIR, "outputs/reports"]:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    ensure_dirs()

    save = not args.no_save

    # ------------------------------------------------------------------
    # ÉTAPE 1 — Ingestion / chargement
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ÉTAPE 1 — Ingestion des données")
    print("=" * 60)

    if args.from_csv:
        print(f"Chargement depuis {args.from_csv}...")
        df_combined = pd.read_csv(args.from_csv, index_col=0, parse_dates=True)
        print(f"  → {df_combined.shape[0]} lignes chargées")
    else:
        df_combined = build_dataset()
        df_combined.to_csv(COMBINED_CSV, index=True)
        print(f"Dataset sauvegardé → {COMBINED_CSV}")

    # ------------------------------------------------------------------
    # ÉTAPE 2 — Analyse exploratoire
    # ------------------------------------------------------------------
    if not args.skip_eda:
        print("\n" + "=" * 60)
        print("ÉTAPE 2 — Analyse exploratoire (EDA)")
        print("=" * 60)

        # Lance tout sauf Granger si --skip-granger
        from src import eda as eda_module
        eda_module.plot_price_vs_return(df_combined, save)
        eda_module.plot_geopolitical_shocks(df_combined, save)
        eda_module.plot_all_crisis_events(df_combined, save)
        eda_module.plot_distributions(df_combined, save)
        eda_module.plot_correlation_matrix(df_combined, save)
        eda_module.plot_vix_sensitivity(df_combined, save)

        if not args.skip_granger:
            eda_module.plot_granger_significance(df_combined, save)
        else:
            print("Test de Granger ignoré (--skip-granger).")
    else:
        print("\nÉtape EDA ignorée (--skip-eda).")

    # ------------------------------------------------------------------
    # ÉTAPE 3 — Feature engineering
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ÉTAPE 3 — Feature engineering (lags)")
    print("=" * 60)

    df_model, feature_cols = build_lag_features(df_combined)
    X_train, X_test, y_train, y_test = split_train_test(df_model, feature_cols)

    # ------------------------------------------------------------------
    # ÉTAPE 4 — Modélisation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ÉTAPE 4 — Modélisation (Logistic Regression)")
    print("=" * 60)

    results = run_modeling(X_train, X_test, y_train, y_test, feature_cols, save)

    # ------------------------------------------------------------------
    # Résumé final
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE TERMINÉ")
    print("=" * 60)
    print(f"  Accuracy finale  : {results['metrics']['accuracy']:.4f}")
    print(f"  Figures          : {FIGURES_DIR}/")
    print(f"  Dataset exporté  : {COMBINED_CSV}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
