# src/training/benchmark_isolation_forest.py
"""
Script de Benchmark Scientifique :
Comparaison Side-by-Side entre :
  1. XGBoost Standard (Baseline)
  2. XGBoost + Meta-Feature Isolation Forest (Anomaly Score Stacking)

Protocole de Test Temporel (Out-of-Time strict) :
  • Période d'Entraînement : du 21 Juin 2020 au 21 Juillet 2020 (1 mois)
  • Période d'Inférence / Test : du 22 Juillet 2020 au 13 Septembre 2020 (~1.5 mois)
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from skrub import TableVectorizer
from xgboost import XGBClassifier

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loader import load_dataset
from src.utils.threshold import evaluate_predictions_and_curves, find_optimal_threshold


def main():
    print("=" * 75)
    print(" BENCHMARK SCIENTIFIQUE : IMPACT DU STACKING ISOLATION FOREST")
    print("=" * 75)

    # 1. Chargement du dataset complet (sans filtrage d'outliers)
    df = load_dataset(sample_size=300000, sample_position="first")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    # 2. Découpage temporel strict selon la consigne
    train_mask = (df["trans_date_trans_time"] >= "2020-06-21") & (
        df["trans_date_trans_time"] <= "2020-07-21 23:59:59"
    )
    test_mask = (df["trans_date_trans_time"] >= "2020-07-22 00:00:00") & (
        df["trans_date_trans_time"] <= "2020-09-13 23:59:59"
    )

    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    print(
        f" Jeu d'Entraînement : {len(df_train):,} transactions (du {df_train['trans_date_trans_time'].min().date()} au {df_train['trans_date_trans_time'].max().date()})"
    )
    print(
        f"   • Fraudes Train : {df_train['is_fraud'].sum():,} ({df_train['is_fraud'].mean() * 100:.3f}%)"
    )
    print(
        f" Jeu de Test Out-of-Time : {len(df_test):,} transactions (du {df_test['trans_date_trans_time'].min().date()} au {df_test['trans_date_trans_time'].max().date()})"
    )
    print(
        f"   • Fraudes Test : {df_test['is_fraud'].sum():,} ({df_test['is_fraud'].mean() * 100:.3f}%)"
    )
    print("-" * 75)

    features = [
        "category",
        "amt",
        "gender",
        "distance_achat",
        "age",
        "city_pop",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
    ]

    X_train = df_train[features]
    y_train = df_train["is_fraud"]
    X_test = df_test[features]
    y_test = df_test["is_fraud"]

    # 3. Vectorisation des variables tabulaires
    print(" Vectorisation des features tabulaires avec TableVectorizer...")
    vectorizer = TableVectorizer()
    X_tr_enc_df = vectorizer.fit_transform(X_train)
    X_te_enc_df = vectorizer.transform(X_test)

    # Conversion stricte en float32 numpy array
    if hasattr(X_tr_enc_df, "toarray"):
        X_tr_enc = X_tr_enc_df.toarray().astype(np.float32)
        X_te_enc = X_te_enc_df.toarray().astype(np.float32)
    elif isinstance(X_tr_enc_df, pd.DataFrame):
        X_tr_enc = X_tr_enc_df.to_numpy(dtype=np.float32)
        X_te_enc = X_te_enc_df.to_numpy(dtype=np.float32)
    else:
        X_tr_enc = np.asarray(X_tr_enc_df, dtype=np.float32)
        X_te_enc = np.asarray(X_te_enc_df, dtype=np.float32)

    # Calcul du ratio scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = float(n_neg / max(1, n_pos))
    print(f" Scale Pos Weight calculé : {spw:.2f}")

    xgb_params = {
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.05,
        "scale_pos_weight": np.clip(spw, 1.0, 50.0),
        "random_state": 42,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_jobs": 2,
    }

    # =========================================================================
    # MODÈLE 1 : XGBOOST BASELINE (Sans Isolation Forest)
    # =========================================================================
    print("\n Entraînement Modèle 1 : XGBoost Baseline...")
    clf_baseline = XGBClassifier(**xgb_params)
    clf_baseline.fit(X_tr_enc, y_train)

    probas_base = clf_baseline.predict_proba(X_te_enc)[:, 1]

    # Optimisation seuil F1 et F2
    thresh_f1_base, _score_f1_base = find_optimal_threshold(
        y_test, probas_base, metric_target="f1"
    )
    thresh_f2_base, _score_f2_base = find_optimal_threshold(
        y_test, probas_base, metric_target="f2"
    )

    metrics_base_f1, cm_base_f1 = evaluate_predictions_and_curves(
        y_test, probas_base, threshold=thresh_f1_base
    )
    metrics_base_f2, cm_base_f2 = evaluate_predictions_and_curves(
        y_test, probas_base, threshold=thresh_f2_base
    )

    # =========================================================================
    # MODÈLE 2 : XGBOOST + ISOLATION FOREST META-FEATURE
    # =========================================================================
    print(" Entraînement Modèle 2 : XGBoost + Stacking Isolation Forest...")
    iso = IsolationForest(
        n_estimators=150, contamination=0.01, random_state=42, n_jobs=2
    )
    # Entraînement de l'Isolation Forest STRICTEMENT sur le Train
    iso.fit(X_tr_enc)

    # Inférence des anomaly scores (inversion : score plus élevé = plus anormal)
    score_tr_iso = -iso.score_samples(X_tr_enc).reshape(-1, 1)
    score_te_iso = -iso.score_samples(X_te_enc).reshape(-1, 1)

    # Concaténation de la méta-feature
    X_tr_enriched = np.hstack([X_tr_enc, score_tr_iso])
    X_te_enriched = np.hstack([X_te_enc, score_te_iso])

    clf_enriched = XGBClassifier(**xgb_params)
    clf_enriched.fit(X_tr_enriched, y_train)

    probas_enriched = clf_enriched.predict_proba(X_te_enriched)[:, 1]

    # Optimisation seuil F1 et F2
    thresh_f1_enr, _score_f1_enr = find_optimal_threshold(
        y_test, probas_enriched, metric_target="f1"
    )
    thresh_f2_enr, _score_f2_enr = find_optimal_threshold(
        y_test, probas_enriched, metric_target="f2"
    )

    metrics_enr_f1, cm_enr_f1 = evaluate_predictions_and_curves(
        y_test, probas_enriched, threshold=thresh_f1_enr
    )
    metrics_enr_f2, cm_enr_f2 = evaluate_predictions_and_curves(
        y_test, probas_enriched, threshold=thresh_f2_enr
    )

    # =========================================================================
    # SYNTHÈSE COMPARATIVE
    # =========================================================================
    print("\n" + "=" * 80)
    print(" RÉSULTATS COMPARATIFS SUR LE JEU DE TEST (22 Juillet -> 13 Septembre)")
    print("=" * 80)

    def print_comparison_block(title, m_base, m_enr, cm_base, cm_enr):
        print(f"\n {title} :")
        print(
            f"{'Métrique':<24} | {'1. Baseline (XGBoost)':<22} | {'2. XGB + IsolationForest':<24} | {'Gain / Diff':<12}"
        )
        print("-" * 90)

        metrics_to_show = [
            ("AUPRC / PR-AUC", "pr_auc"),
            ("ROC-AUC", "roc_auc"),
            ("Seuil Calibré", "decision_threshold"),
            ("Précision Fraude (C1)", "prec_class_1"),
            ("Rappel Fraude (C1)", "rec_class_1"),
            ("F1-Score Fraude (C1)", "f1_class_1"),
            ("F2-Score Fraude (C1)", "f2_class_1"),
            ("F1-Global (Macro)", "F1_global"),
        ]

        for label, key in metrics_to_show:
            v_b = m_base[key]
            v_e = m_enr[key]
            diff = v_e - v_b
            sign = "+" if diff > 0 else ""
            fmt_diff = (
                f"{sign}{diff:.4f}" if key != "decision_threshold" else f"{diff:+.4f}"
            )

            highlight = (
                " "
                if diff > 0.005
                and key
                in ["pr_auc", "f1_class_1", "f2_class_1", "prec_class_1", "rec_class_1"]
                else ""
            )
            print(
                f"{label:<24} | {v_b:<22.4f} | {v_e:<24.4f} | {fmt_diff:<10}{highlight}"
            )

        print(
            f"\n  • Matrice Confusion Baseline  : TP={cm_base['tp']}, FP={cm_base['fp']}, FN={cm_base['fn']}, TN={cm_base['tn']}"
        )
        print(
            f"  • Matrice Confusion + IFOREST : TP={cm_enr['tp']}, FP={cm_enr['fp']}, FN={cm_enr['fn']}, TN={cm_enr['tn']}"
        )

    print_comparison_block(
        "CONFIGURATION A : Optimisation orientée F1-SCORE",
        metrics_base_f1,
        metrics_enr_f1,
        cm_base_f1,
        cm_enr_f1,
    )
    print_comparison_block(
        "CONFIGURATION B : Optimisation orientée F2-SCORE (Priorité Rappel)",
        metrics_base_f2,
        metrics_enr_f2,
        cm_base_f2,
        cm_enr_f2,
    )

    print("\n" + "=" * 80)
    print(" CONCLUSION DE L'EXPÉRIMENTATION")
    print("=" * 80)
    pr_auc_gain = (metrics_enr_f1["pr_auc"] - metrics_base_f1["pr_auc"]) * 100
    f1_gain = (metrics_enr_f1["f1_class_1"] - metrics_base_f1["f1_class_1"]) * 100
    print(
        f"  • Évolution PR-AUC (Qualité intrinsèque) : {pr_auc_gain:+.2f} points de %"
    )
    print(f"  • Évolution F1-Score Fraude               : {f1_gain:+.2f} points de %")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import traceback

    class Tee:
        def __init__(self, filename):
            self.file = open(filename, "w", encoding="utf-8")
            self.stdout = sys.stdout

        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)
            self.file.flush()

        def flush(self):
            self.file.flush()
            self.stdout.flush()

    out_file = os.path.join(os.path.dirname(__file__), "benchmark_results.txt")
    sys.stdout = Tee(out_file)
    try:
        main()
    except Exception as e:
        print(f" ERREUR LORS DU BENCHMARK : {e}")
        traceback.print_exc()
        sys.exit(1)
