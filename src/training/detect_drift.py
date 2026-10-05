# src/training/detect_drift.py

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

# Import du module transverse de features
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sqlalchemy import text

from src.utils.db import get_postgres_engine
from src.utils.features import get_drift_feature_columns, prepare_features


def main():
    parser = argparse.ArgumentParser(
        description="Détecteur de dérive des données et de dégradation des performances du Champion"
    )
    parser.add_argument(
        "--current-date",
        type=str,
        default=None,
        help="Date actuelle simulée au format YYYY-MM-DD (ex: 2020-07-22 ou 2020-12-04). Par défaut: date max des données.",
    )
    parser.add_argument(
        "--current-days",
        type=int,
        default=7,
        help="Nombre de jours pour la période courante (défaut: 7)",
    )
    parser.add_argument(
        "--ref-days-start",
        type=int,
        default=38,
        help="Début de la période de référence en jours d'antériorité (défaut: 38)",
    )
    parser.add_argument(
        "--ref-days-end",
        type=int,
        default=8,
        help="Fin de la période de référence en jours d'antériorité (défaut: 8)",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.33,
        help="Seuil de ratio de variables en dérive pour lever l'alerte (défaut: 0.33 soit 2/6)",
    )
    parser.add_argument(
        "--min-f2",
        type=float,
        default=0.50,
        help="Seuil F2-Score minimal requis pour le Champion sur la période courante (défaut: 0.50)",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.50,
        help="Seuil de Rappel minimal requis sur la classe fraude (défaut: 0.50)",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.20,
        help="Seuil de Précision minimal requis sur la classe fraude (défaut: 0.20)",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.50,
        help="Seuil de F1-Score minimal requis sur la classe fraude (défaut: 0.50)",
    )
    parser.add_argument(
        "--no-check-performance",
        action="store_true",
        help="Désactiver l'audit de performance du Champion en production",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv"))

    if not os.path.exists(csv_path):
        print(f"Erreur : Dataset {csv_path} introuvable.")
        sys.exit(2)

    # Chargement et préparation des features standardisées via src.utils
    df = pd.read_csv(csv_path)
    df = prepare_features(df, include_graph_ids=True)

    max_dataset_date = df["trans_date_trans_time"].max()
    min_dataset_date = df["trans_date_trans_time"].min()

    # Détermination de la date cible
    target_date = None
    if args.current_date:
        try:
            parsed_date = pd.to_datetime(args.current_date)
            if (
                min_dataset_date + pd.Timedelta(days=args.ref_days_start)
                <= parsed_date
                <= max_dataset_date
            ):
                target_date = parsed_date
            else:
                print(
                    f"⚠️ Date demandée ({args.current_date}) hors limites du dataset [{min_dataset_date.date()} -> {max_dataset_date.date()}]."
                )
        except Exception as e:
            print(f"⚠️ Erreur parsing --current-date ({args.current_date}): {e}")

    if target_date is None:
        try:
            engine = get_postgres_engine()
            with engine.connect() as conn:
                max_db_val = conn.execute(
                    text("SELECT MAX(trans_date_trans_time) FROM silver.rawdata")
                ).scalar()
                if max_db_val:
                    t = pd.to_datetime(max_db_val)
                    if t.tzinfo is not None:
                        t = t.tz_localize(None)
                    target_date = t
                    print(
                        f"✅ Date cible détectée depuis PostgreSQL (silver.rawdata) : {target_date}"
                    )
        except Exception as e:
            print(f"ℹ️ PostgreSQL non disponible ({e}).")

    if target_date is None:
        target_date = min_dataset_date + pd.Timedelta(days=args.ref_days_start)
        print(
            f"ℹ️ Date cible par défaut (début + {args.ref_days_start}j) : {target_date}"
        )

    print(
        f"\n======================================================================\n"
        f"📊 AUDIT MLOPS DU {target_date.strftime('%Y-%m-%d')} (DATA DRIFT + PERFORMANCE GATE)\n"
        f"======================================================================"
    )

    # Détermination de la période courante (les 7 derniers jours par défaut)
    current_period_end = target_date
    current_period_start = current_period_end - pd.Timedelta(days=args.current_days)

    # Détermination de la période de référence : fenêtre glissante [-38 jours ; -8 jours]
    reference_period_end = current_period_end - pd.Timedelta(days=args.ref_days_end)
    reference_period_start = current_period_end - pd.Timedelta(days=args.ref_days_start)

    print(
        f"• Période de référence ([-{args.ref_days_start}j; -{args.ref_days_end}j]) : du {reference_period_start} au {reference_period_end}"
    )
    print(
        f"• Période courante    ({args.current_days} jours cibles)               : du {current_period_start} au {current_period_end}"
    )

    # Extraction des sous-ensembles temporels complets
    full_start_df = df[
        (df["trans_date_trans_time"] >= reference_period_start)
        & (df["trans_date_trans_time"] < reference_period_end)
    ].copy()

    full_end_df = df[
        (df["trans_date_trans_time"] >= current_period_start)
        & (df["trans_date_trans_time"] < current_period_end)
    ].copy()

    if len(full_end_df) < 20:
        print(
            f"⚠️ Avertissement : Trop peu de transactions sur la période courante ({len(full_end_df)}). Audit non testable."
        )
        sys.exit(0)

    # ============================================================================
    # 1. ANALYSE DE DATA DRIFT AVEC EVIDENTLY AI
    # ============================================================================
    relevant_columns = get_drift_feature_columns()
    start_df = full_start_df[relevant_columns]
    end_df = full_end_df[relevant_columns]

    # Définition du schéma pour Evidently
    schema = DataDefinition(
        numerical_columns=["amt", "hour_sin", "hour_cos", "distance_achat", "is_fraud"],
        categorical_columns=["gender"],
    )

    eval_data_1 = Dataset.from_pandas(start_df, data_definition=schema)
    eval_data_2 = Dataset.from_pandas(end_df, data_definition=schema)

    report = Report([DataDriftPreset()])
    my_eval = report.run(eval_data_2, eval_data_1)

    metrics_list = my_eval.dict()["metrics"]

    drift_flags = []
    details = {}

    for m in metrics_list:
        name = m["metric_name"]
        if name.startswith("ValueDrift"):
            col = m["config"]["column"]
            val = float(m["value"])
            threshold = float(m["config"]["threshold"])
            method = m["config"]["method"]

            if "distance" in method.lower():
                col_drift = 1.0 if val > threshold else 0.0
            else:
                col_drift = 1.0 if val < threshold else 0.0

            drift_flags.append(col_drift)
            details[col] = {
                "drift_detected": bool(col_drift > 0),
                "metric_value": val,
                "threshold": threshold,
                "method": method,
            }

    mean_drift = np.mean(drift_flags) if drift_flags else 0.0
    drift_detected = mean_drift >= args.drift_threshold

    print("\n--- 1. SYNTHÈSE DU DATA DRIFT (EVIDENTLY AI) ---")
    print(f"Variables testées : {len(drift_flags)}")
    print(
        f"Ratio de variables en dérive : {mean_drift * 100:.1f}% (Seuil d'alerte : {args.drift_threshold * 100:.0f}%)"
    )
    print(
        f"Statut Data Drift : {'🚨 DÉRIVE DÉTECTÉE !' if drift_detected else '✅ STABLE'}"
    )

    for col, detail in details.items():
        status_str = "🚨 Dérivé" if detail["drift_detected"] else "✅ Stable"
        print(
            f"  - Colonne '{col:15s}' ({detail['method']}) : value = {detail['metric_value']:.4f} (seuil: {detail['threshold']}) | Statut : {status_str}"
        )

    # Sauvegarde HTML Evidently
    html_path = os.path.join(script_dir, "evidently_drift_report.html")
    my_eval.save_html(html_path)
    print(f"Rapport HTML Evidently sauvegardé : {html_path}")

    # ============================================================================
    # 2. AUDIT DE PERFORMANCE DU MODÈLE CHAMPION EN PRODUCTION
    # ============================================================================
    perf_degraded = False
    champion_perf_summary = {}

    if not args.no_check_performance:
        print("\n--- 2. CONTRÔLE DES PERFORMANCES DU CHAMPION ACTIF ---")
        try:
            from src.utils.mlflow_manager import load_champion_model
            from src.utils.threshold import evaluate_predictions_and_curves

            champion_model, champion_version_id, decision_threshold = (
                load_champion_model()
            )

            if (
                champion_model is not None
                and len(full_end_df) >= 20
                and "is_fraud" in full_end_df.columns
            ):
                y_true = full_end_df["is_fraud"].astype(int).values
                n_frauds = int(np.sum(y_true))

                # Inférence avec gestion robuste des colonnes d'entrée
                try:
                    y_probas = champion_model.predict_proba(full_end_df)[:, 1]
                except Exception:
                    feature_cols = [
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
                    available_cols = [
                        c for c in feature_cols if c in full_end_df.columns
                    ]
                    y_probas = champion_model.predict_proba(
                        full_end_df[available_cols]
                    )[:, 1]

                perf_metrics, cm_dict = evaluate_predictions_and_curves(
                    y_true, y_probas, threshold=decision_threshold
                )

                f2_val = float(perf_metrics.get("f2_class_1", 0.0))
                rec_val = float(perf_metrics.get("rec_class_1", 0.0))
                prec_val = float(perf_metrics.get("prec_class_1", 0.0))
                f1_val = float(perf_metrics.get("f1_class_1", 0.0))

                reasons = []
                if n_frauds > 0:
                    if f2_val < args.min_f2:
                        reasons.append(f"F2-Score ({f2_val:.4f} < {args.min_f2:.4f})")
                    if rec_val < args.min_recall:
                        reasons.append(
                            f"Rappel C1 ({rec_val:.4f} < {args.min_recall:.4f})"
                        )
                    if prec_val < args.min_precision:
                        reasons.append(
                            f"Précision C1 ({prec_val:.4f} < {args.min_precision:.4f})"
                        )
                    if f1_val < args.min_f1:
                        reasons.append(f"F1-Score ({f1_val:.4f} < {args.min_f1:.4f})")

                perf_degraded = len(reasons) > 0

                champion_perf_summary = {
                    "model_version": champion_version_id,
                    "decision_threshold": decision_threshold,
                    "sample_size": len(full_end_df),
                    "fraud_count": n_frauds,
                    "f2_score": f2_val,
                    "recall": rec_val,
                    "precision": prec_val,
                    "f1_score": f1_val,
                    "pr_auc": float(perf_metrics.get("pr_auc", 0.0)),
                    "confusion_matrix": cm_dict,
                    "performance_degraded": perf_degraded,
                    "degradation_reasons": reasons,
                    "thresholds": {
                        "min_f2": args.min_f2,
                        "min_recall": args.min_recall,
                        "min_precision": args.min_precision,
                        "min_f1": args.min_f1,
                    },
                }

                print(
                    f"Modèle Champion : {champion_version_id} (Seuil calibré : {decision_threshold:.4f})"
                )
                print(
                    f"Échantillon     : {len(full_end_df)} transactions ({n_frauds} fraudes observées)"
                )
                print(
                    f"  • F2-Score   : {f2_val:.4f} (Seuil min: {args.min_f2:.4f})  -> {'🚨 DÉGRADÉ' if f2_val < args.min_f2 else '✅ OK'}"
                )
                print(
                    f"  • Rappel C1  : {rec_val:.4f} (Seuil min: {args.min_recall:.4f})  -> {'🚨 DÉGRADÉ' if rec_val < args.min_recall else '✅ OK'}"
                )
                print(
                    f"  • Précision  : {prec_val:.4f} (Seuil min: {args.min_precision:.4f})  -> {'🚨 DÉGRADÉ' if prec_val < args.min_precision else '✅ OK'}"
                )
                print(
                    f"  • F1-Score   : {f1_val:.4f} (Seuil min: {args.min_f1:.4f})  -> {'🚨 DÉGRADÉ' if f1_val < args.min_f1 else '✅ OK'}"
                )
                print(f"  • PR-AUC     : {perf_metrics.get('pr_auc', 0.0):.4f}")
                print(
                    f"  • Matrice    : TP={cm_dict.get('tp', 0)}, FP={cm_dict.get('fp', 0)}, FN={cm_dict.get('fn', 0)}, TN={cm_dict.get('tn', 0)}"
                )

                if perf_degraded:
                    print(f"🚨 ALERTE PERFORMANCES : {'; '.join(reasons)}")
                else:
                    print("✅ Performances en production conformes aux SLA métier.")
            else:
                print(
                    "ℹ️ Audit de performance ignoré (champion non disponible ou données insuffisantes)."
                )
        except Exception as e:
            print(f"⚠️ Erreur lors du calcul des performances du champion : {e}")

    # ============================================================================
    # 3. VERSION JSON GLOBALE ET PRISE DE DÉCISION
    # ============================================================================
    should_retrain = bool(drift_detected or perf_degraded)

    json_summary = {
        "current_date": target_date.strftime("%Y-%m-%d"),
        "reference_period": f"{reference_period_start.strftime('%Y-%m-%d')} -> {reference_period_end.strftime('%Y-%m-%d')}",
        "current_period": f"{current_period_start.strftime('%Y-%m-%d')} -> {current_period_end.strftime('%Y-%m-%d')}",
        "sample_size": len(full_end_df),
        "reference_size": len(full_start_df),
        "drift_detected": bool(drift_detected),
        "mean_drift_ratio": float(mean_drift),
        "drift_threshold": float(args.drift_threshold),
        "details": details,
        "performance_audit": champion_perf_summary,
        "should_retrain": should_retrain,
    }

    json_path = os.path.join(script_dir, "drift_report.json")
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=4)
    print(f"\nRapport JSON complet sauvegardé : {json_path}")

    print("\n" + "=" * 70)
    print("🎯 DÉCISION D'ORCHESTRATION MLOPS (AIRFLOW TRIGGER)")
    print(
        f"  • Dérive des données P(X) (Evidently AI)    : {'🚨 DÉTECTÉE' if drift_detected else '✅ STABLE'}"
    )
    if not args.no_check_performance and champion_perf_summary:
        print(
            f"  • Dégradation des performances P(Y|X)      : {'🚨 DÉTECTÉE' if perf_degraded else '✅ CONFORME'}"
        )
    print(
        f"  • Action globale                            : {'🚀 RÉENTRAÎNEMENT REQUIS (Exit 1)' if should_retrain else '💤 PAS DE RÉENTRAÎNEMENT (Exit 0)'}"
    )
    print("=" * 70 + "\n")

    # Code de sortie pour Airflow BranchPythonOperator
    if should_retrain:
        sys.exit(1)  # Dérive ou dégradation de perf -> Branche Réentraînement
    else:
        sys.exit(0)  # Stable -> Branche Skip


if __name__ == "__main__":
    main()
