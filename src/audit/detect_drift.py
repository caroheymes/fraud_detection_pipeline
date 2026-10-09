# src/audit/detect_drift.py
"""
Module unifié de détection de dérive (Data Drift, Prediction/Score Drift) et d'audit
des performances (Concept Drift) du modèle Champion en production.

Capacités :
1. Data Drift P(X) : Analyse multidimensionnelle via Evidently AI (KS-test / Wasserstein).
2. Score / Output Drift P(Y_hat) : Population Stability Index (PSI) et suivi du taux d'alerte.
3. Concept Drift P(Y|X) : Dégradation relative par rapport aux métriques de référence du Champion dans MLflow
   (ex: alerte dès que le F2 ou le Rappel baisse de plus de 5% par rapport à l'entraînement initial).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

# Import du module transverse
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.data_loader import load_dataset
from src.utils.features import get_drift_feature_columns
from src.utils.mlflow_manager import MLflowQualityGate, load_champion_model
from src.utils.threshold import evaluate_predictions_and_curves


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> tuple[float, list[dict[str, float]]]:
    """
    Calcule le Population Stability Index (PSI) entre la distribution des probabilités
    de référence (expected) et la distribution actuelle (actual).

    Interprétation :
      - PSI < 0.10  : Distribution stable (aucun changement significatif)
      - 0.10 <= PSI < 0.20 : Dérive modérée (à surveiller)
      - PSI >= 0.20 : Dérive significative de la population (réentraînement recommandé)
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0, []

    # Bins basés sur les percentiles de la référence
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0, []

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Régularisation pour éviter les log(0) ou divisions par zéro
    expected_pct = np.clip(expected_pct, epsilon, 1.0)
    actual_pct = np.clip(actual_pct, epsilon, 1.0)

    # Normalisation
    expected_pct = expected_pct / np.sum(expected_pct)
    actual_pct = actual_pct / np.sum(actual_pct)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    total_psi = float(np.sum(psi_values))

    bucket_details = []
    for i in range(len(psi_values)):
        b_low = float(bins[i]) if not np.isneginf(bins[i]) else 0.0
        b_high = float(bins[i + 1]) if not np.isposinf(bins[i + 1]) else 1.0
        bucket_details.append(
            {
                "bucket_range": f"[{b_low:.4f} - {b_high:.4f}]",
                "expected_pct": round(float(expected_pct[i]) * 100, 2),
                "actual_pct": round(float(actual_pct[i]) * 100, 2),
                "psi_contrib": round(float(psi_values[i]), 5),
            }
        )

    return total_psi, bucket_details


def run_full_drift_audit(
    current_date: str | None = None,
    current_days: int = 7,
    ref_days_start: int = 38,
    ref_days_end: int = 8,
    drift_threshold: float = 0.33,
    psi_threshold: float = 0.20,
    max_relative_perf_drop: float = 0.05,
    min_f2: float = 0.50,
    min_recall: float = 0.50,
    min_precision: float = 0.20,
    min_f1: float = 0.50,
    check_performance: bool = True,
) -> tuple[bool, dict]:
    """
    Exécute l'audit complet 360° de dérive et de performance.
    Retourne (should_retrain: bool, report_dict: dict).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_dir = os.path.abspath(os.path.join(script_dir, "..", "training"))

    # 1. Chargement unifié et synchronisé des données
    print(" Chargement et synchronisation du dataset de référence et de production...")
    df = load_dataset(sample_size=-1, include_graph_ids=True)

    max_dataset_date = df["trans_date_trans_time"].max()
    min_dataset_date = df["trans_date_trans_time"].min()

    # Détermination de la date cible
    target_date = None
    if current_date:
        try:
            parsed_date = pd.to_datetime(current_date)
            if min_dataset_date <= parsed_date <= max_dataset_date:
                target_date = parsed_date
            else:
                print(
                    f" Date demandée ({current_date}) hors limites [{min_dataset_date.date()} -> {max_dataset_date.date()}]."
                )
        except Exception as e:
            print(f" Erreur parsing current_date ({current_date}): {e}")

    if target_date is None:
        target_date = max_dataset_date

    print(
        f"\n======================================================================\n"
        f" AUDIT MLOPS DU {target_date.strftime('%Y-%m-%d')} (DATA DRIFT, SCORE DRIFT & CONCEPT DRIFT)\n"
        f"======================================================================"
    )

    # Découpage temporel
    current_period_end = target_date
    current_period_start = current_period_end - pd.Timedelta(days=current_days)

    reference_period_end = current_period_end - pd.Timedelta(days=ref_days_end)
    reference_period_start = current_period_end - pd.Timedelta(days=ref_days_start)

    # Si la simulation a démarré récemment, ajuster la période de référence
    if reference_period_start < min_dataset_date:
        reference_period_start = min_dataset_date
    if reference_period_end <= reference_period_start:
        reference_period_end = reference_period_start + pd.Timedelta(days=14)

    print(
        f"• Période de référence : du {reference_period_start.strftime('%Y-%m-%d %H:%M')} au {reference_period_end.strftime('%Y-%m-%d %H:%M')}"
    )
    print(
        f"• Période courante    : du {current_period_start.strftime('%Y-%m-%d %H:%M')} au {current_period_end.strftime('%Y-%m-%d %H:%M')}"
    )

    full_start_df = df[
        (df["trans_date_trans_time"] >= reference_period_start)
        & (df["trans_date_trans_time"] < reference_period_end)
    ].copy()

    full_end_df = df[
        (df["trans_date_trans_time"] >= current_period_start)
        & (df["trans_date_trans_time"] <= current_period_end)
    ].copy()

    if len(full_end_df) < 20:
        print(
            f" Avertissement : Trop peu de transactions sur la période courante ({len(full_end_df)}). Audit non testable."
        )
        return False, {"warning": "Not enough transactions in current period"}

    # ============================================================================
    # 1. ANALYSE DU DATA DRIFT P(X) AVEC EVIDENTLY AI
    # ============================================================================
    relevant_columns = get_drift_feature_columns()
    start_df = full_start_df[relevant_columns]
    end_df = full_end_df[relevant_columns]

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
    data_drift_detected = mean_drift >= drift_threshold

    print("\n--- 1. SYNTHÈSE DU DATA DRIFT P(X) (EVIDENTLY AI) ---")
    print(f"Variables testées : {len(drift_flags)}")
    print(
        f"Ratio de variables en dérive : {mean_drift * 100:.1f}% (Seuil d'alerte : {drift_threshold * 100:.0f}%)"
    )
    print(
        f"Statut Data Drift : {' DÉRIVE DÉTECTÉE !' if data_drift_detected else ' STABLE'}"
    )

    for col, detail in details.items():
        status_str = " Dérivé" if detail["drift_detected"] else " Stable"
        print(
            f"  - Colonne '{col:15s}' ({detail['method']}) : value = {detail['metric_value']:.4f} (seuil: {detail['threshold']}) | Statut : {status_str}"
        )

    # Sauvegarde HTML dans src/audit et src/training
    html_path_audit = os.path.join(script_dir, "evidently_drift_report.html")
    my_eval.save_html(html_path_audit)
    if os.path.exists(training_dir):
        try:
            my_eval.save_html(os.path.join(training_dir, "evidently_drift_report.html"))
        except Exception:
            pass

    # ============================================================================
    # 2. CHARGEMENT DU MODÈLE CHAMPION ET INFERENCE
    # ============================================================================
    champion_model, champion_version_id, decision_threshold = load_champion_model()
    qg = MLflowQualityGate()
    champion_info = qg.get_active_champion_info()

    score_drift_detected = False
    psi_score = 0.0
    psi_buckets = []
    cur_alert_rate = 0.0
    ref_alert_rate = 0.0
    alert_rate_drift = False

    perf_degraded = False
    champion_perf_summary = {}

    if champion_model is not None:

        def predict_robust(model, df_input):
            # 1. Si le pipeline Scikit-Learn expose les colonnes exactes attendues lors du fit
            if hasattr(model, "feature_names_in_"):
                expected_cols = [
                    c for c in model.feature_names_in_ if c in df_input.columns
                ]
                try:
                    return model.predict_proba(df_input[expected_cols])[:, 1]
                except Exception:
                    pass

            if hasattr(model, "named_steps") and hasattr(
                model.named_steps.get("preprocessor"), "feature_names_in_"
            ):
                expected_cols = [
                    c
                    for c in model.named_steps["preprocessor"].feature_names_in_
                    if c in df_input.columns
                ]
                try:
                    return model.predict_proba(df_input[expected_cols])[:, 1]
                except Exception:
                    pass

            # 2. Essai avec les colonnes de base historiques
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
            avail = [c for c in feature_cols if c in df_input.columns]
            try:
                return model.predict_proba(df_input[avail])[:, 1]
            except Exception:
                pass

            # 3. Essai standard avec DataFrame complet
            try:
                return model.predict_proba(df_input)[:, 1]
            except Exception:
                pass

            # 4. Essai avec BASE_FEATURE_COLUMNS
            from src.utils.features import BASE_FEATURE_COLUMNS

            base_avail = [c for c in BASE_FEATURE_COLUMNS if c in df_input.columns]
            return model.predict_proba(df_input[base_avail])[:, 1]

        y_probas_ref = predict_robust(champion_model, full_start_df)
        y_probas_cur = predict_robust(champion_model, full_end_df)

        # ========================================================================
        # 3. SCORE / OUTPUT DRIFT P(Y_hat) (POPULATION STABILITY INDEX & ALERT RATE)
        # ========================================================================
        print("\n--- 2. SCORE & OUTPUT DRIFT P(Y_hat) (PSI & ALERT RATE) ---")
        psi_score, psi_buckets = calculate_psi(
            y_probas_ref, y_probas_cur, num_buckets=10
        )
        score_drift_detected = psi_score >= psi_threshold

        ref_alert_rate = float(np.mean(y_probas_ref >= decision_threshold))
        cur_alert_rate = float(np.mean(y_probas_cur >= decision_threshold))

        rel_alert_rate_diff = 0.0
        if ref_alert_rate > 0:
            rel_alert_rate_diff = abs(cur_alert_rate - ref_alert_rate) / ref_alert_rate
            alert_rate_drift = rel_alert_rate_diff > 0.50

        print(
            f"Population Stability Index (PSI) : {psi_score:.4f} (Seuil d'alerte : {psi_threshold:.2f})"
        )
        print(
            f"  • Statut PSI : {' SCORE DRIFT DÉTECTÉ !' if score_drift_detected else ' STABLE'}"
        )
        print(f"Taux d'alerte (Score >= {decision_threshold:.4f}) :")
        print(
            f"  • Référence : {ref_alert_rate * 100:.2f}% ({int(ref_alert_rate * len(full_start_df))} / {len(full_start_df)})"
        )
        print(
            f"  • Courant   : {cur_alert_rate * 100:.2f}% ({int(cur_alert_rate * len(full_end_df))} / {len(full_end_df)})"
        )
        if alert_rate_drift:
            print(
                f"  •  Variation anormale du volume d'alertes : {rel_alert_rate_diff * 100:.1f}%"
            )

        # ========================================================================
        # 4. AUDIT DE PERFORMANCE DU CHAMPION (CONCEPT DRIFT P(Y|X))
        # ========================================================================
        if check_performance and "is_fraud" in full_end_df.columns:
            print("\n--- 3. CONTRÔLE DES PERFORMANCES DU CHAMPION (CONCEPT DRIFT) ---")
            y_true = full_end_df["is_fraud"].astype(int).values
            n_frauds = int(np.sum(y_true))

            perf_metrics, cm_dict = evaluate_predictions_and_curves(
                y_true, y_probas_cur, threshold=decision_threshold
            )

            f2_val = float(perf_metrics.get("f2_class_1", 0.0))
            rec_val = float(perf_metrics.get("rec_class_1", 0.0))
            prec_val = float(perf_metrics.get("prec_class_1", 0.0))
            f1_val = float(perf_metrics.get("f1_class_1", 0.0))
            brier_val = float(perf_metrics.get("brier_score", 0.0))

            ref_metrics = champion_info.get("metrics", {}) if champion_info else {}
            champion_ref_f2 = float(
                ref_metrics.get("f2_class_1")
                or ref_metrics.get("test_f2_score")
                or ref_metrics.get("f2", 0.80)
            )
            champion_ref_rec = float(
                ref_metrics.get("rec_class_1")
                or ref_metrics.get("recall_class_1")
                or ref_metrics.get("recall", 0.85)
            )
            champion_ref_prec = float(
                ref_metrics.get("prec_class_1")
                or ref_metrics.get("precision_class_1")
                or ref_metrics.get("precision", 0.60)
            )
            champion_ref_f1 = float(
                ref_metrics.get("f1_class_1") or ref_metrics.get("f1", 0.70)
            )
            champion_ref_brier = float(ref_metrics.get("brier_score", 0.0))

            rel_f2_drop = (
                (champion_ref_f2 - f2_val) / champion_ref_f2
                if champion_ref_f2 > 0
                else 0.0
            )
            rel_rec_drop = (
                (champion_ref_rec - rec_val) / champion_ref_rec
                if champion_ref_rec > 0
                else 0.0
            )
            rel_prec_drop = (
                (champion_ref_prec - prec_val) / champion_ref_prec
                if champion_ref_prec > 0
                else 0.0
            )
            rel_f1_drop = (
                (champion_ref_f1 - f1_val) / champion_ref_f1
                if champion_ref_f1 > 0
                else 0.0
            )

            reasons = []
            if n_frauds > 0:
                if rel_f2_drop > max_relative_perf_drop:
                    reasons.append(
                        f"Chute relative F2 : {f2_val:.4f} vs ref {champion_ref_f2:.4f} (Baisse: {rel_f2_drop * 100:.1f}% > seuil toléré {max_relative_perf_drop * 100:.1f}%)"
                    )
                if rel_rec_drop > max_relative_perf_drop:
                    reasons.append(
                        f"Chute relative Rappel : {rec_val * 100:.2f}% vs ref {champion_ref_rec * 100:.2f}% (Baisse: {rel_rec_drop * 100:.1f}% > seuil toléré {max_relative_perf_drop * 100:.1f}%)"
                    )
                if rel_f1_drop > max_relative_perf_drop:
                    reasons.append(
                        f"Chute relative F1 : {f1_val:.4f} vs ref {champion_ref_f1:.4f} (Baisse: {rel_f1_drop * 100:.1f}% > seuil toléré {max_relative_perf_drop * 100:.1f}%)"
                    )

                if f2_val < min_f2:
                    reasons.append(
                        f"Plancher absolu F2 franchi ({f2_val:.4f} < {min_f2:.4f})"
                    )
                if rec_val < min_recall:
                    reasons.append(
                        f"Plancher absolu Rappel franchi ({rec_val:.4f} < {min_recall:.4f})"
                    )
                if prec_val < min_precision:
                    reasons.append(
                        f"Plancher absolu Précision franchi ({prec_val:.4f} < {min_precision:.4f})"
                    )
                if f1_val < min_f1:
                    reasons.append(
                        f"Plancher absolu F1 franchi ({f1_val:.4f} < {min_f1:.4f})"
                    )

            perf_degraded = len(reasons) > 0

            champion_perf_summary = {
                "model_version": champion_version_id,
                "decision_threshold": decision_threshold,
                "sample_size": len(full_end_df),
                "fraud_count": n_frauds,
                "observed_metrics": {
                    "f2_score": f2_val,
                    "recall": rec_val,
                    "precision": prec_val,
                    "f1_score": f1_val,
                    "brier_score": brier_val,
                    "pr_auc": float(perf_metrics.get("pr_auc", 0.0)),
                },
                "reference_metrics_mlflow": {
                    "f2_score": champion_ref_f2,
                    "recall": champion_ref_rec,
                    "precision": champion_ref_prec,
                    "f1_score": champion_ref_f1,
                    "brier_score": champion_ref_brier,
                },
                "relative_drops": {
                    "f2_drop_pct": round(rel_f2_drop * 100, 2),
                    "recall_drop_pct": round(rel_rec_drop * 100, 2),
                    "precision_drop_pct": round(rel_prec_drop * 100, 2),
                    "f1_drop_pct": round(rel_f1_drop * 100, 2),
                },
                "confusion_matrix": cm_dict,
                "performance_degraded": perf_degraded,
                "degradation_reasons": reasons,
            }

            print(
                f"Modèle Champion : {champion_version_id} (Seuil calibré : {decision_threshold:.4f})"
            )
            print(
                f"Échantillon     : {len(full_end_df)} transactions ({n_frauds} fraudes)"
            )
            print(
                f"  • F2-Score   : {f2_val:.4f} (Ref MLflow: {champion_ref_f2:.4f}, Delta: {-rel_f2_drop * 100:+.1f}%) -> {' DÉGRADÉ' if rel_f2_drop > max_relative_perf_drop or f2_val < min_f2 else ' OK'}"
            )
            print(
                f"  • Rappel C1  : {rec_val * 100:.2f}% (Ref MLflow: {champion_ref_rec * 100:.2f}%, Delta: {-rel_rec_drop * 100:+.1f}%) -> {' DÉGRADÉ' if rel_rec_drop > max_relative_perf_drop or rec_val < min_recall else ' OK'}"
            )
            print(
                f"  • Précision  : {prec_val * 100:.2f}% (Ref MLflow: {champion_ref_prec * 100:.2f}%, Delta: {-rel_prec_drop * 100:+.1f}%) -> {' DÉGRADÉ' if rel_prec_drop > max_relative_perf_drop or prec_val < min_precision else ' OK'}"
            )
            print(
                f"  • F1-Score   : {f1_val:.4f} (Ref MLflow: {champion_ref_f1:.4f}, Delta: {-rel_f1_drop * 100:+.1f}%) -> {' DÉGRADÉ' if rel_f1_drop > max_relative_perf_drop or f1_val < min_f1 else ' OK'}"
            )
            print(
                f"  • Brier Score: {brier_val:.4f} (Calibration probabiliste, idéal: 0.0000)"
            )
            print(
                f"  • Matrice    : TP={cm_dict.get('tp', 0)}, FP={cm_dict.get('fp', 0)}, FN={cm_dict.get('fn', 0)}, TN={cm_dict.get('tn', 0)}"
            )

            if perf_degraded:
                print(
                    " ALERTE PERFORMANCES (CONCEPT DRIFT) :\n  - "
                    + "\n  - ".join(reasons)
                )
            else:
                print(" Performances en production conformes au benchmark du Champion.")

    # ============================================================================
    # 5. SYNTHÈSE GLOBALE & DÉCISION D'ORCHESTRATION
    # ============================================================================
    should_retrain = bool(
        data_drift_detected or score_drift_detected or alert_rate_drift or perf_degraded
    )

    json_summary = {
        "current_date": target_date.strftime("%Y-%m-%d"),
        "reference_period": f"{reference_period_start.strftime('%Y-%m-%d')} -> {reference_period_end.strftime('%Y-%m-%d')}",
        "current_period": f"{current_period_start.strftime('%Y-%m-%d')} -> {current_period_end.strftime('%Y-%m-%d')}",
        "sample_size": len(full_end_df),
        "reference_size": len(full_start_df),
        "data_drift": {
            "drift_detected": bool(data_drift_detected),
            "mean_drift_ratio": float(mean_drift),
            "threshold": float(drift_threshold),
            "details": details,
        },
        "score_drift": {
            "score_drift_detected": bool(score_drift_detected),
            "psi_score": float(psi_score),
            "psi_threshold": float(psi_threshold),
            "ref_alert_rate": float(ref_alert_rate),
            "cur_alert_rate": float(cur_alert_rate),
            "alert_rate_drift": bool(alert_rate_drift),
            "buckets": psi_buckets,
        },
        "performance_audit": champion_perf_summary,
        "should_retrain": should_retrain,
    }

    # Sauvegarde dans src/audit et src/training
    json_path_audit = os.path.join(script_dir, "drift_report.json")
    with open(json_path_audit, "w") as f:
        json.dump(json_summary, f, indent=4)
    if os.path.exists(training_dir):
        try:
            with open(os.path.join(training_dir, "drift_report.json"), "w") as f:
                json.dump(json_summary, f, indent=4)
        except Exception:
            pass

    print(f"\nRapport JSON complet sauvegardé : {json_path_audit}")

    print("\n" + "=" * 70)
    print(" DÉCISION D'ORCHESTRATION MLOPS (AIRFLOW TRIGGER)")
    print(
        f"  • Dérive des données P(X) (Evidently AI)   : {' DÉTECTÉE' if data_drift_detected else ' STABLE'}"
    )
    print(
        f"  • Dérive des scores P(Y_hat) (PSI & Alerte): {' DÉTECTÉE' if (score_drift_detected or alert_rate_drift) else ' STABLE'}"
    )
    print(
        f"  • Dégradation des perfs P(Y|X) (Concept)   : {' DÉTECTÉE' if perf_degraded else ' CONFORME'}"
    )
    print(
        f"  • Action finale                            : {' RÉENTRAÎNEMENT REQUIS (Exit 1)' if should_retrain else ' PAS DE RÉENTRAÎNEMENT (Exit 0)'}"
    )
    print("=" * 70 + "\n")

    return should_retrain, json_summary


def main():
    parser = argparse.ArgumentParser(
        description="Audit MLOps 360° : Data Drift P(X), Score Drift P(Y_hat) et Concept Drift P(Y|X)"
    )
    parser.add_argument(
        "--current-date",
        type=str,
        default=None,
        help="Date actuelle simulée au format YYYY-MM-DD. Par défaut: date max atteinte dans PostgreSQL/données.",
    )
    parser.add_argument(
        "--current-days",
        type=int,
        default=7,
        help="Nombre de jours pour la fenêtre courante de production (défaut: 7)",
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
        help="Seuil de ratio de variables en dérive P(X) pour lever l'alerte (défaut: 0.33 soit 2/6)",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.20,
        help="Seuil PSI maximal admissible sur les probabilités prédites P(Y_hat) (défaut: 0.20)",
    )
    parser.add_argument(
        "--max-relative-perf-drop",
        type=float,
        default=0.05,
        help="Dégradation relative maximale tolérée (-5% par défaut) par rapport aux perfs du Champion dans MLflow",
    )
    parser.add_argument(
        "--min-f2",
        type=float,
        default=0.50,
        help="Plancher de sécurité F2-Score minimal (défaut: 0.50)",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.50,
        help="Plancher de sécurité Rappel minimal (défaut: 0.50)",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.20,
        help="Plancher de sécurité Précision minimale (défaut: 0.20)",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=0.50,
        help="Plancher de sécurité F1-Score minimal (défaut: 0.50)",
    )
    parser.add_argument(
        "--no-check-performance",
        action="store_true",
        help="Désactiver l'audit de performance du Champion en production",
    )
    args = parser.parse_args()

    should_retrain, _ = run_full_drift_audit(
        current_date=args.current_date,
        current_days=args.current_days,
        ref_days_start=args.ref_days_start,
        ref_days_end=args.ref_days_end,
        drift_threshold=args.drift_threshold,
        psi_threshold=args.psi_threshold,
        max_relative_perf_drop=args.max_relative_perf_drop,
        min_f2=args.min_f2,
        min_recall=args.min_recall,
        min_precision=args.min_precision,
        min_f1=args.min_f1,
        check_performance=not args.no_check_performance,
    )

    if should_retrain:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
