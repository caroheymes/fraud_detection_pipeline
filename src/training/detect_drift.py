# src/training/detect_drift.py

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset


def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def main():
    parser = argparse.ArgumentParser(
        description="Détecteur de dérive des données utilisant le schéma et la logique du user"
    )
    parser.add_argument(
        "--current-date",
        type=str,
        required=True,
        help="Date actuelle simulée au format YYYY-MM-DD (ex: 2020-07-22)",
    )
    args = parser.parse_args()

    print(
        f"--- ANALYSE DE DÉRIVE DES DONNÉES AVEC LE SCHÉMA USER POUR LE : {args.current_date} ---"
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv"))

    if not os.path.exists(csv_path):
        print(f"Erreur : Dataset {csv_path} introuvable.")
        sys.exit(2)

    # Chargement du dataset complet
    df = pd.read_csv(csv_path)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    # Engineering des features requises par le schéma
    dt_col = df["trans_date_trans_time"]
    df["hour_sin"] = np.sin(2 * np.pi * dt_col.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * dt_col.dt.hour / 24.0)
    df["distance_achat"] = haversine_vectorized(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    # Détermination de la période de référence (30 premiers jours du dataset)
    reference_period_start = df["trans_date_trans_time"].min()
    reference_period_end = reference_period_start + pd.Timedelta(days=30)

    # Détermination de la période courante (fenêtre de 1 jour se terminant à la date spécifiée)
    current_period_end = pd.to_datetime(args.current_date)
    current_period_start = current_period_end - pd.Timedelta(days=1)

    print(
        f"Période de référence (30 jours) : du {reference_period_start} au {reference_period_end}"
    )
    print(
        f"Période courante (1 jour) : du {current_period_start} au {current_period_end}"
    )

    # Colonnes pertinentes pour le schéma Evidently
    relevant_columns = [
        "amt",
        "gender",
        "is_fraud",
        "hour_sin",
        "hour_cos",
        "distance_achat",
    ]

    # Filtrage des DataFrames
    start_df = df[df["trans_date_trans_time"] < reference_period_end].copy()
    start_df = start_df[relevant_columns]

    end_df = df[
        (df["trans_date_trans_time"] >= current_period_start)
        & (df["trans_date_trans_time"] < current_period_end)
    ].copy()
    end_df = end_df[relevant_columns]

    if len(end_df) < 20:
        print(
            f"Avertissement : Trop peu de transactions pour cette journée ({len(end_df)}). Dérive non testable."
        )
        sys.exit(0)

    # Définition du schéma pour Evidently
    schema = DataDefinition(
        numerical_columns=["amt", "hour_sin", "hour_cos", "distance_achat", "is_fraud"],
        categorical_columns=["gender"],
    )

    # Création des Dataset Evidently
    eval_data_1 = Dataset.from_pandas(start_df, data_definition=schema)
    eval_data_2 = Dataset.from_pandas(end_df, data_definition=schema)

    # Lancement du rapport
    report = Report([DataDriftPreset()])
    my_eval = report.run(eval_data_2, eval_data_1)

    metrics_list = my_eval.dict()["metrics"]

    # Calcul de la dérive moyenne : si mean(drift_flag) > 0.5 alors drift
    drift_flags = []
    details = {}

    for m in metrics_list:
        name = m["metric_name"]
        if name.startswith("ValueDrift"):
            col = m["config"]["column"]
            val = float(m["value"])
            threshold = float(m["config"]["threshold"])
            method = m["config"]["method"]

            # Pour les métriques de distance, il y a dérive si la distance > seuil
            if "distance" in method.lower():
                col_drift = 1.0 if val > threshold else 0.0
            # Pour les métriques de p-value, il y a dérive si la p-value < seuil
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
    drift_detected = mean_drift > 0.5

    print("\n--- SYNTHÈSE DE LA DÉRIVE (EVIDENTLY AI - SCHÉMA USER) ---")
    print(f"Nombre de variables testées : {len(drift_flags)}")
    print(f"Ratio de variables dérivées : {mean_drift * 100:.1f}%")
    print(
        f"Statut global du dataset : {'🚨 DÉRIVE DÉTECTÉE !' if drift_detected else '✅ STABLE'}"
    )

    for col, detail in details.items():
        status_str = "🚨 Dérivé" if detail["drift_detected"] else "✅ Stable"
        print(
            f"  - Column '{col}' ({detail['method']}) : value = {detail['metric_value']:.4f} (seuil: {detail['threshold']}) | Statut : {status_str}"
        )

    # Sauvegarde HTML autonome
    html_path = os.path.join(script_dir, "evidently_drift_report.html")
    my_eval.save_html(html_path)
    print(f"\nRapport HTML Evidently sauvegardé : {html_path}")

    # Version JSON
    json_summary = {
        "current_date": args.current_date,
        "sample_size": len(end_df),
        "drift_detected": bool(drift_detected),
        "mean_drift_ratio": float(mean_drift),
        "details": details,
    }

    json_path = os.path.join(script_dir, "drift_report.json")
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=4)
    print(f"Rapport JSON sauvegardé : {json_path}")

    # Code de sortie pour Airflow/Shell
    if drift_detected:
        sys.exit(1)  # Dérive détectée -> Réentraînement
    else:
        sys.exit(0)  # Stable


if __name__ == "__main__":
    main()
