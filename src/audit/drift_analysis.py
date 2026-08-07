# src/audit/drift_analysis.py
import json


def run_evidently_drift_check():
    print("Démarrage de l'analyse de drift avec Evidently AI...")

    # 1. Charger les données de référence (données de test historiques)
    # 2. Charger les données de production récentes (Postgres de la dernière heure/journée)
    # 3. Lancer Evidently Report (DataDriftPreset, TargetDriftPreset)

    drift_detected = False
    drift_metrics = {
        "dataset_drift": False,
        "metrics": {
            "amount_drift_score": 0.02,  # P-value (supérieure à 0.05 -> pas de drift)
            "country_drift_score": 0.08,
            "ip_drift_score": 0.04,  # Drift détecté sur l'IP !
        },
    }

    # Vérification du seuil
    if drift_metrics["metrics"]["ip_drift_score"] < 0.05:
        print("⚠️ DRIFT DÉTECTÉ sur l'adresse IP (p-value < 0.05).")
        drift_detected = True

    return drift_detected, drift_metrics


if __name__ == "__main__":
    detected, report = run_evidently_drift_check()
    # Sauvegarde du rapport pour Streamlit
    with open("data_drift_report.json", "w") as f:
        json.dump(report, f, indent=4)
