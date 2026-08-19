# src/audit/drift_analysis.py
import json
import os

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def run_evidently_drift_check():
    print("Démarrage de l'analyse de drift avec Evidently AI...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.join(script_dir, "..", "training", "reference_data.csv")
    
    if not os.path.exists(ref_path):
        print(f"Erreur : Le fichier de référence {ref_path} n'existe pas.")
        return False, {}
        
    try:
        # 1. Charger les données de référence (données de test historiques)
        df_ref = pd.read_csv(ref_path)
        
        # 2. Préparer un échantillon actuel pour l'analyse (par exemple les 1000 dernières lignes)
        df_curr = df_ref.tail(1000).copy()
        df_reference = df_ref.head(1000).copy()
        
        # Définition des variables cibles
        relevant_columns = ['amt', 'gender', 'is_fraud', 'hour_sin', 'hour_cos', 'distance_achat']
        
        df_reference_filtered = df_reference[relevant_columns]
        df_curr_filtered = df_curr[relevant_columns]
        
        # 3. Lancer Evidently Report (DataDriftPreset)
        report = Report(metrics=[
            DataDriftPreset()
        ])
        
        report.run(reference_data=df_reference_filtered, current_data=df_curr_filtered)
        report_dict = report.dict()
        
        # Extraire le statut du drift
        metrics = report_dict["metrics"]
        drift_detected = False
        drift_metrics = {"dataset_drift": False, "metrics": {}}
        
        # Trouver la métrique de drift global du dataset
        for m in metrics:
            if m.get("metric") == "DatasetDriftMetric":
                drift_detected = m["result"]["dataset_drift"]
                drift_metrics["dataset_drift"] = drift_detected
            elif m.get("metric") == "ColumnDriftMetric":
                col = m["result"]["column_name"]
                drift_score = m["result"]["drift_score"]
                drift_metrics["metrics"][f"{col}_drift_score"] = float(drift_score)
                
        print(f"Analyse terminée. Drift global détecté : {drift_detected}")
        return drift_detected, drift_metrics
        
    except Exception as e:
        print(f"Erreur lors de l'exécution d'Evidently : {e}")
        return False, {"error": str(e)}

if __name__ == "__main__":
    detected, report = run_evidently_drift_check()
    # Sauvegarde du rapport pour le tableau de bord
    with open("data_drift_report.json", "w") as f:
        json.dump(report, f, indent=4)
