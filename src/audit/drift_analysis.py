# src/audit/drift_analysis.py
import json
import os

import pandas as pd
try:
    from evidently.presets import DataDriftPreset
    from evidently import Report
except ImportError:
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
        relevant_columns = [
            "amt",
            "gender",
            "is_fraud",
            "hour_sin",
            "hour_cos",
            "distance_achat",
        ]

        df_reference_filtered = df_reference[relevant_columns]
        df_curr_filtered = df_curr[relevant_columns]

        # 3. Lancer Evidently Report (DataDriftPreset)
        try:
            from evidently import DataDefinition, Dataset
            schema = DataDefinition(
                numerical_columns=["amt", "hour_sin", "hour_cos", "distance_achat", "is_fraud"],
                categorical_columns=["gender"],
            )
            eval_ref = Dataset.from_pandas(df_reference_filtered, data_definition=schema)
            eval_curr = Dataset.from_pandas(df_curr_filtered, data_definition=schema)
            report = Report([DataDriftPreset()])
            my_eval = report.run(eval_curr, eval_ref)
            report_dict = my_eval.dict() if hasattr(my_eval, "dict") else my_eval.as_dict()
            
            metrics = report_dict.get("metrics", [])
            drift_flags = []
            drift_metrics = {"dataset_drift": False, "metrics": {}}
            for m in metrics:
                name = m.get("metric_name", "")
                if name.startswith("ValueDrift"):
                    col = m["config"]["column"]
                    val = float(m["value"])
                    threshold = float(m["config"]["threshold"])
                    method = m["config"]["method"]
                    col_drift = 1.0 if ("distance" in method.lower() and val > threshold) or ("distance" not in method.lower() and val < threshold) else 0.0
                    drift_flags.append(col_drift)
                    drift_metrics["metrics"][f"{col}_drift_score"] = val
            drift_detected = bool(len(drift_flags) > 0 and (sum(drift_flags) / len(drift_flags)) > 0.5)
            drift_metrics["dataset_drift"] = drift_detected
        except Exception:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=df_reference_filtered, current_data=df_curr_filtered)
            report_dict = report.as_dict() if hasattr(report, "as_dict") else getattr(report, "dict", lambda: {})()
            metrics = report_dict.get("metrics", [])
            drift_detected = False
            drift_metrics = {"dataset_drift": False, "metrics": {}}
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
