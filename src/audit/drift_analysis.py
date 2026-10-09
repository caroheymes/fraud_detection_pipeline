# src/audit/drift_analysis.py
"""
Module d'audit de dérive des données (Evidently AI) et point d'entrée programmatique
pour l'observabilité du Data Drift, Score Drift et Concept Drift.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Import du module transverse
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.audit.detect_drift import run_full_drift_audit


def run_evidently_drift_check() -> tuple[bool, dict[str, Any]]:
    """
    Exécute un contrôle de dérive Evidently AI et retourne (drift_detected, report_dict)
    compatible avec la suite de tests et les modules d'audit.
    """
    try:
        _should_retrain, report = run_full_drift_audit()

        # Adaptation du dictionnaire pour conformité avec l'interface historique
        data_drift_info = report.get("data_drift", {})
        drift_detected = data_drift_info.get("drift_detected", False)
        details = data_drift_info.get("details", {})

        metrics_flat = {}
        for col, det in details.items():
            metrics_flat[f"{col}_drift_score"] = det.get("metric_value", 0.0)

        compat_report = {
            "dataset_drift": drift_detected,
            "metrics": metrics_flat,
            "full_audit": report,
        }

        # Sauvegarde du rapport JSON
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "data_drift_report.json"), "w") as f:
            json.dump(compat_report, f, indent=4)

        return drift_detected, compat_report
    except Exception as e:
        print(f" Erreur lors de l'analyse Evidently : {e}")
        return False, {"dataset_drift": False, "metrics": {}, "error": str(e)}


if __name__ == "__main__":
    detected, rep = run_evidently_drift_check()
    print(f"Drift détecté : {detected}")
