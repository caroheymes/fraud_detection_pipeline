# src/training/retrain_champion.py
"""
Dispatcher dynamique de réentraînement MLOps :
1. Interroge MLflow pour identifier l'architecture et les métriques du Champion actuel (@champion).
2. Déclenche le pipeline d'entraînement correspondant (HinSAGE+XGBoost ou XGBoost pur).
3. Soumet le nouveau candidat au Quality Gate de promotion.
"""

from __future__ import annotations

import os
import subprocess
import sys

from mlflow.tracking import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")


def get_champion_architecture() -> tuple[str, str]:
    """Interroge MLflow Model Registry pour inspecter le modèle @champion."""
    try:
        client = MlflowClient(tracking_uri=MLFLOW_URI)
        champ = client.get_model_version_by_alias("fraud_detector", "champion")
        run = client.get_run(champ.run_id)
        params = run.data.params

        model_type = params.get("model_type", "").lower()
        target_metric = params.get(
            "optimization_metric_target",
            params.get("target_metric", "f2"),
        ).lower()

        print("=" * 70)
        print("🔍 INSPECTION DU MODÈLE CHAMPION DANS MLFLOW")
        print("  • Modèle        : fraud_detector")
        print(f"  • Version       : {champ.version}")
        print(f"  • Type          : {model_type or 'Inductive GRL (par défaut)'}")
        print(f"  • Métrique cible: {target_metric.upper()}")
        print("=" * 70)
        return model_type, target_metric
    except Exception as e:
        print(f"ℹ️ Aucun alias @champion trouvé ou connexion MLflow indisponible ({e}).")
        print("👉 Basculement sur l'architecture par défaut : Inductive GRL (F2).")
        return "inductive_grl", "f2"


def main():
    model_type, metric_target = get_champion_architecture()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))

    if "xgb" in model_type and "hinsage" not in model_type and "grl" not in model_type:
        print("\n🚀 Lancement du réentraînement XGBoost Tabulaire (optimize_xgb.py)...")
        script_path = os.path.join(script_dir, "optimize_xgb.py")
        cmd = [
            sys.executable,
            script_path,
            "--n-trials",
            "100",
            "--sample-size",
            "-1",
        ]
    else:
        print(
            f"\n🚀 Lancement du réentraînement Inductive GRL HinSAGE + XGBoost (demo_gnn.py, Cible: {metric_target.upper()})..."
        )
        script_path = os.path.join(script_dir, "demo_gnn.py")
        cmd = [
            sys.executable,
            script_path,
            "--metric-target",
            metric_target,
            "--sample-size",
            "-1",
            "--sample-position",
            "last",
            "--n-trials",
            "100",
            "--sampling-ratio",
            "0",
        ]

    print(f"Commande : {' '.join(cmd)}\n")
    res = subprocess.run(cmd, cwd=project_root)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
