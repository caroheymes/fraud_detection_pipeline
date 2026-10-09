# src/training/retrain_champion.py
"""
Dispatcher dynamique de réentraînement MLOps :
1. Interroge MLflow pour identifier l'architecture et les métriques du Champion actuel (@champion).
2. Déclenche le pipeline d'entraînement correspondant (HinSAGE+XGBoost, IsolationForest+XGBoost ou XGBoost pur).
3. Soumet le nouveau candidat au Quality Gate de promotion.
"""

from __future__ import annotations

import argparse
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
        print("[INFO] INSPECTION DU MODELE CHAMPION DANS MLFLOW")
        print("  - Modele        : fraud_detector")
        print(f"  - Version       : {champ.version}")
        print(f"  - Type          : {model_type or 'Inductive GRL (par defaut)'}")
        print(f"  - Metrique cible: {target_metric.upper()}")
        print("=" * 70)
        return model_type, target_metric
    except Exception as e:
        print(
            f"[INFO] Aucun alias @champion trouve ou connexion MLflow indisponible ({e})."
        )
        print("[INFO] Basculement sur l'architecture par defaut : Inductive GRL (F2).")
        return "inductive_grl", "f2"


def main():
    parser = argparse.ArgumentParser(
        description="Dispatcher dynamique de reentrainement MLOps du Champion"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Nombre de trials Optuna pour l'optimisation HPO (defaut: 50)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100000,
        help="Nombre de transactions pour l'entrainement (defaut: 100000 transactions recentes, -1 = tout)",
    )
    parser.add_argument(
        "--sample-position",
        type=str,
        default="last",
        choices=["last", "first"],
        help="Mode d'echantillonnage : 'last' (defaut) ou 'first'",
    )
    args = parser.parse_args()

    model_type, metric_target = get_champion_architecture()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))

    if "iforest" in model_type or "isolation" in model_type:
        print(
            "\n[INFO] Lancement du reentrainement XGBoost + Isolation Forest (optimize_xgb_iforest.py)..."
        )
        script_path = os.path.join(script_dir, "optimize_xgb_iforest.py")
        cmd = [
            sys.executable,
            script_path,
            "--n-trials",
            str(args.n_trials),
            "--sample-size",
            str(args.sample_size),
        ]
    elif (
        "xgb" in model_type and "hinsage" not in model_type and "grl" not in model_type
    ):
        print(
            "\n[INFO] Lancement du reentrainement XGBoost Tabulaire (optimize_xgb.py)..."
        )
        script_path = os.path.join(script_dir, "optimize_xgb.py")
        cmd = [
            sys.executable,
            script_path,
            "--n-trials",
            str(args.n_trials),
            "--sample-size",
            str(args.sample_size),
        ]
    else:
        print(
            f"\n[INFO] Lancement du reentrainement Inductive GRL HinSAGE + XGBoost (demo_gnn.py, Cible: {metric_target.upper()})..."
        )
        script_path = os.path.join(script_dir, "demo_gnn.py")
        cmd = [
            sys.executable,
            script_path,
            "--metric-target",
            metric_target,
            "--sample-size",
            str(args.sample_size),
            "--sample-position",
            args.sample_position,
            "--n-trials",
            str(args.n_trials),
            "--sampling-ratio",
            "0",
        ]

    print(f"Commande : {' '.join(cmd)}\n")
    res = subprocess.run(cmd, cwd=project_root)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
