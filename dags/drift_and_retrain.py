# dags/drift_and_retrain.py

import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator


def refresh_gold_marts_dbt():
    """Tâche Airflow — Exécution de dbt run pour rafraîchir les tables Gold avant l'audit de drift"""
    import os

    print("Exécution de dbt run pour mettre à jour les tables Gold...")
    env = os.environ.copy()
    env["POSTGRES_HOST"] = os.getenv("POSTGRES_HOST", "postgres")
    env["POSTGRES_PORT"] = os.getenv("POSTGRES_PORT", "5432")
    res = subprocess.run(
        ["dbt", "run", "--profiles-dir", "."],
        cwd="/opt/airflow/project/dbt_project",
        env=env,
        capture_output=True,
        text=True,
    )
    print(res.stdout)
    if res.returncode != 0:
        print(f"Erreur dbt run : {res.stderr}")
        raise RuntimeError(f"Échec de dbt run : {res.stderr}")
    print("[INFO] Tables Gold (SLA, marchands, etc.) rafraîchies avec succès par dbt !")


def check_drift_evidently(**context):
    """Exécute le script detect_drift.py dans ray-head avec la date simulée courante"""
    cmd = (
        "docker exec -i fraud-detection-ray-head python src/audit/detect_drift.py "
        "--current-days 7 --ref-days-start 38 --ref-days-end 8 "
        "--max-relative-perf-drop 0.05 --psi-threshold 0.20 "
        "--min-f2 0.50 --min-recall 0.50 --min-precision 0.20 --min-f1 0.50"
    )
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(res.stdout)
    print(res.stderr)
    return "trigger_hpo_and_retrain" if res.returncode == 1 else "skip_retrain"


def trigger_hpo_and_retrain():
    """Exécute le pipeline de réentraînement dynamique selon l'architecture du Champion actif dans MLflow"""
    cmd = "docker exec -i fraud-detection-ray-head python src/training/retrain_champion.py"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(res.stdout)
    print(res.stderr)
    if res.returncode != 0:
        raise RuntimeError(
            f"Échec de l'optimisation/réentraînement du modèle : {res.stderr or res.stdout}"
        )


def export_shap_rules():
    """Exécute le script export_rules.py pour extraire les seuils, mettre à jour Redis et le JSON local"""
    cmd = "docker exec -i fraud-detection-ray-head python src/explain/export_rules.py"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(res.stdout)
    print(res.stderr)
    if res.returncode != 0:
        raise RuntimeError(
            f"Échec de l'export des règles de suspicion : {res.stderr or res.stdout}"
        )


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2020, 7, 20),
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

with DAG(
    "drift_and_retrain_loop",
    default_args=default_args,
    description="Vérification périodique du drift/perfs et réentraînement HPO si nécessaire",
    schedule="*/30 * * * *",  # Toutes les 30 minutes
    catchup=False,
) as dag:
    dbt_task = PythonOperator(
        task_id="refresh_gold_marts_dbt",
        python_callable=refresh_gold_marts_dbt,
    )

    audit_task = BranchPythonOperator(
        task_id="audit_drift",
        python_callable=check_drift_evidently,
    )

    train_task = PythonOperator(
        task_id="trigger_hpo_and_retrain",
        python_callable=trigger_hpo_and_retrain,
    )

    skip_task = EmptyOperator(
        task_id="skip_retrain",
    )

    export_rules_task = PythonOperator(
        task_id="export_rules",
        python_callable=export_shap_rules,
    )

    dbt_task >> audit_task
    audit_task >> [train_task, skip_task]
    train_task >> export_rules_task
