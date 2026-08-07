# dags/drift_and_retrain.py

import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator


def check_drift_evidently(**context):
    """Exécute le script detect_drift.py dans ray-head avec la date courante"""
    print("Context keys:", list(context.keys()))
    logical_date = (
        context.get("logical_date") or context.get("dag_run").logical_date
        if context.get("dag_run")
        else None
    )
    if logical_date:
        if hasattr(logical_date, "strftime"):
            ds = logical_date.strftime("%Y-%m-%d")
        else:
            ds = str(logical_date)[:10]
    else:
        ds = datetime.now().strftime("%Y-%m-%d")
    print(f"Using date for drift test: {ds}")
    cmd = f"docker exec -t fraud-detection-ray-head python src/training/detect_drift.py --current-date {ds}"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(res.stdout)
    print(res.stderr)
    return "trigger_hpo_and_retrain" if res.returncode == 1 else "skip_retrain"


def trigger_hpo_and_retrain():
    """Exécute l'optimisation XGBoost, le réentraînement et la promotion sur Ray/MLflow"""
    cmd = "docker exec -t fraud-detection-ray-head python src/training/optimize_xgb.py --n-trials 10 --sample-size -1"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(res.stdout)
    print(res.stderr)
    if res.returncode != 0:
        raise RuntimeError(
            f"Échec de l'optimisation/réentraînement XGBoost : {res.stderr}"
        )


def export_shap_rules():
    """Exécute le script export_rules.py pour extraire les seuils, mettre à jour Redis et le JSON local"""
    cmd = "docker exec -t fraud-detection-ray-head python src/explain/export_rules.py"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(res.stdout)
    print(res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"Échec de l'export des règles de suspicion : {res.stderr}")


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
    description="Vérification quotidienne du drift et réentraînement HPO si nécessaire",
    schedule="0 2 * * *",  # Se déclenche tous les jours à 2 heures du matin
    catchup=False,
) as dag:
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
        trigger_rule="none_failed_min_one_success",  # S'exécute si train_task ou skip_task réussit sans erreur
    )

    audit_task >> [train_task, skip_task]
    [train_task, skip_task] >> export_rules_task
