"""
dag_monitoring.py

=================
DAG Airflow unifié d'observabilité, de détection de dérive et de réentraînement conditionnel ("Self-Healing").
Il s'exécute quotidiennement à 21h00, après la période d'activité diurne (08h00 - 20h00).

Workflow :
1. trigger_evidently_monitoring : Soumet le script d'analyse d'observabilité Evidently AI à Ray.
2. evaluate_drift_and_performance : Analyse le rapport JSON produit.
   - Branche A -> trigger_retraining_on_ray (GPU) >> validate_and_promote_model (Validation & Promotion)
   - Branche B -> skip_retraining (Fin silencieuse si le modèle est performant)
"""

import json
import logging
import os
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.hooks.base import BaseHook
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

# Configuration du Logger
logger = logging.getLogger("airflow.task")

# Seuils par défaut
MAE_THRESHOLD = 4.5
P_VALUE_THRESHOLD = 0.05


def evaluate_drift_and_performance(**kwargs):
    """
    Analyse le fichier JSON produit par Evidently AI pour décider si le réentraînement est requis.
    Le réentraînement est déclenché UNIQUEMENT si la précision du modèle se dégrade (MAE > 4.5 km/h).
    La dérive de données est logguée comme avertissement mais ne déclenche pas le réentraînement.
    """
    metrics_path = "/opt/airflow/project/data/out/monitoring_metrics_morning.json"
    logger.info(f"Lecture des métriques de monitoring depuis : {metrics_path}")

    if not os.path.exists(metrics_path):
        logger.warning(
            f"⚠️ Fichier de métriques introuvable à {metrics_path}. Déclenchement du réentraînement par sécurité."
        )
        return "trigger_retraining_on_ray"

    try:
        with open(metrics_path, encoding="utf-8") as f:
            metrics_data = json.load(f)

        mae = None

        metrics_list = metrics_data.get("metrics", [])
        for m in metrics_list:
            metric_name = m.get("metric_name", "")
            if "MAE(regression_name" in metric_name:
                val = m.get("value", {})
                mae = val.get("mean") if isinstance(val, dict) else val

        logger.info(f"Métrique extraite -> MAE: {mae} km/h")

        # 2. Condition de déclenchement du réentraînement basé sur la performance réelle du modèle
        if mae is not None and mae > MAE_THRESHOLD:
            logger.info(
                f"🚨 RÉENTRAÎNEMENT REQUIS : Précision dégradée (MAE de {mae:.2f} km/h > {MAE_THRESHOLD:.2f} km/h)."
            )
            return "trigger_retraining_on_ray"
        else:
            logger.info(
                f"✅ Modèle performant (MAE de {mae if mae is not None else 0:.2f} km/h <= {MAE_THRESHOLD:.2f} km/h). "
                "Pas de réentraînement requis."
            )
            return "skip_retraining"

    except Exception as e:
        logger.error(
            f"❌ Erreur lors de l'analyse du fichier de métriques : {e}. Déclenchement de sécurité."
        )
        return "trigger_retraining_on_ray"


def submit_ray_job_callable(script_name, env_overrides=None, **kwargs):
    """
    Soumet et suit un script Python sur le cluster Ray via l'API Jobs.
    """
    ray_dashboard_url = "http://ray-head:8265"
    submit_url = f"{ray_dashboard_url}/api/jobs/"
    import time

    # Récupération de la connexion Postgres d'Airflow
    try:
        conn = BaseHook.get_connection("postgres_default")
        db_host = conn.host or "postgres"
        db_port = str(conn.port) or "5432"
        db_user = conn.login or "lyonflow"
        db_password = conn.password or ""
        db_name = conn.schema or "lyonflow"
    except Exception as e:
        logger.warning(
            f"⚠️ Connexion Airflow 'postgres_default' introuvable ({e}). Repli sur l'environnement..."
        )
        db_host = os.getenv("POSTGRES_HOST", "postgres")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        db_user = os.getenv("POSTGRES_USER", "lyonflow")
        db_password = os.getenv("POSTGRES_PASSWORD", "")
        db_name = os.getenv("POSTGRES_DB", "lyonflow")

    # Construction des variables d'environnement pour Ray
    env_vars = {
        "POSTGRES_HOST": db_host,
        "POSTGRES_PORT": db_port,
        "POSTGRES_USER": db_user,
        "POSTGRES_PASSWORD": db_password,
        "POSTGRES_DB": db_name,
    }

    # Appliquer les surcharges d'environnement si fournies
    if env_overrides:
        env_vars.update(env_overrides)

    payload = {
        "entrypoint": f"cd /home/ray/project && python {script_name}",
        "runtime_env": {"env_vars": env_vars},
    }

    logger.info(f"Soumission du script '{script_name}' à Ray sur {submit_url}...")
    response = requests.post(
        submit_url, json=payload, headers={"Connection": "close"}, timeout=(5, 30)
    )
    response.raise_for_status()
    job_id = response.json()["job_id"]
    logger.info(f"Job Ray soumis avec succès. ID : {job_id}")

    # Suivi du statut
    status_url = f"{ray_dashboard_url}/api/jobs/{job_id}"
    while True:
        time.sleep(15)
        try:
            status_resp = requests.get(
                status_url, headers={"Connection": "close"}, timeout=(5, 15)
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()
            status = status_data["status"]
            logger.info(f"État du Job Ray {job_id} : {status}")
        except Exception as e:
            logger.warning(
                f"⚠️ Erreur lors de la récupération du statut du Job Ray {job_id} : {e}. "
                "Nouvelle tentative au prochain cycle..."
            )
            continue

        if status == "SUCCEEDED":
            logger.info(f"🟢 Le Job Ray '{script_name}' s'est complété avec succès !")
            break
        elif status in ["FAILED", "STOPPED"]:
            error_msg = (
                f"🔴 Le Job Ray '{script_name}' a échoué avec le statut : {status}."
            )
            logger.error(error_msg)
            try:
                logs_resp = requests.get(
                    f"{status_url}/logs",
                    headers={"Connection": "close"},
                    timeout=(5, 30),
                )
                if logs_resp.status_code == 200:
                    logger.error(
                        f"Logs du job Ray :\n{logs_resp.json().get('logs', '')}"
                    )
            except Exception as le:
                logger.warning(f"Impossible de récupérer les logs du job : {le}")
            raise Exception(error_msg)


# Configuration par défaut du DAG
default_args = {
    "owner": "lyonflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="lyonflow_monitoring_pipeline",
    default_args=default_args,
    description="Pipeline unifié d observabilite, de détection de dérive et de réentraînement STGCN",
    schedule="0 21 * * *",  # S'exécute chaque jour à 21h00 locale/UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["lyonflow", "monitoring", "retraining", "self-healing", "mlops"],
) as dag:
    # 1. Étape de Monitoring (Evidently AI)
    run_monitoring = PythonOperator(
        task_id="trigger_evidently_monitoring",
        python_callable=submit_ray_job_callable,
        op_kwargs={
            "script_name": "utils/monitoring_evidently.py",
            "env_overrides": {"DATA_FOLDER_OUT": "/home/ray/project/data/out"},
        },
    )

    # 2. Branchement de dérive
    evaluate_metrics = BranchPythonOperator(
        task_id="evaluate_drift_and_performance",
        python_callable=evaluate_drift_and_performance,
    )

    # 3. Réentraînement sur GPU
    trigger_retraining = PythonOperator(
        task_id="trigger_retraining_on_ray",
        python_callable=submit_ray_job_callable,
        op_kwargs={
            "script_name": "training/stgcn/run_retraining.py",
            "env_overrides": {"EPOCHS": "100", "HORIZONS": "6,12,36"},
        },
    )

    # 4. Validation & Promotion
    validate_and_promote = PythonOperator(
        task_id="validate_and_promote_model",
        python_callable=submit_ray_job_callable,
        op_kwargs={
            "script_name": "training/stgcn/validate_and_promote.py",
            "env_overrides": {"HORIZONS": "6,12,36"},
        },
    )

    # 5. Fin silencieuse si pas de dérive
    skip_retraining = EmptyOperator(
        task_id="skip_retraining",
    )

    # Définition des dépendances du workflow unifié
    run_monitoring >> evaluate_metrics
    evaluate_metrics >> [trigger_retraining, skip_retraining]
    trigger_retraining >> validate_and_promote
