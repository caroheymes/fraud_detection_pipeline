#  dag_pipeline.py
# docker exec -t fraud-detection-ray-head python -m py_compile dags/dag_pipeline.py
# docker logs fraud-detection-airflow-webserver mot de passe

import json
import logging
import os

# import requests
import sys
from datetime import datetime, timedelta

# import numpy as np
# import pandas as pd
import pytz
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from sqlalchemy import text

# Import du socle transverse de base de données
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.db import get_postgres_engine

# ============================================================================
# LOGGING & CORE CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "/opt/airflow/project/data"


# ============================================================================
# CORE PIPELINE PIPES (EXECUTED AS PYTHON TASKS)
# ============================================================================
def ingest_data_from_queue(ti):
    """Tâche Airflow #1 — Ingestion temps réel des fichiers par lots dans ./data/queue"""
    queue_dir = os.path.join(OUTPUT_DIR, "queue")
    if not os.path.exists(queue_dir):
        raise FileNotFoundError(f"Le répertoire {queue_dir} n'existe pas.")

    batch_size = 100
    filenames = []

    with os.scandir(queue_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".csv"):
                try:
                    if entry.stat().st_size == 0:
                        os.remove(entry.path)
                        continue
                    filenames.append(entry.name)
                    if len(filenames) >= batch_size:
                        break
                except Exception:
                    pass

    if not filenames:
        logger.info(
            "Le répertoire ./data/queue est vide. Aucun fichier à ingérer pour ce cycle."
        )
        batch_info_path = os.path.join(OUTPUT_DIR, "current_batch.json")
        with open(batch_info_path, "w") as f:
            json.dump([], f)
        return

    # Connexion à PostgreSQL
    logger.info("Connecting to PostgreSQL container...")
    engine = get_postgres_engine()
    try:
        timezone = pytz.timezone("Europe/Paris")
        fetched_at = datetime.now(timezone)

        with engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS bronze;"))
            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS silver.ingested_file (
                    id SERIAL PRIMARY KEY,
                    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    file_name VARCHAR(255) NOT NULL
                );
            """)
            )

            # Insertion en batch dans ingested_file
            logger.info(
                f"Inserting {len(filenames)} data file_names in silver.ingested_file ..."
            )
            insert_query = text("""
                INSERT INTO silver.ingested_file (fetched_at, file_name)
                VALUES (:fetched_at, :file_name);
            """)

            params = [
                {"fetched_at": fetched_at, "file_name": name} for name in filenames
            ]
            conn.execute(insert_query, params)

        logger.info(f"🟢 Ingestion of {len(filenames)} files successfully registered!")
        batch_info_path = os.path.join(OUTPUT_DIR, "current_batch.json")
        with open(batch_info_path, "w") as f:
            json.dump(filenames, f)
    except Exception as e:
        logger.error(
            f"❌ Erreur lors de l'insertion dans la table silver.ingested_file : {e}"
        )
        raise
    finally:
        engine.dispose()


# ============================================================================
# INFERENCE PIPELINE (EXECUTED AS PYTHON TASK)
# ============================================================================


def trigger_batch_prediction(ti):
    """Soumission et surveillance d'un job d'inférence en batch http://ray-head:8000/predict_batch"""
    logger.info("Starting trigger_batch_prediction task...")
    import pandas as pd
    import requests

    batch_info_path = os.path.join(OUTPUT_DIR, "current_batch.json")
    if not os.path.exists(batch_info_path):
        logger.warning(
            f"Le fichier d'information du batch {batch_info_path} n'existe pas. Inférence ignorée."
        )
        return

    with open(batch_info_path, "r") as f:
        filenames = json.load(f)

    # Lire tous les fichiers du batch et les concaténer en évitant le segfault
    dfs = []
    numeric_cols = [
        "amt",
        "lat",
        "long",
        "city_pop",
        "unix_time",
        "merch_lat",
        "merch_long",
        "is_fraud",
    ]
    for filename in filenames:
        file_path = os.path.join(OUTPUT_DIR, "queue", filename)
        if os.path.exists(file_path):
            if os.path.getsize(file_path) == 0:
                logger.warning(
                    f"Fichier 0-octet ignoré lors de la prédiction : {filename}"
                )
                continue
            try:
                # Lecture en string pour contourner le segfault de pandas sur cc_num à 19 chiffres
                df = pd.read_csv(file_path, dtype=str)
                if df.empty or len(df.columns) == 0:
                    logger.warning(f"Fichier sans colonnes/données ignoré : {filename}")
                    continue
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                dfs.append(df)
            except pd.errors.EmptyDataError:
                logger.warning(
                    f"Fichier vide sans entête ignoré (EmptyDataError) : {filename}"
                )
                continue
            except Exception as e:
                logger.error(f"Erreur lors de la lecture du fichier {filename} : {e}")
                raise

    if not dfs:
        logger.warning("Aucune donnée trouvée dans le batch de fichiers.")
        return

    data = pd.concat(dfs, ignore_index=True)
    data_json = data.to_dict(orient="records")

    submit_url = os.getenv(
        "FASTAPI_PREDICT_BATCH_URL", "http://ray-head:8000/predict_batch"
    )
    data_payload = {"transactions": data_json}

    logger.info(
        f"Soumission du batch de prédiction ({len(data_json)} transactions de {len(filenames)} fichiers) à {submit_url}..."
    )
    r = requests.post(submit_url, json=data_payload, timeout=60)
    r.raise_for_status()
    result = r.json()
    logger.info(f"Prédictions API reçues avec succès (status: {result.get('status')})")


# ============================================================================
# SUPPRESSION DU FICHIER DE LA FILE D'ATTENTE APRÈS TRAITEMENT
# ============================================================================


def delete_processed_file(ti):
    """Tâche Airflow — Suppression des fichiers du batch après traitement"""
    logger.info("Suppression des fichiers du batch après traitement...")

    batch_info_path = os.path.join(OUTPUT_DIR, "current_batch.json")
    if not os.path.exists(batch_info_path):
        logger.warning(
            f"Le fichier d'information du batch {batch_info_path} n'existe pas. Rien à supprimer."
        )
        return

    with open(batch_info_path, "r") as f:
        filenames = json.load(f)

    errors = []
    for filename in filenames:
        file_path = os.path.join(OUTPUT_DIR, "queue", filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Fichier {filename} supprimé physiquement de la queue.")
            else:
                logger.info(f"Fichier {filename} déjà supprimé ou absent de la queue.")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du fichier {filename} : {e}")
            errors.append(filename)

    if errors:
        raise RuntimeError(
            f"Erreur lors de la suppression de {len(errors)} fichiers dans la queue."
        )

    if os.path.exists(batch_info_path):
        os.remove(batch_info_path)


def check_queue_func():
    """Tâche de décision : Reste-t-il des fichiers à traiter ?"""
    queue_dir = os.path.join(OUTPUT_DIR, "queue")
    if os.path.exists(queue_dir):
        with os.scandir(queue_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".csv"):
                    return "trigger_next_run"
    logger.info("Plus aucun fichier dans la queue. Fin de la simulation.")
    return "end_simulation"


# ============================================================================
# AIRFLOW DAG ORCHESTRATION LAYOUT
# ============================================================================
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(seconds=5),
}

with DAG(
    dag_id="batch_prediction_pipeline",
    default_args=default_args,
    description="Inférence périodique sur les données de fraude",
    schedule="* * * * *",  # Toutes les minutes
    start_date=datetime(2019, 1, 1),
    catchup=False,
    max_active_runs=1,  # IMPORTANT : Traite les fichiers un par un
    tags=["fraud-detection", "pipeline", "ingest", "predict"],
) as dag:
    ingest_task = PythonOperator(
        task_id="ingest_data_from_queue",
        python_callable=ingest_data_from_queue,
    )

    predict_task = PythonOperator(
        task_id="batch_predict_with_ray",
        python_callable=trigger_batch_prediction,
    )

    delete_task = PythonOperator(
        task_id="delete_processed_file",
        python_callable=delete_processed_file,
    )

    # Tâche d'évaluation de la boucle
    check_queue_task = BranchPythonOperator(
        task_id="check_queue",
        python_callable=check_queue_func,
    )

    # Si la queue contient des fichiers, on auto-déclenche ce DAG
    trigger_next_run = TriggerDagRunOperator(
        task_id="trigger_next_run",
        trigger_dag_id="batch_prediction_pipeline",
        wait_for_completion=False,
    )

    # Si la queue est vide, fin de la simulation
    end_simulation = EmptyOperator(
        task_id="end_simulation",
    )

    # Définition des dépendances séquentielles
    ingest_task >> predict_task >> delete_task >> check_queue_task
    check_queue_task >> [trigger_next_run, end_simulation]
