# dags/predict_batch_v2.py

import glob
import json
import logging
import os
from datetime import datetime, timedelta

import pytz
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from sqlalchemy import create_engine, text

# ============================================================================
# LOGGING & CORE CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DB_USER = os.getenv("POSTGRES_USER", "fraud-detection")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fraud-detection_password")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_DB = os.getenv("POSTGRES_DB", "fraud-detection")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DB}"
)
OUTPUT_DIR = "/opt/airflow/project/data"


# ============================================================================
# CORE PIPELINE PIPES (INGESTION TEMPS RÉEL)
# ============================================================================
def ingest_data_from_queue(ti):
    """Tâche Airflow #1 — Ingestion temps réel des fichiers par lots dans ./data/queue"""
    queue_files = glob.glob(os.path.join(OUTPUT_DIR, "queue", "*.csv"))
    queue_files.sort()
    if not queue_files:
        raise FileNotFoundError(
            "Le répertoire ./data/queue est vide. Aucune donnée à ingérer."
        )

    # Prendre un lot (batch) de max 100 fichiers pour accélérer le traitement
    batch_size = 100
    batch_files = queue_files[:batch_size]
    filenames = [os.path.basename(f) for f in batch_files]

    # Connexion à PostgreSQL
    logger.info("Connecting to PostgreSQL container...")
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
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
# INFERENCE PIPELINE (ASYNC VIA GREQUESTS)
# ============================================================================
def trigger_batch_prediction(ti):
    """Soumission en parallèle via GRequests d'un job d'inférence en batch"""
    logger.info("Starting trigger_batch_prediction (GRequests v2)...")
    import grequests
    import pandas as pd

    batch_info_path = os.path.join(OUTPUT_DIR, "current_batch.json")
    with open(batch_info_path, "r") as f:
        filenames = json.load(f)

    # Préparer les requêtes en parallèle pour chaque fichier du lot
    reqs = []
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
    submit_url = os.getenv(
        "FASTAPI_PREDICT_BATCH_URL", "http://ray-head:8000/predict_batch"
    )

    for filename in filenames:
        file_path = os.path.join(OUTPUT_DIR, "queue", filename)
        if os.path.exists(file_path):
            try:
                # Lecture en string pour contourner le segfault de pandas sur cc_num à 19 chiffres
                df = pd.read_csv(file_path, dtype=str)
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                data_json = df.to_dict(orient="records")
                payload = {"transactions": data_json}

                # Ajout de la requête POST non envoyée à la liste grequests
                reqs.append(grequests.post(submit_url, json=payload, timeout=60))
            except Exception as e:
                logger.error(f"Erreur lors de la lecture du fichier {filename} : {e}")
                raise

    if not reqs:
        logger.warning("Aucune donnée trouvée dans le batch de fichiers.")
        return

    logger.info(
        f"Envoi de {len(reqs)} requêtes de prédiction en parallèle via GRequests à {submit_url}..."
    )
    responses = grequests.map(reqs)

    logger.info(
        "Toutes les requêtes parallèles ont été émises. Vérification des réponses..."
    )
    for i, r in enumerate(responses):
        if r is None:
            logger.error(
                f"La requête {i} pour le fichier {filenames[i]} a échoué (Pas de réponse / Timeout)"
            )
            raise RuntimeError(
                f"La requête {i} pour le fichier {filenames[i]} a échoué"
            )
        r.raise_for_status()
        result = r.json()
        logger.info(
            f"Prédictions API [{i}] ({filenames[i]}) reçues avec succès (status: {result.get('status')})"
        )


# ============================================================================
# SUPPRESSION DU FICHIER DE LA FILE D'ATTENTE APRÈS TRAITEMENT
# ============================================================================
def delete_processed_file(ti):
    """Tâche Airflow — Suppression des fichiers du batch après traitement"""
    logger.info("Suppression des fichiers du batch après traitement...")

    batch_info_path = os.path.join(OUTPUT_DIR, "current_batch.json")
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
                logger.warning(f"Fichier {filename} introuvable pour suppression.")
                errors.append(filename)
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
    queue_files = glob.glob(os.path.join(OUTPUT_DIR, "queue", "*.csv"))
    if queue_files:
        logger.info(
            f"Il reste {len(queue_files)} fichier(s) dans la queue. Relance du DAG."
        )
        return "trigger_next_run"
    else:
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
    dag_id="test_ingest_parallel_requests",
    default_args=default_args,
    description="Inférence parallélisée avec GRequests sur les données de fraude",
    schedule=None,
    start_date=datetime(2019, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["fraud-detection", "pipeline", "grequests", "predict"],
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
        trigger_dag_id="predict_batchV2",
        wait_for_completion=False,
    )

    # Si la queue est vide, on s'arrête proprement
    end_simulation = EmptyOperator(
        task_id="end_simulation",
    )

    # Définition des dépendances séquentielles
    ingest_task >> predict_task >> delete_task >> check_queue_task
    check_queue_task >> [trigger_next_run, end_simulation]
