# dags/ingestion_feedback.py
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "ingestion_feedback_labels",
    default_args=default_args,
    description="DAG d'ingestion des fraudes réelles (labels/feedback) depuis les banques",
    schedule="@weekly",
    catchup=False,
) as dag:

    def ingest_labels():
        # Récupération des données bancaires et mise à jour de Postgres
        print("Importation des rejets bancaires et des signalements marchands...")

    ingest_task = PythonOperator(
        task_id="ingest_labels_to_postgres",
        python_callable=ingest_labels,
    )
