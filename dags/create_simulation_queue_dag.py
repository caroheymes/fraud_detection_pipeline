# dags/create_simulation_queue_dag.py

import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

def run_create_simulation_queue():
    """Exécute le script create_simulation_queue.py dans le conteneur fraud-detection-ray-head."""
    # Exécution dans le conteneur Ray comme suggéré par la documentation et les exemples du projet
    cmd = "docker exec -t fraud-detection-ray-head python src/training/create_simulation_queue.py --duration-value 1 --duration-unit hours --steps 60"
    print(f"Exécution de la commande : {cmd}")
    
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    print("STDOUT :")
    print(res.stdout)
    print("STDERR :")
    print(res.stderr)
    
    if res.returncode != 0:
        raise RuntimeError(
            f"Le script create_simulation_queue.py a échoué avec le code de retour {res.returncode}"
        )

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}

with DAG(
    dag_id="create_simulation_queue_dag",
    default_args=default_args,
    description="Génère la file d'attente de simulation toutes les minutes",
    schedule=timedelta(minutes=1),
    catchup=False,
    max_active_runs=1,
    tags=["simulation", "fraud-detection"],
) as dag:

    run_script_task = PythonOperator(
        task_id="run_create_simulation_queue",
        python_callable=run_create_simulation_queue,
    )
