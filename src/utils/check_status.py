# src/utils/check_status.py
import os
import sys

import pandas as pd
from mlflow.tracking import MlflowClient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.db import get_postgres_engine

client = MlflowClient()

print("=== REGISTERED MODEL ALIASES ===")
try:
    rm = client.get_registered_model("fraud_detector")
    print("Aliases:", rm.aliases)
except Exception as e:
    print("Error getting registered model:", e)

print("\n=== MODEL VERSIONS & METRICS ===")
try:
    models = client.search_model_versions("name='fraud_detector'")
    for mv in sorted(models, key=lambda x: int(x.version)):
        run = client.get_run(mv.run_id)
        m = run.data.metrics
        p = run.data.params
        print(
            f"Version {mv.version:>2} | Run ID: {mv.run_id[:8]} | Run Name: {run.info.run_name:<30} | F2 C1: {m.get('f2_class_1', 0.0):.4f} | F1 C1: {m.get('f1_class_1', 0.0):.4f} | Rec C1: {m.get('rec_class_1', 0.0):.4f} | Prec C1: {m.get('prec_class_1', 0.0):.4f} | ROC-AUC: {m.get('roc_auc', 0.0):.4f}"
        )
except Exception as e:
    print("Error search_model_versions:", e)

print("\n=== POSTGRESQL (silver.rawdata) INFERENCES ===")
eng = get_postgres_engine()
df_mods = pd.read_sql(
    "SELECT model_version, COUNT(*) as count, MIN(trans_date_trans_time) as min_date, MAX(trans_date_trans_time) as max_date, COUNT(CASE WHEN is_fraud::int = 1 AND prediction::int = 1 THEN 1 END) as tp, COUNT(CASE WHEN is_fraud::int = 0 AND prediction::int = 1 THEN 1 END) as fp, COUNT(CASE WHEN is_fraud::int = 1 AND prediction::int = 0 THEN 1 END) as fn, COUNT(CASE WHEN is_fraud::int = 0 AND prediction::int = 0 THEN 1 END) as tn FROM silver.rawdata GROUP BY 1 ORDER BY min_date ASC",
    eng,
)
print(df_mods)

print("\n=== DATA QUEUE STATUS ===")
queue_dir = os.path.join(project_root, "data/queue")
if os.path.exists(queue_dir):
    files = [f for f in os.listdir(queue_dir) if f.endswith(".csv")]
    print(f"Files waiting in data/queue: {len(files)}")
    if files:
        print(f"  First 3 files: {files[:3]}")
else:
    print("Queue dir does not exist.")
