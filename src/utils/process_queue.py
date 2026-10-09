# src/utils/process_queue.py
import glob
import os
import time

import pandas as pd
import requests

queue_dir = "/home/ray/project/data/queue"
files = sorted(glob.glob(os.path.join(queue_dir, "*.csv")))
print(f"=== PROCESSING {len(files)} QUEUE FILES VIA FASTAPI ===")

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

for f_path in files:
    df = pd.read_csv(f_path, dtype=str)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    data_json = df.to_dict(orient="records")
    res = requests.post(
        "http://localhost:8000/predict_batch",
        json={"transactions": data_json},
        timeout=120,
    )
    status_msg = (
        res.json().get("status")
        if res.status_code == 200
        else f"HTTP {res.status_code}"
    )
    print(
        f"[{os.path.basename(f_path)}] {len(df)} transactions -> Response: {status_msg}"
    )
    if res.status_code == 200 and res.json().get("status") == "success":
        os.remove(f_path)

# Wait 2 seconds for background tasks to commit to Postgres
time.sleep(2)
print("=== QUEUE PROCESSING COMPLETE ===")
