# src/training/benchmark_latency.py

import os
import sys
import time

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd


def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def main():
    print("--- BENCHMARK DE LATENCE D'INFÉRENCE XGBOOST ---")

    # 1. Chargement du modèle depuis MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    # Charger depuis le run contenant le pipeline sklearn
    run_id = "a7122e479bc34cbdba6ae605abf62343"
    model_uri = f"runs:/{run_id}/model"

    print(f"Chargement du pipeline depuis : {model_uri}...")
    try:
        pipeline = mlflow.sklearn.load_model(model_uri)
        print("Pipeline chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

    # 2. Préparation d'une transaction unique
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv"))

    if not os.path.exists(csv_path):
        print(f"Erreur : Dataset {csv_path} introuvable.")
        sys.exit(1)

    df_raw = pd.read_csv(csv_path, nrows=5)

    # Feature Engineering de la ligne de transaction
    df_raw["trans_date_trans_time"] = pd.to_datetime(df_raw["trans_date_trans_time"])
    df_raw["hour_sin"] = np.sin(
        2 * np.pi * df_raw["trans_date_trans_time"].dt.hour / 24.0
    )
    df_raw["hour_cos"] = np.cos(
        2 * np.pi * df_raw["trans_date_trans_time"].dt.hour / 24.0
    )
    df_raw["weekday_sin"] = np.sin(
        2 * np.pi * df_raw["trans_date_trans_time"].dt.dayofweek / 7.0
    )
    df_raw["weekday_cos"] = np.cos(
        2 * np.pi * df_raw["trans_date_trans_time"].dt.dayofweek / 7.0
    )
    df_raw["month_sin"] = np.sin(
        2 * np.pi * df_raw["trans_date_trans_time"].dt.month / 12.0
    )
    df_raw["month_cos"] = np.cos(
        2 * np.pi * df_raw["trans_date_trans_time"].dt.month / 12.0
    )
    df_raw["distance_achat"] = haversine_vectorized(
        df_raw["lat"], df_raw["long"], df_raw["merch_lat"], df_raw["merch_long"]
    )
    dob_col = pd.to_datetime(df_raw["dob"])
    df_raw["age"] = (
        2020 - dob_col.dt.year
    )  # On simule l'âge en 2020 pour rester cohérent

    # Colonnes attendues par le pipeline
    base_cols = [
        "category",
        "amt",
        "gender",
        "distance_achat",
        "age",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
    ]

    # Prendre une seule ligne sous forme de DataFrame (pour respecter les types skrub)
    one_row = df_raw[base_cols].iloc[[0]]

    # 3. Mesures de latence
    print("\nLancement du benchmark...")

    # Premier appel (Cold start)
    start_cold = time.perf_counter()
    pred_cold = pipeline.predict_proba(one_row)
    cold_latency = (time.perf_counter() - start_cold) * 1000
    print(f"Latence au premier appel (cold start) : {cold_latency:.2f} ms")

    # Appels chauds en boucle (1000 itérations)
    n_iters = 1000
    latencies = []

    for _ in range(n_iters):
        t_start = time.perf_counter()
        pipeline.predict_proba(one_row)
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)

    latencies = np.array(latencies)

    print("\n--- STATISTIQUES SUR 1000 APPELS CHAUDS ---")
    print(f"Moyenne : {np.mean(latencies):.4f} ms")
    print(f"Médiane : {np.median(latencies):.4f} ms")
    print(f"Min : {np.min(latencies):.4f} ms")
    print(f"Max : {np.max(latencies):.4f} ms")
    print(f"Percentile 95 (P95) : {np.percentile(latencies, 95):.4f} ms")
    print(f"Percentile 99 (P99) : {np.percentile(latencies, 99):.4f} ms")
    print(f"Écart-type : {np.std(latencies):.4f} ms")


if __name__ == "__main__":
    main()
