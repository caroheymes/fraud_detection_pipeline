# src/dashboard/scratch/backfill_predictions.py

import os
import time

import mlflow
import pandas as pd
import psycopg2
from mlflow.tracking import MlflowClient
from psycopg2.extras import execute_batch


def calculate_age(dob, trans_date):
    try:
        dob_dt = pd.to_datetime(dob)
        trans_dt = pd.to_datetime(trans_date)
        return (trans_dt - dob_dt).dt.days / 365.25
    except Exception:
        return 35.0


def calculate_distance(lat1, lon1, lat2, lon2):
    import numpy as np

    # Formule de Haversine
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def main():
    print("🚀 Démarrage du backfill des prédictions...")
    start_time = time.time()

    # 1. Connexion Postgres
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        database=os.getenv("POSTGRES_DB", "fraud-detection"),
        user=os.getenv("POSTGRES_USER", "fraud-detection"),
        password=os.getenv("POSTGRES_PASSWORD", "fraud-detection_password"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )
    cursor = conn.cursor()

    # 2. Récupérer le modèle champion MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = MlflowClient()
    try:
        version_details = client.get_model_version_by_alias(
            "fraud_detector", "champion"
        )
        champion_run_id = version_details.run_id
        version_num = version_details.version
        print(
            f"📦 Chargement du modèle champion Version {version_num} (Run ID: {champion_run_id})..."
        )
        model = mlflow.sklearn.load_model(f"runs:/{champion_run_id}/model")
    except Exception as e:
        print(f"⚠️ Impossible de charger l'alias champion : {e}")
        print("Repli sur le dernier run disponible...")
        experiment = client.get_experiment_by_name("Default")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
        )
        champion_run_id = runs[0].info.run_id
        model = mlflow.sklearn.load_model(f"runs:/{champion_run_id}/model")
        version_num = "Dernier Run"

    # 3. Charger toutes les transactions
    print("📥 Lecture des transactions depuis PostgreSQL...")
    query = """
        SELECT 
            trans_num, trans_date_trans_time, category, amt, gender, 
            lat, long, merch_lat, merch_long, dob, city_pop
        FROM silver.rawdata
    """
    df = pd.read_sql_query(query, conn)
    print(f"📊 {len(df)} transactions chargées.")

    if len(df) == 0:
        print("❌ Aucune transaction trouvée.")
        return

    # 4. Préparation des features
    print("⚙️ Préparation des features pour le modèle...")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    # Calcul de la distance d'achat et de l'âge (si non présents ou pour recalcule propre)
    df["distance_achat"] = calculate_distance(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    df["age"] = calculate_age(df["dob"], df["trans_date_trans_time"])

    # Variables cycliques
    df["hour_sin"] = np.sin(2 * np.pi * df["trans_date_trans_time"].dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["trans_date_trans_time"].dt.hour / 24.0)

    df["weekday_sin"] = np.sin(2 * np.pi * df["trans_date_trans_time"].dt.weekday / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["trans_date_trans_time"].dt.weekday / 7.0)

    df["month_sin"] = np.sin(2 * np.pi * df["trans_date_trans_time"].dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["trans_date_trans_time"].dt.month / 12.0)

    features_list = [
        "category",
        "amt",
        "gender",
        "distance_achat",
        "age",
        "city_pop",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
    ]

    # 5. Prédiction
    print("🧠 Calcul des prédictions avec le modèle champion...")
    X = df[features_list]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    df["new_pred"] = preds
    df["new_prob"] = probs

    # 6. Mise à jour de la base par lots
    print("📤 Écriture des nouvelles prédictions dans PostgreSQL...")
    update_query = """
        UPDATE silver.rawdata
        SET prediction = %s, prediction_proba = %s, model_version = %s
        WHERE trans_num = %s
    """

    # Préparer les données pour execute_batch
    data_to_update = list(
        zip(
            df["new_pred"].astype(int).tolist(),
            df["new_prob"].astype(float).tolist(),
            [f"Version {version_num}"] * len(df),
            df["trans_num"].tolist(),
        )
    )

    batch_size = 5000
    for i in range(0, len(data_to_update), batch_size):
        batch = data_to_update[i : i + batch_size]
        execute_batch(cursor, update_query, batch)
        conn.commit()
        print(
            f"Progress : {min(i + batch_size, len(data_to_update))}/{len(data_to_update)} transactions mises à jour."
        )

    cursor.close()
    conn.close()
    print(
        f"🎉 Backfill terminé avec succès en {time.time() - start_time:.2f} secondes !"
    )


if __name__ == "__main__":
    import numpy as np

    main()
