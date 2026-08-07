# src/training/test_cartesian.py

import json
import os
import sys
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from skrub import TableVectorizer
from xgboost import XGBClassifier


# --- 1. DÉFINITION DE LA PROJECTION CARTÉSIENNE 3D ---
def latlon_to_cartesian(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return x, y, z


# --- 2. DÉFINITION DE LA DISTANCE HAVERSINE ---
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def main():
    print(
        "--- DÉMARRAGE DU TEST COMPARATIF AVEC PROJECTION CARTÉSIENNE DES COORDONNÉES ---"
    )

    # Configuration du tracking MLflow vers l'expérience centralisée "fraud_detection"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("fraud_detection")

    # Chargement du dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv"))
    if not os.path.exists(csv_path):
        print(f"Erreur : Fichier {csv_path} introuvable.")
        sys.exit(1)

    df_raw = pd.read_csv(csv_path)
    df = df_raw.sample(n=30000, random_state=42).reset_index(drop=True)

    # Feature Engineering temporel et âge
    dt_col = pd.to_datetime(df["trans_date_trans_time"])
    df["hour_sin"] = np.sin(2 * np.pi * dt_col.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * dt_col.dt.hour / 24.0)
    df["weekday_sin"] = np.sin(2 * np.pi * dt_col.dt.dayofweek / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * dt_col.dt.dayofweek / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * dt_col.dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * dt_col.dt.month / 12.0)

    df["distance_achat"] = haversine_vectorized(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    dob_col = pd.to_datetime(df["dob"])
    df["age"] = datetime.now().year - dob_col.dt.year

    # Application de la projection cartésienne
    print(
        "Calcul de la projection cartésienne 3D (x, y, z) pour l'acheteur et le marchand..."
    )
    x, y, z = latlon_to_cartesian(df["lat"], df["long"])
    df["x"] = x
    df["y"] = y
    df["z"] = z

    mx, my, mz = latlon_to_cartesian(df["merch_lat"], df["merch_long"])
    df["merch_x"] = mx
    df["merch_y"] = my
    df["merch_z"] = mz

    # Filtrage des outliers
    df = df[df.amt > df.amt.mean() - 3 * df.amt.std()].reset_index(drop=True)
    print(f"Dimensions après ingénierie : {df.shape}")

    # Définition des 4 listes de colonnes avec projection cartésienne
    feature_sets_cartesian = {
        "FeatureSet_1_Cartesian": [
            "merchant",
            "category",
            "amt",
            "first",
            "last",
            "gender",
            "street",
            "city",
            "state",
            "zip",
            "city_pop",
            "job",
            "dob",
            "trans_num",
            "unix_time",
            "distance_achat",
            "age",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
            "x",
            "y",
            "z",
            "merch_x",
            "merch_y",
            "merch_z",
        ],
        "FeatureSet_2_No_Personal_IDs_Cartesian": [
            "category",
            "amt",
            "gender",
            "city",
            "state",
            "city_pop",
            "distance_achat",
            "age",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
            "x",
            "y",
            "z",
            "merch_x",
            "merch_y",
            "merch_z",
        ],
        "FeatureSet_3_No_State_City_Cartesian": [
            "category",
            "amt",
            "gender",
            "city_pop",
            "distance_achat",
            "age",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
            "x",
            "y",
            "z",
            "merch_x",
            "merch_y",
            "merch_z",
        ],
        "FeatureSet_4_No_CityPop_Cartesian": [
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
            "x",
            "y",
            "z",
            "merch_x",
            "merch_y",
            "merch_z",
        ],
    }

    results = {}

    for name, cols in feature_sets_cartesian.items():
        print(f"\n--- Évaluation de {name} ({len(cols)} variables) ---")
        X = df[cols]
        y = df["is_fraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        vectorizer = TableVectorizer()
        X_train_encoded = vectorizer.fit_transform(X_train)
        X_test_encoded = vectorizer.transform(X_test)

        clf = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            tree_method="hist",
        )
        clf.fit(X_train_encoded, y_train)

        y_pred = clf.predict(X_test_encoded)

        prec_c1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_c1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_c1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_glob = f1_score(y_test, y_pred, average="macro", zero_division=0)
        rec_glob = recall_score(y_test, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        metrics = {
            "accuracy": float(acc),
            "prec_class_1": float(prec_c1),
            "rec_class_1": float(rec_c1),
            "f1_class_1": float(f1_c1),
            "F1_global": float(f1_glob),
            "recall_global": float(rec_glob),
        }

        results[name] = metrics

        # Enregistrement dans MLflow
        with mlflow.start_run(run_name=name):
            mlflow.log_param("num_features", len(cols))
            mlflow.log_params({"features_used": ", ".join(cols[:5]) + "..."})
            mlflow.log_param("lat_lon_representation", "cartesian_3d")
            mlflow.log_metrics(metrics)

        print(
            f"  -> Enregistré dans MLflow : F1 C1 = {f1_c1:.4f} | F1 Global = {f1_glob:.4f}"
        )

    # Sauvegarde des résultats sous forme de fichier JSON comparatif
    json_path = os.path.join(script_dir, "metrics_cartesian_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nComparatif cartésien exporté en JSON : {json_path}")


if __name__ == "__main__":
    main()
