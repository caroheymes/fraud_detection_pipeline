# src/training/demo_v2.py

import argparse
import json
import os
import sys
from datetime import datetime

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from skrub import TableVectorizer
from xgboost import XGBClassifier


# --- 1. DÉFINITION DE LA DISTANCE HAVERSINE VECTORISÉE ---
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0  # Rayon de la Terre en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


# --- 2. PROJECTION CARTÉSIENNE 3D ---
def latlon_to_cartesian(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return x, y, z


# --- 3. SOUS-ÉCHANTILLONNAGE MAJORITAIRE ---
def get_moderate_sampled_data(X_train_df, y_train_series, target_ratio=0.05):
    if target_ratio <= 0.0 or target_ratio >= 1.0:
        return X_train_df, y_train_series

    train_df = pd.concat([X_train_df, y_train_series], axis=1)
    fraud = train_df[train_df["is_fraud"] == 1]
    normal = train_df[train_df["is_fraud"] == 0]

    n_fraud = len(fraud)
    n_normal_required = int(n_fraud * (1.0 / target_ratio - 1.0))

    if n_normal_required < len(normal):
        normal_sampled = normal.sample(n=n_normal_required, random_state=42)
    else:
        normal_sampled = normal

    sampled_df = pd.concat([fraud, normal_sampled]).sample(frac=1.0, random_state=42)
    return sampled_df.drop(columns=["is_fraud"]), sampled_df["is_fraud"]


def main():
    # Définition des arguments de la ligne de commande pour paramétrer l'entraînement
    parser = argparse.ArgumentParser(
        description="Script d'entraînement MLOps XGBoost V2"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="XGBoost_Run_" + datetime.now().strftime("%H%M%S"),
        help="Nom explicite du run dans MLflow",
    )
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=0.0,
        help="Ratio d'échantillon de fraude (ex: 0.05, 0.10. 0.0 = pas de sampling)",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=3.0,
        help="Seuil en écarts-types pour le filtrage des montants aberrants",
    )
    parser.add_argument(
        "--geo-mode",
        type=str,
        default="none",
        choices=["none", "raw", "cartesian", "h3"],
        help="Mode de représentation géographique (none, raw, cartesian, h3)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Nombre d'estimateurs (arbres) pour XGBoost",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Profondeur maximale des arbres de décision",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Taux d'apprentissage du modèle",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30000,
        help="Taille de l'échantillon de données utilisé",
    )
    args = parser.parse_args()

    print(f"--- DÉMARRAGE DU RUN MLflow : {args.run_name} ---")

    # Configuration MLflow vers l'expérience centralisée
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("fraud_detection")

    # Chargement des données
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv"))
    if not os.path.exists(csv_path):
        print(f"Erreur : Dataset {csv_path} introuvable.")
        sys.exit(1)

    print(f"Chargement de {args.sample_size} lignes de données...")
    df_raw = pd.read_csv(csv_path)
    df = df_raw.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    # Feature Engineering de base
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

    # Filtrage des outliers montants basé sur l'argument utilisateur
    mean_amt = df.amt.mean()
    std_amt = df.amt.std()
    initial_len = len(df)
    df = df[df.amt > mean_amt - args.outlier_threshold * std_amt].reset_index(drop=True)
    print(f"Outliers filtrés : {initial_len - len(df)} lignes supprimées.")

    # Liste des variables de base à conserver
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

    # Application de la stratégie géographique choisie
    if args.geo_mode == "raw":
        base_cols += ["lat", "long", "merch_lat", "merch_long", "city_pop"]
    elif args.geo_mode == "cartesian":
        x, y, z = latlon_to_cartesian(df["lat"], df["long"])
        df["x"], df["y"], df["z"] = x, y, z
        mx, my, mz = latlon_to_cartesian(df["merch_lat"], df["merch_long"])
        df["merch_x"], df["merch_y"], df["merch_z"] = mx, my, mz
        base_cols += ["x", "y", "z", "merch_x", "merch_y", "merch_z", "city_pop"]
    elif args.geo_mode == "h3":
        import h3

        df["h3_client"] = [
            h3.latlng_to_cell(lat, lon, 5) for lat, lon in zip(df.lat, df.long)
        ]
        df["h3_merchant"] = [
            h3.latlng_to_cell(lat, lon, 5)
            for lat, lon in zip(df.merch_lat, df.merch_long)
        ]
        base_cols += ["h3_client", "h3_merchant", "city_pop"]
    else:
        # "none" : On conserve seulement city_pop
        base_cols += ["city_pop"]

    X = df[base_cols]
    y = df["is_fraud"]

    # Division Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Application du sampling modéré sur le train
    if args.sampling_ratio > 0.0:
        print(
            f"Application de l'échantillonnage modéré (ratio cible : {args.sampling_ratio * 100:.1f}%)"
        )
        X_train_sampled, y_train_sampled = get_moderate_sampled_data(
            X_train, y_train, target_ratio=args.sampling_ratio
        )
    else:
        X_train_sampled, y_train_sampled = X_train, y_train

    # Vectorisation Skrub
    print("Vectorisation des variables avec skrub.TableVectorizer...")
    vectorizer = TableVectorizer()
    X_train_encoded = vectorizer.fit_transform(X_train_sampled)
    X_test_encoded = vectorizer.transform(X_test)

    # Suivi MLflow
    with mlflow.start_run(run_name=args.run_name):
        print("Entraînement de XGBoost...")
        clf = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=42,
            tree_method="hist",
        )
        clf.fit(X_train_encoded, y_train_sampled)

        # Prédictions
        y_pred = clf.predict(X_test_encoded)

        # Calcul des 5 métriques clés + accuracy
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

        # Calcul de la matrice de confusion
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        confusion_dict = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

        print("\nMétriques obtenues :")
        for k, v in metrics.items():
            print(f"  {k} : {v:.4f}")

        # Enregistrement des paramètres de choix dans MLflow
        mlflow.log_param("sampling_ratio", args.sampling_ratio)
        mlflow.log_param("outlier_threshold", args.outlier_threshold)
        mlflow.log_param("geo_representation_mode", args.geo_mode)
        mlflow.log_param("num_features", len(base_cols))
        mlflow.log_params(clf.get_params())

        # Enregistrement des métriques dans MLflow
        mlflow.log_metrics(metrics)

        # Enregistrement de la matrice de confusion
        temp_json_path = f"confusion_matrix_{args.run_name}.json"
        with open(temp_json_path, "w") as f:
            json.dump(confusion_dict, f, indent=4)
        mlflow.log_artifact(temp_json_path)
        os.remove(temp_json_path)

        # Enregistrement du modèle
        mlflow.xgboost.log_model(clf, artifact_path="model")
        print("\nModèle et choix enregistrés dans MLflow !")

    # Actualisation automatique des métadonnées globales MLflow (tags de synthèse)
    try:
        from update_experiment_metadata import main as update_metadata

        update_metadata()
    except Exception as e:
        print(
            f"Avertissement : Impossible de mettre à jour la page d'accueil MLflow : {e}"
        )


if __name__ == "__main__":
    main()
