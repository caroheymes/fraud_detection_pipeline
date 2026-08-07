# src/training/demo.py

import os
import sys
from datetime import datetime

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


# --- FONCTIONS DE PREPARATION VECTORISÉES (OPTIMISÉES) ---
def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Calcul de distance Haversine vectorisé pour des performances maximales."""
    R = 6371.0  # Rayon de la Terre en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def main():
    print("--- DÉMARRAGE DE L'ENTRAÎNEMENT DU MODÈLE BASELINE ---")

    # 1. Chargement des données robuste
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv"))

    print(f"Chargement du fichier CSV : {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Erreur : Le fichier {csv_path} n'existe pas !")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Dimensions initiales : {df.shape}")
    print(df.head())

    # 2. Feature Engineering vectorisé (100x plus rapide que les list comprehensions)
    print("Ingénierie des caractéristiques temporelles et géographiques...")

    # Conversion en datetime
    dt_col = pd.to_datetime(df["trans_date_trans_time"])

    # Encodage cyclique (remplace encode_cyclical_datetime)
    df["hour_sin"] = np.sin(2 * np.pi * dt_col.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * dt_col.dt.hour / 24.0)
    df["weekday_sin"] = np.sin(2 * np.pi * dt_col.dt.dayofweek / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * dt_col.dt.dayofweek / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * dt_col.dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * dt_col.dt.month / 12.0)

    # Calcul de la distance haversine vectorisé
    df["distance_achat"] = haversine_vectorized(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    # Calcul de l'âge
    dob_col = pd.to_datetime(df["dob"])
    df["age"] = datetime.now().year - dob_col.dt.year

    # Suppression des outliers sur le montant (amt) à moins de 3 écarts-types
    mean_amt = df["amt"].mean()
    std_amt = df["amt"].std()
    df = df[df["amt"] > (mean_amt - 3 * std_amt)]
    print(f"Dimensions après filtrage des outliers : {df.shape}")

    # 3. Définition des features (X) et de la cible (y)
    X = df[
        [
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
    ].copy()

    y = df["is_fraud"].copy()

    # Encodage des colonnes catégorielles pour XGBoost (natif)
    X["category"] = X["category"].astype("category")
    X["gender"] = X["gender"].astype("category")

    # Division Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(
        f"Taille d'entraînement : {X_train.shape[0]} lignes | Taille de test : {X_test.shape[0]} lignes"
    )

    # 4. Entraînement et suivi avec MLflow
    model_name = "XGBClassifier"

    # L'expérience MLflow est "Default"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("Default")

    with mlflow.start_run(run_name="Baseline_XGBoost_Real_Data"):
        print(f"Entraînement du modèle {model_name}...")

        # Initialisation du classifieur XGBoost avec support des variables catégorielles
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            enable_categorical=True,
            tree_method="hist",  # Optimisé pour les performances
            random_state=42,
        )

        clf.fit(X_train, y_train)

        # Prédictions et évaluation
        y_pred = clf.predict(X_test)

        prec_c1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_c1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_c1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_glob = f1_score(y_test, y_pred, average="macro", zero_division=0)
        rec_glob = recall_score(y_test, y_pred, average="macro", zero_division=0)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "prec_class_1": prec_c1,
            "rec_class_1": rec_c1,
            "f1_class_1": f1_c1,
            "F1_global": f1_glob,
            "recall_global": rec_glob,
        }

        print("\nMétriques obtenues :")
        for k, v in metrics.items():
            print(f"  {k} : {v:.4f}")

        # Enregistrement des paramètres et des métriques
        mlflow.log_params(clf.get_params())
        mlflow.log_metrics(metrics)

        # Log du modèle et enregistrement dans le Model Registry
        mlflow.xgboost.log_model(
            clf,
            artifact_path="model",
            registered_model_name=f"{model_name}_Baseline_Model",
        )

        print("\nModèle et métriques enregistrés dans MLflow !")

        # Export des métriques en JSON
        import json

        metrics_json_path = os.path.join(script_dir, "metrics_baseline.json")
        with open(metrics_json_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Métriques exportées en JSON dans : {metrics_json_path}")


if __name__ == "__main__":
    main()
