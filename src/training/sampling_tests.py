# src/training/sampling_tests.py

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
from skrub import TableVectorizer


# --- DISTANCE HAVERSINE VECTORISÉE ---
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0  # Rayon de la Terre en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def main():
    print("--- DÉBUT DES TESTS D'OVER ET UNDER SAMPLING ---")

    # 1. Chargement robuste des données
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv"))

    if not os.path.exists(csv_path):
        print(f"Erreur : Fichier {csv_path} introuvable.")
        sys.exit(1)

    print("Chargement de fraudTest.csv...")
    df = pd.read_csv(csv_path)
    print(f"Dimensions du dataset : {df.shape}")

    # 2. Préparation des variables (Feature Engineering)
    print("Ingénierie des caractéristiques...")
    dt_col = pd.to_datetime(df["trans_date_trans_time"])

    # Encodage temporel cyclique
    df["hour_sin"] = np.sin(2 * np.pi * dt_col.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * dt_col.dt.hour / 24.0)
    df["weekday_sin"] = np.sin(2 * np.pi * dt_col.dt.dayofweek / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * dt_col.dt.dayofweek / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * dt_col.dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * dt_col.dt.month / 12.0)

    # Distance et âge
    df["distance_achat"] = haversine_vectorized(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    dob_col = pd.to_datetime(df["dob"])
    df["age"] = datetime.now().year - dob_col.dt.year

    # Filtrage des outliers (amt)
    df = df[df.amt > df.amt.mean() - 3 * df.amt.std()]

    # Sélection des features selon vos specs
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
    ]
    y = df["is_fraud"].reset_index(drop=True)

    # Division Train / Test stratifiée (pour conserver le ratio de fraude dans le test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3. Vectorisation avec SKRUB (TableVectorizer)
    print("Vectorisation des variables avec skrub.TableVectorizer...")
    vectorizer = TableVectorizer()
    X_train_encoded = vectorizer.fit_transform(X_train_raw)
    X_test_encoded = vectorizer.transform(X_test_raw)

    # Conversion en DataFrame pour manipulation de sampling plus aisée
    X_train_df = pd.DataFrame(
        X_train_encoded, columns=vectorizer.get_feature_names_out()
    ).reset_index(drop=True)
    y_train_series = pd.Series(y_train).reset_index(drop=True)

    # Dataset d'entraînement combiné
    train_df = pd.concat([X_train_df, y_train_series], axis=1)

    fraud_train = train_df[train_df["is_fraud"] == 1]
    normal_train = train_df[train_df["is_fraud"] == 0]

    print(
        f"Dataset d'entraînement initial : {len(normal_train)} sains, {len(fraud_train)} fraudes"
    )

    # Configuration MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("Default")

    # ================= TEST 1 : UNDER-SAMPLING =================
    print("\n--- TEST 1 : UNDER-SAMPLING ---")
    # Sous-échantillonnage de la classe majoritaire pour atteindre un ratio 1:1
    normal_under = normal_train.sample(n=len(fraud_train), random_state=42)
    train_under = pd.concat([fraud_train, normal_under]).sample(
        frac=1.0, random_state=42
    )

    X_train_under = train_under.drop(columns=["is_fraud"])
    y_train_under = train_under["is_fraud"]
    print(
        f"Dimensions après Under-sampling : {X_train_under.shape[0]} lignes ({len(y_train_under[y_train_under == 1])} fraudes)"
    )

    with mlflow.start_run(run_name="Under_Sampling_Test"):
        clf_under = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        clf_under.fit(X_train_under, y_train_under)

        # Évaluation sur le dataset de test original (non-échantillonné)
        y_pred = clf_under.predict(X_test_encoded)

        prec_c1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_c1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_c1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_glob = f1_score(y_test, y_pred, average="macro", zero_division=0)
        rec_glob = recall_score(y_test, y_pred, average="macro", zero_division=0)

        metrics_under = {
            "accuracy": accuracy_score(y_test, y_pred),
            "prec_class_1": prec_c1,
            "rec_class_1": rec_c1,
            "f1_class_1": f1_c1,
            "F1_global": f1_glob,
            "recall_global": rec_glob,
        }

        print("Métriques Under-Sampling :")
        for k, v in metrics_under.items():
            print(f"  {k} : {v:.4f}")

        mlflow.log_params(clf_under.get_params())
        mlflow.log_param("sampling_strategy", "under-sampling")
        mlflow.log_metrics(metrics_under)
        mlflow.xgboost.log_model(clf_under, artifact_path="model")

    # ================= TEST 2 : OVER-SAMPLING =================
    print("\n--- TEST 2 : OVER-SAMPLING ---")
    # Sur-échantillonnage de la classe minoritaire pour atteindre un ratio 1:1 (avec remplacement)
    fraud_over = fraud_train.sample(n=len(normal_train), replace=True, random_state=42)
    train_over = pd.concat([normal_train, fraud_over]).sample(frac=1.0, random_state=42)

    X_train_over = train_over.drop(columns=["is_fraud"])
    y_train_over = train_over["is_fraud"]
    print(
        f"Dimensions après Over-sampling : {X_train_over.shape[0]} lignes ({len(y_train_over[y_train_over == 1])} fraudes)"
    )

    with mlflow.start_run(run_name="Over_Sampling_Test"):
        clf_over = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        clf_over.fit(X_train_over, y_train_over)

        # Évaluation sur le dataset de test original (non-échantillonné)
        y_pred = clf_over.predict(X_test_encoded)

        prec_c1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_c1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_c1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_glob = f1_score(y_test, y_pred, average="macro", zero_division=0)
        rec_glob = recall_score(y_test, y_pred, average="macro", zero_division=0)

        metrics_over = {
            "accuracy": accuracy_score(y_test, y_pred),
            "prec_class_1": prec_c1,
            "rec_class_1": rec_c1,
            "f1_class_1": f1_c1,
            "F1_global": f1_glob,
            "recall_global": rec_glob,
        }

        print("Métriques Over-Sampling :")
        for k, v in metrics_over.items():
            print(f"  {k} : {v:.4f}")

        mlflow.log_params(clf_over.get_params())
        mlflow.log_param("sampling_strategy", "over-sampling")
        mlflow.log_metrics(metrics_over)
        mlflow.xgboost.log_model(clf_over, artifact_path="model")

    print("\n--- LES DEUX TESTS SONT TERMINÉS ET ENREGISTRÉS DANS MLFLOW ---")

    # Export des métriques en JSON
    import json

    combined_metrics = {"under_sampling": metrics_under, "over_sampling": metrics_over}
    metrics_json_path = os.path.join(script_dir, "metrics_sampling.json")
    with open(metrics_json_path, "w") as f:
        json.dump(combined_metrics, f, indent=4)
    print(f"Métriques de sampling exportées en JSON dans : {metrics_json_path}")


if __name__ == "__main__":
    main()
