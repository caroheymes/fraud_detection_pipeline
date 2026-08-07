# src/training/train_initial.py
# docker exec -t fraud-detection-ray-head python src/training/train_initial.py

import json
import os
import sys
from datetime import datetime, timedelta

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from skrub import TableVectorizer
from xgboost import XGBClassifier


def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def main():
    print("--- ENTRAÎNEMENT INITIAL (30 PREMIERS JOURS) ---")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("fraud_detection")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv"))
    if not os.path.exists(csv_path):
        print(f"Erreur : Dataset {csv_path} introuvable.")
        sys.exit(1)

    print("Chargement complet du dataset...")
    df = pd.read_csv(csv_path)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    # Détermination de la période de 30 jours
    min_date = df["trans_date_trans_time"].min()
    limit_date = min_date + timedelta(days=30)
    print(f"Période d'entraînement initial : du {min_date} au {limit_date}")

    # Filtrage des données des 30 premiers jours
    df_initial = df[df["trans_date_trans_time"] <= limit_date].reset_index(drop=True)
    print(f"Nombre de transactions pour l'entraînement initial : {len(df_initial)}")

    # Ingestion des colonnes calculées (Feature Engineering)
    df_initial["hour_sin"] = np.sin(
        2 * np.pi * df_initial["trans_date_trans_time"].dt.hour / 24.0
    )
    df_initial["hour_cos"] = np.cos(
        2 * np.pi * df_initial["trans_date_trans_time"].dt.hour / 24.0
    )
    df_initial["weekday_sin"] = np.sin(
        2 * np.pi * df_initial["trans_date_trans_time"].dt.dayofweek / 7.0
    )
    df_initial["weekday_cos"] = np.cos(
        2 * np.pi * df_initial["trans_date_trans_time"].dt.dayofweek / 7.0
    )
    df_initial["month_sin"] = np.sin(
        2 * np.pi * df_initial["trans_date_trans_time"].dt.month / 12.0
    )
    df_initial["month_cos"] = np.cos(
        2 * np.pi * df_initial["trans_date_trans_time"].dt.month / 12.0
    )

    df_initial["distance_achat"] = haversine_vectorized(
        df_initial["lat"],
        df_initial["long"],
        df_initial["merch_lat"],
        df_initial["merch_long"],
    )
    dob_col = pd.to_datetime(df_initial["dob"])
    df_initial["age"] = datetime.now().year - dob_col.dt.year

    # Filtrage des outliers montants (seuil à 3.0)
    df_initial = df_initial[
        df_initial.amt > df_initial.amt.mean() - 3.0 * df_initial.amt.std()
    ].reset_index(drop=True)

    # Sauvegarde des données initiales préparées pour servir de référence d'observabilité
    ref_path = os.path.join(script_dir, "reference_data.csv")
    df_initial.to_csv(ref_path, index=False)
    print(f"Jeu de données de référence sauvegardé : {ref_path}")

    # Variables d'entraînement (Jeu 6)
    features = [
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

    X = df_initial[features]
    y = df_initial["is_fraud"]

    # Division Train / Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Échantillonnage léger (5%)
    from optimize_xgb import get_moderate_sampled_data

    X_train_sampled, y_train_sampled = get_moderate_sampled_data(
        X_train, y_train, target_ratio=0.05
    )

    # Vectorisation TableVectorizer
    vectorizer = TableVectorizer()
    X_train_encoded = vectorizer.fit_transform(X_train_sampled)
    X_test_encoded = vectorizer.transform(X_test)

    # Run MLflow
    with mlflow.start_run(run_name="XGB_Initial_30_Days") as run:
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            tree_method="hist",
        )
        clf.fit(X_train_encoded, y_train_sampled)

        y_pred = clf.predict(X_test_encoded)

        prec_c1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_c1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_c1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        f2_c1 = fbeta_score(y_test, y_pred, beta=2, pos_label=1, zero_division=0)
        f1_glob = f1_score(y_test, y_pred, average="macro", zero_division=0)
        rec_glob = recall_score(y_test, y_pred, average="macro", zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        metrics = {
            "accuracy": float(acc),
            "prec_class_1": float(prec_c1),
            "rec_class_1": float(rec_c1),
            "f1_class_1": float(f1_c1),
            "f2_class_1": float(f2_c1),
            "F1_global": float(f1_glob),
            "recall_global": float(rec_glob),
        }

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        confusion_dict = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

        # Enregistrement des paramètres et des métriques
        mlflow.log_params(clf.get_params())
        mlflow.log_param("training_window", "30_days_initial")
        mlflow.log_metrics(metrics)

        # Enregistrement de la matrice de confusion
        temp_json = "confusion_matrix_initial.json"
        with open(temp_json, "w") as f:
            json.dump(confusion_dict, f, indent=4)
        mlflow.log_artifact(temp_json)
        os.remove(temp_json)

        # Enregistrement du modèle avec enregistrement dans le Model Registry MLflow
        # mlflow.xgboost.log_model(
        #     clf,
        #     artifact_path="model",
        #     registered_model_name="XGB_Fraud_Model"
        # )
        # print("\nModèle initial v1.0 entraîné et enregistré dans MLflow Registry sous le nom 'XGB_Fraud_Model' !")

        # Enregistrement du modèle avec enregistrement dans le Model Registry MLflow
        mlflow.xgboost.log_model(
            clf, artifact_path="model", registered_model_name="XGB_Fraud_Model"
        )
        print(
            "\nModèle initial v1.0 entraîné et enregistré dans MLflow Registry sous le nom 'XGB_Fraud_Model' !"
        )

        # --- NOUVEAU : Promotion automatique en Production ---
        try:
            client = MlflowClient()

            versions = client.get_latest_versions("XGB_Fraud_Model", stages=["None"])
            if versions:
                latest_version = versions[0].version
                client.transition_model_version_stage(
                    name="XGB_Fraud_Model",
                    version=latest_version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                print(
                    f"Modèle 'XGB_Fraud_Model' version {latest_version} promu en stage 'Production' avec succès !"
                )
        except Exception as e:
            print(
                f"Avertissement : Échec de la promotion automatique en Production : {e}"
            )

    # Actualisation des métadonnées
    try:
        from update_experiment_metadata import main as update_metadata

        update_metadata()
    except Exception:
        pass


if __name__ == "__main__":
    main()
