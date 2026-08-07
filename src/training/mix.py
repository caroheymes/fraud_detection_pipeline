# src/training/demo.py
# src/training/train.py

#   • 🚀 Airflow Webserver : http://localhost:8082 (ou 8081 selon vos redirections de
#   ports)
#   • 📊 Streamlit (Visualisation) : http://localhost:8511
#   • 🧠 Ray Dashboard : http://localhost:8266
#   • 📈 MLflow Tracking : http://localhost:5001
import os
from datetime import datetime

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import ray
from haversine import haversine
from sklearn import model_selection
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer
from xgboost import XGBClassifier

# 1. Connexion au cluster Ray local
# En indiquant "auto", Ray se connecte au cluster existant démarré par Docker
ray.init(address="auto", ignore_reinit_error=True)


# 2. Définition de la tâche d'entraînement qui va s'exécuter sur le GPU
@ray.remote(num_gpus=1)  # Indique à Ray de planifier cette tâche sur le worker avec GPU
def train_model_on_gpu(params):
    # Initialisation de MLflow au sein du Worker Ray
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))

    # Le nom de l'experiment correspond au nom du modèle testé
    model_name = params["model_type"]
    mlflow.set_experiment(model_name)

    # Démarrage du run dans l'expérience du modèle
    with mlflow.start_run(
        run_name=f"Run_{model_name}_depth_{params.get('max_depth', 'default')}"
    ):
        print(f"Début de l'entraînement sur GPU pour {model_name}...")

        # Log des paramètres
        mlflow.log_params(params)

        # =======================================================================================
        # Features
        # =======================================================================================

        # transformation datetime en sin cos
        def encode_cyclical_datetime(dt_str):
            # 1. Conversion en objet Datetime
            dt = pd.to_datetime(dt_str, format="%Y-%m-%d %H:%M:%S", errors="coerce")

            # 2. Extraction des valeurs temporelles
            # Heure décimale (Heure + Minute/60 + Seconde/3600)
            decimal_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
            weekday = dt.weekday()  # Lundi = 0, Dimanche = 6
            month = dt.month  # Janvier = 1, Décembre = 12

            # 3. Calcul de l'encodage cyclique (Sin / Cos)
            features = {
                # Heure (période = 24)
                "hour_sin": np.sin(2 * np.pi * decimal_hour / 24.0),
                "hour_cos": np.cos(2 * np.pi * decimal_hour / 24.0),
                # Jour de la semaine (période = 7)
                "weekday_sin": np.sin(2 * np.pi * weekday / 7.0),
                "weekday_cos": np.cos(2 * np.pi * weekday / 7.0),
                # Mois (période = 12)
                "month_sin": np.sin(2 * np.pi * month / 12.0),
                "month_cos": np.cos(2 * np.pi * month / 12.0),
            }

            return features

        # Chargement des données
        df = pd.read_csv("../fraudTest.csv")

        print(df.head())

        features_list = [
            encode_cyclical_datetime(elem) for elem in df.trans_date_trans_time
        ]
        df_features = pd.DataFrame(features_list)
        df = pd.concat([df, df_features], axis=1)

        features_list = [
            encode_cyclical_datetime(elem) for elem in df.trans_date_trans_time
        ]
        df_features = pd.DataFrame(features_list)
        df = pd.concat([df, df_features], axis=1)

        df["distance_achat"] = [
            haversine((lat, long), (merchant_lat, merchant_long))
            for lat, long, merchant_lat, merchant_long in zip(
                df.lat, df.long, df.merch_lat, df.merch_long
            )
        ]
        df["age"] = [
            datetime.now().year - datetime.strptime(elem, "%Y-%m-%d").year
            for elem in df.dob
        ]

        # surppresion des outliers sur amount moins de 3 écarts type
        df = df[df.amt > df.amt.mean() - 2 * df.amt.std()]

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

        y = df["is_fraud"]

        # # XGBoost
        # model = XGBClassifier()
        model = model_selection.models[params["model_type"]]
        preprocessor = TableVectorizer()
        pipeline_model = Pipeline([("preprocess", preprocessor), ("classifier", model)])
        # train test split stratifié

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # cv
        cv_results = cross_validate(
            pipeline_model,
            X_train,
            y_train,
            cv=5,
            scoring=["precision", "recall", "f1", "roc_auc"],
            return_train_score=True,
        )

        # entraîner le modèle
        pipeline_model.fit(X_train, y_train)

        # Prediction sur le train
        y_pred_train = pipeline_model.predict(X_train)

        # Prediction sur le test
        y_pred_test = pipeline_model.predict(X_test)

        # metric test
        test_precision = precision_score(y_test, y_pred_test)
        test_recall = recall_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)
        test_roc_auc = roc_auc_score(y_test, y_pred_test)

        # # Simulation d'entraînement et métriques
        # # (À remplacer par votre vrai code d'entraînement)
        # accuracy = 0.975 if model_name == "XGBoost" else 0.958
        # precision = 0.942 if model_name == "XGBoost" else 0.925

        # je voudrais surtout la précision sur la classe    1
        test_precision_1 = precision_score(y_test, y_pred_test, pos_label=1)
        test_recall_1 = recall_score(y_test, y_pred_test, pos_label=1)
        test_f1_1 = f1_score(y_test, y_pred_test, pos_label=1)

        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_precision_1", test_precision_1)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_recall_1", test_recall_1)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_f1_1", test_f1_1)
        mlflow.log_metric("test_roc_auc", test_roc_auc)

        # Simulation d'enregistrement du code
        mlflow.log_artifact(__file__, artifact_path="model_code")

        return f"Entraînement réussi pour {model_name} (test_precision_1: {test_precision}, test_F1_1: {test_f1_1})"


def main():
    # Tester différents modèles et paramètres
    models = {
        "HistGradientBoosting": HistGradientBoostingClassifier(class_weight="balanced"),
        "XGBoost": XGBClassifier(scale_pos_weight=0.4),
    }
    configs = [
        {"model_type": "HistGradientBoosting", "max_depth": 6, "learning_rate": 0.1},
        {"model_type": "RandomForest", "n_estimators": 100},
        {"model_type": "XGBoost", "max_depth": 10, "learning_rate": 0.05},
    ]

    # Lancement asynchrone sur le cluster Ray
    futures = [train_model_on_gpu.remote(cfg) for cfg in configs]

    # Récupération et affichage des résultats
    results = ray.get(futures)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
