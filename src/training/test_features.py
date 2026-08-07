# src/training/test_features.py

import json
import os
import sys
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import skore
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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


def main():
    print("--- DÉMARRAGE DU TEST COMPARATIF DE SÉLECTION DE VARIABLES (XGBOOST) ---")

    # Configuration du tracking MLflow vers l'expérience centralisée "fraud_detection"
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("fraud_detection")

    # Initialisation du projet Skore local
    script_dir = os.path.dirname(os.path.abspath(__file__))
    skore_project_path = os.path.join(script_dir, "fraud_detection_skore")
    project = skore.Project(skore_project_path)
    print(f"Projet Skore initialisé à l'emplacement : {skore_project_path}")

    # Chargement du fichier CSV
    csv_path = os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv"))
    if not os.path.exists(csv_path):
        print(f"Erreur : Fichier {csv_path} introuvable.")
        sys.exit(1)

    print("Chargement du dataset...")
    full_df = pd.read_csv(csv_path)

    # Échantillon de 30 000 transactions pour des métriques représentatives et rapides
    df = full_df.sample(n=30000, random_state=42).reset_index(drop=True)

    # Feature Engineering de base
    print("Calcul des features temporelles, géographiques et d'âge...")
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

    # Retrait des outliers sur amt
    df = df[df.amt > df.amt.mean() - 3 * df.amt.std()].reset_index(drop=True)
    print(f"Dimensions après ingénierie et filtrage : {df.shape}")

    # Définition des 6 listes de colonnes à tester
    feature_sets = {
        "FeatureSet_1_All_Columns": [
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
            "lat",
            "long",
            "city_pop",
            "job",
            "dob",
            "trans_num",
            "unix_time",
            "merch_lat",
            "merch_long",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
            "distance_achat",
            "age",
        ],
        "FeatureSet_2_No_Personal_IDs": [
            "category",
            "amt",
            "gender",
            "city",
            "state",
            "lat",
            "long",
            "city_pop",
            "merch_lat",
            "merch_long",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
            "distance_achat",
            "age",
        ],
        "FeatureSet_3_No_State_City": [
            "category",
            "amt",
            "gender",
            "lat",
            "long",
            "city_pop",
            "merch_lat",
            "merch_long",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
            "distance_achat",
            "age",
        ],
        "FeatureSet_4_No_CityPop": [
            "category",
            "amt",
            "gender",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
            "distance_achat",
            "age",
        ],
        "FeatureSet_5_Only_State_City": [
            "category",
            "amt",
            "gender",
            "city",
            "state",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
            "distance_achat",
            "age",
        ],
        "FeatureSet_6_Only_CityPop": [
            "category",
            "amt",
            "gender",
            "city_pop",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
            "distance_achat",
            "age",
        ],
    }

    results = {}

    for name, cols in feature_sets.items():
        print(f"\n--- Évaluation de {name} ({len(cols)} variables) ---")

        # Sélection des colonnes pour X
        X = df[cols]
        y = df["is_fraud"]

        # Split Train / Test (70 / 30)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Encodage avec skrub TableVectorizer
        vectorizer = TableVectorizer()
        X_train_encoded = vectorizer.fit_transform(X_train)
        X_test_encoded = vectorizer.transform(X_test)

        # Entraînement XGBoost
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            tree_method="hist",
        )
        clf.fit(X_train_encoded, y_train)

        # Prédiction
        y_pred = clf.predict(X_test_encoded)

        # Calcul des métriques
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
            mlflow.log_metrics(metrics)

        # Enregistrement dans Skore sous forme d'EstimatorReport
        try:
            report = skore.EstimatorReport(
                clf,
                X_train=X_train_encoded,
                y_train=y_train,
                X_test=X_test_encoded,
                y_test=y_test,
                pos_label=1,
            )
            project.put(name, report)
            print(f"  -> Enregistré dans Skore : '{name}'")
        except Exception as e:
            print(f"  -> Erreur d'enregistrement Skore : {e}")
        print(
            f"  -> Enregistré dans MLflow : F1 C1 = {f1_c1:.4f} | F1 Global = {f1_glob:.4f}"
        )

    # Sauvegarde des résultats sous forme de fichier JSON comparatif
    json_path = os.path.join(script_dir, "metrics_features_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nComparatif exporté en JSON : {json_path}")

    # Génération automatique du rapport HTML autonome à la fin
    try:
        summary = project.summarize()
        html_content = summary._repr_html_()
        html_path = os.path.join(script_dir, "skore_report.html")
        with open(html_path, "w") as f:
            f.write(html_content)
        print(f"Rapport HTML Skore généré avec succès dans : {html_path}")
    except Exception as e:
        print(f"Erreur de génération du rapport HTML Skore : {e}")


if __name__ == "__main__":
    main()
