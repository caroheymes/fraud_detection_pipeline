# src/training/optimize_xgb.py
# docker exec -t fraud-detection-ray-head python src/training/optimize_xgb.py --n-trials 50 --sample-size -1

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer
from xgboost import XGBClassifier


# --- 1. HAVERSINE DISTANCE ---
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


# --- 2. MODERATE SAMPLING ---
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
    parser = argparse.ArgumentParser(
        description="Optimisation des hyperparamètres XGBoost avec Optuna"
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Nombre d'essais d'optimisation"
    )
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=0.05,
        help="Ratio d'échantillonnage de la fraude sur le train (ex: 0.05. 0.0 = pas de sampling)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30000,
        help="Taille du jeu de données pour l'optimisation",
    )
    args = parser.parse_args()

    # Configuration MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("fraud_detection")

    # Chargement et préparation des données depuis Postgres ou CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv"))

    db_user = os.getenv("POSTGRES_USER", "fraud-detection")
    db_password = os.getenv("POSTGRES_PASSWORD", "fraud-detection_password")
    db_host = os.getenv("POSTGRES_HOST", "postgres")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_db = os.getenv("POSTGRES_DB", "fraud-detection")
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_db}"

    df_raw = None
    try:
        import sqlalchemy

        engine = sqlalchemy.create_engine(db_url)
        with engine.connect() as conn:
            # Vérifier si la table existe et contient des données
            count = (
                conn.execute(
                    sqlalchemy.text("SELECT COUNT(*) FROM silver.rawdata")
                ).scalar()
                or 0
            )
            if count > 0:
                df_raw = pd.read_sql("SELECT * FROM silver.rawdata", engine)
                print(
                    f"Loaded {len(df_raw)} transactions from PostgreSQL (silver.rawdata)."
                )
    except Exception as e:
        print(f"PostgreSQL connection failed or table empty: {e}")

    if df_raw is None:
        if not os.path.exists(csv_path):
            print(f"Erreur : Dataset {csv_path} introuvable.")
            sys.exit(1)
        df_raw = pd.read_csv(csv_path)
        print(f"Loaded {len(df_raw)} transactions from CSV file.")
    if args.sample_size == -1:
        df = df_raw.reset_index(drop=True)
    else:
        df = df_raw.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    # Filtrer pour n'utiliser que les 30 premiers jours (évite le lookahead bias / data leakage) si chargé du CSV
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    if "logged_at" in df.columns:
        print(
            f"Taille finale retenue pour l'optimisation (toutes les données Postgres) : {len(df)} lignes."
        )
    else:
        start_date = df["trans_date_trans_time"].min()
        end_date = start_date + timedelta(days=30)
        df = df[
            (df["trans_date_trans_time"] >= start_date)
            & (df["trans_date_trans_time"] < end_date)
        ].reset_index(drop=True)
        print(
            f"Taille finale retenue pour l'optimisation (30 premiers jours) : {len(df)} lignes."
        )

    # Feature Engineering de base (On exclut les coordonnées car la distance seule s'est avérée meilleure)
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

    # Filtrage des outliers montants (seuil fixe à 3.0)
    df = df[df.amt > df.amt.mean() - 3.0 * df.amt.std()].reset_index(drop=True)

    # Variables utilisées pour l'optimisation (Jeu 6 : Distance + CityPop)
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

    X = df[features]
    y = df["is_fraud"]

    # Séparation train global / test final pour évaluation finale
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(
        f"Stratified Cross-Validation (3 Folds) démarrée sur {len(X_train_full)} lignes."
    )
    print("Métriques cible : F2-Score sur la classe 1 (Fraude).")

    # Définition de la fonction objectif d'Optuna
    def objective(trial):
        # Espace de recherche hyperparamètres
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 100.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "tree_method": "hist",
        }

        # 3-Fold Stratified Cross-Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f2_scores = []
        f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X_train_full, y_train_full)
        ):
            # Extraction des données du pli
            X_tr, y_tr = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
            X_val, y_val = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]

            # Application du sampling uniquement sur le pli d'entraînement
            if args.sampling_ratio > 0.0:
                X_tr_sampled, y_tr_sampled = get_moderate_sampled_data(
                    X_tr, y_tr, target_ratio=args.sampling_ratio
                )
            else:
                X_tr_sampled, y_tr_sampled = X_tr, y_tr

            # Vectorisation avec TableVectorizer pour le pli
            vectorizer = TableVectorizer()
            X_tr_encoded = vectorizer.fit_transform(X_tr_sampled)
            X_val_encoded = vectorizer.transform(X_val)

            # Entraînement du modèle
            clf = XGBClassifier(**params)
            clf.fit(X_tr_encoded, y_tr_sampled)

            # Prédictions et calcul des scores F2 et F1 sur la classe 1
            y_pred = clf.predict(X_val_encoded)
            f2 = fbeta_score(y_val, y_pred, beta=2, pos_label=1, zero_division=0)
            f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
            f2_scores.append(f2)
            f1_scores.append(f1)

        mean_f2 = np.mean(f2_scores)
        mean_f1 = np.mean(f1_scores)

        # Enregistrement du F1-score comme attribut d'essai pour récupération par le callback
        trial.set_user_attr("mean_f1_class_1", float(mean_f1))
        return mean_f2

    # Lancement du Run Parent dans MLflow
    parent_run_name = f"Optuna_XGBoost_Search_{datetime.now().strftime('%m%d_%H%M')}"
    with mlflow.start_run(run_name=parent_run_name) as parent_run_name:
        print(f"Enregistrement du run parent dans MLflow : {parent_run_name}")

        # Log des paramètres généraux de l'étude
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("sampling_ratio_train", args.sampling_ratio)
        mlflow.log_param("optimization_metric", "F2_score_class_1")
        mlflow.log_param("validation_strategy", "Stratified_3Fold_CV")

        # Callback pour logger chaque essai dans MLflow comme un sous-run imbriqué
        def mlflow_trial_callback(study, trial):
            with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
                # Log des hyperparamètres testés
                mlflow.log_params(trial.params)
                # Log de la performance moyenne obtenue (F2 et F1)
                mlflow.log_metric("mean_f2_class_1", trial.value)
                if "mean_f1_class_1" in trial.user_attrs:
                    mlflow.log_metric(
                        "mean_f1_class_1", trial.user_attrs["mean_f1_class_1"]
                    )
                mlflow.set_tag("trial_status", str(trial.state))

        # Création et lancement de l'étude Optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective, n_trials=args.n_trials, callbacks=[mlflow_trial_callback]
        )

        print("\nOptimisation terminée !")
        print(f"Meilleur F2-Score obtenu : {study.best_value:.4f}")
        print("Meilleurs hyperparamètres :")
        for k, v in study.best_params.items():
            print(f"  {k} : {v}")

        # Entraînement final avec les meilleurs hyperparamètres sur tout le train
        print(
            "\nEntraînement final du meilleur modèle sur l'ensemble du jeu d'entraînement..."
        )
        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["tree_method"] = "hist"

        # Sampling modéré sur l'ensemble du train
        if args.sampling_ratio > 0.0:
            X_train_full_sampled, y_train_full_sampled = get_moderate_sampled_data(
                X_train_full, y_train_full, target_ratio=args.sampling_ratio
            )
        else:
            X_train_full_sampled, y_train_full_sampled = X_train_full, y_train_full

        # Vectorisation finale
        vectorizer = TableVectorizer()
        pipeline = Pipeline(
            [("preprocessor", vectorizer), ("model", XGBClassifier(**best_params))]
        )

        # Entraînement du pipeline sur les données brutes échantillonnées
        pipeline.fit(X_train_full_sampled, y_train_full_sampled)

        # Évaluation finale sur le test set brut (le pipeline gère la transformation en interne !)
        y_pred_final = pipeline.predict(X_test_final)

        prec_c1 = precision_score(
            y_test_final, y_pred_final, pos_label=1, zero_division=0
        )
        rec_c1 = recall_score(y_test_final, y_pred_final, pos_label=1, zero_division=0)
        f1_c1 = f1_score(y_test_final, y_pred_final, pos_label=1, zero_division=0)
        f2_c1 = fbeta_score(
            y_test_final, y_pred_final, beta=2, pos_label=1, zero_division=0
        )
        f1_glob = f1_score(y_test_final, y_pred_final, average="macro", zero_division=0)
        rec_glob = recall_score(
            y_test_final, y_pred_final, average="macro", zero_division=0
        )
        acc = accuracy_score(y_test_final, y_pred_final)

        metrics = {
            "accuracy": float(acc),
            "prec_class_1": float(prec_c1),
            "rec_class_1": float(rec_c1),
            "f1_class_1": float(f1_c1),
            "f2_class_1": float(f2_c1),
            "F1_global": float(f1_glob),
            "recall_global": float(rec_glob),
        }

        tn, fp, fn, tp = confusion_matrix(y_test_final, y_pred_final).ravel()
        confusion_dict = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

        # Log des métriques finales et du modèle gagnant dans le run parent
        mlflow.log_params(study.best_params)
        mlflow.log_metrics(metrics)

        # Log de la matrice de confusion
        temp_json = "confusion_matrix_best_optuna.json"
        with open(temp_json, "w") as f:
            json.dump(confusion_dict, f, indent=4)
        mlflow.log_artifact(temp_json)
        os.remove(temp_json)

        # Sauvegarde du pipeline Scikit-Learn complet (format pickle requis pour TableVectorizer)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            serialization_format="pickle",
            registered_model_name="fraud_detector",
        )
        print(
            "\nMeilleur pipeline final (préprocesseur + modèle) enregistré dans MLflow avec succès !"
        )

        # Promotion automatique avec l'alias 'champion'
        try:
            client = MlflowClient()
            # Récupérer les versions du modèle générique 'fraud_detector'
            versions = client.get_latest_versions("fraud_detector", stages=["None"])
            if versions:
                latest_version = versions[0].version
                # Assigner l'alias 'champion' à la nouvelle version
                client.set_registered_model_alias(
                    name="fraud_detector", alias="champion", version=latest_version
                )
                print(
                    f"Modèle 'fraud_detector' version {latest_version} promu comme 'champion' avec succès !"
                )
        except Exception as e:
            print(
                f"Avertissement : Échec de la promotion automatique en champion : {e}"
            )
    # Sauvegarde du nouveau jeu de données comme référence d'observabilité
    try:
        ref_path = os.path.join(script_dir, "reference_data.csv")
        # df contient les données réelles avec toutes les colonnes calculées (hour_sin, distance_achat, etc.)
        df.to_csv(ref_path, index=False)
        print(f"Jeu de données de référence d'observabilité mis à jour : {ref_path}")
    except Exception as e:
        print(f"Avertissement : Échec de la mise à jour de reference_data.csv : {e}")

    # Mise à jour globale des métadonnées
    try:
        from update_experiment_metadata import main as update_metadata

        update_metadata()
    except Exception as e:
        print(f"Avertissement : Mise à jour des tags échouée : {e}")


if __name__ == "__main__":
    main()
