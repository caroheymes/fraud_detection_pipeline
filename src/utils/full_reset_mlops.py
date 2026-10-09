# src/utils/full_reset_mlops.py
"""
Script de Grand Reset MLOps :
1. Purge et réinitialisation complète de MLflow (via SQL sur la base mlflow).
2. Purge de la base PostgreSQL (Conservation du mois de Juillet 2020 comme baseline).
3. Vidage de la file d'attente de simulation (data/queue).
4. Entraînement d'un Modèle Champion Initial propre (Version 1, TimeSeriesSplit, sans leakage).
5. Rechargement à chaud de l'API FastAPI et synchronisation Redis.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mlflow
from sqlalchemy import create_engine, text

from src.utils.db import get_postgres_engine


def reset_mlflow():
    print("=" * 70)
    print(" 1. PURGE ET RÉINITIALISATION DE MLFLOW")
    print("=" * 70)
    # Connexion directe à la base postgres 'mlflow'
    pg_host = os.getenv("POSTGRES_HOST", "postgres")
    pg_port = os.getenv("POSTGRES_PORT", "5432")
    pg_user = os.getenv("POSTGRES_USER", "fraud-detection")
    pg_pwd = os.getenv("POSTGRES_PASSWORD", "fraud-detection_password")

    mlflow_db_url = f"postgresql://{pg_user}:{pg_pwd}@{pg_host}:{pg_port}/mlflow"
    mlflow_engine = create_engine(mlflow_db_url)

    with mlflow_engine.connect() as conn:
        try:
            print("   • Truncate des tables MLflow (runs, metrics, models, tags)...")
            truncate_query = text("""
                TRUNCATE TABLE 
                    runs, experiment_tags, latest_metrics, metrics, params, tags, 
                    model_version_tags, model_versions, registered_model_aliases, 
                    registered_model_tags, registered_models, logged_models, 
                    logged_model_metrics, logged_model_params, logged_model_tags 
                CASCADE;
            """)
            conn.execute(truncate_query)
            conn.commit()
            print("    Tables de traçabilité MLflow vidées avec succès.")
        except Exception as e:
            print(f"    Erreur truncate MLflow : {e}")

        try:
            # Réactiver ou créer l'expérience fraud_detection
            conn.execute(
                text(
                    "UPDATE experiments SET lifecycle_stage = 'active' WHERE name IN ('fraud_detection', 'Default');"
                )
            )
            conn.commit()
            # S'assurer que l'expérience 'fraud_detection' existe bien
            exp_exists = conn.execute(
                text("SELECT COUNT(*) FROM experiments WHERE name = 'fraud_detection';")
            ).scalar()
            if not exp_exists:
                conn.execute(
                    text("""
                    INSERT INTO experiments (name, artifact_location, lifecycle_stage, creation_time, last_update_time)
                    VALUES ('fraud_detection', '/mlflow/artifacts', 'active', :ts, :ts);
                """),
                    {"ts": int(time.time() * 1000)},
                )
                conn.commit()
                print("    Expérience 'fraud_detection' créée dans la base MLflow.")
            else:
                print("    Expérience 'fraud_detection' réactivée et prête.")
        except Exception as e:
            print(f"    Erreur configuration expérience : {e}")

    # Réinitialisation du cache MLflow côté client
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    print("    Client MLflow synchronisé.")


def reset_postgres_database(cutoff_date: str = "2020-07-31 23:59:59"):
    print("\n" + "=" * 70)
    print(f" 2. PURGE DE POSTGRESQL (Cutoff: {cutoff_date})")
    print("=" * 70)
    engine = get_postgres_engine()
    with engine.connect() as conn:
        # 2.1 silver.rawdata
        try:
            cnt_del = conn.execute(
                text(
                    "SELECT COUNT(*) FROM silver.rawdata WHERE trans_date_trans_time > :cutoff"
                ),
                {"cutoff": cutoff_date},
            ).scalar()
            if cnt_del and cnt_del > 0:
                conn.execute(
                    text(
                        "DELETE FROM silver.rawdata WHERE trans_date_trans_time > :cutoff"
                    ),
                    {"cutoff": cutoff_date},
                )
                conn.commit()
                print(f"    {cnt_del:,} transactions supprimées de silver.rawdata.")
            else:
                print("    Aucune transaction après la date de coupure.")

            total_rem = conn.execute(
                text("SELECT COUNT(*) FROM silver.rawdata")
            ).scalar()
            max_dt = conn.execute(
                text("SELECT MAX(trans_date_trans_time) FROM silver.rawdata")
            ).scalar()
            min_dt = conn.execute(
                text("SELECT MIN(trans_date_trans_time) FROM silver.rawdata")
            ).scalar()
            print(
                f"    Données restantes dans silver.rawdata : {total_rem:,} transactions (de {min_dt} à {max_dt})."
            )
        except Exception as e:
            print(f"    Erreur purge silver.rawdata : {e}")

        # 2.2 silver.ingested_file
        try:
            conn.execute(
                text("TRUNCATE TABLE silver.ingested_file RESTART IDENTITY CASCADE;")
            )
            conn.commit()
            print("    Table silver.ingested_file réinitialisée.")
        except Exception as e:
            print(f"    silver.ingested_file : {e}")


def reset_queue():
    print("\n" + "=" * 70)
    print(" 3. VIDAGE DU DOSSIER DE FILE D'ATTENTE (data/queue)")
    print("=" * 70)
    queue_dir = os.path.join(project_root, "data/queue")
    if os.path.exists(queue_dir):
        files = [f for f in os.listdir(queue_dir) if f.endswith(".csv")]
        print(f"   • Suppression de {len(files)} fichiers CSV dans {queue_dir}...")
        for f in files:
            try:
                os.remove(os.path.join(queue_dir, f))
            except Exception:
                pass
        print("    Dossier data/queue vidé et prêt pour de nouvelles simulations.")
    else:
        os.makedirs(queue_dir, exist_ok=True)
        print("    Dossier data/queue créé.")


def train_clean_initial_champion():
    print("\n" + "=" * 70)
    print(" 4. ENTRAÎNEMENT DU MODÈLE INITIAL PROPRE (Version 1)")
    print("=" * 70)
    train_cmd = [
        sys.executable,
        os.path.join(project_root, "src/training/optimize_xgb.py"),
        "--n-trials",
        "20",
        "--sample-size",
        "30000",
        "--metric-target",
        "f2",
        "--sampling-ratio",
        "0.05",
    ]
    print(f"Exécution de : {' '.join(train_cmd)}")
    res = subprocess.run(train_cmd)
    if res.returncode != 0:
        raise RuntimeError(
            f"Échec de l'entraînement du modèle initial : code {res.returncode}"
        )
    print(" Modèle Version 1 entraîné et enregistré avec succès dans MLflow !")


def update_baseline_predictions():
    print("\n" + "=" * 70)
    print(" 4.5 MISE À JOUR DES PRÉDICTIONS DE BASELINE DANS POSTGRESQL")
    print("=" * 70)
    try:
        import pandas as pd

        from src.utils.db import get_postgres_engine
        from src.utils.features import prepare_features
        from src.utils.mlflow_manager import load_champion_model

        eng = get_postgres_engine()
        df_raw = pd.read_sql(
            "SELECT * FROM silver.rawdata ORDER BY trans_date_trans_time ASC", eng
        )
        if df_raw.empty:
            print("    Aucune donnée dans silver.rawdata.")
            return

        model, active_name, threshold = load_champion_model()
        df_proc = prepare_features(df_raw)
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
        X_proc = df_proc[features]
        probas = model.predict_proba(X_proc)[:, 1]
        preds = (probas >= threshold).astype(int)

        temp_df = pd.DataFrame(
            {
                "trans_num": df_raw["trans_num"],
                "prediction": preds,
                "prediction_proba": probas,
                "model_version": active_name,
            }
        )
        temp_df.to_sql("temp_preds_v1", eng, if_exists="replace", index=False)

        with eng.connect() as conn:
            conn.execute(
                text("""
                UPDATE silver.rawdata r
                SET 
                    prediction = t.prediction,
                    prediction_proba = t.prediction_proba,
                    model_version = t.model_version
                FROM temp_preds_v1 t
                WHERE r.trans_num = t.trans_num;
            """)
            )
            conn.execute(text("DROP TABLE IF EXISTS temp_preds_v1;"))
            conn.commit()

        print(
            f"    {len(df_raw):,} transactions mises à jour avec les prédictions du modèle Champion '{active_name}'."
        )
    except Exception as e:
        print(f"    Erreur mise à jour prédictions baseline : {e}")


def sync_and_reload():
    print("\n" + "=" * 70)
    print(" 5. RECHARGEMENT API ET SYNCHRONISATION REDIS")
    print("=" * 70)
    try:
        from src.utils.api_reloader import reload_serving_api

        reload_serving_api()
    except Exception as e:
        print(f" Erreur reload API : {e}")

    try:
        export_cmd = [
            sys.executable,
            os.path.join(project_root, "src/explain/export_rules.py"),
        ]
        res = subprocess.run(export_cmd)
        if res.returncode == 0:
            print(" Règles de suspicion SHAP synchronisées dans Redis.")
        else:
            print(" Avertissement export rules.")
    except Exception as e:
        print(f" Erreur export rules : {e}")


def main():
    print(" LANCEMENT DU GRAND RESET MLOPS DU PROJET \n")
    reset_mlflow()
    reset_postgres_database(cutoff_date="2020-07-31 23:59:59")
    reset_queue()
    train_clean_initial_champion()
    update_baseline_predictions()
    sync_and_reload()
    print("\n" + "=" * 70)
    print(" GRAND RESET TERMINÉ AVEC SUCCÈS !")
    print("   • Base PostgreSQL initialisée avec Juillet 2020")
    print("   • MLflow reparti à neuf (Modèle v1 = Champion propre)")
    print("   • API d'inférence et Redis synchronisés")
    print("=" * 70)


if __name__ == "__main__":
    main()
