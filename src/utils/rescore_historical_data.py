# src/training/rescore_historical_data.py
"""
Script MLOps de ré-inférence historique (In-Place Batch Re-scoring).
Permet de ré-évaluer l'historique PostgreSQL (silver.rawdata) à partir d'une date cible
avec le modèle Champion actif dans MLflow (ex. fraud_detector_v12) et d'actualiser dbt.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

# Enregistrement des classes pour le dé-pickle HinSAGE GRL
import __main__

try:
    import src.training.inductive_grl as ig

    __main__.InductiveGRLPipeline = ig.InductiveGRLPipeline
    __main__.HinSAGERepresentationLearner = ig.HinSAGERepresentationLearner
    __main__.HinSAGEPyTorchNet = ig.HinSAGEPyTorchNet
    __main__.FocalLoss = ig.FocalLoss
except Exception:
    pass

from src.utils.db import get_postgres_engine
from src.utils.features import prepare_features


def main():
    parser = argparse.ArgumentParser(
        description="Ré-inférence historique in-place avec le modèle Champion MLflow"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-08-20",
        help="Date de début de ré-inférence (ex: 2020-08-20 pour Datemin + 30 jours)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Taille des paquets de mise à jour en base (défaut: 10000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Seuil de décision binaire (défaut: 0.15 pour priorité au Rappel/F2)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  🚀 RÉ-INFÉRENCE HISTORIQUE MLOPS (In-Place Batch Rescoring)")
    print(f"  Date de départ : {args.start_date}")
    print(f"  Seuil de décision : {args.threshold}")
    print(f"  Taille des lots : {args.batch_size}")
    print("=" * 70)

    # 1. Chargement du modèle Champion depuis MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    try:
        version_details = client.get_model_version_by_alias(
            "fraud_detector", "champion"
        )
        champion_version_num = version_details.version
        champion_run_id = version_details.run_id
        champion_tag = f"fraud_detector_v{champion_version_num}"
        print(
            f"\n📦 Modèle Champion détecté : '{champion_tag}' (Run ID: {champion_run_id})"
        )
        model = mlflow.sklearn.load_model(f"runs:/{champion_run_id}/model")
    except Exception as e:
        print(
            f"⚠️ Erreur alias champion : {e}. Recherche du dernier modèle de 'fraud_detection'..."
        )
        exp = client.get_experiment_by_name("fraud_detection")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id], order_by=["start_time DESC"]
        )
        champion_run_id = runs[0].info.run_id
        champion_tag = "fraud_detector_v12"
        model = mlflow.sklearn.load_model(f"runs:/{champion_run_id}/model")

    print(f"✅ Modèle chargé avec succès : {type(model).__name__}")

    # 2. Récupération des transactions cibles depuis PostgreSQL
    engine = get_postgres_engine()
    conn = engine.raw_connection()
    query_count = (
        "SELECT count(*) FROM silver.rawdata WHERE trans_date_trans_time >= %s"
    )
    with conn.cursor() as cur:
        cur.execute(query_count, (args.start_date,))
        total_rows = cur.fetchone()[0]

    print(
        f"\n📊 Transactions à ré-inférer depuis le {args.start_date} : {total_rows:,}"
    )
    if total_rows == 0:
        print("Aucune transaction trouvée pour cette période.")
        conn.close()
        return

    # 3. Traitement par lots (Batch Processing)
    offset = 0
    t_start = time.time()
    total_fraud_detected = 0

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

    while offset < total_rows:
        batch_query = f"""
            SELECT 
                trans_num, cc_num, merchant, category, amt, gender, lat, long, 
                city_pop, dob, trans_date_trans_time, merch_lat, merch_long, is_fraud
            FROM silver.rawdata
            WHERE trans_date_trans_time >= %(start_date)s
            ORDER BY trans_date_trans_time ASC
            LIMIT {args.batch_size} OFFSET {offset}
        """
        df_batch = pd.read_sql_query(
            batch_query, conn, params={"start_date": args.start_date}
        )
        if df_batch.empty:
            break

        # Feature Engineering unifié via src.utils.features
        df_batch = prepare_features(df_batch, include_graph_ids=True)

        # Inférence avec le modèle Champion
        t0_inf = time.time()
        if hasattr(model, "predict_proba"):
            if hasattr(model, "hinsage"):
                # Pipeline GRL HinSAGE
                probs = model.predict_proba(df_batch)[:, 1]
            else:
                # Pipeline Scikit-Learn standard
                probs = model.predict_proba(df_batch[features_list])[:, 1]
        else:
            preds_raw = model.predict(df_batch[features_list])
            probs = preds_raw.astype(float)

        t_inf_elapsed = max(0.5, (time.time() - t0_inf) * 1000.0 / len(df_batch))

        # Décision calibrée (Option C : seuil 0.15)
        binary_preds = (probs >= args.threshold).astype(int)
        total_fraud_detected += int(binary_preds.sum())

        # Règles Fast-Pass de suspicion
        fast_pass_susp = (
            (df_batch["amt"] > 900.0) | (df_batch["distance_achat"] > 150.0)
        ).astype(int)
        fast_pass_scores = np.where(fast_pass_susp == 1, 0.85, 0.05)

        # Mise à jour PostgreSQL par lots
        update_data = [
            (
                int(pred),
                float(prob),
                champion_tag,
                int(fps),
                float(fpsc),
                float(t_inf_elapsed),
                str(tnum),
            )
            for pred, prob, fps, fpsc, tnum in zip(
                binary_preds,
                probs,
                fast_pass_susp,
                fast_pass_scores,
                df_batch["trans_num"],
            )
        ]

        update_query = """
            UPDATE silver.rawdata AS r
            SET 
                prediction = u.pred,
                prediction_proba = u.prob,
                model_version = u.mver,
                fast_pass_suspicion = u.fps,
                fast_pass_score = u.fpsc,
                prediction_latency_ms = u.lat
            FROM (VALUES %s) AS u(pred, prob, mver, fps, fpsc, lat, tnum)
            WHERE r.trans_num = u.tnum;
        """
        import psycopg2.extras

        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                UPDATE silver.rawdata AS r
                SET 
                    prediction = v.pred,
                    prediction_proba = v.prob,
                    model_version = v.mver,
                    fast_pass_suspicion = v.fps,
                    fast_pass_score = v.fpsc,
                    prediction_latency_ms = v.lat
                FROM (VALUES %s) AS v(pred, prob, mver, fps, fpsc, lat, tnum)
                WHERE r.trans_num = v.tnum
                """,
                update_data,
                template="(%s, %s, %s, %s, %s, %s, %s)",
                page_size=args.batch_size,
            )
        conn.commit()

        offset += len(df_batch)
        pct = (offset / total_rows) * 100.0
        print(
            f" -> Progression : {offset:,}/{total_rows:,} transactions traitées ({pct:.1f}%)..."
        )

    conn.close()
    elapsed_total = time.time() - t_start
    print(f"\n🎉 Ré-inférence terminée en {elapsed_total:.2f}s !")
    print(f"Total fraudes détectées sur le segment : {total_fraud_detected:,}")

    # 4. Actualisation des Data Marts Gold via dbt run
    print("\n🔄 Exécution de 'dbt run' pour actualiser les tables Gold...")
    env = os.environ.copy()
    env["POSTGRES_HOST"] = os.getenv("POSTGRES_HOST", "postgres")
    env["POSTGRES_PORT"] = os.getenv("POSTGRES_PORT", "5432")

    dbt_dir = "/opt/airflow/project/dbt_project"
    if not os.path.exists(dbt_dir):
        dbt_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../dbt_project")
        )

    try:
        res = subprocess.run(
            ["dbt", "run", "--profiles-dir", "."],
            cwd=dbt_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        print(res.stdout)
        if res.returncode == 0:
            print(
                "✅ Toutes les tables Gold (SLA, marchands, Pareto) sont 100% à jour !"
            )
        else:
            print(f"⚠️ Avertissement dbt : {res.stderr}")
    except Exception as dbt_err:
        print(f"⚠️ dbt non trouvé en local ou erreur d'exécution : {dbt_err}")


if __name__ == "__main__":
    main()
