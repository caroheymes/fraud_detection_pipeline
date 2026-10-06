# src/api/main.py
# docker exec -it fraud-detection-ray-head uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# http://localhost:8001/docs#/
#
# paramètres de connexion à PostgreSQL :
#   Hôte (Host) : localhost (ou 127.0.0.1)
#   Port : 5433
#   Base de données (Database) : fraud-detection
#   Username : fraud-detection
#   Password : fraud-detection_password

import json
import os
import time
from datetime import datetime

import httpx
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from fastapi import BackgroundTasks, FastAPI, Header
from pydantic import BaseModel
from sqlalchemy import text

from src.utils.db import get_postgres_engine, get_redis_client
from src.utils.features import haversine_vectorized
from src.utils.mlflow_manager import load_champion_model as fetch_champion_model

# --- 1. CONFIGURATION POSTGRESQL & REDIS VIA SRC.UTILS.DB ---
db_engine = get_postgres_engine()
redis_client = get_redis_client()
if redis_client is not None:
    print("Connexion globale à Redis pour l'API initialisée.")
else:
    print("Avertissement : Connexion à Redis impossible pour l'API.")

# --- 2. CONFIGURATION DE L'APPLICATION FASTAPI ---
active_decision_threshold: float = 0.50

app = FastAPI(
    title="API de Détection de Fraude - MLOps",
    description="Inférence en temps réel avec double scoring : Règles Redis (Fast Pass) + XGBoost.",
    version="1.2.0",
)


# --- 3. DÉFINITION DES SCHÉMAS PYDANTIC & EXEMPLES ---
class TransactionInput(BaseModel):
    trans_date_trans_time: str
    cc_num: int
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: int
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    trans_num: str
    unix_time: int
    merch_lat: float
    merch_long: float
    is_fraud: int


class TransactionBatch(BaseModel):
    transactions: list[TransactionInput]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transactions": [
                        {
                            "trans_date_trans_time": "2020-07-22 14:05:00",
                            "cc_num": 423578912345,
                            "merchant": "fraud_gas_station",
                            "category": "gas_transport",
                            "amt": 85.50,
                            "first": "Caro",
                            "last": "MS",
                            "gender": "F",
                            "street": "12 rue de la Paix",
                            "city": "Lyon",
                            "state": "Rhone",
                            "zip": 69000,
                            "lat": 45.764043,
                            "long": 4.835659,
                            "city_pop": 513000,
                            "job": "Data Ingé",
                            "dob": "1985-04-12",
                            "trans_num": "test_tx_001",
                            "unix_time": 1595426700,
                            "merch_lat": 45.768000,
                            "merch_long": 4.840000,
                            "is_fraud": 0,
                        }
                    ]
                }
            ]
        }
    }


class WebhookData(BaseModel):
    transaction_id: str
    cc_num_sha256: str
    amount: float
    category: str
    merchant: str
    prediction: int
    prediction_proba: float
    explications_shap: dict[str, float]


class WebhookPayload(BaseModel):
    event: str
    timestamp: str
    data: WebhookData


class WebhookResponse(BaseModel):
    status: str
    message: str


class WebhookRequest(BaseModel):
    transaction_id: str


# Variables globales pour le modèle ML
model_pipeline = None
model_run_id = "unknown"
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.50"))


# --- 4.5. CALCUL SHAP EN TEMPS RÉEL (EXPLICABILITÉ) ---
def compute_shap_values(model_pipeline, X):
    try:
        if hasattr(model_pipeline, "named_steps"):
            preprocessor = model_pipeline.named_steps["preprocessor"]
            predictor = model_pipeline.named_steps["model"]
            X_enc = preprocessor.transform(X)
            feature_names = list(preprocessor.get_feature_names_out())
        elif hasattr(model_pipeline, "ae_extractor"):
            predictor = model_pipeline.classifier
            X_enc = model_pipeline.ae_extractor.transform(X)
            try:
                feature_names = list(model_pipeline.get_feature_names_out())
            except Exception:
                feature_names = [f"feat_{i}" for i in range(X_enc.shape[1])]
        elif hasattr(model_pipeline, "hinsage"):
            predictor = model_pipeline.classifier
            test_embeddings = model_pipeline.hinsage.transform(X)
            X_enc = model_pipeline._prepare_features(X, test_embeddings)
            try:
                feature_names = list(model_pipeline.get_feature_names_out())
            except Exception:
                feature_names = [f"feat_{i}" for i in range(X_enc.shape[1])]
        elif hasattr(model_pipeline, "classifier"):
            predictor = model_pipeline.classifier
            X_enc = X
            feature_names = list(X.columns)
        else:
            return [{} for _ in range(len(X))]

        # Convertir en DataFrame pour l'explication si c'est un tableau numpy
        if not isinstance(X_enc, pd.DataFrame):
            X_enc_df = pd.DataFrame(X_enc, columns=feature_names)
        else:
            X_enc_df = X_enc

        # Explainer Tree SHAP
        explainer = shap.TreeExplainer(predictor)
        raw_shap = explainer.shap_values(X_enc_df)

        # Adapter la dimension des SHAP values selon le format retourné
        if isinstance(raw_shap, list):
            if len(raw_shap) == 2:
                raw_shap = raw_shap[1]
            else:
                raw_shap = raw_shap[0]
        elif len(raw_shap.shape) == 3:
            raw_shap = raw_shap[:, :, 1]

        # Extraire les features d'intérêt pour chaque ligne
        shap_dicts = []
        for i in range(len(X)):
            row_dict = {}
            for col in [
                "amt",
                "distance_achat",
                "age",
                "city_pop",
                "hour_sin",
                "hour_cos",
                "weekday_sin",
                "weekday_cos",
                "month_sin",
                "month_cos",
            ]:
                if col in feature_names:
                    idx = feature_names.index(col)
                    val = raw_shap[i, idx]
                    if hasattr(val, "__len__"):
                        val = np.ravel(val)[0]
                    row_dict[col] = float(val)
                else:
                    row_dict[col] = 0.0
            shap_dicts.append(row_dict)
        return shap_dicts
    except Exception as e:
        print(f"[SHAP API Engine] Échec du calcul SHAP : {e}")
        return [{} for _ in range(len(X))]


# --- 5. LOGGING ASYNCHRONE DANS POSTGRESQL (INSERT-ONLY) ---
def save_predictions_to_db(
    transactions_list: list,
    predictions: list,
    probabilities: list,
    model_version: str,
    fast_pass_suspicions: list,
    fast_pass_scores: list,
    prediction_latency_ms: float,
    shap_values_list: list,
):
    query = text("""
        INSERT INTO silver.rawdata (
            trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender,
            street, city, state, zip, lat, long, city_pop, job, dob, trans_num,
            unix_time, merch_lat, merch_long, is_fraud, prediction, prediction_proba, model_version,
            fast_pass_suspicion, fast_pass_score, prediction_latency_ms, shap_values
        ) VALUES (
            :trans_date_trans_time, :cc_num, :merchant, :category, :amt, :first, :last, :gender,
            :street, :city, :state, :zip, :lat, :long, :city_pop, :job, :dob, :trans_num,
            :unix_time, :merch_lat, :merch_long, :is_fraud, :prediction, :prediction_proba, :model_version,
            :fast_pass_suspicion, :fast_pass_score, :prediction_latency_ms, :shap_values
        ) ON CONFLICT (trans_num) DO NOTHING;
    """)

    params_list = []
    for i, t in enumerate(transactions_list):
        t_param = t.copy()
        t_param["prediction"] = int(predictions[i])
        t_param["prediction_proba"] = float(probabilities[i])
        t_param["model_version"] = model_version
        t_param["fast_pass_suspicion"] = int(fast_pass_suspicions[i])
        t_param["fast_pass_score"] = int(fast_pass_scores[i])
        t_param["prediction_latency_ms"] = float(prediction_latency_ms)
        t_param["shap_values"] = (
            json.dumps(shap_values_list[i]) if shap_values_list[i] is not None else None
        )
        params_list.append(t_param)

    try:
        with db_engine.connect() as conn:
            conn.execute(query, params_list)
            conn.commit()
        print(
            f"[Postgres MLOps] Ingestion réussie pour {len(transactions_list)} transactions (XGBoost + Fast Pass + SHAP + Latency)."
        )
    except Exception as e:
        print(f"[Postgres MLOps] Erreur d'écriture dans la base : {e}")


# --- 5.5. ENVOI DE WEBHOOK AU MARCHAND EN CAS DE FRAUDE (ASYNCHRONE) ---
def send_fraud_webhook(
    transaction_data: dict, prediction: int, probability: float, shap_values: dict
):
    webhook_url = os.getenv(
        "MERCHANT_WEBHOOK_URL", "http://localhost:8000/mock-merchant-webhook"
    )

    payload = {
        "transaction_id": str(transaction_data.get("trans_num")),
    }
    headers = {
        "Content-Type": "application/json",
        "x-merchant-token": "demo-secret-key-123",
    }

    try:
        response = httpx.post(webhook_url, json=payload, headers=headers, timeout=5.0)
        if response.status_code in [200, 201, 202]:
            print(
                f"[Webhook MLOps] Notification envoyée avec succès au marchand pour la transaction {transaction_data.get('trans_num')}."
            )
        else:
            print(
                f"[Webhook MLOps] Échec de l'envoi du webhook (code {response.status_code})."
            )
    except Exception as e:
        print(
            f"[Webhook MLOps] Erreur lors de l'envoi du webhook vers {webhook_url} : {e}"
        )


# --- 6. GESTION DU MODÈLE CHAMPION & SYNCHRONISATION AUTOMATIQUE ---
def init_postgres_schema():
    """Initialisation et migration sécurisée du schéma PostgreSQL."""
    try:
        with db_engine.connect() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS silver;"))
            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS silver.rawdata (
                    trans_date_trans_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    cc_num BIGINT,
                    merchant VARCHAR(255),
                    category VARCHAR(255),
                    amt NUMERIC(10, 2),
                    first VARCHAR(255),
                    last VARCHAR(255),
                    gender VARCHAR(10),
                    street VARCHAR(255),
                    city VARCHAR(255),
                    state VARCHAR(50),
                    zip INT,
                    lat NUMERIC(10, 6),
                    long NUMERIC(10, 6),
                    city_pop INT,
                    job VARCHAR(255),
                    dob DATE,
                    trans_num VARCHAR(255) PRIMARY KEY,
                    unix_time BIGINT,
                    merch_lat NUMERIC(10, 6),
                    merch_long NUMERIC(10, 6),
                    is_fraud INT,
                    prediction INT,
                    prediction_proba NUMERIC(5, 4),
                    model_version VARCHAR(50),
                    fast_pass_suspicion INT,
                    fast_pass_score INT,
                    prediction_latency_ms NUMERIC(10, 4),
                    shap_values JSONB,
                    logged_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            )

            # Colonnes d'observabilité
            for col_sql in [
                "ALTER TABLE silver.rawdata ADD COLUMN IF NOT EXISTS prediction_latency_ms NUMERIC(10, 4);",
                "ALTER TABLE silver.rawdata ADD COLUMN IF NOT EXISTS shap_values JSONB;",
                "ALTER TABLE silver.rawdata ADD COLUMN IF NOT EXISTS fast_pass_suspicion INT DEFAULT 0;",
                "ALTER TABLE silver.rawdata ADD COLUMN IF NOT EXISTS fast_pass_score INT DEFAULT 0;",
            ]:
                try:
                    conn.execute(text(col_sql))
                except Exception:
                    pass
            conn.commit()
            print("[Postgres MLOps] Schéma silver.rawdata validé.")
    except Exception as e:
        print(f"[Postgres MLOps] Erreur schéma : {e}")


def load_champion_model():
    """Charge ou recharge le modèle Champion actif depuis MLflow via le gestionnaire MLOps."""
    global model_pipeline
    global model_run_id
    global active_decision_threshold

    loaded_model, new_version_id, new_threshold = fetch_champion_model(
        model_name="fraud_detector",
        alias="champion",
        fallback_uri="runs:/dba1e5b2807b4785a89dc0d23a247c17/model",
    )
    if loaded_model is not None:
        model_pipeline = loaded_model
        model_run_id = new_version_id
        active_decision_threshold = float(new_threshold)
    return model_run_id


async def auto_sync_champion_loop(check_interval_seconds: int = 10):
    """Tâche d'arrière-plan surveillant automatiquement les promotions dans MLflow."""
    global model_run_id
    print(
        f"[MLOps Auto-Sync] 🔄 Boucle de synchronisation automatique activée ({check_interval_seconds}s intervalle)."
    )
    import asyncio

    while True:
        try:
            await asyncio.sleep(check_interval_seconds)
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            version_details = client.get_model_version_by_alias(
                "fraud_detector", "champion"
            )
            target_id = f"fraud_detector_v{version_details.version}"

            if model_run_id != target_id:
                print(
                    f"[MLOps Auto-Sync] 🔔 Nouvelle version Champion détectée dans MLflow : {target_id} (actuelle en RAM: {model_run_id}). Rechargement automatique..."
                )
                load_champion_model()
        except Exception:
            pass


@app.on_event("startup")
async def startup_event():
    import asyncio

    print("--- DÉMARRAGE DE L'API FRAUD DETECTION MLOPS ---")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    init_postgres_schema()
    load_champion_model()

    # Démarrage de la synchronisation automatique en arrière-plan
    asyncio.create_task(auto_sync_champion_loop(check_interval_seconds=10))


# --- 7. ROUTES HTTP ---
@app.get("/")
def read_root():
    return {
        "message": "Bienvenue sur l'API de Détection de Fraude - MLOps. Utilisez /predict_batch pour l'inférence",
        "active_model": model_run_id,
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "active_model": model_run_id}


@app.get("/model-info")
def model_info():
    return {
        "status": "success",
        "active_model_version": model_run_id,
        "decision_threshold": active_decision_threshold,
    }


@app.post("/reload-model")
def reload_model():
    try:
        current_v = load_champion_model()
        return {
            "status": "success",
            "message": f"Modèle rechargé avec succès en mémoire : {current_v}",
            "active_model": current_v,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erreur lors du rechargement du modèle : {e}",
        }


@app.post("/predict_batch")
def predict_batch(batch: TransactionBatch, background_tasks: BackgroundTasks):
    global model_pipeline
    global model_run_id
    global redis_client

    if model_pipeline is None:
        return {"status": "error", "message": "Le modèle n'est pas chargé en mémoire."}

    if not batch.transactions:
        return {"status": "success", "predictions": []}

    # 1. Conversion du batch Pydantic en DataFrame pandas
    transactions_list = [t.dict() for t in batch.transactions]
    df = pd.DataFrame(transactions_list)

    # 2. Feature Engineering
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    df["hour_sin"] = np.sin(2 * np.pi * df["trans_date_trans_time"].dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["trans_date_trans_time"].dt.hour / 24.0)
    df["weekday_sin"] = np.sin(
        2 * np.pi * df["trans_date_trans_time"].dt.dayofweek / 7.0
    )
    df["weekday_cos"] = np.cos(
        2 * np.pi * df["trans_date_trans_time"].dt.dayofweek / 7.0
    )
    df["month_sin"] = np.sin(2 * np.pi * df["trans_date_trans_time"].dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["trans_date_trans_time"].dt.month / 12.0)

    df["distance_achat"] = haversine_vectorized(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    dob_col = pd.to_datetime(df["dob"])
    df["age"] = 2020 - dob_col.dt.year

    # 3. Sélection des variables pour le Pipeline ML
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

    # ==========================================================
    # 4. ÉVALUATION DU SCORE DE SUSPICION (REDIS FAST PASS)
    # ==========================================================
    fast_pass_suspicions = []
    fast_pass_scores = []

    # Lecture en temps réel des règles compilées de suspicion depuis Redis
    rules = None
    if redis_client is not None:
        try:
            rules_raw = redis_client.get("fraud_rules:config")
            if rules_raw:
                rules = json.loads(rules_raw)
        except Exception as redis_err:
            print(
                f"[Redis Rule Engine] Échec de la récupération des seuils : {redis_err}"
            )

    # Calcul systématique du score pour chaque transaction
    for i, row in df.iterrows():
        suspicion = 0
        score = 0

        if rules:
            thresholds = rules.get("thresholds", {})
            suspicious_categories = rules.get("suspicious_categories", [])
            suspicious_hours = rules.get("suspicious_hours", [])
            suspicious_weekdays = rules.get("suspicious_weekdays", [])

            # Extraction des valeurs
            amt = float(row["amt"])
            distance_achat = float(row["distance_achat"])
            age = int(row["age"])
            city_pop = int(row["city_pop"])
            category = str(row["category"])
            hour = int(row["trans_date_trans_time"].hour)
            weekday = int(row["trans_date_trans_time"].dayofweek)

            # Évaluation du score
            if amt > thresholds.get("amt_max", 300.0):
                score += 2
            if distance_achat > thresholds.get("distance_achat_max", 50.0):
                score += 2
            if category in suspicious_categories:
                score += 1
            if hour in suspicious_hours:
                score += 1
            if weekday in suspicious_weekdays:
                score += 1
            if age > thresholds.get("age_max", 38.0):
                score += 1
            if city_pop > thresholds.get("city_pop_max", 3600.0):
                score += 1

            # Seuil de déclenchement suspicion Fast Pass
            if score >= 4:
                suspicion = 1

        fast_pass_suspicions.append(suspicion)
        fast_pass_scores.append(score)

    # ==========================================================
    # 5. INFÉRENCE SYSTÉMATIQUE XGBOOST AVEC LATENCE
    # ==========================================================
    start_time = time.time()
    try:
        if (
            hasattr(model_pipeline, "hinsage")
            or hasattr(model_pipeline, "mu_loss_")
            or hasattr(model_pipeline, "classifier")
        ):
            probabilities = model_pipeline.predict_proba(df)[:, 1]
        else:
            probabilities = model_pipeline.predict_proba(X)[:, 1]
        predictions = (probabilities >= active_decision_threshold).astype(int)
    except Exception as ml_err:
        return {
            "status": "error",
            "message": f"Erreur pendant l'inférence XGBoost : {ml_err}",
        }
    end_time = time.time()
    prediction_latency_ms = ((end_time - start_time) * 1000.0) / max(1, len(df))

    # ==========================================================
    # 5.5 CALCUL SÉLECTIF DES CONTRIBUTIONS SHAP LOCALES
    # ==========================================================
    shap_values_list = [None] * len(df)
    suspicious_indices = [
        idx
        for idx in range(len(df))
        if predictions[idx] == 1 or fast_pass_suspicions[idx] == 1
    ]
    if suspicious_indices:
        X_suspicious = X.iloc[suspicious_indices]
        shap_suspicious = compute_shap_values(model_pipeline, X_suspicious)
        for idx, s_idx in enumerate(suspicious_indices):
            shap_values_list[s_idx] = shap_suspicious[idx]

    # ==========================================================
    # 6. ENREGISTREMENT ASYNCHRONE DANS LA BASE POSTGRES
    # ==========================================================
    background_tasks.add_task(
        save_predictions_to_db,
        transactions_list,
        list(predictions),
        list(probabilities),
        model_run_id,
        fast_pass_suspicions,
        fast_pass_scores,
        prediction_latency_ms,
        shap_values_list,
    )

    # 6.5. ENVOI DES WEBHOOKS POUR LES TRANSACTIONS FRAUDULEUSES
    for i, t in enumerate(batch.transactions):
        if predictions[i] == 1:
            background_tasks.add_task(
                send_fraud_webhook,
                transactions_list[i],
                predictions[i],
                probabilities[i],
                shap_values_list[i],
            )

    # 7. Préparation de la réponse de l'API
    results = []
    for i, t in enumerate(batch.transactions):
        results.append(
            {
                "transaction_id": t.trans_num,
                "prediction": int(predictions[i]),
                "prediction_proba": float(probabilities[i]),
                "fast_pass_suspicion": int(fast_pass_suspicions[i]),
                "fast_pass_score": int(fast_pass_scores[i]),
                "model_version": model_run_id,
            }
        )

    return {"status": "success", "predictions": results}


# --- 8. ENDPOINT DE SIMULATION DE RÉCEPTEUR WEBHOOK MARCHAND ---
@app.post(
    "/mock-merchant-webhook",
    response_model=WebhookPayload,
    summary="Mock de réception de webhook marchand sécurisé",
    description="Simule l'écouteur du marchand recevant les alertes de transactions suspectes. Valide la présence d'un en-tête d'authentification simulated X-Merchant-Token et renvoie le payload complet du webhook après récupération des détails de transaction dans PostgreSQL.",
)
def mock_merchant_webhook(
    payload: WebhookRequest,
    x_merchant_token: str = Header(
        ..., description="Token d'authentification simulé du marchand (ex: secret_key)"
    ),
):
    global redis_client
    print(
        f"[Mock Merchant Server] Requête de webhook reçue pour la transaction ID: {payload.transaction_id}"
    )

    # Valeurs par défaut (fallback)
    import hashlib

    cc_num_sha256 = hashlib.sha256(b"423578912345").hexdigest()
    amount = 949.99
    category = "misc_net"
    merchant = "fraud_Ferry, Lynch and Kautzer"
    prediction = 1
    prediction_proba = 0.9962
    explications_shap = {
        "amt": 0.15,
        "distance_achat": 0.35,
        "age": 0.05,
        "city_pop": 0.01,
        "hour_sin": 0.04,
        "hour_cos": -0.02,
    }
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Essayer de récupérer les données réelles de la transaction dans PostgreSQL
    try:
        with db_engine.connect() as conn:
            query = text("""
                SELECT cc_num, amt, category, merchant, prediction, prediction_proba, shap_values, trans_date_trans_time
                FROM silver.rawdata
                WHERE trans_num = :trans_num
            """)
            result = conn.execute(
                query, {"trans_num": payload.transaction_id}
            ).fetchone()
            if result:
                cc_num_sha256 = hashlib.sha256(str(result[0]).encode()).hexdigest()
                amount = float(result[1])
                category = str(result[2])
                merchant = str(result[3])
                prediction = int(result[4])
                prediction_proba = float(result[5])

                shap_str = result[6]
                if shap_str:
                    try:
                        explications_shap = json.loads(shap_str)
                    except Exception:
                        pass

                timestamp = pd.to_datetime(result[7]).isoformat() + "Z"
    except Exception as db_err:
        print(f"[Mock Merchant Server] Échec de la requête Postgres : {db_err}")

    # Construction du payload complet de webhook
    webhook_payload = WebhookPayload(
        event="transaction.suspecte",
        timestamp=timestamp,
        data=WebhookData(
            transaction_id=payload.transaction_id,
            cc_num_sha256=cc_num_sha256,
            amount=amount,
            category=category,
            merchant=merchant,
            prediction=prediction,
            prediction_proba=prediction_proba,
            explications_shap=explications_shap,
        ),
    )

    # Écriture dans Redis pour l'affichage en direct sur le Dashboard
    if redis_client is not None:
        try:
            import time

            now = time.time()
            if redis_client.type("merchant_webhook_alerts") == "list":
                redis_client.delete("merchant_webhook_alerts")
            redis_client.zadd(
                "merchant_webhook_alerts", {json.dumps(webhook_payload.dict()): now}
            )
            redis_client.zremrangebyscore(
                "merchant_webhook_alerts", "-inf", now - 86400
            )
        except Exception as redis_err:
            print(
                f"[Mock Merchant Server] Échec de l'écriture dans Redis : {redis_err}"
            )

    return webhook_payload
