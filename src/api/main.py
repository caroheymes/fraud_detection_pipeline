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
import redis
import shap
from fastapi import BackgroundTasks, FastAPI, Header
from pydantic import BaseModel
from sqlalchemy import create_engine, text

# --- 1. CONFIGURATION POSTGRESQL ---
pg_user = os.getenv("POSTGRES_USER", "fraud-detection")
pg_password = os.getenv("POSTGRES_PASSWORD", "fraud-detection_password")
pg_host = os.getenv("POSTGRES_HOST", "postgres")
pg_port = os.getenv("POSTGRES_PORT", "5432")
pg_db = os.getenv("POSTGRES_DB", "fraud-detection")

DATABASE_URL = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
db_engine = create_engine(DATABASE_URL)

# --- 1.5. CONFIGURATION REDIS ---
redis_host = os.getenv("REDIS_HOST", "redis")
redis_client = None
try:
    redis_client = redis.Redis(host=redis_host, port=6379, db=0, decode_responses=True)
    print("Connexion globale à Redis pour l'API initialisée.")
except Exception as re_err:
    print(f"Avertissement : Connexion à Redis impossible pour l'API : {re_err}")

# --- 2. CONFIGURATION DE L'APPLICATION FASTAPI ---
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


# --- 4. FONCTIONS DE CALCUL AUXILIAIRES ---
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0  # Rayon de la Terre en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


# --- 4.5. CALCUL SHAP EN TEMPS RÉEL (EXPLICABILITÉ) ---
def compute_shap_values(model_pipeline, X):
    try:
        preprocessor = model_pipeline.named_steps["preprocessor"]
        predictor = model_pipeline.named_steps["model"]

        # Encodage des features
        X_enc = preprocessor.transform(X)
        feature_names = list(preprocessor.get_feature_names_out())

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
            ]:
                if col in feature_names:
                    idx = feature_names.index(col)
                    row_dict[col] = float(raw_shap[i, idx])
                else:
                    row_dict[col] = 0.0
            shap_dicts.append(row_dict)
        return shap_dicts
    except Exception as e:
        print(f"[SHAP API Engine] Échec du calcul SHAP : {e}")
        # Repli sur des valeurs vides en cas d'erreur
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
        t_param["shap_values"] = json.dumps(shap_values_list[i])
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
        "event": "transaction.suspecte",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": {
            "transaction_id": transaction_data.get("trans_num"),
            "amount": float(transaction_data.get("amt", 0.0)),
            "category": transaction_data.get("category"),
            "merchant": transaction_data.get("merchant"),
            "prediction": int(prediction),
            "prediction_proba": float(probability),
            "explications_shap": shap_values,
        },
    }

    try:
        response = httpx.post(webhook_url, json=payload, timeout=5.0)
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


# --- 6. INITIALISATION AU DÉMARRAGE ---
@app.on_event("startup")
def startup_event():
    global model_pipeline
    global model_run_id
    print("--- DÉMARRAGE DE L'API : CHARGEMENT DU MODÈLE CHAMPION ---")

    # 1. Configuration MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    # 2. Initialisation de la base de données & Alteration du schéma
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

            # Ajout sécurisé des colonnes si la table pré-existait sans elles
            try:
                conn.execute(
                    text(
                        "ALTER TABLE silver.rawdata ADD COLUMN IF NOT EXISTS prediction_latency_ms NUMERIC(10, 4);"
                    )
                )
                conn.execute(
                    text(
                        "ALTER TABLE silver.rawdata ADD COLUMN IF NOT EXISTS shap_values JSONB;"
                    )
                )
                conn.execute(
                    text(
                        "ALTER TABLE silver.rawdata ADD COLUMN IF NOT EXISTS fast_pass_suspicion INT;"
                    )
                )
                conn.execute(
                    text(
                        "ALTER TABLE silver.rawdata ADD COLUMN IF NOT EXISTS fast_pass_score INT;"
                    )
                )
                conn.commit()
            except Exception as schema_err:
                print(
                    f"[Postgres Schema Update] Erreur de mise à niveau de table : {schema_err}"
                )

            # S'assurer que les deux colonnes d'observabilité Fast Pass existent dans la table
            conn.execute(
                text(
                    "ALTER TABLE silver.rawdata ADD COLUMN IF NOT EXISTS fast_pass_suspicion INT DEFAULT 0;"
                )
            )
            conn.execute(
                text(
                    "ALTER TABLE silver.rawdata ADD COLUMN IF NOT EXISTS fast_pass_score INT DEFAULT 0;"
                )
            )
            conn.commit()
            print(
                "Schéma de la base PostgreSQL validé (Insert-only avec colonnes Fast Pass)."
            )
    except Exception as e:
        print(f"Erreur critique lors de la connexion/migration de PostgreSQL : {e}")

    # 3. Récupération du modèle champion depuis le registre MLflow
    try:
        model_uri = "models:/fraud_detector@champion"
        model_pipeline = mlflow.sklearn.load_model(model_uri)

        # Résolution dynamique de version du modèle champion
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            version_details = client.get_model_version_by_alias(
                "fraud_detector", "champion"
            )
            model_run_id = f"fraud_detector_v{version_details.version}"
        except Exception:
            model_run_id = "fraud_detector_champion"

        print(
            f"Modèle '{model_uri}' chargé avec l'identifiant version '{model_run_id}'."
        )
    except Exception as e:
        print(f"Impossible de charger le modèle champion : {e}")
        print("Tentative de chargement du modèle de fallback...")
        try:
            model_uri = "runs:/dba1e5b2807b4785a89dc0d23a247c17/model"
            model_pipeline = mlflow.sklearn.load_model(model_uri)
            model_run_id = "runs_dba1e5b2"
            print("Modèle de secours chargé en mémoire.")
        except Exception as fallback_err:
            print(
                f"Erreur critique lors du chargement du modèle de secours : {fallback_err}"
            )


# --- 7. ROUTES HTTP ---
@app.get("/")
def read_root():
    return {
        "message": "Bienvenue sur l'API de Détection de Fraude - MLOps. Utilisez /predict_batch pour l'inférence"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/reload-model")
def reload_model():
    try:
        startup_event()
        return {
            "status": "success",
            "message": "Nouveau modèle chargé en mémoire avec succès.",
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
        predictions = model_pipeline.predict(X)
        probabilities = model_pipeline.predict_proba(X)[:, 1]
    except Exception as ml_err:
        return {
            "status": "error",
            "message": f"Erreur pendant l'inférence XGBoost : {ml_err}",
        }
    end_time = time.time()
    prediction_latency_ms = ((end_time - start_time) * 1000.0) / max(1, len(df))

    # ==========================================================
    # 5.5 CALCUL DES CONTRIBUTIONS SHAP LOCALES
    # ==========================================================
    shap_values_list = compute_shap_values(model_pipeline, X)

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
    description="Simule l'écouteur du marchand recevant les alertes de transactions suspectes. Valide la présence d'un en-tête d'authentification simulated X-Merchant-Token et renvoie le payload complet du webhook après récupération des détails de transaction dans PostgreSQL."
)
def mock_merchant_webhook(
    payload: WebhookRequest,
    x_merchant_token: str = Header(..., description="Token d'authentification simulé du marchand (ex: secret_key)")
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
        "hour_cos": -0.02
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
            result = conn.execute(query, {"trans_num": payload.transaction_id}).fetchone()
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
            explications_shap=explications_shap
        )
    )

    # Écriture dans Redis pour l'affichage en direct sur le Dashboard
    if redis_client is not None:
        try:
            import time
            now = time.time()
            if redis_client.type("merchant_webhook_alerts") == "list":
                redis_client.delete("merchant_webhook_alerts")
            redis_client.zadd("merchant_webhook_alerts", {json.dumps(webhook_payload.dict()): now})
            redis_client.zremrangebyscore("merchant_webhook_alerts", "-inf", now - 86400)
        except Exception as redis_err:
            print(f"[Mock Merchant Server] Échec de l'écriture dans Redis : {redis_err}")

    return webhook_payload
