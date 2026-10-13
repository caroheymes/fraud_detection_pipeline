# tests/test_api.py
import json
import os
import sys
from unittest.mock import MagicMock

import numpy as np

# Ajouter la racine du projet au path pour éviter les erreurs d'import en CI/CD
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Création d'un faux modèle pour simuler l'inférence XGBoost
class DummyXGBoostModel:
    def predict(self, X):
        # Renvoie 0 pour toutes les lignes
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        # Renvoie une probabilité nulle de fraude (classe 1) pour toutes les lignes
        probas = np.zeros((len(X), 2))
        probas[:, 0] = 1.0  # classe saine
        probas[:, 1] = 0.0  # classe fraude
        return probas


# Mock du pipeline de modèle global
dummy_pipeline = MagicMock()
dummy_pipeline.predict = DummyXGBoostModel().predict
dummy_pipeline.predict_proba = DummyXGBoostModel().predict_proba

# Importation de l'API et de ses composants
from fastapi.testclient import TestClient

import src.api.main as api_module
from src.api.main import app, haversine_vectorized

# Configuration des variables globales de l'API pour les tests (injection de mocks)
api_module.model_pipeline = dummy_pipeline
api_module.model_run_id = "test_xgb_champion_v1"
api_module.db_engine = MagicMock()

# Mock de Redis pour retourner des règles configurées et gérer les pipelines de vélocité
mock_redis = MagicMock()
mock_pipe = MagicMock()
mock_pipe.execute.return_value = [2, True]
mock_redis.pipeline.return_value = mock_pipe
mock_redis.get.return_value = json.dumps(
    {
        "thresholds": {
            "amt_max": 300.0,
            "distance_achat_max": 50.0,
            "age_max": 38.0,
            "city_pop_max": 3600.0,
            "user_daily_tx_count_max": 3,
        },
        "suspicious_categories": ["travel", "food_dining"],
        "suspicious_hours": [3, 22, 23],
        "suspicious_weekdays": [5, 6],
        "suspicious_months": [6],
    }
)
api_module.redis_client = mock_redis

client = TestClient(app)

# ==========================================================
# 2. TESTS UNITAIRES MATHÉMATIQUES & UTILITAIRES
# ==========================================================


def test_haversine_distance():
    """Vérifie le calcul de la distance physique Haversine (ex: Lyon à Paris)"""
    lat_lyon, lon_lyon = 45.764043, 4.835659
    lat_paris, lon_paris = 48.856614, 2.352222

    distance = haversine_vectorized(lat_lyon, lon_lyon, lat_paris, lon_paris)

    # La distance orthodromique théorique Lyon-Paris est d'environ 391 km
    assert 385.0 < distance < 395.0


def test_cyclical_inverse_retransformation():
    """Valide les calculs d'arctan2 pour reconstruire l'heure à partir de sin/cos"""
    # Encodage de 23 heures
    hour = 23
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)

    # Retransformation inverse (arctan2)
    angle = np.arctan2(hour_sin, hour_cos) % (2 * np.pi)
    decoded_hour = int(np.round(angle * 12.0 / np.pi) % 24)

    assert decoded_hour == hour


# ==========================================================
# 3. TESTS D'INTÉGRATION FASTAPI
# ==========================================================


def test_api_health():
    """Vérifie que la route de diagnostic /health fonctionne et renvoie l'active_model"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["active_model"] == "test_xgb_champion_v1"


def test_api_model_info():
    """Vérifie la route /model-info"""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["active_model_version"] == "test_xgb_champion_v1"
    assert "decision_threshold" in data


def test_api_reload_model():
    """Vérifie la route /reload-model"""
    response = client.post("/reload-model")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_api_predict_batch_schema():
    """Envoie une transaction et vérifie le bon typage de la réponse de predict_batch"""
    payload = {
        "transactions": [
            {
                "trans_date_trans_time": "2020-07-22 23:05:00",  # Heure suspecte (23)
                "cc_num": 123456789,
                "merchant": "fraud_gas_station",
                "category": "travel",  # Catégorie suspecte
                "amt": 500.0,  # Montant suspect (> 300)
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
                "trans_num": "test_tx_ci_001",
                "unix_time": 1595426700,
                "merch_lat": 45.768000,
                "merch_long": 4.840000,
                "is_fraud": 0,
            }
        ]
    }

    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert len(data["predictions"]) == 1

    prediction_result = data["predictions"][0]
    assert prediction_result["transaction_id"] == "test_tx_ci_001"
    assert "prediction" in prediction_result
    assert "prediction_proba" in prediction_result

    # Vérification que le Fast Pass s'est déclenché dans les métriques renvoyées
    assert prediction_result["fast_pass_suspicion"] == 1
    assert prediction_result["fast_pass_score"] >= 4


def test_ingest_parallel_requests():
    """Simule des requêtes d'ingestion parallèles pour valider la robustesse de l'API sous charge"""
    import concurrent.futures

    payload = {
        "transactions": [
            {
                "trans_date_trans_time": "2020-07-22 23:05:00",
                "cc_num": 123456789,
                "merchant": "fraud_gas_station",
                "category": "travel",
                "amt": 500.0,
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
                "trans_num": "test_tx_parallel",
                "unix_time": 1595426700,
                "merch_lat": 45.768000,
                "merch_long": 4.840000,
                "is_fraud": 0,
            }
        ]
    }

    num_threads = 10

    def send_request(i):
        req_payload = payload.copy()
        req_payload["transactions"] = [t.copy() for t in payload["transactions"]]
        req_payload["transactions"][0]["trans_num"] = f"test_tx_parallel_{i}"
        return client.post("/predict_batch", json=req_payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(send_request, i) for i in range(num_threads)]
        results = [f.result() for f in futures]

    for response in results:
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["predictions"]) == 1


def test_user_velocity_and_hash_pseudonymization():
    """Valide le hachage HMAC RGPD et la restitution de la vélocité utilisateur."""
    from src.utils.features import compute_user_hash

    # 1. Vérification de l'idempotence et du masquage des données nominatives
    h1 = compute_user_hash("Caro", "MS", "1985-04-12")
    h2 = compute_user_hash("Caro", "MS", "1985-04-12")
    h3 = compute_user_hash("Autre", "Client", "1990-01-01")

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # SHA-256 hex string

    # 2. Vérification que /predict_batch retourne la feature de vélocité
    payload = {
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
                "trans_num": "test_velocity_tx_001",
                "unix_time": 1595426700,
                "merch_lat": 45.768000,
                "merch_long": 4.840000,
                "is_fraud": 0,
            }
        ]
    }
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert "user_daily_tx_count" in res_data["predictions"][0]
    assert res_data["predictions"][0]["user_daily_tx_count"] == 2  # mock_pipe returns 2


def test_user_velocity_redis_fallback():
    """Vérifie que la fonction de vélocité retourne 1 par défaut en cas d'absence ou panne de Redis."""
    from src.api.main import get_and_increment_user_velocity

    # 1. Cas sans connexion Redis
    res_none = get_and_increment_user_velocity("test_user_hash_123", "2020-07-22", redis_conn=None)
    assert res_none == 1

    # 2. Cas avec exception levée par Redis
    failing_redis = MagicMock()
    failing_redis.pipeline.side_effect = RuntimeError("Redis connection lost")
    res_err = get_and_increment_user_velocity("test_user_hash_123", "2020-07-22", redis_conn=failing_redis)
    assert res_err == 1


def test_predict_batch_inference_error():
    """Vérifie que l'API renvoie un payload d'erreur clair lors d'un échec d'inférence du modèle."""
    failing_model = MagicMock()
    failing_model.predict_proba.side_effect = RuntimeError("Inference tensor shape mismatch")

    original_pipeline = api_module.model_pipeline
    api_module.model_pipeline = failing_model
    try:
        payload = {
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
                    "trans_num": "test_err_tx_001",
                    "unix_time": 1595426700,
                    "merch_lat": 45.768000,
                    "merch_long": 4.840000,
                    "is_fraud": 0,
                }
            ]
        }
        response = client.post("/predict_batch", json=payload)
        assert response.status_code == 200
        res_data = response.json()
        assert res_data["status"] == "error"
        assert "message" in res_data
        assert "Inference tensor shape mismatch" in res_data["message"]
    finally:
        api_module.model_pipeline = original_pipeline


def test_mock_merchant_webhook_validation(monkeypatch):
    """Vérifie le contrôle d'accès par jeton et le traitement du webhook marchand."""
    # Test sans header de jeton -> 422 Unprocessable Entity
    resp_no_token = client.post(
        "/mock-merchant-webhook", json={"transaction_id": "tx_test_123"}
    )
    assert resp_no_token.status_code == 422

    # Mock DB pour vérifier la récupération et le parsing shap_values sous forme de dict
    class MockResult:
        def fetchone(self):
            return (
                "1234567890123456",  # cc_num
                150.0,  # amt
                "misc_net",  # category
                "fraud_merchant",  # merchant
                1,  # prediction
                0.95,  # prediction_proba
                {"amt": 0.42, "distance_achat": 0.15},  # shap_values as dict
                "2023-01-01 12:00:00",  # trans_date_trans_time
            )

    class MockConn:
        def execute(self, query, params=None):
            return MockResult()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class MockEngine:
        def connect(self):
            return MockConn()

    monkeypatch.setattr(api_module, "db_engine", MockEngine())

    # Test avec header de jeton valide
    headers = {"x-merchant-token": "demo-secret-key-123"}
    resp_valid = client.post(
        "/mock-merchant-webhook",
        json={"transaction_id": "tx_test_123"},
        headers=headers,
    )
    assert resp_valid.status_code == 200
    data = resp_valid.json()
    assert data["event"] == "transaction.suspecte"
    assert data["data"]["transaction_id"] == "tx_test_123"
    assert data["data"]["amount"] == 150.0
    assert data["data"]["category"] == "misc_net"
    assert data["data"]["explications_shap"] == {"amt": 0.42, "distance_achat": 0.15}


def test_compute_shap_values_real_pipeline():
    """Vérifie le calcul unifié des attributions locales SHAP."""
    import pandas as pd
    from xgboost import XGBClassifier

    from src.api.main import compute_shap_values

    cols = ["amt", "distance_achat"]
    X_train = np.array([[10.0, 1.0], [500.0, 50.0], [20.0, 2.0], [800.0, 80.0]])
    y_train = np.array([0, 1, 0, 1])

    xgb = XGBClassifier(n_estimators=5, max_depth=2, random_state=42)
    xgb.fit(X_train, y_train)

    X_test_df = pd.DataFrame(X_train, columns=cols)
    shap_results = compute_shap_values(xgb, X_test_df)

    assert isinstance(shap_results, list)
    assert len(shap_results) == 4
    for row in shap_results:
        assert isinstance(row, dict)
        assert len(row) > 0
        assert "amt" in row
        assert "distance_achat" in row


def test_predict_batch_malformed_dob():
    """Vérifie la robustesse de l'inférence face à des dates de naissance invalides/malformées."""
    payload = {
        "transactions": [
            {
                "trans_date_trans_time": "2020-07-22 14:05:00",
                "cc_num": 123456789,
                "merchant": "valid_merchant",
                "category": "misc_net",
                "amt": 45.0,
                "first": "Alice",
                "last": "Durand",
                "gender": "F",
                "street": "1 rue test",
                "city": "Paris",
                "state": "IDF",
                "zip": 75001,
                "lat": 48.85,
                "long": 2.35,
                "city_pop": 2000000,
                "job": "Ingénieure",
                "dob": "not-a-valid-date-format",
                "trans_num": "tx_malformed_dob_001",
                "unix_time": 1595426700,
                "merch_lat": 48.86,
                "merch_long": 2.36,
                "is_fraud": 0,
            }
        ]
    }
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert len(res_data["predictions"]) == 1
    assert "prediction" in res_data["predictions"][0]


def test_isolation_forest_xgboost_pipeline_execution():
    """Vérifie l'entraînement et l'inférence du pipeline hybride Isolation Forest + XGBoost."""
    import pandas as pd

    from src.training.optimize_xgb_iforest import IsolationForestXGBoostPipeline

    df_synth = pd.DataFrame(
        {
            "amt": [15.0, 25.0, 1200.0, 30.0, 950.0],
            "category": ["grocery_pos", "gas_transport", "shopping_net", "grocery_pos", "shopping_net"],
            "gender": ["M", "F", "F", "M", "M"],
        }
    )
    y = np.array([0, 0, 1, 0, 1])

    pipe = IsolationForestXGBoostPipeline(
        n_estimators=5,
        max_depth=2,
        iso_estimators=10,
        random_state=42,
    )
    pipe.fit(df_synth, y)

    probas = pipe.predict_proba(df_synth)
    assert probas.shape == (5, 2)
    assert np.all((probas >= 0.0) & (probas <= 1.0))

    feature_names = pipe.get_feature_names_out()
    assert "iso_forest_anomaly_score" in feature_names


