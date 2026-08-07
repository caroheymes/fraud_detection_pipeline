# tests/test_api.py
import json
import os
import sys
from unittest.mock import MagicMock

import numpy as np

# Ajouter la racine du projet au path pour éviter les erreurs d'import en CI/CD
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 1. Mock de MLflow, de Redis et de SHAP avant d'importer l'API pour désactiver les connexions réseau
sys.modules['mlflow'] = MagicMock()
sys.modules['mlflow.sklearn'] = MagicMock()
sys.modules['redis'] = MagicMock()
sys.modules['shap'] = MagicMock()
sys.modules['sqlalchemy'] = MagicMock()

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

# Mock de Redis pour retourner des règles configurées
mock_redis = MagicMock()
mock_redis.get.return_value = json.dumps({
    "thresholds": {
        "amt_max": 300.0,
        "distance_achat_max": 50.0,
        "age_max": 38.0,
        "city_pop_max": 3600.0
    },
    "suspicious_categories": ["travel", "food_dining"],
    "suspicious_hours": [3, 22, 23],
    "suspicious_weekdays": [5, 6],
    "suspicious_months": [6]
})
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
    """Vérifie que la route de diagnostic /health fonctionne"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_api_predict_batch_schema():
    """Envoie une transaction et vérifie le bon typage de la réponse de predict_batch"""
    payload = {
        "transactions": [
            {
                "trans_date_trans_time": "2020-07-22 23:05:00", # Heure suspecte (23)
                "cc_num": 123456789,
                "merchant": "fraud_gas_station",
                "category": "travel", # Catégorie suspecte
                "amt": 500.0, # Montant suspect (> 300)
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
                "is_fraud": 0
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
