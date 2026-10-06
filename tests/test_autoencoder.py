# tests/test_autoencoder.py
"""
Tests unitaires pour le détecteur d'anomalies Auto-encodeur semi-supervisé.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.autoencoder import AutoencoderFraudDetector, AutoencoderNet
from src.utils.threshold import evaluate_predictions_and_curves, find_optimal_threshold


@pytest.fixture
def synthetic_fraud_dataset():
    """Génère un dataset synthétique avec 95% de transactions normales et 5% d'anomalies extrêmes."""
    np.random.seed(42)
    n_samples = 300

    # Normal transactions
    amt = np.random.exponential(scale=50.0, size=n_samples)
    dist = np.random.normal(loc=10.0, scale=3.0, size=n_samples)
    is_fraud = np.zeros(n_samples, dtype=int)

    # Injecter 15 fraudes extrêmes
    fraud_indices = np.random.choice(n_samples, size=15, replace=False)
    amt[fraud_indices] = np.random.uniform(800.0, 2000.0, size=15)
    dist[fraud_indices] = np.random.uniform(80.0, 300.0, size=15)
    is_fraud[fraud_indices] = 1

    df = pd.DataFrame(
        {
            "trans_date_trans_time": pd.date_range(
                "2020-01-01", periods=n_samples, freq="h"
            ),
            "amt": amt,
            "category": np.random.choice(
                ["grocery_pos", "gas_transport", "misc_net"], size=n_samples
            ),
            "gender": np.random.choice(["M", "F"], size=n_samples),
            "lat": 45.76 + np.random.normal(0, 0.05, size=n_samples),
            "long": 4.83 + np.random.normal(0, 0.05, size=n_samples),
            "merch_lat": 45.76 + np.random.normal(0, 0.05, size=n_samples),
            "merch_long": 4.83 + np.random.normal(0, 0.05, size=n_samples),
            "city_pop": np.random.randint(1000, 500000, size=n_samples),
            "dob": "1990-05-15",
            "is_fraud": is_fraud,
        }
    )
    return df


def test_autoencoder_net_architecture():
    """Vérifie la dimension des tenseurs d'entrée, de sortie et de l'espace latent."""
    batch_size = 16
    in_features = 20
    latent_dim = 6
    hidden_dim = 32

    net = AutoencoderNet(
        in_features=in_features, hidden_dim=hidden_dim, latent_dim=latent_dim
    )
    dummy_input = torch.randn(batch_size, in_features)

    reconstructed, latent = net(dummy_input)

    assert reconstructed.shape == (batch_size, in_features)
    assert latent.shape == (batch_size, latent_dim)


def test_autoencoder_fraud_detector_fit_and_predict(synthetic_fraud_dataset):
    """Vérifie l'entraînement, la fonction de décision et les probabilités retournées."""
    df = synthetic_fraud_dataset
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    detector = AutoencoderFraudDetector(
        hidden_dim=32,
        latent_dim=4,
        epochs=5,
        batch_size=64,
        lr=0.005,
    )
    detector.fit(X, y)

    # 1. Decision function (Erreurs de reconstruction MSE)
    errors = detector.decision_function(X)
    assert len(errors) == len(df)
    assert np.all(errors >= 0)

    # Les erreurs moyennes sur les fraudes doivent être supérieures aux erreurs sur les légitimes
    mean_legit_error = np.mean(errors[y == 0])
    mean_fraud_error = np.mean(errors[y == 1])
    assert mean_fraud_error > mean_legit_error

    # 2. Predict proba
    probas = detector.predict_proba(X)
    assert probas.shape == (len(df), 2)
    assert np.all((probas >= 0.0) & (probas <= 1.0))
    np.testing.assert_allclose(np.sum(probas, axis=1), 1.0, atol=1e-5)

    # 3. Predict avec seuil
    preds = detector.predict(X)
    assert len(preds) == len(df)
    assert set(np.unique(preds)).issubset({0, 1})


def test_autoencoder_threshold_tuning_integration(synthetic_fraud_dataset):
    """Vérifie la calibration du seuil optimal et le calcul complet des métriques."""
    df = synthetic_fraud_dataset
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    detector = AutoencoderFraudDetector(
        hidden_dim=32,
        latent_dim=4,
        epochs=10,
        batch_size=64,
        lr=0.005,
    )
    detector.fit(X, y)

    probas = detector.predict_proba(X)[:, 1]
    opt_thresh, score = find_optimal_threshold(y, probas, metric_target="f2")

    assert 0.05 <= opt_thresh <= 0.95
    assert score >= 0.0

    metrics, cm = evaluate_predictions_and_curves(y, probas, threshold=opt_thresh)
    assert "f2_class_1" in metrics
    assert "auprc" in metrics
    assert "decision_threshold" in metrics
    assert metrics["decision_threshold"] == opt_thresh
    assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == len(df)


def test_autoencoder_xgboost_pipeline(synthetic_fraud_dataset):
    """Vérifie le pipeline hybride combinant Auto-encodeur et XGBoost."""
    from src.training.autoencoder_xgb import AutoencoderXGBoostPipeline

    df = synthetic_fraud_dataset
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    pipe = AutoencoderXGBoostPipeline(
        hidden_dim=32,
        latent_dim=4,
        ae_epochs=5,
        ae_batch_size=64,
        xgb_params={"n_estimators": 20, "max_depth": 3, "random_state": 42},
    )
    pipe.fit(X, y)

    probas = pipe.predict_proba(X)
    assert probas.shape == (len(df), 2)
    assert np.all((probas >= 0.0) & (probas <= 1.0))

    preds = pipe.predict(X)
    assert len(preds) == len(df)
    assert set(np.unique(preds)).issubset({0, 1})
