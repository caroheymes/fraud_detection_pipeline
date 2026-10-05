# tests/test_mlflow_manager.py
import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.features import (
    prepare_features,
)
from src.utils.mlflow_manager import MLflowQualityGate


# ============================================================================
# 1. TESTS UNITAIRES FEATURES TRANSVERSES
# ============================================================================
def test_prepare_features_pipeline():
    """Valide l'ensemble du pipeline prepare_features (cyclique, spatial, démographique, graph ids)."""
    df = pd.DataFrame(
        {
            "cc_num": [111, 222],
            "merchant": ["merch_A", "merch_B"],
            "amt": [50.0, 120.0],
            "lat": [48.85, 45.76],
            "long": [2.35, 4.83],
            "merch_lat": [48.86, 45.77],
            "merch_long": [2.36, 4.84],
            "trans_date_trans_time": ["2020-05-15 14:30:00", "2020-06-20 22:15:00"],
            "dob": ["1990-05-15", "1980-01-01"],
            "is_fraud": [0, 1],
        }
    )

    df_res = prepare_features(df, include_graph_ids=True)

    # Vérification des colonnes attendues
    assert "distance_achat" in df_res.columns
    assert "hour_sin" in df_res.columns
    assert "hour_cos" in df_res.columns
    assert "weekday_sin" in df_res.columns
    assert "weekday_cos" in df_res.columns
    assert "month_sin" in df_res.columns
    assert "month_cos" in df_res.columns
    assert "age" in df_res.columns
    assert "client_node" in df_res.columns
    assert "merchant_node" in df_res.columns
    assert "fraud_label" in df_res.columns

    # Vérification du calcul d'âge cohérent avec l'année de la transaction (2020)
    assert df_res["age"].iloc[0] == 2020 - 1990  # 30 ans
    assert df_res["age"].iloc[1] == 2020 - 1980  # 40 ans

    # Vérification distance non nulle
    assert df_res["distance_achat"].iloc[0] > 0.0


# ============================================================================
# 2. TESTS UNITAIRES MLFLOW QUALITY GATE
# ============================================================================
@patch("src.utils.mlflow_manager.MlflowClient")
@patch("src.utils.mlflow_manager.mlflow")
def test_mlflow_quality_gate_metric_resolution(mock_mlflow, mock_client_cls):
    """Vérifie la sélection de la métrique cible selon la configuration (F2, F1, Recall)."""
    gate_f2 = MLflowQualityGate(metric_target="f2")
    metrics = {"f2_class_1": 0.85, "f1_class_1": 0.75, "accuracy": 0.99}
    assert gate_f2.get_target_metric_key(metrics) == "f2_class_1"

    gate_f1 = MLflowQualityGate(metric_target="f1")
    assert gate_f1.get_target_metric_key(metrics) == "f1_class_1"


@patch("src.utils.mlflow_manager.MlflowClient")
@patch("src.utils.mlflow_manager.mlflow")
def test_mlflow_quality_gate_promotion_superior(mock_mlflow, mock_client_cls):
    """Vérifie qu'un candidat avec un meilleur score est promu avec l'alias @champion."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    # Simulation d'un champion existant avec F2 = 0.70
    mock_champ = MagicMock()
    mock_champ.version = "10"
    mock_champ.run_id = "run_champ_10"
    mock_client.get_model_version_by_alias.return_value = mock_champ

    mock_run = MagicMock()
    mock_run.data.metrics = {"f2_class_1": 0.70}
    mock_run.data.params = {"model_type": "XGBClassifier"}
    mock_client.get_run.return_value = mock_run

    # Simulation de la recherche de la version créée (V11)
    mock_v11 = MagicMock()
    mock_v11.version = 11
    mock_client.search_model_versions.return_value = [mock_v11]

    gate = MLflowQualityGate(model_name="fraud_detector", metric_target="f2")
    gate.client = mock_client

    candidate_metrics = {"f2_class_1": 0.85}  # 0.85 > 0.70 -> Promotion attendue
    promoted, version = gate.log_and_evaluate(
        model=MagicMock(),
        metrics=candidate_metrics,
    )

    assert promoted is True
    assert version == "11"
    mock_client.set_registered_model_alias.assert_called_with(
        name="fraud_detector", alias="champion", version="11"
    )


@patch("src.utils.mlflow_manager.MlflowClient")
@patch("src.utils.mlflow_manager.mlflow")
def test_mlflow_quality_gate_rejection_inferior(mock_mlflow, mock_client_cls):
    """Vérifie qu'un candidat avec un score inférieur ou égal n'est PAS promu."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    # Simulation d'un champion existant avec F2 = 0.80
    mock_champ = MagicMock()
    mock_champ.version = "10"
    mock_champ.run_id = "run_champ_10"
    mock_client.get_model_version_by_alias.return_value = mock_champ

    mock_run = MagicMock()
    mock_run.data.metrics = {"f2_class_1": 0.80}
    mock_run.data.params = {"model_type": "XGBClassifier"}
    mock_client.get_run.return_value = mock_run

    mock_v11 = MagicMock()
    mock_v11.version = 11
    mock_client.search_model_versions.return_value = [mock_v11]

    gate = MLflowQualityGate(model_name="fraud_detector", metric_target="f2")
    gate.client = mock_client

    candidate_metrics = {"f2_class_1": 0.65}  # 0.65 <= 0.80 -> Pas de promotion
    promoted, version = gate.log_and_evaluate(
        model=MagicMock(),
        metrics=candidate_metrics,
    )

    assert promoted is False
    assert version == "11"
    # set_registered_model_alias ne doit PAS être appelé pour le candidat
    mock_client.set_registered_model_alias.assert_not_called()


@patch("src.utils.mlflow_manager.MlflowClient")
@patch("src.utils.mlflow_manager.mlflow")
def test_load_champion_model_success(mock_mlflow, mock_client_cls):
    """Vérifie le chargement réussi du modèle champion depuis le registry."""
    from src.utils.mlflow_manager import load_champion_model

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_version = MagicMock()
    mock_version.version = "12"
    mock_version.run_id = "run_12"
    mock_client.get_model_version_by_alias.return_value = mock_version

    mock_run = MagicMock()
    mock_run.data.params = {"decision_threshold": "0.35"}
    mock_run.data.metrics = {"f2_class_1": 0.88}
    mock_client.get_run.return_value = mock_run

    mock_model = MagicMock()
    mock_mlflow.sklearn.load_model.return_value = mock_model

    loaded_model, version_id, threshold = load_champion_model(
        model_name="fraud_detector"
    )

    assert loaded_model == mock_model
    assert version_id == "fraud_detector_v12"
    assert threshold == 0.35
    mock_mlflow.sklearn.load_model.assert_called_with("models:/fraud_detector@champion")
