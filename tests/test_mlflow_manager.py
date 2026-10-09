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
    """Valide l'ensemble du pipeline prepare_features (cyclique, spatial, démographique, vélocité, graph ids)."""
    df = pd.DataFrame(
        {
            "cc_num": [111, 222, 111],
            "first": ["Alice", "Bob", "Alice"],
            "last": ["Smith", "Jones", "Smith"],
            "merchant": ["merch_A", "merch_B", "merch_C"],
            "amt": [50.0, 120.0, 30.0],
            "lat": [48.85, 45.76, 48.85],
            "long": [2.35, 4.83, 2.35],
            "merch_lat": [48.86, 45.77, 48.87],
            "merch_long": [2.36, 4.84, 2.37],
            "trans_date_trans_time": [
                "2020-05-15 14:30:00",
                "2020-06-20 22:15:00",
                "2020-05-15 18:45:00",
            ],
            "dob": ["1990-05-15", "1980-01-01", "1990-05-15"],
            "is_fraud": [0, 1, 0],
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
    assert "user_hash" in df_res.columns
    assert "user_daily_tx_count" in df_res.columns
    assert "client_node" in df_res.columns
    assert "merchant_node" in df_res.columns
    assert "fraud_label" in df_res.columns

    # Vérification du calcul d'âge cohérent avec l'année de la transaction (2020)
    assert df_res["age"].iloc[0] == 2020 - 1990  # 30 ans
    assert df_res["age"].iloc[1] == 2020 - 1980  # 40 ans

    # Vérification du cumul séquentiel temporel de la vélocité (Alice : 1ère transaction=1, 2ème transaction le même jour=2)
    assert df_res["user_daily_tx_count"].iloc[0] == 1
    assert df_res["user_daily_tx_count"].iloc[2] == 2
    assert df_res["user_daily_tx_count"].iloc[1] == 1  # Bob

    # Vérification distance non nulle
    assert df_res["distance_achat"].iloc[0] > 0.0


def test_centralized_feature_definitions_and_sampling():
    """Valide les constantes et méthodes de sampling réutilisables du module transverse features."""
    from src.utils.features import (
        get_base_feature_columns,
        get_feature_groups,
        get_feature_labels,
        get_moderate_sampled_data,
        get_moderate_sampled_df,
    )

    base_cols = get_base_feature_columns()
    assert "user_daily_tx_count" in base_cols
    assert "amt" in base_cols
    assert "category" in base_cols

    groups = get_feature_groups()
    assert "Comportement & Vélocité" in groups
    assert "user_daily_tx_count" in groups["Comportement & Vélocité"]

    labels = get_feature_labels()
    assert labels["user_daily_tx_count"] == "Nombre d'achats du jour (Vélocité)"

    # Test sampling modéré
    df_imbalanced = pd.DataFrame(
        {
            "amt": [10.0] * 100 + [500.0] * 5,
            "is_fraud": [0] * 100 + [1] * 5,
        }
    )
    X = df_imbalanced[["amt"]]
    y = df_imbalanced["is_fraud"]

    _X_s, y_s = get_moderate_sampled_data(X, y, target_ratio=0.10)
    # Ratio attendu = 5 / (5 + 45) = 10%
    assert y_s.sum() == 5
    assert len(y_s) == 50

    df_s = get_moderate_sampled_df(
        df_imbalanced, target_ratio=0.10, label_col="is_fraud"
    )
    assert df_s["is_fraud"].sum() == 5
    assert len(df_s) == 50


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

    gate_brier = MLflowQualityGate(metric_target="brier")
    metrics_with_brier = {"f2_class_1": 0.85, "brier_score": 0.0007}
    assert gate_brier.get_target_metric_key(metrics_with_brier) == "brier_score"


@patch("src.utils.mlflow_manager.MlflowClient")
@patch("src.utils.mlflow_manager.mlflow")
def test_mlflow_quality_gate_brier_minimization_promotion(mock_mlflow, mock_client_cls):
    """Vérifie qu'un candidat avec un Brier Score plus FAIBLE (meilleure calibration) est promu."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    # Simulation d'un champion existant avec Brier Score = 0.0020
    mock_champ = MagicMock()
    mock_champ.version = "10"
    mock_champ.run_id = "run_champ_10"
    mock_client.get_model_version_by_alias.return_value = mock_champ

    mock_run = MagicMock()
    mock_run.data.metrics = {"brier_score": 0.0020}
    mock_run.data.params = {"model_type": "XGBClassifier"}
    mock_client.get_run.return_value = mock_run

    # Simulation de la recherche de la version créée (V11)
    mock_v11 = MagicMock()
    mock_v11.version = 11
    mock_client.search_model_versions.return_value = [mock_v11]

    gate = MLflowQualityGate(model_name="fraud_detector", metric_target="brier")
    gate.client = mock_client

    candidate_metrics = {
        "brier_score": 0.0008
    }  # 0.0008 < 0.0020 -> Promotion attendue (minimisation)
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

    candidate_metrics = {
        "f2_class_1": 0.85,
        "prec_class_1": 0.75,
        "rec_class_1": 0.88,
        "brier_score": 0.0015,
    }  # 0.85 > 0.70 et garde-fous validés -> Promotion attendue
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
def test_mlflow_quality_gate_rejection_guardrail_precision(
    mock_mlflow, mock_client_cls
):
    """Vérifie le rejet d'un candidat avec un meilleur F2 mais dont la précision viole le garde-fou (< 50%)."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_champ = MagicMock()
    mock_champ.version = "10"
    mock_champ.run_id = "run_champ_10"
    mock_client.get_model_version_by_alias.return_value = mock_champ

    mock_run = MagicMock()
    mock_run.data.metrics = {"f2_class_1": 0.70}
    mock_run.data.params = {"model_type": "XGBClassifier"}
    mock_client.get_run.return_value = mock_run

    mock_v11 = MagicMock()
    mock_v11.version = 11
    mock_client.search_model_versions.return_value = [mock_v11]

    gate = MLflowQualityGate(
        model_name="fraud_detector", metric_target="f2", min_precision=0.50
    )
    gate.client = mock_client

    # F2 = 0.85 (> 0.70), mais précision = 40% (< 50%)
    candidate_metrics = {
        "f2_class_1": 0.85,
        "prec_class_1": 0.40,
        "rec_class_1": 0.90,
        "brier_score": 0.0020,
    }
    promoted, version = gate.log_and_evaluate(
        model=MagicMock(),
        metrics=candidate_metrics,
    )

    assert promoted is False
    assert version == "11"
    mock_client.set_registered_model_alias.assert_not_called()


@patch("src.utils.mlflow_manager.MlflowClient")
@patch("src.utils.mlflow_manager.mlflow")
def test_mlflow_quality_gate_rejection_guardrail_brier(mock_mlflow, mock_client_cls):
    """Vérifie le rejet d'un candidat avec un meilleur F2 mais dont le Brier Score est trop élevé (> 0.0050)."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_champ = MagicMock()
    mock_champ.version = "10"
    mock_champ.run_id = "run_champ_10"
    mock_client.get_model_version_by_alias.return_value = mock_champ

    mock_run = MagicMock()
    mock_run.data.metrics = {"f2_class_1": 0.70}
    mock_run.data.params = {"model_type": "XGBClassifier"}
    mock_client.get_run.return_value = mock_run

    mock_v11 = MagicMock()
    mock_v11.version = 11
    mock_client.search_model_versions.return_value = [mock_v11]

    gate = MLflowQualityGate(
        model_name="fraud_detector", metric_target="f2", max_brier=0.0050
    )
    gate.client = mock_client

    # F2 = 0.85 (> 0.70), mais Brier = 0.0085 (> 0.0050)
    candidate_metrics = {
        "f2_class_1": 0.85,
        "prec_class_1": 0.75,
        "rec_class_1": 0.88,
        "brier_score": 0.0085,
    }
    promoted, version = gate.log_and_evaluate(
        model=MagicMock(),
        metrics=candidate_metrics,
    )

    assert promoted is False
    assert version == "11"
    mock_client.set_registered_model_alias.assert_not_called()


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
