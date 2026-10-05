# src/utils/__init__.py
"""
Package d'utilitaires MLOps transverses :
- features: Feature engineering unifié (Haversine, cyclical time sin/cos, graph nodes)
- mlflow_manager: MLflow Quality Gate & Model Registry
- api_reloader: Hot reload de l'API de serving
- db: Connecteurs PostgreSQL et Redis
"""

from src.utils.api_reloader import reload_serving_api
from src.utils.data_loader import load_dataset
from src.utils.db import get_postgres_engine, get_redis_client
from src.utils.features import (
    compute_cyclical_time_features,
    compute_demographic_features,
    compute_spatial_features,
    get_drift_feature_columns,
    haversine_vectorized,
    prepare_features,
    prepare_graph_identifiers,
)
from src.utils.mlflow_manager import MLflowQualityGate, load_champion_model
from src.utils.threshold import evaluate_predictions_and_curves, find_optimal_threshold

__all__ = [
    "MLflowQualityGate",
    "compute_cyclical_time_features",
    "compute_demographic_features",
    "compute_spatial_features",
    "evaluate_predictions_and_curves",
    "find_optimal_threshold",
    "get_drift_feature_columns",
    "get_postgres_engine",
    "get_redis_client",
    "haversine_vectorized",
    "load_champion_model",
    "load_dataset",
    "prepare_features",
    "prepare_graph_identifiers",
    "reload_serving_api",
]
