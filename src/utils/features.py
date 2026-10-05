# src/utils/features.py
"""
Module universel de Feature Engineering pour le Train et le Serving (Zero Train-Serving Skew).
Centralise le calcul des distances spatiales, composantes temporelles cycliques et standardisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_vectorized(
    lat1: pd.Series | np.ndarray | float,
    lon1: pd.Series | np.ndarray | float,
    lat2: pd.Series | np.ndarray | float,
    lon2: pd.Series | np.ndarray | float,
) -> pd.Series | np.ndarray | float:
    """
    Calcule la distance Haversine en kilomètres entre deux points géographiques (latitude/longitude).
    Fonctionne de manière vectorisée avec Pandas Series et NumPy Arrays.
    """
    R = 6371.0  # Rayon de la Terre en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def compute_cyclical_time_features(
    df: pd.DataFrame, datetime_col: str = "trans_date_trans_time"
) -> pd.DataFrame:
    """
    Extrait et encode les caractéristiques temporelles en composantes cycliques (sin/cos).
    """
    df_out = df.copy()
    if datetime_col in df_out.columns:
        dt_col = pd.to_datetime(df_out[datetime_col])
        df_out[datetime_col] = dt_col
        df_out["hour"] = dt_col.dt.hour
        df_out["day"] = dt_col.dt.dayofweek
        if "hour_sin" not in df_out.columns:
            df_out["hour_sin"] = np.sin(2 * np.pi * dt_col.dt.hour / 24.0)
            df_out["hour_cos"] = np.cos(2 * np.pi * dt_col.dt.hour / 24.0)
        if "weekday_sin" not in df_out.columns:
            df_out["weekday_sin"] = np.sin(2 * np.pi * dt_col.dt.dayofweek / 7.0)
            df_out["weekday_cos"] = np.cos(2 * np.pi * dt_col.dt.dayofweek / 7.0)
        if "month_sin" not in df_out.columns:
            df_out["month_sin"] = np.sin(2 * np.pi * dt_col.dt.month / 12.0)
            df_out["month_cos"] = np.cos(2 * np.pi * dt_col.dt.month / 12.0)
    elif "hour" in df_out.columns:
        if "hour_sin" not in df_out.columns:
            df_out["hour_sin"] = np.sin(2 * np.pi * df_out["hour"] / 24.0)
            df_out["hour_cos"] = np.cos(2 * np.pi * df_out["hour"] / 24.0)
    return df_out


def compute_demographic_features(
    df: pd.DataFrame, datetime_col: str = "trans_date_trans_time"
) -> pd.DataFrame:
    """
    Calcule les variables démographiques telles que l'âge à partir de la date de naissance (dob).
    Utilise l'année de la transaction si disponible pour éviter le data leakage/dérive temporelle,
    sinon l'année courante.
    """
    df_out = df.copy()
    if "age" not in df_out.columns and "dob" in df_out.columns:
        from datetime import datetime

        dob_dt = pd.to_datetime(df_out["dob"])
        if datetime_col in df_out.columns:
            ref_year = pd.to_datetime(df_out[datetime_col]).dt.year
        else:
            ref_year = datetime.now().year
        df_out["age"] = ref_year - dob_dt.dt.year
    return df_out


def compute_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la distance d'achat client-marchand si les coordonnées sont disponibles.
    """
    df_out = df.copy()
    coords = {"lat", "long", "merch_lat", "merch_long"}
    if coords.issubset(df_out.columns) and "distance_achat" not in df_out.columns:
        df_out["distance_achat"] = haversine_vectorized(
            df_out["lat"].astype(float),
            df_out["long"].astype(float),
            df_out["merch_lat"].astype(float),
            df_out["merch_long"].astype(float),
        )
    elif "distance_km" in df_out.columns and "distance_achat" not in df_out.columns:
        df_out["distance_achat"] = df_out["distance_km"]
    return df_out


def prepare_graph_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise les identifiants de nœuds client, marchand et labels pour les modèles de graphe (GNN).
    """
    df_out = df.copy()
    if "client_node" not in df_out.columns:
        if "cc_num" in df_out.columns:
            df_out["client_node"] = df_out["cc_num"].astype(str)
        else:
            df_out["client_node"] = [f"c_{i}" for i in range(len(df_out))]

    if "merchant_node" not in df_out.columns:
        if "merchant" in df_out.columns:
            df_out["merchant_node"] = df_out["merchant"].astype(str)
        else:
            df_out["merchant_node"] = [f"m_{i % 100}" for i in range(len(df_out))]

    if "fraud_label" not in df_out.columns:
        if "is_fraud" in df_out.columns:
            df_out["fraud_label"] = df_out["is_fraud"].astype(int)
        else:
            df_out["fraud_label"] = 0

    return df_out


def prepare_features(
    df: pd.DataFrame,
    include_graph_ids: bool = True,
    datetime_col: str = "trans_date_trans_time",
) -> pd.DataFrame:
    """
    Pipeline unifié complet de préparation des données pour entraînement et inférence.
    """
    df_ready = compute_cyclical_time_features(df, datetime_col=datetime_col)
    df_ready = compute_spatial_features(df_ready)
    df_ready = compute_demographic_features(df_ready, datetime_col=datetime_col)
    if include_graph_ids:
        df_ready = prepare_graph_identifiers(df_ready)
    return df_ready


def get_drift_feature_columns() -> list[str]:
    """Retourne la liste standard des colonnes surveillées pour la détection de dérive (Evidently)."""
    return ["amt", "gender", "is_fraud", "hour_sin", "hour_cos", "distance_achat"]
