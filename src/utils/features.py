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
    Gère gracieusement les valeurs manquantes ou formats invalides (NaN/NaT).
    """
    df_out = df.copy()
    if datetime_col in df_out.columns:
        dt_col = pd.to_datetime(df_out[datetime_col], errors="coerce")
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
        h = pd.to_numeric(df_out["hour"], errors="coerce")
        if "hour_sin" not in df_out.columns:
            df_out["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
            df_out["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    return df_out


def compute_demographic_features(
    df: pd.DataFrame, datetime_col: str = "trans_date_trans_time"
) -> pd.DataFrame:
    """
    Calcule les variables démographiques telles que l'âge à partir de la date de naissance (dob).
    Utilise l'année de la transaction si disponible pour éviter le data leakage/dérive temporelle,
    sinon l'année courante. Robuste aux NaNs et formats de dates erronés.
    """
    df_out = df.copy()
    if "age" not in df_out.columns and "dob" in df_out.columns:
        from datetime import datetime

        dob_dt = pd.to_datetime(df_out["dob"], errors="coerce")
        if datetime_col in df_out.columns:
            ref_year = pd.to_datetime(df_out[datetime_col], errors="coerce").dt.year
        else:
            ref_year = datetime.now().year
        df_out["age"] = ref_year - dob_dt.dt.year
    return df_out


def compute_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la distance d'achat client-marchand si les coordonnées sont disponibles.
    Gère les valeurs non numériques ou manquantes de façon sécurisée (NaN).
    """
    df_out = df.copy()
    coords = {"lat", "long", "merch_lat", "merch_long"}
    if coords.issubset(df_out.columns) and "distance_achat" not in df_out.columns:
        lat1 = pd.to_numeric(df_out["lat"], errors="coerce")
        lon1 = pd.to_numeric(df_out["long"], errors="coerce")
        lat2 = pd.to_numeric(df_out["merch_lat"], errors="coerce")
        lon2 = pd.to_numeric(df_out["merch_long"], errors="coerce")
        df_out["distance_achat"] = haversine_vectorized(lat1, lon1, lat2, lon2)
    elif "distance_km" in df_out.columns and "distance_achat" not in df_out.columns:
        df_out["distance_achat"] = pd.to_numeric(df_out["distance_km"], errors="coerce")
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


def compute_user_hash(
    first: str | pd.Series,
    last: str | pd.Series,
    dob: str | pd.Series,
    secret_salt: str | None = None,
) -> str | pd.Series:
    """
    Calcule le hash cryptographique pseudonymisé de l'utilisateur conforme RGPD / PCI-DSS :
    HMAC-SHA256(first_name + last_name + DOB, SECRET_SALT).
    Ne stocke aucune donnée nominative en clair et protège contre les rainbow tables.
    """
    import hashlib
    import hmac
    import os

    salt = (
        secret_salt or os.getenv("USER_HASH_SALT", "fraud_detection_secret_salt_2026")
    ).encode("utf-8")

    if isinstance(first, pd.Series):
        first_s = first.fillna("").astype(str).str.strip().str.lower()
        last_s = last.fillna("").astype(str).str.strip().str.lower()
        dob_s = dob.fillna("").astype(str).str.strip()

        hashes = [
            hmac.new(salt, f"{f}_{l}_{d}".encode(), hashlib.sha256).hexdigest()
            for f, l, d in zip(first_s, last_s, dob_s)
        ]
        return pd.Series(hashes, index=first.index)
    else:
        msg = f"{str(first).strip().lower()}_{str(last).strip().lower()}_{str(dob).strip()}".encode()
        return hmac.new(salt, msg, hashlib.sha256).hexdigest()


def compute_user_velocity_features(
    df: pd.DataFrame,
    datetime_col: str = "trans_date_trans_time",
    secret_salt: str | None = None,
) -> pd.DataFrame:
    """
    Calcule la vélocité des transactions par utilisateur (nombre d'achats journaliers cumulés).
    Utilise un compteur séquentiel temporel (cumcount + 1) pour éviter tout data leakage lors de l'entraînement.
    """
    import hashlib
    import hmac
    import os

    df_out = df.copy()

    # 1. Vérification / Calcul du user_hash
    if "user_hash" not in df_out.columns:
        if {"first", "last", "dob"}.issubset(df_out.columns):
            df_out["user_hash"] = compute_user_hash(
                df_out["first"], df_out["last"], df_out["dob"], secret_salt=secret_salt
            )
        elif "full_name" in df_out.columns and "dob" in df_out.columns:
            salt = (
                secret_salt
                or os.getenv("USER_HASH_SALT", "fraud_detection_secret_salt_2026")
            ).encode("utf-8")
            df_out["user_hash"] = [
                hmac.new(
                    salt,
                    f"{str(fn).strip().lower()}_{str(d).strip()}".encode(),
                    hashlib.sha256,
                ).hexdigest()
                for fn, d in zip(df_out["full_name"], df_out["dob"])
            ]
        elif "cc_num" in df_out.columns:
            salt = (
                secret_salt
                or os.getenv("USER_HASH_SALT", "fraud_detection_secret_salt_2026")
            ).encode("utf-8")
            df_out["user_hash"] = [
                hmac.new(salt, str(c).encode("utf-8"), hashlib.sha256).hexdigest()
                for c in df_out["cc_num"]
            ]
        else:
            df_out["user_daily_tx_count"] = 1
            return df_out

    # 2. Calcul du compteur cumulé temporel
    if datetime_col in df_out.columns and "user_daily_tx_count" not in df_out.columns:
        df_out[datetime_col] = pd.to_datetime(df_out[datetime_col])
        df_out["_tx_date_temp"] = df_out[datetime_col].dt.date

        # Ordonnancement strict dans le temps pour reproduire le flux réel d'inférence
        df_out["_orig_seq_pos"] = np.arange(len(df_out))
        df_sorted = df_out.sort_values(by=datetime_col)
        df_sorted["user_daily_tx_count"] = (
            df_sorted.groupby(["user_hash", "_tx_date_temp"]).cumcount() + 1
        )
        df_out = (
            df_sorted.sort_values(by="_orig_seq_pos")
            .drop(columns=["_tx_date_temp", "_orig_seq_pos"])
        )
    elif "user_daily_tx_count" not in df_out.columns:
        df_out["user_daily_tx_count"] = 1

    return df_out


# Standardized Feature Definitions & Schemas across all models and serving
BASE_FEATURE_COLUMNS: list[str] = [
    "category",
    "amt",
    "gender",
    "distance_achat",
    "age",
    "city_pop",
    "user_daily_tx_count",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
]

CATEGORICAL_FEATURE_COLUMNS: list[str] = [
    "category",
    "gender",
]

NUMERICAL_FEATURE_COLUMNS: list[str] = [
    "amt",
    "distance_achat",
    "age",
    "city_pop",
    "user_daily_tx_count",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "Heure": ["hour_sin", "hour_cos"],
    "Jour de la semaine": ["weekday_sin", "weekday_cos"],
    "Mois de l'année": ["month_sin", "month_cos"],
    "Comportement & Vélocité": ["user_daily_tx_count"],
}

FEATURE_LABELS: dict[str, str] = {
    "amt": "Montant (€)",
    "distance_achat": "Distance d'achat (km)",
    "age": "Âge du client",
    "city_pop": "Population de la ville",
    "category": "Catégorie d'achat",
    "gender": "Genre",
    "user_daily_tx_count": "Nombre d'achats du jour (Vélocité)",
}

DRIFT_FEATURE_COLUMNS: list[str] = [
    "amt",
    "gender",
    "is_fraud",
    "hour_sin",
    "hour_cos",
    "distance_achat",
    "user_daily_tx_count",
]


def get_base_feature_columns() -> list[str]:
    """Retourne la liste standard des variables explicatives pour tous les modèles."""
    return list(BASE_FEATURE_COLUMNS)


def get_categorical_feature_columns() -> list[str]:
    """Retourne la liste des colonnes catégorielles."""
    return list(CATEGORICAL_FEATURE_COLUMNS)


def get_numerical_feature_columns() -> list[str]:
    """Retourne la liste des colonnes numériques continues."""
    return list(NUMERICAL_FEATURE_COLUMNS)


def get_feature_groups() -> dict[str, list[str]]:
    """Retourne les regroupements de variables pour l'explicabilité Shapash / SHAP."""
    return {k: list(v) for k, v in FEATURE_GROUPS.items()}


def get_feature_labels() -> dict[str, str]:
    """Retourne le dictionnaire des libellés métier en français pour les dashboards et graphiques."""
    return dict(FEATURE_LABELS)


def get_drift_feature_columns() -> list[str]:
    """Retourne la liste standard des colonnes surveillées pour la détection de dérive (Evidently)."""
    return list(DRIFT_FEATURE_COLUMNS)


def get_moderate_sampled_data(
    X_train_df: pd.DataFrame,
    y_train_series: pd.Series,
    target_ratio: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Rééchantillonne modérément le jeu d'entraînement pour obtenir le ratio cible de fraudes.
    Si target_ratio <= 0.0 ou >= 1.0, retourne les données inchangées.
    """
    if target_ratio <= 0.0 or target_ratio >= 1.0:
        return X_train_df, y_train_series

    label_name = y_train_series.name or "is_fraud"
    train_df = pd.concat([X_train_df, y_train_series], axis=1)
    fraud = train_df[train_df[label_name] == 1]
    normal = train_df[train_df[label_name] == 0]

    n_fraud = len(fraud)
    n_normal_required = int(n_fraud * (1.0 / target_ratio - 1.0))

    if n_normal_required < len(normal):
        normal_sampled = normal.sample(n=n_normal_required, random_state=random_state)
    else:
        normal_sampled = normal

    sampled_df = (
        pd.concat([fraud, normal_sampled])
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    return sampled_df.drop(columns=[label_name]), sampled_df[label_name]


def get_moderate_sampled_df(
    df: pd.DataFrame,
    target_ratio: float = 0.05,
    label_col: str = "fraud_label",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Rééchantillonne un DataFrame complet pour atteindre le ratio cible de transactions positives.
    """
    if target_ratio <= 0.0 or target_ratio >= 1.0:
        return df

    target_name = (
        label_col
        if label_col in df.columns
        else ("is_fraud" if "is_fraud" in df.columns else None)
    )
    if not target_name:
        return df

    fraud = df[df[target_name] == 1]
    normal = df[df[target_name] == 0]

    n_fraud = len(fraud)
    n_normal_required = int(n_fraud * (1.0 / target_ratio - 1.0))

    if n_normal_required < len(normal):
        normal_sampled = normal.sample(n=n_normal_required, random_state=random_state)
    else:
        normal_sampled = normal

    return (
        pd.concat([fraud, normal_sampled])
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


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
    df_ready = compute_user_velocity_features(df_ready, datetime_col=datetime_col)
    if include_graph_ids:
        df_ready = prepare_graph_identifiers(df_ready)
    return df_ready
