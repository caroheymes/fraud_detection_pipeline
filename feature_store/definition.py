# cd feature_store
# docker exec -t fraud-detection-airflow-scheduler sh -c "cd  /opt/airflow/project/feature_store && /usr/python/bin/feast apply"
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from feast import Entity, FeatureView, Field, RequestSource, ValueType
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Int64, String

# 1. L'Entité : Notre clé pivot pour rechercher les données (le numéro de carte)
card_entity = Entity(
    name="cc_num",
    value_type=ValueType.STRING,
    description="Numéro de carte de crédit du client",
)

# 2. La Source Offline : Où se trouvent nos données historiques dans Postgres
postgres_source = PostgreSQLSource(
    name="postgres_rawdata",
    query="SELECT cc_num, gender, dob, city_pop, logged_at FROM silver.rawdata",
    timestamp_field="logged_at",
)

# 3. La Feature View (Les colonnes à charger dans Redis pour chaque cc_num)
card_user_view = FeatureView(
    name="card_user_features",
    entities=[card_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="gender", dtype=String),
        Field(name="dob", dtype=String),
        Field(name="city_pop", dtype=Int64),
    ],
    online=True,
    source=postgres_source,
)

# 4. Source des paramètres de la requête temps réel (transmis lors du paiement)
transaction_request_source = RequestSource(
    name="transaction_request",
    schema=[
        Field(name="amt", dtype=Float32),
        Field(name="category", dtype=String),
        Field(name="trans_date_trans_time", dtype=String),
        Field(name="lat", dtype=Float32),
        Field(name="long", dtype=Float32),
        Field(name="merch_lat", dtype=Float32),
        Field(name="merch_long", dtype=Float32),
    ],
)


# Fonction pour calculer la distance
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


# 5. La vue "À la demande" : combine Redis (card_user_view) et la requête (transaction_request_source)
@on_demand_feature_view(
    sources=[card_user_view, transaction_request_source],
    schema=[
        Field(name="category", dtype=String),
        Field(name="amt", dtype=Float32),
        Field(name="gender", dtype=String),
        Field(name="distance_achat", dtype=Float32),
        Field(name="age", dtype=Int64),
        Field(name="city_pop", dtype=Int64),
        Field(name="hour_sin", dtype=Float32),
        Field(name="hour_cos", dtype=Float32),
        Field(name="weekday_sin", dtype=Float32),
        Field(name="weekday_cos", dtype=Float32),
        Field(name="month_sin", dtype=Float32),
        Field(name="month_cos", dtype=Float32),
    ],
)
def get_derived_features(inputs: pd.DataFrame) -> pd.DataFrame:
    # Si le DataFrame d'entrée est vide (cas du Dry Run Feast), on renvoie une ligne fictive bien typée
    if inputs.empty:
        return pd.DataFrame(
            {
                "category": ["shopping_net"],
                "amt": [0.0],
                "gender": ["F"],
                "distance_achat": [0.0],
                "age": [30],
                "city_pop": [10000],
                "hour_sin": [0.0],
                "hour_cos": [0.0],
                "weekday_sin": [0.0],
                "weekday_cos": [0.0],
                "month_sin": [0.0],
                "month_cos": [0.0],
            }
        ).astype(
            {
                "category": "string",
                "amt": np.float32,
                "gender": "string",
                "distance_achat": np.float32,
                "age": np.int64,
                "city_pop": np.int64,
                "hour_sin": np.float32,
                "hour_cos": np.float32,
                "weekday_sin": np.float32,
                "weekday_cos": np.float32,
                "month_sin": np.float32,
                "month_cos": np.float32,
            }
        )

    # Sinon, on procède aux calculs normaux sur les données réelles
    output = pd.DataFrame()

    # 1. Variables directes avec nettoyage et casting explicite
    output["category"] = inputs["category"].fillna("unknown").astype("string")
    output["amt"] = inputs["amt"].fillna(0.0).astype(np.float32)
    output["gender"] = inputs["gender"].fillna("unknown").astype("string")
    output["city_pop"] = inputs["city_pop"].fillna(0).astype(np.int64)

    # 2. Calcul de la distance d'achat
    lat1 = inputs["lat"].fillna(0.0).astype(np.float32)
    lon1 = inputs["long"].fillna(0.0).astype(np.float32)
    lat2 = inputs["merch_lat"].fillna(0.0).astype(np.float32)
    lon2 = inputs["merch_long"].fillna(0.0).astype(np.float32)
    output["distance_achat"] = haversine_vectorized(lat1, lon1, lat2, lon2).astype(
        np.float32
    )

    # 3. Calcul de l'âge
    dob_dt = pd.to_datetime(inputs["dob"], errors="coerce").fillna(
        pd.Timestamp("1980-01-01")
    )
    output["age"] = (datetime.now().year - dob_dt.dt.year).astype(np.int64)

    # 4. Encodages temporels cycliques
    dt_col = pd.to_datetime(inputs["trans_date_trans_time"], errors="coerce").fillna(
        pd.Timestamp("2020-01-01")
    )
    output["hour_sin"] = np.sin(2 * np.pi * dt_col.dt.hour / 24.0).astype(np.float32)
    output["hour_cos"] = np.cos(2 * np.pi * dt_col.dt.hour / 24.0).astype(np.float32)
    output["weekday_sin"] = np.sin(2 * np.pi * dt_col.dt.dayofweek / 7.0).astype(
        np.float32
    )
    output["weekday_cos"] = np.cos(2 * np.pi * dt_col.dt.dayofweek / 7.0).astype(
        np.float32
    )
    output["month_sin"] = np.sin(2 * np.pi * dt_col.dt.month / 12.0).astype(np.float32)
    output["month_cos"] = np.cos(2 * np.pi * dt_col.dt.month / 12.0).astype(np.float32)

    return output
