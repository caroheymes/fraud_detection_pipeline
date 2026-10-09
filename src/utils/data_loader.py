# src/utils/data_loader.py
"""
Chargeur de données universel pour l'entraînement et l'évaluation des modèles de fraude.
Gère l'historique complet depuis le début (2020-06-21) jusqu'à la date maximale atteinte
dans PostgreSQL (silver.rawdata), l'échantillonnage chronologique ou aléatoire,
le feature engineering unifié et le filtrage des outliers.
"""

from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import pandas as pd
from sqlalchemy import text

from src.utils.db import get_postgres_engine
from src.utils.features import prepare_features


def load_dataset(
    sample_size: int = -1,
    sample_position: str = "last",
    include_graph_ids: bool = False,
    filter_outliers: bool = False,
    csv_fallback_path: str | None = None,
) -> pd.DataFrame:
    """
    Charge et prépare l'ensemble du dataset transactionnel disponible :
    Depuis le début de l'historique (2020-06-21) jusqu'à la date max de PostgreSQL (silver.rawdata).

    Arguments:
        sample_size: Nombre de lignes à échantillonner (-1 = tout le dataset disponible).
        sample_position: Mode d'échantillonnage si sample_size > 0 :
            - "last"   : Les N transactions les plus récentes (défaut).
            - "first"  : Les N transactions les plus anciennes.
            - "random" : Échantillon aléatoire uniforme reproductible (seed=42).
        include_graph_ids: Si True, génère et inclut client_node, merchant_node, fraud_label.
        filter_outliers: Conservé pour rétrocompatibilité (aucun filtrage n'est appliqué pour préserver 100% des fraudes).
        csv_fallback_path: Chemin alternatif pour le CSV de base si non standard.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        csv_fallback_path,
        os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv")),
        os.path.abspath(os.path.join(script_dir, "../training/reference_data.csv")),
        os.path.abspath("fraudTest.csv"),
        os.path.abspath("data/fraudTest.csv"),
    ]
    csv_path = None
    for p in candidates:
        if p and os.path.exists(p):
            csv_path = p
            break

    df_csv = None
    if csv_path:
        df_csv = pd.read_csv(csv_path)
        if "trans_date_trans_time" in df_csv.columns:
            df_csv["trans_date_trans_time"] = pd.to_datetime(
                df_csv["trans_date_trans_time"]
            )

    # 1. Vérification de la date maximale atteinte dans PostgreSQL (silver.rawdata)
    max_db_date = None
    df_pg = None
    try:
        engine = get_postgres_engine()
        with engine.connect() as conn:
            max_db_val = conn.execute(
                text("SELECT MAX(trans_date_trans_time) FROM silver.rawdata")
            ).scalar()
            if max_db_val:
                t = pd.to_datetime(max_db_val)
                if t.tzinfo is not None:
                    t = t.tz_localize(None)
                max_db_date = t
                df_pg = pd.read_sql("SELECT * FROM silver.rawdata", engine)
                if "trans_date_trans_time" in df_pg.columns:
                    df_pg["trans_date_trans_time"] = pd.to_datetime(
                        df_pg["trans_date_trans_time"]
                    )
                    if df_pg["trans_date_trans_time"].dt.tz is not None:
                        df_pg["trans_date_trans_time"] = df_pg[
                            "trans_date_trans_time"
                        ].dt.tz_localize(None)
                print(
                    f" PostgreSQL connecté : {len(df_pg):,} transactions récentes (Date max: {max_db_date})."
                )
    except Exception as e:
        print(f" PostgreSQL non accessible ({e}), utilisation du fichier CSV seul.")

    # 2. Fusion de l'historique de base et des données streaming
    if df_csv is not None and df_pg is not None and max_db_date is not None:
        # Données synchronisées avec la date max de PostgreSQL
        df_csv_filtered = df_csv[df_csv["trans_date_trans_time"] <= max_db_date]
        df = pd.concat([df_csv_filtered, df_pg], ignore_index=True)
        if "trans_num" in df.columns:
            df = df.drop_duplicates(subset=["trans_num"], keep="last")
        else:
            df = df.drop_duplicates()

        # Si l'utilisateur demande un volume supérieur au périmètre PostgreSQL (ex: sample_size=500000)
        if sample_size > len(df) and len(df_csv) > len(df):
            print(
                f" Volume demandé ({sample_size:,} lignes) supérieur au périmètre PostgreSQL ({len(df):,}). "
                f"Extension automatique sur l'historique complet fraudTest.csv ({len(df_csv):,} lignes disponibles).",
                flush=True,
            )
            df = pd.concat([df_csv, df_pg], ignore_index=True)
            if "trans_num" in df.columns:
                df = df.drop_duplicates(subset=["trans_num"], keep="last")
            else:
                df = df.drop_duplicates()
        else:
            print(
                f" Fusion synchronisée : Historique ({df_csv['trans_date_trans_time'].min().date()} -> {max_db_date.date()}) = {len(df):,} transactions disponibles.",
                flush=True,
            )
        del df_csv_filtered
        del df_pg
        del df_csv
    elif df_pg is not None and len(df_pg) > 0:
        df = df_pg
    elif df_csv is not None:
        df = df_csv
    else:
        raise FileNotFoundError(
            " Impossible de charger les données : ni PostgreSQL ni fraudTest.csv ne sont accessibles."
        )

    # 3. Tri chronologique et échantillonnage
    import gc

    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        df.sort_values("trans_date_trans_time", inplace=True)
        df.reset_index(drop=True, inplace=True)

    # 4. Échantillonnage chronologique ou aléatoire
    if 0 < sample_size < len(df):
        if sample_position == "last":
            print(
                f" Échantillonnage chronologique : {sample_size:,} DERNIÈRES transactions (les plus récentes)...",
                flush=True,
            )
            df = df.iloc[-sample_size:].reset_index(drop=True)
        elif sample_position == "first":
            print(
                f" Échantillonnage chronologique : {sample_size:,} PREMIÈRES transactions (les plus anciennes)...",
                flush=True,
            )
            df = df.iloc[:sample_size].reset_index(drop=True)
        else:
            print(
                f" Échantillonnage aléatoire : {sample_size:,} transactions (seed=42)...",
                flush=True,
            )
            df = (
                df.sample(n=sample_size, random_state=42)
                .sort_values("trans_date_trans_time")
                .reset_index(drop=True)
            )

    gc.collect()

    # 5. Feature Engineering unifié sur l'échantillon ciblé
    df = prepare_features(df, include_graph_ids=include_graph_ids)

    # 6. Tri chronologique final
    if "trans_date_trans_time" in df.columns:
        df.sort_values("trans_date_trans_time", inplace=True)
        df.reset_index(drop=True, inplace=True)

    fraud_col = "fraud_label" if "fraud_label" in df.columns else "is_fraud"
    fraud_count = df[fraud_col].sum() if fraud_col in df.columns else 0
    fraud_pct = (df[fraud_col].mean() * 100) if fraud_col in df.columns else 0.0

    date_range_info = ""
    if "trans_date_trans_time" in df.columns and len(df) > 0:
        d_min = str(df["trans_date_trans_time"].min())[:10]
        d_max = str(df["trans_date_trans_time"].max())[:10]
        date_range_info = f" [du {d_min} au {d_max}]"

    print(
        f" Données prêtes : {df.shape}{date_range_info} (dont {fraud_count:,} fraudes, {fraud_pct:.3f}%)",
        flush=True,
    )
    return df
