# src/training/preprocessing.py

import os
from datetime import datetime

import numpy as np
import pandas as pd
from skrub import TableVectorizer


# --- 1. DÉFINITION DE LA DISTANCE HAVERSINE VECTORISÉE ---
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0  # Rayon de la Terre en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


# --- 2. FONCTION PRINCIPALE DE CHARGEMENT ET D'INGÉNIERIE ---
def load_and_preprocess_data(csv_path, sample_size=None, random_state=42):
    """
    Charge le dataset fraudTest.csv, effectue l'ingénierie des features (cyclique, distance, âge),
    et filtre les valeurs aberrantes (outliers) sur le montant.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Le fichier dataset est introuvable à l'emplacement : {csv_path}"
        )

    print(f"Chargement du dataset : {csv_path}...")
    df = pd.read_csv(csv_path)

    # Optionnel : échantillonnage pour les modèles lourds (GNN)
    if sample_size is not None:
        print(
            f"Échantillonnage de {sample_size} lignes pour l'évaluation comparative..."
        )
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    print("Calcul des caractéristiques temporelles (sin/cos de l'heure, jour, mois)...")
    dt_col = pd.to_datetime(df["trans_date_trans_time"])
    df["hour_sin"] = np.sin(2 * np.pi * dt_col.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * dt_col.dt.hour / 24.0)
    df["weekday_sin"] = np.sin(2 * np.pi * dt_col.dt.dayofweek / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * dt_col.dt.dayofweek / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * dt_col.dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * dt_col.dt.month / 12.0)

    print("Calcul de la distance Haversine entre l'acheteur et le commerçant...")
    df["distance_achat"] = haversine_vectorized(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    print("Calcul de l'âge du client...")
    dob_col = pd.to_datetime(df["dob"])
    df["age"] = datetime.now().year - dob_col.dt.year

    print("Filtrage des outliers sur le montant (amt > mean - 3*std)...")
    initial_shape = df.shape[0]
    df = df[df.amt > df.amt.mean() - 3 * df.amt.std()].reset_index(drop=True)
    filtered_shape = df.shape[0]
    print(
        f"Lignes filtrées : {initial_shape - filtered_shape} supprimées | {filtered_shape} restantes."
    )

    # Définition des variables explicatives X et de la cible y
    feature_cols = [
        "category",
        "amt",
        "gender",
        "distance_achat",
        "age",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
    ]

    X = df[feature_cols]
    y = df["is_fraud"].copy()

    return df, X, y


# --- 3. VECTORISATION DES CARACTÉRISTIQUES CATÉGORIELLES ---
def vectorize_features(X_train, X_test):
    """
    Vectorise les variables à l'aide de skrub.TableVectorizer.
    """
    print("Vectorisation des variables avec skrub.TableVectorizer...")
    vectorizer = TableVectorizer()
    X_train_encoded = vectorizer.fit_transform(X_train)
    X_test_encoded = vectorizer.transform(X_test)

    # Conversion en DataFrames avec les noms des caractéristiques pour faciliter la manipulation
    feature_names = vectorizer.get_feature_names_out()
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=feature_names)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=feature_names)

    return X_train_encoded_df, X_test_encoded_df, vectorizer


# --- 4. DEGRÉ D'ADJACENCE DU GRAPHE POUR LES MODÈLES DE GRAPHE ---
def build_adjacency_matrix(df):
    """
    Construit et normalise la matrice d'adjacence basée sur cc_num (transactions partagées).
    """
    print("Construction de la matrice d'adjacence du graphe bancaire...")
    num_nodes = len(df)
    cc_nums = df["cc_num"].values

    adj = np.eye(num_nodes)
    for val in np.unique(cc_nums):
        idx = np.where(cc_nums == val)[0]
        if len(idx) > 1:
            for i in idx:
                for j in idx:
                    adj[i, j] = 1.0

    # Normalisation D^-1 * A (moyenne des voisins)
    rowsum = adj.sum(axis=1)
    d_inv = np.power(rowsum, -1.0, where=rowsum > 0)
    d_inv[rowsum == 0] = 0.0
    adj_norm = np.diag(d_inv).dot(adj)

    return adj_norm
