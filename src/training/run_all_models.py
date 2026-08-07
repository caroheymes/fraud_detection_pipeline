# src/training/run_all_models.py

import json
import os
import sys
from datetime import datetime

import mlflow
import mlflow.pytorch
import mlflow.xgboost
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from skrub import TableVectorizer
from torch import nn
from xgboost import XGBClassifier


# --- 1. DÉFINITION DE LA DISTANCE HAVERSINE VECTORISÉE ---
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0  # Rayon de la Terre en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


# --- 2. DÉFINITION DES MODÈLES GNN (GCN et GraphSAGE) ---
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj_norm):
        h = self.linear(x)
        return torch.mm(adj_norm, h)


class FraudGCN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_classes=2):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_classes)

    def forward(self, x, adj_norm):
        h = F.relu(self.gcn1(x, adj_norm))
        logits = self.gcn2(h, adj_norm)
        return logits


class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x, adj_norm):
        # adj_norm correspond à la moyenne des voisins D^-1 * A
        neighbor_agg = torch.mm(adj_norm, x)
        combined = torch.cat([x, neighbor_agg], dim=1)
        return self.linear(combined)


class GraphSAGEModel(nn.Module):
    def __init__(self, in_features, hidden_dim, embedding_dim):
        super().__init__()
        self.sage1 = GraphSAGELayer(in_features, hidden_dim)
        self.sage2 = GraphSAGELayer(hidden_dim, embedding_dim)

    def forward(self, x, adj):
        h = F.relu(self.sage1(x, adj))
        embeddings = self.sage2(h, adj)
        return embeddings


# --- 3. FONCTION D'ÉCHANTILLONNAGE MODÉRÉ (SOUS-ÉCHANTILLONNAGE MAJORITAIRE) ---
def get_moderate_sampled_data(X_train_df, y_train_series, target_ratio=0.05):
    """
    Échantillonnage modéré pour que la classe fraude (1) représente exactement target_ratio.
    """
    train_df = pd.concat([X_train_df, y_train_series], axis=1)
    fraud = train_df[train_df["is_fraud"] == 1]
    normal = train_df[train_df["is_fraud"] == 0]

    n_fraud = len(fraud)
    # Calcul du nombre requis de cas sains
    n_normal_required = int(n_fraud * (1.0 / target_ratio - 1.0))

    if n_normal_required < len(normal):
        normal_sampled = normal.sample(n=n_normal_required, random_state=42)
    else:
        normal_sampled = normal

    sampled_df = pd.concat([fraud, normal_sampled]).sample(frac=1.0, random_state=42)
    return sampled_df.drop(columns=["is_fraud"]), sampled_df["is_fraud"]


def main():
    print(
        "--- DÉMARRAGE DE LA CAMPAGNE D'ENTRAÎNEMENT DES MODÈLES (5% & 10% SAMPLING) ---"
    )

    # Configuration du tracking MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("Default")

    # Chargement du fichier CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, "../../fraudTest.csv"))

    if not os.path.exists(csv_path):
        print(f"Erreur : Fichier {csv_path} introuvable.")
        sys.exit(1)

    print("Chargement de fraudTest.csv...")
    full_df = pd.read_csv(csv_path)

    # Pour le GNN et pour éviter l'explosion mémoire/temps sur CPU, nous prenons un échantillon de 5 000 transactions
    # (ce qui évite la saturation RAM de la matrice d'adjacence tout en restant représentatif)
    print("Échantillonnage de 5 000 lignes pour l'évaluation comparative...")
    df = full_df.sample(n=5000, random_state=42).reset_index(drop=True)
    print(f"Dimensions réduites : {df.shape}")

    # 4. Feature Engineering vectorisé
    print("Ingénierie des caractéristiques temporelles, géographiques et d'âge...")
    dt_col = pd.to_datetime(df["trans_date_trans_time"])
    df["hour_sin"] = np.sin(2 * np.pi * dt_col.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * dt_col.dt.hour / 24.0)
    df["weekday_sin"] = np.sin(2 * np.pi * dt_col.dt.dayofweek / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * dt_col.dt.dayofweek / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * dt_col.dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * dt_col.dt.month / 12.0)

    df["distance_achat"] = haversine_vectorized(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    dob_col = pd.to_datetime(df["dob"])
    df["age"] = datetime.now().year - dob_col.dt.year

    # Outliers
    df = df[df.amt > df.amt.mean() - 3 * df.amt.std()]
    print(f"Dimensions après outliers : {df.shape}")

    # 5. Construction du Graphe de transactions basé sur cc_num (même carte bleue)
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

    # Normalisation degré D^-1 * A (moyenne des voisins)
    rowsum = adj.sum(axis=1)
    d_inv = np.power(rowsum, -1.0, where=rowsum > 0)
    d_inv[rowsum == 0] = 0.0
    adj_norm = np.diag(d_inv).dot(adj)

    # 6. Sélection des variables brutes et vectorisation avec skrub TableVectorizer
    X = df[
        [
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
    ]
    y = df["is_fraud"].reset_index(drop=True)

    print("Vectorisation des colonnes avec skrub.TableVectorizer...")
    vectorizer = TableVectorizer()
    X_encoded = vectorizer.fit_transform(X)
    X_encoded_df = pd.DataFrame(
        X_encoded, columns=vectorizer.get_feature_names_out()
    ).reset_index(drop=True)

    # Division Train / Test sur les indices pour pouvoir aligner la matrice d'adjacence
    indices = np.arange(num_nodes)
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, y, test_size=0.3, random_state=42, stratify=y
    )

    X_train_raw = X_encoded_df.iloc[train_idx].reset_index(drop=True)
    X_test_raw = X_encoded_df.iloc[test_idx].reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # 7. Préparation des Graph Embeddings (GraphSAGE) sur le train et test
    print("Calcul des embeddings GraphSAGE...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Variables sous forme de tenseurs
    X_tensor = torch.FloatTensor(X_encoded.values).to(device)
    adj_tensor = torch.FloatTensor(adj_norm).to(device)

    # Entraînement rapide de GraphSAGE non-supervisé ou auto-encodeur
    # Ici, nous utilisons un modèle GraphSAGE simple à 2 couches pour générer des représentations
    sage_model = GraphSAGEModel(
        in_features=X_encoded.shape[1], hidden_dim=16, embedding_dim=8
    ).to(device)
    sage_model.eval()
    with torch.no_grad():
        X_embeddings_tensor = sage_model(X_tensor, adj_tensor)
        X_embeddings = X_embeddings_tensor.cpu().numpy()

    # Alignement des embeddings avec train / test
    X_train_emb = X_embeddings[train_idx]
    X_test_emb = X_embeddings[test_idx]

    X_train_combined = np.hstack([X_train_raw.values, X_train_emb])
    X_test_combined = np.hstack([X_test_raw.values, X_test_emb])

    # 8. Grille de tests : Modèles et Ratios d'échantillonnage
    ratios = [0.05, 0.10]

    # Dictionnaire de stockage de toutes les métriques pour export JSON final
    all_experiments_results = {}

    for ratio in ratios:
        ratio_pct = int(ratio * 100)
        print(
            f"\n================ RATIO ÉCHANTILLONNAGE : {ratio_pct}% ================"
        )

        # Sous-échantillonnage modéré sur les données d'entraînement
        X_train_sampled, y_train_sampled = get_moderate_sampled_data(
            X_train_raw, y_train, target_ratio=ratio
        )

        # Pour le modèle combiné GraphSAGE + XGBoost, on fait le sampling également
        X_train_comb_df = pd.DataFrame(X_train_combined)
        X_train_comb_sampled, y_train_comb_sampled = get_moderate_sampled_data(
            X_train_comb_df, y_train, target_ratio=ratio
        )

        # --- MODÈLE 1 : XGBOOST ---
        model_name = f"XGBoost_Sampling_{ratio_pct}pc"
        print(f"Entraînement de {model_name}...")
        clf_xgb = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        clf_xgb.fit(X_train_sampled, y_train_sampled)
        preds_xgb = clf_xgb.predict(X_test_raw)

        metrics_xgb = calculate_and_log_metrics(y_test, preds_xgb, model_name, ratio)
        all_experiments_results[model_name] = metrics_xgb

        # --- MODÈLE 2 : HISTGRADIENTBOOSTING ---
        model_name = f"HistGradientBoosting_Sampling_{ratio_pct}pc"
        print(f"Entraînement de {model_name}...")
        clf_hgb = HistGradientBoostingClassifier(
            max_iter=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        clf_hgb.fit(X_train_sampled, y_train_sampled)
        preds_hgb = clf_hgb.predict(X_test_raw)

        metrics_hgb = calculate_and_log_metrics(y_test, preds_hgb, model_name, ratio)
        all_experiments_results[model_name] = metrics_hgb

        # --- MODÈLE 3 : GNN (GCN) ---
        model_name = f"GNN_GCN_Sampling_{ratio_pct}pc"
        print(f"Entraînement de {model_name}...")
        # Pour le GCN, on entraîne sur les indices d'entraînement échantillonnés
        sampled_train_indices = (
            X_train_sampled.index.values
        )  # indices locaux échantillonnés

        # Pour simplifier, nous créons un sous-graphe ou nous masquons les pertes lors de l'entraînement
        gcn_model = FraudGCN(
            in_features=X_encoded.shape[1], hidden_dim=16, out_classes=2
        ).to(device)
        optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Entraînement rapide sur 50 époques
        sampled_train_nodes = train_idx[X_train_sampled.index.values]
        y_train_sampled_tensor = torch.LongTensor(y_train_sampled.values).to(device)

        gcn_model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            out = gcn_model(X_tensor, adj_tensor)
            # Calcul de la perte uniquement sur les nœuds d'entraînement échantillonnés
            loss = criterion(out[sampled_train_nodes], y_train_sampled_tensor)
            loss.backward()
            optimizer.step()

        gcn_model.eval()
        with torch.no_grad():
            out_logits = gcn_model(X_tensor, adj_tensor)
            _, preds_gcn_all = torch.max(out_logits, dim=1)
            preds_gcn_test = preds_gcn_all[test_idx].cpu().numpy()

        metrics_gcn = calculate_and_log_metrics(
            y_test, preds_gcn_test, model_name, ratio
        )
        all_experiments_results[model_name] = metrics_gcn

        # --- MODÈLE 4 : NVIDIA HYBRIDE (GraphSAGE + XGBoost) ---
        model_name = f"NVIDIA_GraphSAGE_XGBoost_Sampling_{ratio_pct}pc"
        print(f"Entraînement de {model_name}...")
        clf_hybrid = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        clf_hybrid.fit(X_train_comb_sampled, y_train_comb_sampled)
        preds_hybrid = clf_hybrid.predict(X_test_combined)

        metrics_hybrid = calculate_and_log_metrics(
            y_test, preds_hybrid, model_name, ratio
        )
        all_experiments_results[model_name] = metrics_hybrid

    # 9. Sauvegarde locale des résultats au format JSON
    metrics_json_path = os.path.join(script_dir, "metrics_all_models.json")
    with open(metrics_json_path, "w") as f:
        json.dump(all_experiments_results, f, indent=4)
    print(
        f"\nToutes les métriques ont été exportées avec succès dans : {metrics_json_path}"
    )

    # 10. Lancement de la mise à jour des métadonnées MLflow pour la page d'accueil
    print("\nMise à jour automatique de la page des expériences MLflow...")
    from update_experiment_metadata import main as update_metadata

    update_metadata()


def calculate_and_log_metrics(y_true, y_pred, model_name, target_ratio):
    """
    Calcule toutes les métriques de classification pour la classe 1 et globales,
    génère la matrice de confusion et logue le tout dans MLflow.
    """
    prec_c1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_c1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_c1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_glob = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rec_glob = recall_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    # Calcul de la matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    confusion_dict = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    metrics = {
        "accuracy": float(acc),
        "prec_class_1": float(prec_c1),
        "rec_class_1": float(rec_c1),
        "f1_class_1": float(f1_c1),
        "F1_global": float(f1_glob),
        "recall_global": float(rec_glob),
    }

    # Logging dans MLflow sous l'expérience Default
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name.split("_")[0])
        mlflow.log_param("target_ratio", target_ratio)
        mlflow.log_metrics(metrics)

        # Enregistrement de la matrice de confusion en tant qu'artéfact JSON
        temp_json_path = f"confusion_matrix_{model_name}.json"
        with open(temp_json_path, "w") as f:
            json.dump(confusion_dict, f, indent=4)
        mlflow.log_artifact(temp_json_path)
        os.remove(temp_json_path)  # Nettoyage local

    # On ajoute la matrice de confusion au dictionnaire pour l'export JSON local
    metrics["confusion_matrix"] = confusion_dict
    return metrics


if __name__ == "__main__":
    main()
