# src/training/demo_nvidia_gnn.py

import os

import mlflow
import numpy as np
import ray
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from xgboost import XGBClassifier

# 1. Initialisation de Ray
ray.init(address="auto", ignore_reinit_error=True)


# --- DÉFINITION DE GRAPHSAGE EN PUR PYTORCH ---
# Selon la spécification NVIDIA, nous utilisons l'architecture GraphSAGE
# pour générer des embeddings de nœuds qui capturent le voisinage.
class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # GraphSAGE concatène les features du nœud et de ses voisins
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x, adj):
        # Agréger les features des voisins (Moyenne simple par multiplication matricielle avec l'adjacence)
        # adj est normalisée par degré : D^-1 * A
        neighbor_agg = torch.spmm(adj, x)

        # Concaténation [x_self || x_neighbors]
        combined = torch.cat([x, neighbor_agg], dim=1)

        # Transformation linéaire + Activation
        out = self.linear(combined)
        return out


class GraphSAGEModel(nn.Module):
    def __init__(self, in_features, hidden_dim, embedding_dim):
        super().__init__()
        self.sage1 = GraphSAGELayer(in_features, hidden_dim)
        self.sage2 = GraphSAGELayer(hidden_dim, embedding_dim)

    def forward(self, x, adj):
        h = F.relu(self.sage1(x, adj))
        embeddings = self.sage2(h, adj)
        return embeddings


# --- ENTRENEMENT DE LA PIPELINE HYBRIDE SUR LE CLUSTER RAY GPU ---
@ray.remote(num_gpus=1)
def run_nvidia_fraud_pipeline(config):
    # Initialisation MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("Default")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exécution du pipeline NVIDIA sur le périphérique : {device}")

    # 1. Génération d'un Dataset Graphe / Tabulaire avec du SIGNAL
    num_nodes = 2000
    in_features = 10  # Caractéristiques tabulaires de base (montant, heure, etc.)

    # Features tabulaires (X)
    X_raw = np.random.randn(num_nodes, in_features).astype(np.float32)

    # Création d'une vraie corrélation pour les labels (signal)
    # Si la variable 0 (ex: montant) + variable 1 (ex: risque) dépassent un seuil -> Forte probabilité de fraude
    logits = X_raw[:, 0] * 2.0 + X_raw[:, 1] * 1.5 - 1.5
    probs = 1 / (1 + np.exp(-logits))
    y = np.random.binomial(1, probs)  # Tirage de Bernoulli

    # Matrice d'adjacence : Connecter fortement les nœuds ayant le même label (homophilie de réseau)
    # Cela simule des réseaux de fraudeurs connectés entre eux (IPs, cartes, etc.)
    adj = np.eye(num_nodes)
    for i in range(num_nodes):
        if y[i] == 1:
            peers = np.where(y == 1)[0]
            if len(peers) > 1:
                # On connecte le fraudeur à d'autres fraudeurs aléatoires
                targets = np.random.choice(
                    peers, size=min(4, len(peers)), replace=False
                )
                for t in targets:
                    adj[i, t] = 1.0
                    adj[t, i] = 1.0
        else:
            peers = np.where(y == 0)[0]
            if len(peers) > 1:
                targets = np.random.choice(
                    peers, size=min(2, len(peers)), replace=False
                )
                for t in targets:
                    adj[i, t] = 1.0
                    adj[t, i] = 1.0

    # Normalisation par degré (Moyenne des voisins)
    rowsum = adj.sum(axis=1)
    d_inv = np.power(rowsum, -1.0, where=rowsum > 0)
    d_inv[rowsum == 0] = 0.0
    adj_norm = np.diag(d_inv).dot(adj)

    # Passage sur le GPU/Périphérique cible
    X_tensor = torch.FloatTensor(X_raw).to(device)
    adj_tensor = torch.FloatTensor(adj_norm).to(device)

    # 2. Étape A : Entraînement de GraphSAGE pour générer des embeddings
    sage_config = config["gnn"]
    sage_model = GraphSAGEModel(
        in_features=in_features,
        hidden_dim=sage_config["hidden_channels"],
        embedding_dim=sage_config["hidden_channels"],
    ).to(device)

    # Optimisation non supervisée simplifiée (ici simulée ou entraînée rapidement)
    # Dans la pratique, GraphSAGE peut être entraîné avec une loss de classification ou contrastive.
    # Ici, nous l'entraînons brièvement pour projeter les nœuds dans un espace latent structuré.
    optimizer = torch.optim.Adam(sage_model.parameters(), lr=0.01)
    sage_model.train()
    for _ in range(sage_config["num_epochs"]):
        optimizer.zero_grad()
        embeddings = sage_model(X_tensor, adj_tensor)
        # Loss simplifiée pour structurer l'embedding (proximité des voisins)
        neighbor_loss = F.mse_loss(torch.spmm(adj_tensor, embeddings), embeddings)
        neighbor_loss.backward()
        optimizer.step()

    # Extraction des embeddings finaux et conversion en NumPy
    sage_model.eval()
    with torch.no_grad():
        X_embeddings = sage_model(X_tensor, adj_tensor).cpu().numpy()

    # Concaténation des caractéristiques brutes (tabulaires) avec les embeddings GNN
    X_combined = np.hstack([X_raw, X_embeddings])

    # Division Train / Test
    X_train_raw, X_test_raw, X_train_comb, X_test_comb, y_train, y_test = (
        train_test_split(X_raw, X_combined, y, test_size=0.3, random_state=42)
    )

    # 3. Étape B : Comparaison des deux modèles avec MLflow

    # --- RUN 1 : BASELINE (XGBoost sur données tabulaires brutes uniquement) ---
    with mlflow.start_run(run_name="Baseline_XGBoost"):
        mlflow.log_params(config["xgb"])
        mlflow.log_param("pipeline_type", "Raw_Tabular_Only")

        xgb_raw = XGBClassifier(
            max_depth=config["xgb"]["max_depth"],
            learning_rate=config["xgb"]["learning_rate"],
            n_estimators=config["xgb"]["num_boost_round"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        xgb_raw.fit(X_train_raw, y_train)

        preds_raw = xgb_raw.predict(X_test_raw)

        # Log des métriques
        precision_c1 = precision_score(y_test, preds_raw, pos_label=1, zero_division=0)
        recall_c1 = recall_score(y_test, preds_raw, pos_label=1, zero_division=0)
        f1_glob = f1_score(y_test, preds_raw, average="macro", zero_division=0)

        metrics_baseline = {
            "accuracy": accuracy_score(y_test, preds_raw),
            "precision_class_1": precision_c1,
            "recall_class_1": recall_c1,
            "f1_global": f1_glob,
            "f1_score": f1_score(y_test, preds_raw, zero_division=0),
        }
        mlflow.log_metrics(metrics_baseline)
        print("Baseline XGBoost entraîné.")

    # --- RUN 2 : HYBRIDE (XGBoost sur données tabulaires + Embeddings GraphSAGE) ---
    with mlflow.start_run(run_name="NVIDIA_GraphSAGE_XGBoost"):
        # Log de tous les paramètres du pipeline hybride
        mlflow.log_params(config["xgb"])
        mlflow.log_params(config["gnn"])
        mlflow.log_param("pipeline_type", "GraphSAGE_Embeddings_XGBoost")

        xgb_hybrid = XGBClassifier(
            max_depth=config["xgb"]["max_depth"],
            learning_rate=config["xgb"]["learning_rate"],
            n_estimators=config["xgb"]["num_boost_round"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        xgb_hybrid.fit(X_train_comb, y_train)

        preds_hybrid = xgb_hybrid.predict(X_test_comb)

        # Log des métriques
        precision_c1_hyb = precision_score(
            y_test, preds_hybrid, pos_label=1, zero_division=0
        )
        recall_c1_hyb = recall_score(y_test, preds_hybrid, pos_label=1, zero_division=0)
        f1_glob_hyb = f1_score(y_test, preds_hybrid, average="macro", zero_division=0)

        metrics_hybrid = {
            "accuracy": accuracy_score(y_test, preds_hybrid),
            "precision_class_1": precision_c1_hyb,
            "recall_class_1": recall_c1_hyb,
            "f1_global": f1_glob_hyb,
            "f1_score": f1_score(y_test, preds_hybrid, zero_division=0),
        }
        mlflow.log_metrics(metrics_hybrid)

        # Enregistrement et publication dans le Registre de Modèles MLflow (Model Registry)
        mlflow.xgboost.log_model(
            xgb_hybrid,
            artifact_path="xgboost_model",
            registered_model_name="NVIDIA_XGBoost_Fraud_Model",
        )
        mlflow.pytorch.log_model(
            sage_model,
            artifact_path="graphsage_model",
            registered_model_name="NVIDIA_GraphSAGE_Embedding_Model",
            serialization_format="pickle",
        )

        print("Pipeline NVIDIA GraphSAGE + XGBoost entraîné.")
        return {"baseline": metrics_baseline, "hybrid": metrics_hybrid}


def main():
    # Spécifications de configuration basées sur le blueprint NVIDIA
    config = {
        "gnn": {
            "hidden_channels": 16,
            "dropout_prob": 0.1,
            "fan_out": 16,
            "num_epochs": 20,
        },
        "xgb": {"max_depth": 6, "learning_rate": 0.2, "num_boost_round": 100},
    }

    print("Démarrage de l'entraînement du pipeline NVIDIA sur Ray...")
    # Exécuter sur le cluster Ray (le GPU sera utilisé sur le worker)
    future = run_nvidia_fraud_pipeline.remote(config)
    result = ray.get(future)
    print("Pipeline NVIDIA GNN+XGBoost terminé.")
    print(
        f"Baseline -> F1 Global: {result['baseline']['f1_global']:.4f} | Recall C1: {result['baseline']['recall_class_1']:.4f}"
    )
    print(
        f"Hybride -> F1 Global: {result['hybrid']['f1_global']:.4f} | Recall C1: {result['hybrid']['recall_class_1']:.4f}"
    )

    # Export des métriques en JSON
    import json

    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_json_path = os.path.join(script_dir, "metrics_nvidia_gnn.json")
    with open(metrics_json_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Métriques NVIDIA GNN exportées en JSON dans : {metrics_json_path}")


if __name__ == "__main__":
    main()
