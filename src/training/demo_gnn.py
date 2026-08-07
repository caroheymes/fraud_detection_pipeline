# src/training/demo_gnn.py

import os

import mlflow
import numpy as np
import ray
import torch
import torch.nn.functional as F
from torch import nn

# 1. Connexion au cluster Ray
ray.init(address="auto", ignore_reinit_error=True)


# --- DÉFINITION D'UN GNN SIMPLE EN PYTORCH (Sans librairie externe additionnelle) ---
# Nous implémentons une couche de Graph Convolution Network (GCN) de base.
# Formule : H = Activation( D^-1 * A * H * W )
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj_norm):
        # Propagation des caractéristiques à travers les voisins (médiane pondérée par la matrice d'adjacence normalisée)
        support = self.linear(x)
        out = torch.spmm(adj_norm, support)
        return out


class FraudGCN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_classes):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_classes)

    def forward(self, x, adj_norm):
        h = F.relu(self.gcn1(x, adj_norm))
        # Sortie Logits pour classification binaire (Fraude vs Non-Fraude)
        logits = self.gcn2(h, adj_norm)
        return logits


# --- TÂCHE D'ENTRAÎNEMENT DU GNN SUR LE CLUSTER RAY AVEC GPU ---
@ray.remote(num_gpus=1)
def train_gnn_on_gpu(config):
    # Initialisation de MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("Default")

    # Sélection de la carte GPU si disponible (configurée dans Docker-compose)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exécution du GNN sur le périphérique : {device}")

    # 1. Génération de données de graphe fictives avec du SIGNAL
    num_nodes = 1000
    num_features = 16  # ex: montant, heure, score de risque

    # Caractéristiques des nœuds
    x_np = np.random.randn(num_nodes, num_features).astype(np.float32)

    # Création d'une vraie corrélation pour les labels (signal)
    logits = x_np[:, 0] * 2.5 + x_np[:, 1] * 2.0 - 1.0
    probs = 1 / (1 + np.exp(-logits))
    y_np = np.random.binomial(1, probs)  # Tirage de Bernoulli

    # Passage sur GPU
    x = torch.FloatTensor(x_np).to(device)
    y = torch.LongTensor(y_np).to(device)

    # Matrice d'adjacence : Connecter fortement les nœuds ayant le même label (homophilie de réseau)
    adj = np.eye(num_nodes)
    for i in range(num_nodes):
        if y_np[i] == 1:
            peers = np.where(y_np == 1)[0]
            if len(peers) > 1:
                targets = np.random.choice(
                    peers, size=min(4, len(peers)), replace=False
                )
                for t in targets:
                    adj[i, t] = 1.0
                    adj[t, i] = 1.0
        else:
            peers = np.where(y_np == 0)[0]
            if len(peers) > 1:
                targets = np.random.choice(
                    peers, size=min(2, len(peers)), replace=False
                )
                for t in targets:
                    adj[i, t] = 1.0
                    adj[t, i] = 1.0

    # Normalisation symétrique de la matrice d'adjacence (D^-1/2 * A * D^-1/2)
    rowsum = adj.sum(axis=1)
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
    d_inv_sqrt[rowsum == 0] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    adj_norm_tensor = torch.FloatTensor(adj_norm).to(device)

    # 2. Instanciation du modèle GNN
    model = FraudGCN(
        in_features=num_features, hidden_dim=config["hidden_dim"], out_classes=2
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 3. Suivi MLflow
    with mlflow.start_run(
        run_name=f"GCN_hidden_{config['hidden_dim']}_lr_{config['lr']}"
    ):
        mlflow.log_params(config)
        mlflow.log_param("device", str(device))

        # Boucle d'entraînement
        model.train()
        for epoch in range(1, config["epochs"] + 1):
            optimizer.zero_grad()
            out = model(x, adj_norm_tensor)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            # Évaluation rapide
            _, preds = torch.max(out, dim=1)
            correct = (preds == y).sum().item()
            acc = correct / num_nodes

            # Logger les métriques toutes les 10 époques dans MLflow
            if epoch % 10 == 0:
                mlflow.log_metric("loss", loss.item(), step=epoch)
                mlflow.log_metric("accuracy", acc, step=epoch)
                print(
                    f"Époque {epoch}/{config['epochs']} | Loss: {loss.item():.4f} | Acc: {acc:.4f}"
                )

        # Enregistrement final du modèle GNN dans MLflow
        mlflow.pytorch.log_model(
            model, "gcn_fraud_model", serialization_format="pickle"
        )

        # Calcul des métriques finales
        model.eval()
        with torch.no_grad():
            final_out = model(x, adj_norm_tensor)
            final_loss = criterion(final_out, y).item()
            _, final_preds = torch.max(final_out, dim=1)
            final_acc = (final_preds == y).sum().item() / num_nodes

        # Calcul des métriques par classe (Classe 1 : Fraudes)
        y_true_np = y.cpu().numpy()
        y_pred_np = final_preds.cpu().numpy()

        from sklearn.metrics import f1_score, precision_score, recall_score

        precision_c1 = precision_score(
            y_true_np, y_pred_np, pos_label=1, zero_division=0
        )
        recall_c1 = recall_score(y_true_np, y_pred_np, pos_label=1, zero_division=0)
        f1_glob = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)

        mlflow.log_metric("final_loss", final_loss)
        mlflow.log_metric("accuracy", final_acc)
        mlflow.log_metric("precision_class_1", precision_c1)
        mlflow.log_metric("recall_class_1", recall_c1)
        mlflow.log_metric("f1_global", f1_glob)

    return {
        "config": config,
        "metrics": {
            "accuracy": final_acc,
            "precision_class_1": precision_c1,
            "recall_class_1": recall_c1,
            "f1_global": f1_glob,
        },
    }


def main():
    # Configurations d'hyperparamètres pour nos tests de modèles GNN
    configs = [
        {"hidden_dim": 32, "lr": 0.01, "epochs": 100},
        {"hidden_dim": 64, "lr": 0.005, "epochs": 100},
    ]

    print("Soumission des jobs d'entraînement GNN sur le cluster Ray...")
    # Lancement parallèle sur Ray
    futures = [train_gnn_on_gpu.remote(cfg) for cfg in configs]

    # Attente des résultats
    results = ray.get(futures)
    for r in results:
        print(
            f"GNN entraîné avec config {r['config']} -> Acc: {r['metrics']['accuracy']:.4f}"
        )

    # Export des métriques en JSON
    import json

    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_json_path = os.path.join(script_dir, "metrics_gnn.json")
    with open(metrics_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Métriques GNN exportées en JSON dans : {metrics_json_path}")


if __name__ == "__main__":
    main()
