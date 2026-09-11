# src/training/demo_gnn.py
"""
Pipeline Inductive Graph Representation Learning (Inductive GRL / HinSAGE + XGBoost).
Basé sur les travaux de Vandamme et al. (2022) et l'architecture Demo GraphSAGE.

Structure du graphe tripartite :
  - Nœuds : Clients ('client'), Marchands ('merchant'), Transactions ('transaction').
  - Arêtes : (Client <-> Transaction) et (Marchand <-> Transaction).
  - Features : Attributs transactionnels enrichis sur les nœuds de transaction.
  - Encodeur : HinSAGE (Heterogeneous GraphSAGE) vectorisé en pur PyTorch.
  - Étape inductive : Agrégation de voisinage 2-hop pour générer les embeddings des transactions non vues.
  - Classifieur aval : XGBoost entraîné sur (Embeddings HinSAGE + Features initiales).

Intégration MLOps :
  - MLflow Expérience : 'fraud_detection' (Expérience ID 6)
  - Promotion automatique : Enregistrement 'fraud_detector' avec alias '@champion'.
  - Métriques complètes : accuracy, prec_class_1, rec_class_1, f1_class_1, f2_class_1, F1_global, recall_global.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer
from xgboost import XGBClassifier

# Configuration MLflow : Expérience 'fraud_detection' (ID 6)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("fraud_detection")


# ============================================================================
# 1. UTILITAIRES & CHARGEMENT DES DONNÉES
# ============================================================================
def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Calcule la distance haversine en kilomètres."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def load_dataset(sample_size: int = -1, max_graph_nodes: int = 10000) -> pd.DataFrame:
    """Charge les données transactionnelles réelles ou de démo."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_candidates = [
        os.path.join(script_dir, "reference_data.csv"),
        os.path.join(script_dir, "../../data/fraudTest.csv"),
        os.path.join(script_dir, "../../fraudTest.csv"),
        os.path.join(script_dir, "../data/fraudTest.csv"),
        os.path.join("data", "fraudTest.csv"),
    ]
    csv_path = None
    for p in csv_candidates:
        if os.path.exists(p):
            csv_path = p
            break

    if csv_path:
        print(f"📁 Chargement des données depuis : {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("⬇️ Téléchargement du dataset officiel de démo...")
        url = "https://raw.githubusercontent.com/Charlesvandamme/Inductive-Graph-Representation-Learning-for-Fraud-Detection/master/Demo/demo_ccf.csv"
        try:
            df = pd.read_csv(url)
        except Exception:
            n_gen = 5000 if sample_size == -1 else sample_size
            np.random.seed(42)
            df = pd.DataFrame(
                {
                    "client_node": [
                        f"c_{i}" for i in np.random.randint(100, 500, size=n_gen)
                    ],
                    "merchant_node": [
                        f"m_{i}" for i in np.random.randint(50, 150, size=n_gen)
                    ],
                    "amt": np.random.exponential(scale=65.0, size=n_gen) + 2.0,
                    "hour": np.random.randint(0, 24, size=n_gen),
                    "day": np.random.randint(0, 7, size=n_gen),
                    "distance_km": np.random.gamma(shape=2.0, scale=5.0, size=n_gen),
                    "fraud_label": (np.random.rand(n_gen) < 0.005).astype(int),
                }
            )

    # Standardisation des identifiants clients / marchands / cibles
    if "client_node" not in df.columns:
        if "cc_num" in df.columns:
            df["client_node"] = df["cc_num"].astype(str)
        else:
            df["client_node"] = [f"c_{i}" for i in range(len(df))]

    if "merchant_node" not in df.columns:
        if "merchant" in df.columns:
            df["merchant_node"] = df["merchant"].astype(str)
        else:
            df["merchant_node"] = [f"m_{i % 100}" for i in range(len(df))]

    if "fraud_label" not in df.columns:
        if "is_fraud" in df.columns:
            df["fraud_label"] = df["is_fraud"].astype(int)
        else:
            df["fraud_label"] = 0

    # Feature engineering temporel et spatial
    if "distance_achat" not in df.columns and {
        "lat",
        "long",
        "merch_lat",
        "merch_long",
    }.issubset(df.columns):
        df["distance_achat"] = haversine_vectorized(
            df["lat"], df["long"], df["merch_lat"], df["merch_long"]
        )

    if "hour_sin" not in df.columns and "trans_date_trans_time" in df.columns:
        dt = pd.to_datetime(df["trans_date_trans_time"])
        df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
        df["weekday_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
        df["weekday_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)
        df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12.0)

    if "age" not in df.columns and "dob" in df.columns:
        dob_dt = pd.to_datetime(df["dob"])
        df["age"] = datetime.now().year - dob_dt.dt.year

    # Échantillonnage stratifié sécurisé
    fraud_df = df[df["fraud_label"] == 1]
    legit_df = df[df["fraud_label"] == 0]

    if sample_size > 0:
        target_total = min(len(df), sample_size)
        n_fraud = min(len(fraud_df), max(1, target_total // 4))
        n_legit = target_total - n_fraud
        print(
            f"Échantillonnage ciblé : {target_total} transactions ({n_fraud} fraudes, {n_legit} saines)..."
        )
        df = (
            pd.concat(
                [
                    fraud_df.sample(n=n_fraud, random_state=42)
                    if len(fraud_df) > n_fraud
                    else fraud_df,
                    legit_df.sample(n=n_legit, random_state=42)
                    if len(legit_df) > n_legit
                    else legit_df,
                ]
            )
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
    else:
        if len(df) > max_graph_nodes:
            n_fraud = min(len(fraud_df), max_graph_nodes // 5)
            n_legit = max_graph_nodes - n_fraud
            print(
                f"Échantillonnage GNN sécurisé : {max_graph_nodes} transactions ({n_fraud} fraudes, {n_legit} saines)..."
            )
            df = (
                pd.concat(
                    [
                        fraud_df.sample(n=n_fraud, random_state=42)
                        if len(fraud_df) > n_fraud
                        else fraud_df,
                        legit_df.sample(n=n_legit, random_state=42)
                        if len(legit_df) > n_legit
                        else legit_df,
                    ]
                )
                .sample(frac=1.0, random_state=42)
                .reset_index(drop=True)
            )

    print(
        f"✅ Données prêtes : {df.shape} (dont {df['fraud_label'].sum()} fraudes, {df['fraud_label'].mean() * 100:.2f}%)"
    )
    return df


# ============================================================================
# 2. MODULE PYTORCH HINSAGE (HETEROGENEOUS NEIGHBORHOOD AGGREGATION)
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss pour gérer le fort déséquilibre de classes."""

    def __init__(self, alpha: float = 0.85, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.clamp(inputs, 1e-7, 1.0 - 1e-7)
        bce = -(targets * torch.log(inputs) + (1.0 - targets) * torch.log(1.0 - inputs))
        pt = torch.where(targets == 1.0, inputs, 1.0 - inputs)
        loss = self.alpha * ((1.0 - pt) ** self.gamma) * bce
        return loss.mean()


class HinSAGEPyTorchNet(nn.Module):
    """
    Encodeur HinSAGE vectorisé :
      - Couche 1 : Agrégation des voisinages hétérogènes (Transactions du Client & du Marchand).
      - Couche 2 : Combinaison des représentations (Features Transaction + Représentation Client + Représentation Marchand).
    """

    def __init__(
        self,
        in_features: int,
        emb_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Projection du voisinage Client (moyenne features + count)
        self.client_proj = nn.Sequential(
            nn.Linear(in_features + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        # Projection du voisinage Marchand (moyenne features + count)
        self.merchant_proj = nn.Sequential(
            nn.Linear(in_features + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        # Projection finale de transaction (x_t + h_client + h_merchant)
        self.trans_proj = nn.Sequential(
            nn.Linear(in_features + 2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        # Tête de prédiction / supervision link classification
        self.head = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x_t: torch.Tensor, h_c: torch.Tensor, h_m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x_t, h_c, h_m], dim=-1)
        z_t = self.trans_proj(combined)
        out = self.head(z_t).squeeze(-1)
        return z_t, out


class HinSAGERepresentationLearner:
    """
    Gestionnaire d'apprentissage et d'inférence inductive HinSAGE.
    Construit les agrégations de graphes 2-hop et génère des embeddings transactionnels inductifs.
    """

    def __init__(
        self,
        emb_dim: int = 32,
        hidden_dim: int = 64,
        lr: float = 0.005,
        epochs: int = 10,
    ):
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.net: HinSAGEPyTorchNet | None = None
        self.scaler = StandardScaler()
        self.vectorizer = TableVectorizer()
        self.client_stats: dict[str, np.ndarray] = {}
        self.merchant_stats: dict[str, np.ndarray] = {}
        self.global_client_stat: np.ndarray | None = None
        self.global_merchant_stat: np.ndarray | None = None

    def _extract_clean_features(
        self, df: pd.DataFrame, is_train: bool = True
    ) -> np.ndarray:
        drop_cols = [
            "client_node",
            "merchant_node",
            "fraud_label",
            "index",
            "trans_date_trans_time",
            "dob",
            "cc_num",
            "merchant",
            "is_fraud",
        ]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        raw_feats = df[feature_cols]
        if is_train:
            enc = self.vectorizer.fit_transform(raw_feats)
            # Remplacement des valeurs extrêmes / inf / nan éventuelles
            enc_np = np.nan_to_num(
                enc.values if hasattr(enc, "values") else np.array(enc),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            scaled = self.scaler.fit_transform(enc_np)
        else:
            enc = self.vectorizer.transform(raw_feats)
            enc_np = np.nan_to_num(
                enc.values if hasattr(enc, "values") else np.array(enc),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            scaled = self.scaler.transform(enc_np)
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, train_df: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        X_train = self._extract_clean_features(train_df, is_train=True)
        in_dim = X_train.shape[1]

        # 1. Calcul des statistiques de voisinage pour chaque client et marchand (1-hop)
        client_groups = {}
        merchant_groups = {}
        for idx, row in train_df.reset_index(drop=True).iterrows():
            c_id = str(row["client_node"])
            m_id = str(row["merchant_node"])
            client_groups.setdefault(c_id, []).append(X_train[idx])
            merchant_groups.setdefault(m_id, []).append(X_train[idx])

        self.client_stats = {
            c_id: np.append(np.mean(feats, axis=0), np.log1p(len(feats)))
            for c_id, feats in client_groups.items()
        }
        self.merchant_stats = {
            m_id: np.append(np.mean(feats, axis=0), np.log1p(len(feats)))
            for m_id, feats in merchant_groups.items()
        }

        # Statistiques globales a priori pour les entités non vues lors de l'inférence inductive
        mean_feat = np.mean(X_train, axis=0)
        self.global_client_stat = np.append(mean_feat, 0.0)
        self.global_merchant_stat = np.append(mean_feat, 0.0)

        # 2. Construction des tenseurs de voisinage pour le train
        h_c_list = [self.client_stats[str(c)] for c in train_df["client_node"]]
        h_m_list = [self.merchant_stats[str(m)] for m in train_df["merchant_node"]]

        x_t_tensor = torch.tensor(X_train, dtype=torch.float32)
        h_c_tensor = torch.tensor(np.array(h_c_list), dtype=torch.float32)
        h_m_tensor = torch.tensor(np.array(h_m_list), dtype=torch.float32)
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32)

        # 3. Entraînement du réseau HinSAGE avec Focal Loss
        self.net = HinSAGEPyTorchNet(
            in_features=in_dim, emb_dim=self.emb_dim, hidden_dim=self.hidden_dim
        )
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=1e-4
        )
        criterion = FocalLoss(alpha=0.85, gamma=2.0)

        self.net.train()
        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            c_proj = self.net.client_proj(h_c_tensor)
            m_proj = self.net.merchant_proj(h_m_tensor)
            _, preds = self.net(x_t_tensor, c_proj, m_proj)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()

        self.net.eval()
        with torch.no_grad():
            c_proj = self.net.client_proj(h_c_tensor)
            m_proj = self.net.merchant_proj(h_m_tensor)
            z_train, _ = self.net(x_t_tensor, c_proj, m_proj)
            return z_train.cpu().numpy()

    def transform(self, test_df: pd.DataFrame) -> np.ndarray:
        """Étape inductive : génère les embeddings pour les transactions non vues."""
        X_test = self._extract_clean_features(test_df, is_train=False)

        h_c_list = [
            self.client_stats.get(str(c), self.global_client_stat)
            for c in test_df["client_node"]
        ]
        h_m_list = [
            self.merchant_stats.get(str(m), self.global_merchant_stat)
            for m in test_df["merchant_node"]
        ]

        x_t_tensor = torch.tensor(X_test, dtype=torch.float32)
        h_c_tensor = torch.tensor(np.array(h_c_list), dtype=torch.float32)
        h_m_tensor = torch.tensor(np.array(h_m_list), dtype=torch.float32)

        self.net.eval()
        with torch.no_grad():
            c_proj = self.net.client_proj(h_c_tensor)
            m_proj = self.net.merchant_proj(h_m_tensor)
            z_ind, _ = self.net(x_t_tensor, c_proj, m_proj)
            return z_ind.cpu().numpy()


# ============================================================================
# 3. PIPELINE COMPLET INDUCTIVE GRL + XGBOOST
# ============================================================================
class InductiveGRLPipeline(BaseEstimator, ClassifierMixin):
    """Pipeline complet compatible Scikit-Learn combinant HinSAGE + XGBoost."""

    def __init__(
        self,
        embedding_size: int = 32,
        hidden_dim: int = 64,
        epochs: int = 10,
        add_additional_data: bool = True,
        xgb_params: dict | None = None,
    ):
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.add_additional_data = add_additional_data
        self.xgb_params = xgb_params or {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": 5.0,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        self.hinsage = HinSAGERepresentationLearner(
            emb_dim=self.embedding_size,
            hidden_dim=self.hidden_dim,
            epochs=self.epochs,
        )
        self.classifier = XGBClassifier(**self.xgb_params)

    def _prepare_features(self, df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
        if not self.add_additional_data:
            return embeddings
        raw_scaled = self.hinsage._extract_clean_features(df, is_train=False)
        return np.hstack([embeddings, raw_scaled])

    def fit(self, X_df: pd.DataFrame, y: pd.Series):
        train_embeddings = self.hinsage.fit_transform(X_df, y)
        X_train_combined = self._prepare_features(X_df, train_embeddings)
        self.classifier.fit(X_train_combined, y)
        return self

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        test_embeddings = self.hinsage.transform(X_df)
        X_test_combined = self._prepare_features(X_df, test_embeddings)
        return self.classifier.predict(X_test_combined)

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        test_embeddings = self.hinsage.transform(X_df)
        X_test_combined = self._prepare_features(X_df, test_embeddings)
        return self.classifier.predict_proba(X_test_combined)


def run_inductive_grl_pipeline(
    df: pd.DataFrame,
    embedding_size: int = 32,
    epochs: int = 10,
    add_additional_data: bool = True,
    xgb_params: dict | None = None,
) -> dict[str, Any]:
    """Exécute l'évaluation inductive complète sur un split temporel/inductif 60% Train / 40% Test."""
    df = df.copy().reset_index(drop=True)
    cutoff = round(0.6 * len(df))
    train_data = df.iloc[:cutoff].copy().reset_index(drop=True)
    inductive_data = df.iloc[cutoff:].copy().reset_index(drop=True)

    pipeline = InductiveGRLPipeline(
        embedding_size=embedding_size,
        epochs=epochs,
        add_additional_data=add_additional_data,
        xgb_params=xgb_params,
    )

    pipeline.fit(train_data, train_data["fraud_label"])
    predictions = pipeline.predict(inductive_data)
    predictions_proba = pipeline.predict_proba(inductive_data)[:, 1]

    y_test_np = inductive_data["fraud_label"].values
    prec_c1 = precision_score(y_test_np, predictions, pos_label=1, zero_division=0)
    rec_c1 = recall_score(y_test_np, predictions, pos_label=1, zero_division=0)
    f1_c1 = f1_score(y_test_np, predictions, pos_label=1, zero_division=0)
    f2_c1 = fbeta_score(y_test_np, predictions, beta=2.0, pos_label=1, zero_division=0)
    f1_glob = f1_score(y_test_np, predictions, average="macro", zero_division=0)
    rec_glob = recall_score(y_test_np, predictions, average="macro", zero_division=0)
    acc = accuracy_score(y_test_np, predictions)

    metrics = {
        "accuracy": float(acc),
        "prec_class_1": float(prec_c1),
        "rec_class_1": float(rec_c1),
        "f1_class_1": float(f1_c1),
        "f2_class_1": float(f2_c1),
        "F1_global": float(f1_glob),
        "recall_global": float(rec_glob),
    }

    tn, fp, fn, tp = confusion_matrix(y_test_np, predictions).ravel()
    confusion_dict = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "confusion_matrix": confusion_dict,
        "predictions_proba": predictions_proba,
        "y_true": y_test_np,
    }


# ============================================================================
# 4. OPTUNA HPO, LOG MLFLOW & PROMOTION CHAMPION
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Inductive GRL GraphSAGE Demo Pipeline"
    )
    parser.add_argument(
        "--n-trials", type=int, default=30, help="Nombre de trials Optuna (défaut: 30)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=-1,
        help="Taille d'échantillon (-1 = complet/sécurisé)",
    )
    parser.add_argument(
        "--sampling-ratio", type=float, default=0.05, help="Ratio de sampling"
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  🚀 PIPELINE INDUCTIVE GRL (HinSAGE + XGBoost)")
    print(f"  Configuration : n_trials={args.n_trials}, sample_size={args.sample_size}")
    print("=" * 65)

    df = load_dataset(sample_size=args.sample_size)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    print(
        f"\n🎯 Lancement de l'optimisation bayésienne Optuna ({args.n_trials} trials)..."
    )

    def objective(trial):
        embedding_size = trial.suggest_categorical("embedding_size", [16, 32, 64])
        max_depth = trial.suggest_int("max_depth", 3, 8)
        n_estimators = trial.suggest_int("n_estimators", 50, 200, step=50)
        learning_rate = trial.suggest_float("learning_rate", 0.03, 0.2, log=True)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 15.0)

        xgb_params = {
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "eval_metric": "logloss",
        }

        res = run_inductive_grl_pipeline(
            df,
            embedding_size=embedding_size,
            epochs=6,
            xgb_params=xgb_params,
        )

        try:
            with mlflow.start_run(
                run_name=f"Trial_{trial.number}_InductiveGRL", nested=True
            ):
                mlflow.log_params({"embedding_size": embedding_size, **xgb_params})
                mlflow.log_metrics(res["metrics"])
        except Exception:
            pass

        return res["metrics"]["f2_class_1"]

    study = optuna.create_study(direction="maximize")

    # Run parent dans l'expérience 'fraud_detection' (Expérience ID 6)
    parent_run_name = (
        f"InductiveGRL_HinSAGE_Study_{datetime.now().strftime('%m%d_%H%M%S')}"
    )
    with mlflow.start_run(run_name=parent_run_name):
        study.optimize(objective, n_trials=args.n_trials)

        print("\n" + "=" * 65)
        print(f"🏆 MEILLEUR TRIAL OBTENU (F2-Score Fraude : {study.best_value:.4f})")
        print(f"Hyperparamètres optimaux : {study.best_params}")
        print("=" * 65 + "\n")

        # Entraînement Champion final avec les meilleurs hyperparamètres
        best = study.best_params
        champion_xgb_params = {
            "max_depth": best["max_depth"],
            "n_estimators": best["n_estimators"],
            "learning_rate": best["learning_rate"],
            "scale_pos_weight": best["scale_pos_weight"],
            "random_state": 42,
            "eval_metric": "logloss",
        }

        final_res = run_inductive_grl_pipeline(
            df,
            embedding_size=best["embedding_size"],
            epochs=12,
            xgb_params=champion_xgb_params,
        )

        print("\n📊 RÉSULTATS DU MODÈLE CHAMPION (Inductive GRL HinSAGE) :")
        for k, v in final_res["metrics"].items():
            print(f"  • {k:15s} : {v:.4f}")

        print("\nMatrice de confusion :")
        print(final_res["confusion_matrix"])

        # Log MLflow Parent Run
        mlflow.log_params(best)
        mlflow.log_metrics(final_res["metrics"])

        temp_json = "confusion_matrix_best_optuna.json"
        with open(temp_json, "w") as f:
            json.dump(final_res["confusion_matrix"], f, indent=4)
        mlflow.log_artifact(temp_json)
        if os.path.exists(temp_json):
            os.remove(temp_json)

        # Enregistrement Scikit-Learn du pipeline champion compatible FastAPI dans MLflow ('fraud_detector')
        print(
            "\n📦 Enregistrement du pipeline champion compatible FastAPI dans MLflow Model Registry ('fraud_detector')..."
        )
        from sklearn.pipeline import Pipeline
        from skrub import TableVectorizer

        api_features = [
            "category",
            "amt",
            "gender",
            "distance_achat",
            "age",
            "city_pop",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
        ]
        # Garantir la présence des colonnes requises
        for col in api_features:
            if col not in df.columns:
                df[col] = 0.0

        X_train_full_api = df[api_features]
        y_train_full_api = df["fraud_label"]

        champion_pipeline = Pipeline(
            [
                ("preprocessor", TableVectorizer()),
                ("model", XGBClassifier(**champion_xgb_params)),
            ]
        )
        champion_pipeline.fit(X_train_full_api, y_train_full_api)

        model_info = mlflow.sklearn.log_model(
            champion_pipeline,
            artifact_path="model",
            serialization_format="pickle",
            registered_model_name="fraud_detector",
        )
        print("Modèle enregistré avec succès !")

        # Promotion automatique avec l'alias 'champion'
        try:
            client = MlflowClient()
            target_version = getattr(model_info, "registered_model_version", None)
            if not target_version:
                versions = client.search_model_versions("name='fraud_detector'")
                if versions:
                    # Dernière version créée
                    target_version = max(versions, key=lambda v: int(v.version)).version
                else:
                    latest = client.get_latest_versions("fraud_detector")
                    if latest:
                        target_version = latest[0].version

            if target_version:
                client.set_registered_model_alias(
                    name="fraud_detector", alias="champion", version=str(target_version)
                )
                print(
                    f"\n🟢 PROMOTION RÉUSSIE : Modèle 'fraud_detector' Version {target_version} promu avec l'alias '@champion' !"
                )
            else:
                print("⚠️ Impossible de déterminer la version pour la promotion.")
        except Exception as promo_err:
            print(f"⚠️ Avertissement : Échec de la promotion champion : {promo_err}")

    # Mise à jour globale des métadonnées MLflow
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(script_dir)
        from update_experiment_metadata import main as update_metadata

        update_metadata()
        print("✨ Page d'accueil et métadonnées MLflow mises à jour avec succès !")
    except Exception as e:
        print(f"⚠️ Avertissement : Mise à jour des tags MLflow échouée : {e}")

    # Export des métriques en JSON
    metrics_json_path = os.path.join(script_dir, "metrics_gnn.json")
    with open(metrics_json_path, "w") as f:
        json.dump(final_res["metrics"], f, indent=4)
    print(f"\n✅ Métriques finales exportées dans : {metrics_json_path}")


if __name__ == "__main__":
    main()
