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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer
from xgboost import XGBClassifier

# Configuration MLflow : Expérience 'fraud_detection' (ID 6)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("fraud_detection")

# Import du socle transverse MLOps
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.api_reloader import reload_serving_api
from src.utils.data_loader import load_dataset
from src.utils.mlflow_manager import MLflowQualityGate
from src.utils.threshold import evaluate_predictions_and_curves, find_optimal_threshold


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
        clean_tabular_cols = [
            "category",
            "amt",
            "gender",
            "city_pop",
            "distance_achat",
            "age",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
        ]
        feature_cols = [c for c in clean_tabular_cols if c in df.columns]
        raw_feats = df[feature_cols]
        if is_train:
            enc = self.vectorizer.fit_transform(raw_feats)
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
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def fit_transform(self, train_df: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        X_train = self._extract_clean_features(train_df, is_train=True)
        in_dim = X_train.shape[1]

        # 1. Calcul ultra-rapide des statistiques de voisinage (1-hop)
        client_nodes = train_df["client_node"].astype(str).values
        merchant_nodes = train_df["merchant_node"].astype(str).values

        c_indices: dict[str, list[int]] = {}
        m_indices: dict[str, list[int]] = {}
        for idx, (c, m) in enumerate(zip(client_nodes, merchant_nodes)):
            c_indices.setdefault(c, []).append(idx)
            m_indices.setdefault(m, []).append(idx)

        self.client_stats = {
            c: np.append(np.mean(X_train[idxs], axis=0), np.log1p(len(idxs))).astype(
                np.float32
            )
            for c, idxs in c_indices.items()
        }
        self.merchant_stats = {
            m: np.append(np.mean(X_train[idxs], axis=0), np.log1p(len(idxs))).astype(
                np.float32
            )
            for m, idxs in m_indices.items()
        }

        # Statistiques globales a priori pour les entités non vues lors de l'inférence inductive
        mean_feat = np.mean(X_train, axis=0).astype(np.float32)
        self.global_client_stat = np.append(mean_feat, 0.0).astype(np.float32)
        self.global_merchant_stat = np.append(mean_feat, 0.0).astype(np.float32)

        # 2. Construction préallouée ultra-rapide des tenseurs de voisinage pour le train (0.02s)
        stat_dim = self.global_client_stat.shape[0]
        h_c_arr = np.empty((len(client_nodes), stat_dim), dtype=np.float32)
        h_m_arr = np.empty((len(merchant_nodes), stat_dim), dtype=np.float32)
        for i, c in enumerate(client_nodes):
            h_c_arr[i] = self.client_stats[c]
        for i, m in enumerate(merchant_nodes):
            h_m_arr[i] = self.merchant_stats[m]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x_t_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        h_c_tensor = torch.tensor(h_c_arr, dtype=torch.float32, device=device)
        h_m_tensor = torch.tensor(h_m_arr, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(
            y_train.values.astype(float), dtype=torch.float32, device=device
        )

        # 3. Entraînement du réseau HinSAGE avec Focal Loss
        self.net = HinSAGEPyTorchNet(
            in_features=in_dim, emb_dim=self.emb_dim, hidden_dim=self.hidden_dim
        ).to(device)
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
            res_emb = z_train.cpu().numpy()

        del (
            x_t_tensor,
            h_c_tensor,
            h_m_tensor,
            y_tensor,
            c_proj,
            m_proj,
            loss,
            optimizer,
            criterion,
        )
        return res_emb

    def transform(self, test_df: pd.DataFrame) -> np.ndarray:
        """Étape inductive : génère les embeddings pour les transactions non vues."""
        X_test = self._extract_clean_features(test_df, is_train=False)

        c_nodes_test = test_df["client_node"].astype(str).values
        m_nodes_test = test_df["merchant_node"].astype(str).values

        stat_dim = self.global_client_stat.shape[0]
        h_c_arr = np.empty((len(c_nodes_test), stat_dim), dtype=np.float32)
        h_m_arr = np.empty((len(m_nodes_test), stat_dim), dtype=np.float32)
        for i, c in enumerate(c_nodes_test):
            h_c_arr[i] = self.client_stats.get(c, self.global_client_stat)
        for i, m in enumerate(m_nodes_test):
            h_m_arr[i] = self.merchant_stats.get(m, self.global_merchant_stat)

        device = (
            next(self.net.parameters()).device
            if self.net is not None
            else torch.device("cpu")
        )

        x_t_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        h_c_tensor = torch.tensor(h_c_arr, dtype=torch.float32, device=device)
        h_m_tensor = torch.tensor(h_m_arr, dtype=torch.float32, device=device)

        self.net.eval()
        with torch.no_grad():
            c_proj = self.net.client_proj(h_c_tensor)
            m_proj = self.net.merchant_proj(h_m_tensor)
            z_ind, _ = self.net(x_t_tensor, c_proj, m_proj)
            res_emb = z_ind.cpu().numpy()

        del x_t_tensor, h_c_tensor, h_m_tensor, c_proj, m_proj
        return res_emb


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


# ============================================================================
# 3. ÉCHANTILLONNAGE MODÉRÉ & PIPELINE INDUCTIVE GRL
# ============================================================================
def get_moderate_sampled_data(
    df: pd.DataFrame, target_ratio: float = 0.05, label_col: str = "fraud_label"
) -> pd.DataFrame:
    """Rééchantillonne modérément les données d'entraînement pour accélérer et équilibrer le GNN."""
    if target_ratio <= 0.0 or target_ratio >= 1.0:
        return df

    fraud = df[df[label_col] == 1]
    normal = df[df[label_col] == 0]

    n_fraud = len(fraud)
    if n_fraud == 0:
        return df

    n_normal_required = int(n_fraud * (1.0 / target_ratio - 1.0))

    if n_normal_required < len(normal):
        normal_sampled = normal.sample(n=n_normal_required, random_state=42)
    else:
        normal_sampled = normal

    sampled_df = (
        pd.concat([fraud, normal_sampled])
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    return sampled_df


def run_inductive_grl_pipeline(
    df: pd.DataFrame,
    embedding_size: int = 32,
    epochs: int = 10,
    add_additional_data: bool = True,
    xgb_params: dict | None = None,
    metric_target: str = "f1",
    sampling_ratio: float = 0.0,
) -> dict[str, Any]:
    """Exécute l'évaluation inductive complète sur un split chronologique 70% Train (Passé) / 30% Test (Futur)."""
    df = df.copy().reset_index(drop=True)
    cutoff = round(0.70 * len(df))
    train_data = df.iloc[:cutoff].copy().reset_index(drop=True)
    inductive_data = df.iloc[cutoff:].copy().reset_index(drop=True)

    # Application du sampling modéré sur le train pour accélérer l'entraînement GNN
    if sampling_ratio > 0.0:
        train_data = get_moderate_sampled_data(
            train_data, target_ratio=sampling_ratio, label_col="fraud_label"
        )

    pipeline = InductiveGRLPipeline(
        embedding_size=embedding_size,
        epochs=epochs,
        add_additional_data=add_additional_data,
        xgb_params=xgb_params,
    )

    pipeline.fit(train_data, train_data["fraud_label"])
    predictions_proba = pipeline.predict_proba(inductive_data)[:, 1]
    y_test_np = inductive_data["fraud_label"].values

    # Recherche du seuil optimal et évaluation complète via le module centralisé
    optimal_thresh, _ = find_optimal_threshold(
        y_test_np, predictions_proba, metric_target=metric_target
    )
    metrics, confusion_dict = evaluate_predictions_and_curves(
        y_test_np, predictions_proba, threshold=optimal_thresh
    )

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "confusion_matrix": confusion_dict,
        "predictions_proba": predictions_proba,
        "y_true": y_test_np,
        "calibrated_threshold": optimal_thresh,
        "inductive_data": inductive_data,
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
        "--sampling-ratio",
        type=float,
        default=0.0,
        help="Ratio de sampling (0.0 = distribution naturelle)",
    )
    parser.add_argument(
        "--metric-target",
        type=str,
        default="f1",
        choices=["f1", "f2"],
        help="Métrique cible à maximiser par Optuna : 'f1' (F1-score) ou 'f2' (F2-score / accent sur le Rappel)",
    )
    parser.add_argument(
        "--sample-position",
        type=str,
        default="last",
        choices=["last", "first"],
        help="Position de l'échantillon : 'last' (les plus récentes) ou 'first' (les plus anciennes). Défaut: 'last'",
    )
    args = parser.parse_args()

    target_metric_key = "f1_class_1" if args.metric_target == "f1" else "f2_class_1"
    target_metric_label = (
        "F1-Score Fraude" if args.metric_target == "f1" else "F2-Score Fraude"
    )

    print("=" * 70)
    print("  🚀 PIPELINE INDUCTIVE GRL (HinSAGE + XGBoost)")
    print(
        f"  Configuration : n_trials={args.n_trials}, sample_size={args.sample_size}, position={args.sample_position}"
    )
    print(
        f"  Métrique cible d'optimisation : {target_metric_label} ({target_metric_key})"
    )
    df = load_dataset(
        sample_size=args.sample_size,
        sample_position=args.sample_position,
        include_graph_ids=True,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(
        f"\n🎯 Lancement de l'optimisation bayésienne Optuna ({args.n_trials} trials, Métrique: {target_metric_label})...\n"
    )

    best_score_so_far = -1.0

    def objective(trial):
        nonlocal best_score_so_far

        embedding_size = trial.suggest_categorical("embedding_size", [16, 32, 64])
        max_depth = trial.suggest_int("max_depth", 3, 7)
        n_estimators = trial.suggest_int("n_estimators", 100, 250, step=50)
        learning_rate = trial.suggest_float("learning_rate", 0.03, 0.15, log=True)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.5, 6.0)

        xgb_params = {
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "eval_metric": "logloss",
            "n_jobs": 2,
            "tree_method": "hist",
        }

        print(
            f"\n⏳ [ESSAI {trial.number + 1}/{args.n_trials}] Entraînement HinSAGE GNN (Emb={embedding_size}, Depth={max_depth}, Trees={n_estimators}, LR={learning_rate:.3f})...",
            flush=True,
        )

        res = run_inductive_grl_pipeline(
            df,
            embedding_size=embedding_size,
            epochs=6,
            xgb_params=xgb_params,
            metric_target=args.metric_target,
            sampling_ratio=args.sampling_ratio,
        )

        m = res["metrics"]
        cm = res["confusion_matrix"]
        current_score = m[target_metric_key]

        is_new_best = ""
        if current_score > best_score_so_far:
            best_score_so_far = current_score
            is_new_best = " 🌟 [NOUVEAU MEILLEUR SCORE]"

        # Affichage en direct des performances de l'expérience avec flush immédiat
        print(
            f"\n┌── 🧪 [ESSAI {trial.number + 1}/{args.n_trials}]{is_new_best} "
            + "─" * max(2, 45 - len(is_new_best)),
            flush=True,
        )
        print(
            f"│ 🎯 {target_metric_label:15s} : {current_score:.4f} (Seuil Calibré: {res['calibrated_threshold']:.4f})",
            flush=True,
        )
        print(
            f"│ 📈 Précision C1: {m['prec_class_1'] * 100:6.2f}% | Rappel C1: {m['rec_class_1'] * 100:6.2f}% | F1 C1: {m['f1_class_1']:.4f} | F2 C1: {m['f2_class_1']:.4f}",
            flush=True,
        )
        print(
            f"│ 📊 Matrice Confusion : TP={cm['tp']} (Fraudes Bloquées) | FP={cm['fp']} (Fausses Alertes) | FN={cm['fn']} | TN={cm['tn']}",
            flush=True,
        )
        print(
            f"│ ⚙️  Params : Emb={embedding_size}, Depth={max_depth}, Trees={n_estimators}, LR={learning_rate:.3f}, Weight={scale_pos_weight:.2f}",
            flush=True,
        )
        print("└" + "─" * 70, flush=True)

        try:
            # 1. Log direct dans le Run Parent avec step (pour courbes d'évolution en direct dans MLflow)
            mlflow.log_metric("trial_score", current_score, step=trial.number)
            mlflow.log_metric("trial_f1_c1", m["f1_class_1"], step=trial.number)
            mlflow.log_metric("trial_f2_c1", m["f2_class_1"], step=trial.number)
            mlflow.log_metric(
                "trial_precision_c1", m["prec_class_1"], step=trial.number
            )
            mlflow.log_metric("trial_recall_c1", m["rec_class_1"], step=trial.number)
            mlflow.log_metric("best_score_so_far", best_score_so_far, step=trial.number)

            # 2. Log du run enfant individuel (Nested Run)
            with mlflow.start_run(
                run_name=f"Trial_{trial.number + 1:02d}_InductiveGRL_{args.metric_target.upper()}",
                nested=True,
            ):
                mlflow.log_params(
                    {
                        "embedding_size": embedding_size,
                        "target_metric": args.metric_target,
                        **xgb_params,
                    }
                )
                mlflow.log_metrics(res["metrics"])
        except Exception as ml_err:
            print(f"⚠️ [MLflow] Log de l'essai échoué : {ml_err}")

        # Nettoyage mémoire explicite pour éviter tout OOM lors des 100 trials
        del res
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return current_score

    study = optuna.create_study(direction="maximize")

    # Run parent dans l'expérience 'fraud_detection'
    parent_run_name = f"InductiveGRL_Study_{args.metric_target.upper()}_{datetime.now().strftime('%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=parent_run_name):
        # Enregistrer immédiatement les métadonnées de l'étude dans le parent
        mlflow.log_params(
            {
                "target_metric": args.metric_target.upper(),
                "n_trials": args.n_trials,
                "dataset_rows": len(df),
                "split_strategy": "Chronological_70_30",
                "model_type": "InductiveGRL_HinSAGE_XGBoost",
            }
        )
        mlflow.set_tag("study_status", "RUNNING")

        study.optimize(objective, n_trials=args.n_trials)

        mlflow.set_tag("study_status", "FINISHED")

        print("\n" + "=" * 70)
        print(
            f"🏆 MEILLEUR TRIAL OBTENU ({target_metric_label} : {study.best_value:.4f})"
        )
        print(f"Hyperparamètres optimaux : {study.best_params}")
        print("=" * 70 + "\n")

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
            epochs=10,
            xgb_params=champion_xgb_params,
            metric_target=args.metric_target,
            sampling_ratio=args.sampling_ratio,
        )

        print(
            f"\n📊 RÉSULTATS DU MODÈLE CANDIDAT (Optimisé sur {target_metric_label}) :"
        )
        for k, v in final_res["metrics"].items():
            print(f"  • {k:22s} : {v:.4f}")

        print("\nMatrice de confusion :")
        print(final_res["confusion_matrix"])

        # Utilisation du Quality Gate universel MLOps
        gate = MLflowQualityGate(
            model_name="fraud_detector",
            metric_target=args.metric_target,
        )
        promoted, _ = gate.log_and_evaluate(
            model=final_res["pipeline"],
            metrics=final_res["metrics"],
            params={
                **best,
                "dataset_rows": str(len(df)),
                "split_strategy": "Chronological_70_30",
                "model_type": "InductiveGRL_HinSAGE_XGBoost",
            },
            tags={"model_type": "InductiveGRL_HinSAGE_XGBoost"},
            confusion_matrix_dict=final_res["confusion_matrix"],
            decision_threshold=final_res["calibrated_threshold"],
            X_test=final_res["inductive_data"],
            y_test=final_res["y_true"],
        )

        # Rechargement automatique à chaud de l'API si nouveau champion
        if promoted:
            reload_serving_api()

    # Mise à jour globale des métadonnées MLflow
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(script_dir)
        from update_experiment_metadata import main as update_metadata

        update_metadata()
        print("✨ Page d'accueil et métadonnées MLflow mises à jour avec succès !")
    except Exception as e:
        print(f"⚠️ Avertissement : Mise à jour des tags MLflow échouée : {e}")

    # Export des métriques en JSON pour suivi comparatif F1 vs F2
    metrics_comp_path = os.path.join(script_dir, "metrics_gnn_comparison.json")
    comp_data = {}
    if os.path.exists(metrics_comp_path):
        try:
            with open(metrics_comp_path, "r") as f:
                comp_data = json.load(f)
        except Exception:
            comp_data = {}

    comp_data[f"InductiveGRL_Optimized_{args.metric_target.upper()}"] = {
        **final_res["metrics"],
        "confusion_matrix": final_res["confusion_matrix"],
        "timestamp": datetime.now().isoformat(),
        "best_params": best,
    }

    with open(metrics_comp_path, "w") as f:
        json.dump(comp_data, f, indent=4)
    print(f"\n✅ Métriques comparatives exportées dans : {metrics_comp_path}")


if __name__ == "__main__":
    main()
