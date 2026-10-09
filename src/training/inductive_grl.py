# src/training/inductive_grl.py
"""
Inductive Graph Representation Learning (HinSAGE + XGBoost) Pipeline.
Fournit une classe InductiveGRLPipeline compatible Scikit-Learn pour l'inférence temps réel et batch.
"""

import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer
from xgboost import XGBClassifier

# Import du module features transverse
from src.utils.features import BASE_FEATURE_COLUMNS, prepare_features


class FocalLoss(nn.Module):
    """Focal Loss pour la classification binaire très déséquilibrée."""

    def __init__(self, alpha: float = 0.80, gamma: float = 2.0):
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
    Réseau HinSAGE en PyTorch pour l'agrégation de voisinage hétérogène :
      - Projections Client (voisinage 1-hop)
      - Projections Marchand (voisinage 1-hop)
      - Projection Transaction (x_trans + h_client + h_merchant)
    """

    def __init__(
        self,
        in_features: int,
        emb_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.client_proj = nn.Sequential(
            nn.Linear(in_features + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.merchant_proj = nn.Sequential(
            nn.Linear(in_features + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.trans_proj = nn.Sequential(
            nn.Linear(in_features + 2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )
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
        self.tabular_feature_names: list[str] = []

    def _prepare_df(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Standardise les features, identifiants de graphe et variables temporelles/géographiques."""
        return prepare_features(df_in, include_graph_ids=True)

    def _extract_clean_features(
        self, df_prepared: pd.DataFrame, is_train: bool = True
    ) -> np.ndarray:
        if not is_train and hasattr(self.vectorizer, "feature_names_in_"):
            candidate_cols = list(self.vectorizer.feature_names_in_)
        else:
            candidate_cols = BASE_FEATURE_COLUMNS
        present_cols = [c for c in candidate_cols if c in df_prepared.columns]
        raw_feats = df_prepared[present_cols]

        if is_train:
            enc = self.vectorizer.fit_transform(raw_feats)
            enc_np = np.nan_to_num(
                enc.values if hasattr(enc, "values") else np.array(enc),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            scaled = self.scaler.fit_transform(enc_np)
            self.tabular_feature_names = list(self.vectorizer.get_feature_names_out())
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
        df_prep = self._prepare_df(train_df)
        X_train = self._extract_clean_features(df_prep, is_train=True)
        in_dim = X_train.shape[1]

        # 1. Calcul ultra-rapide des statistiques de voisinage (1-hop)
        client_nodes = df_prep["client_node"].astype(str).values
        merchant_nodes = df_prep["merchant_node"].astype(str).values

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

        # 3. Entraînement PyTorch
        self.net = HinSAGEPyTorchNet(
            in_features=in_dim, emb_dim=self.emb_dim, hidden_dim=self.hidden_dim
        ).to(device)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=1e-4
        )
        criterion = FocalLoss(alpha=0.80, gamma=2.0)

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
            result = z_train.cpu().numpy().astype(np.float32)

        del (
            x_t_tensor,
            h_c_tensor,
            h_m_tensor,
            y_tensor,
            h_c_arr,
            h_m_arr,
            c_proj,
            m_proj,
            z_train,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return result

    def transform(self, test_df: pd.DataFrame) -> np.ndarray:
        """Étape inductive pour transactions en streaming ou batch."""
        df_prep = self._prepare_df(test_df)
        X_test = self._extract_clean_features(df_prep, is_train=False)

        c_nodes_test = df_prep["client_node"].astype(str).values
        m_nodes_test = df_prep["merchant_node"].astype(str).values

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
            result = z_ind.cpu().numpy().astype(np.float32)

        del x_t_tensor, h_c_tensor, h_m_tensor, h_c_arr, h_m_arr, c_proj, m_proj, z_ind
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return result


class InductiveGRLPipeline(BaseEstimator, ClassifierMixin):
    """
    Pipeline complet Scikit-Learn sérialisable intégrant HinSAGE + XGBoost.
    """

    def __init__(
        self,
        embedding_size: int = 32,
        hidden_dim: int = 64,
        epochs: int = 10,
        add_additional_data: bool = True,
        decision_threshold: float = 0.88,
        xgb_params: dict | None = None,
    ):
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.add_additional_data = add_additional_data
        self.decision_threshold = decision_threshold
        self.xgb_params = xgb_params or {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.08,
            "scale_pos_weight": 2.5,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        self.hinsage = HinSAGERepresentationLearner(
            emb_dim=self.embedding_size,
            hidden_dim=self.hidden_dim,
            epochs=self.epochs,
        )
        self.classifier = XGBClassifier(**self.xgb_params)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "decision_threshold" not in self.__dict__:
            self.decision_threshold = 0.5
        if "xgb_params" not in self.__dict__:
            self.xgb_params = {}
        if "add_additional_data" not in self.__dict__:
            self.add_additional_data = True

    def _prepare_features(self, df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
        if not self.add_additional_data:
            return embeddings
        df_prep = self.hinsage._prepare_df(df)
        raw_scaled = self.hinsage._extract_clean_features(df_prep, is_train=False)
        return np.hstack([embeddings, raw_scaled])

    def fit(self, X_df: pd.DataFrame, y: pd.Series):
        y_series = pd.Series(y).reset_index(drop=True)
        train_embeddings = self.hinsage.fit_transform(X_df, y_series)
        X_train_combined = self._prepare_features(X_df, train_embeddings)
        self.classifier.fit(X_train_combined, y_series)
        return self

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        test_embeddings = self.hinsage.transform(X_df)
        X_test_combined = self._prepare_features(X_df, test_embeddings)
        return self.classifier.predict_proba(X_test_combined)

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        probas = self.predict_proba(X_df)[:, 1]
        return (probas >= getattr(self, "decision_threshold", 0.5)).astype(int)

    def get_feature_names_out(self):
        emb_names = [f"hinsage_emb_{i}" for i in range(self.embedding_size)]
        if getattr(self, "add_additional_data", True):
            if (
                hasattr(self.hinsage, "tabular_feature_names")
                and self.hinsage.tabular_feature_names
            ):
                return emb_names + list(self.hinsage.tabular_feature_names)
            elif hasattr(self.hinsage, "vectorizer") and hasattr(
                self.hinsage.vectorizer, "get_feature_names_out"
            ):
                try:
                    return emb_names + list(
                        self.hinsage.vectorizer.get_feature_names_out()
                    )
                except Exception:
                    pass
        return emb_names


# Forcer le namespace canonique pour garantir une sérialisation pickle portable hors __main__
FocalLoss.__module__ = "src.training.inductive_grl"
HinSAGEPyTorchNet.__module__ = "src.training.inductive_grl"
HinSAGERepresentationLearner.__module__ = "src.training.inductive_grl"
InductiveGRLPipeline.__module__ = "src.training.inductive_grl"
