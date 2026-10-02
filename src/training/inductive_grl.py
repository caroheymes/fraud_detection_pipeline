# src/training/inductive_grl.py
"""
Inductive Graph Representation Learning (HinSAGE + XGBoost) Pipeline.
Fournit une classe InductiveGRLPipeline compatible Scikit-Learn pour l'inférence temps réel et batch.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer
from xgboost import XGBClassifier


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Calcule la distance haversine en kilomètres."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


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
        df = df_in.copy()
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

        if "distance_achat" not in df.columns and {
            "lat",
            "long",
            "merch_lat",
            "merch_long",
        }.issubset(df.columns):
            df["distance_achat"] = haversine_vectorized(
                df["lat"].astype(float),
                df["long"].astype(float),
                df["merch_lat"].astype(float),
                df["merch_long"].astype(float),
            )

        if "trans_date_trans_time" in df.columns:
            dt = pd.to_datetime(df["trans_date_trans_time"])
            if "hour_sin" not in df.columns:
                df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
                df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
            if "weekday_sin" not in df.columns:
                df["weekday_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
                df["weekday_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)
            if "month_sin" not in df.columns:
                df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
                df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12.0)

        if "age" not in df.columns and "dob" in df.columns:
            dob_dt = pd.to_datetime(df["dob"])
            df["age"] = datetime.now().year - dob_dt.dt.year

        return df

    def _extract_clean_features(
        self, df_prepared: pd.DataFrame, is_train: bool = True
    ) -> np.ndarray:
        candidate_cols = [
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

        # 1. Calcul des statistiques de voisinage pour chaque client et marchand (1-hop)
        client_groups = {}
        merchant_groups = {}
        for idx, row in df_prep.reset_index(drop=True).iterrows():
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

        mean_feat = np.mean(X_train, axis=0)
        self.global_client_stat = np.append(mean_feat, 0.0)
        self.global_merchant_stat = np.append(mean_feat, 0.0)

        # 2. Construction des tenseurs
        h_c_list = [self.client_stats[str(c)] for c in df_prep["client_node"]]
        h_m_list = [self.merchant_stats[str(m)] for m in df_prep["merchant_node"]]

        x_t_tensor = torch.tensor(X_train, dtype=torch.float32)
        h_c_tensor = torch.tensor(np.array(h_c_list), dtype=torch.float32)
        h_m_tensor = torch.tensor(np.array(h_m_list), dtype=torch.float32)
        y_tensor = torch.tensor(y_train.values.astype(float), dtype=torch.float32)

        # 3. Entraînement PyTorch
        self.net = HinSAGEPyTorchNet(
            in_features=in_dim, emb_dim=self.emb_dim, hidden_dim=self.hidden_dim
        )
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
            return z_train.cpu().numpy()

    def transform(self, test_df: pd.DataFrame) -> np.ndarray:
        """Étape inductive pour transactions en streaming ou batch."""
        df_prep = self._prepare_df(test_df)
        X_test = self._extract_clean_features(df_prep, is_train=False)

        h_c_list = [
            self.client_stats.get(str(c), self.global_client_stat)
            for c in df_prep["client_node"]
        ]
        h_m_list = [
            self.merchant_stats.get(str(m), self.global_merchant_stat)
            for m in df_prep["merchant_node"]
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
        return (probas >= self.decision_threshold).astype(int)

    def get_feature_names_out(self):
        emb_names = [f"hinsage_emb_{i}" for i in range(self.embedding_size)]
        if self.add_additional_data:
            return emb_names + self.hinsage.tabular_feature_names
        return emb_names
