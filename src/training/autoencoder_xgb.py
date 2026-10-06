# src/training/autoencoder_xgb.py
"""
Pipeline Hybride Semi-Supervisé / Supervisé : Auto-encodeur PyTorch + XGBoost.
L'Auto-encodeur est entraîné sur les transactions saines (classe 0) pour extraire :
  1. L'erreur de reconstruction MSE (signal d'anomalie non-supervisé).
  2. Le z-score d'anomalie normalisé et le log-MSE.
  3. Les embeddings de l'espace latent (bottleneck z).
Ces représentations enrichies sont ensuite combinées aux variables tabulaires et fournies à XGBoost.
Compatible Scikit-Learn, MLflow et l'API d'inférence FastAPI.
"""

from __future__ import annotations

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from skrub import TableVectorizer
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from src.utils.features import prepare_features


class AutoencoderNet(nn.Module):
    """Réseau Auto-encodeur à goulot d'étranglement (Bottleneck)."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(latent_dim * 2, 16)),
            nn.BatchNorm1d(max(latent_dim * 2, 16)),
            nn.LeakyReLU(0.2),
            nn.Linear(max(latent_dim * 2, 16), latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(latent_dim * 2, 16)),
            nn.BatchNorm1d(max(latent_dim * 2, 16)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(max(latent_dim * 2, 16), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, in_features),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class AutoencoderFeatureLearner:
    """
    Module d'extraction d'anomalies et d'embeddings par Auto-encodeur.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        lr: float = 0.001,
        epochs: int = 15,
        batch_size: int = 512,
        dropout: float = 0.1,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state

        self.vectorizer = TableVectorizer()
        self.scaler = StandardScaler()
        self.net: AutoencoderNet | None = None
        self.tabular_feature_names: list[str] = []
        self.mu_loss_: float = 0.0
        self.std_loss_: float = 1.0

    def _prepare_df(self, df_in: pd.DataFrame) -> pd.DataFrame:
        return prepare_features(df_in, include_graph_ids=False)

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

        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def fit_transform(
        self, X_df: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> np.ndarray:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        df_prep = self._prepare_df(X_df)
        y_arr = np.asarray(y)

        # 1. Extraction et normalisation des features tabulaires
        X_all_scaled = self._extract_clean_features(df_prep, is_train=True)
        in_dim = X_all_scaled.shape[1]

        # 2. Filtrage des données saines (classe 0) pour l'entraînement de l'Auto-encodeur
        legit_mask = y_arr == 0
        X_legit_scaled = X_all_scaled[legit_mask]
        if len(X_legit_scaled) == 0:
            X_legit_scaled = X_all_scaled

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AutoencoderNet(
            in_features=in_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(device)

        dataset = TensorDataset(torch.tensor(X_legit_scaled, dtype=torch.float32))
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=1e-5
        )
        criterion = nn.MSELoss()

        self.net.train()
        for epoch in range(1, self.epochs + 1):
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                reconstructed, _ = self.net(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()

        # 3. Calcul des métriques de référence normales et transformation complète
        self.net.eval()
        with torch.no_grad():
            x_legit_tensor = torch.tensor(
                X_legit_scaled, dtype=torch.float32, device=device
            )
            rec_legit, _ = self.net(x_legit_tensor)
            losses_legit = (
                torch.mean((x_legit_tensor - rec_legit) ** 2, dim=1).cpu().numpy()
            )
            self.mu_loss_ = float(np.mean(losses_legit))
            self.std_loss_ = float(np.std(losses_legit) + 1e-6)

            # Inférence des features d'anomalie sur TOUTES les transactions d'entraînement
            x_all_tensor = torch.tensor(
                X_all_scaled, dtype=torch.float32, device=device
            )
            reconstructed_all, latent_all = self.net(x_all_tensor)
            mse_errors = (
                torch.mean((x_all_tensor - reconstructed_all) ** 2, dim=1).cpu().numpy()
            )
            latent_np = latent_all.cpu().numpy()

        log_mse = np.log1p(mse_errors).reshape(-1, 1)
        z_score = ((mse_errors - self.mu_loss_) / self.std_loss_).reshape(-1, 1)
        mse_col = mse_errors.reshape(-1, 1)

        # Matrice finale : [Features Tabulaires Scalées, MSE, Log(1+MSE), Z-Score, Embeddings Latents]
        return np.hstack([X_all_scaled, mse_col, log_mse, z_score, latent_np])

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        df_prep = self._prepare_df(X_df)
        X_scaled = self._extract_clean_features(df_prep, is_train=False)

        device = (
            next(self.net.parameters()).device
            if self.net is not None
            else torch.device("cpu")
        )
        self.net.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            reconstructed, latent = self.net(x_tensor)
            mse_errors = (
                torch.mean((x_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            )
            latent_np = latent.cpu().numpy()

        log_mse = np.log1p(mse_errors).reshape(-1, 1)
        z_score = ((mse_errors - self.mu_loss_) / self.std_loss_).reshape(-1, 1)
        mse_col = mse_errors.reshape(-1, 1)

        return np.hstack([X_scaled, mse_col, log_mse, z_score, latent_np])

    def get_feature_names_out(self) -> list[str]:
        base_names = list(self.tabular_feature_names)
        anomaly_names = ["ae_mse_error", "ae_log_mse", "ae_zscore_anomaly"]
        latent_names = [f"ae_latent_{i}" for i in range(self.latent_dim)]
        return base_names + anomaly_names + latent_names


class AutoencoderXGBoostPipeline(BaseEstimator, ClassifierMixin):
    """
    Pipeline Hybride Sérialisable Auto-encodeur + XGBoost.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        ae_lr: float = 0.001,
        ae_epochs: int = 15,
        ae_batch_size: int = 512,
        ae_dropout: float = 0.1,
        decision_threshold: float = 0.50,
        xgb_params: dict | None = None,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.ae_lr = ae_lr
        self.ae_epochs = ae_epochs
        self.ae_batch_size = ae_batch_size
        self.ae_dropout = ae_dropout
        self.decision_threshold = decision_threshold
        self.random_state = random_state
        self.xgb_params = xgb_params or {
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.08,
            "scale_pos_weight": 5.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 42,
            "tree_method": "hist",
            "eval_metric": "logloss",
        }

        self.ae_extractor = AutoencoderFeatureLearner(
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            lr=self.ae_lr,
            epochs=self.ae_epochs,
            batch_size=self.ae_batch_size,
            dropout=self.ae_dropout,
            random_state=self.random_state,
        )
        self.classifier = XGBClassifier(**self.xgb_params)

    def fit(self, X_df: pd.DataFrame, y: pd.Series | np.ndarray):
        y_arr = np.asarray(y)
        X_enriched = self.ae_extractor.fit_transform(X_df, y_arr)
        self.classifier.fit(X_enriched, y_arr)
        return self

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        X_enriched = self.ae_extractor.transform(X_df)
        return self.classifier.predict_proba(X_enriched)

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        probas = self.predict_proba(X_df)[:, 1]
        thresh = getattr(self, "decision_threshold", 0.50)
        return (probas >= thresh).astype(int)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return self.ae_extractor.get_feature_names_out()


# Namespace canonique pour sérialisation pickle portable
AutoencoderNet.__module__ = "src.training.autoencoder_xgb"
AutoencoderFeatureLearner.__module__ = "src.training.autoencoder_xgb"
AutoencoderXGBoostPipeline.__module__ = "src.training.autoencoder_xgb"
