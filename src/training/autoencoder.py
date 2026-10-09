# src/training/autoencoder.py
"""
Auto-encodeur PyTorch pour la Détection d'Anomalies / Fraude Semi-supervisée.
Entraîné exclusivement sur les transactions légitimes (classe 0) pour modéliser le comportement normal.
Lors de l'inférence, une erreur de reconstruction anormale caractérise une fraude potentielle.
Compatible avec Scikit-Learn, MLflow et l'API d'inférence FastAPI.
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

from src.utils.features import BASE_FEATURE_COLUMNS, prepare_features


class AutoencoderNet(nn.Module):
    """
    Réseau Auto-encodeur symétrique à goulot d'étranglement (Bottleneck) en PyTorch.
    Compresse l'espace d'entrée vers un espace latent compact, puis le reconstruit.
    """

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


class AutoencoderFraudDetector(BaseEstimator, ClassifierMixin):
    """
    Classifieur / Détecteur d'anomalies basé sur un Auto-encodeur semi-supervisé.
    Entraîne le réseau uniquement sur les transactions saines (y == 0).
    Fournit `decision_function`, `predict_proba` et `predict` compatibles Scikit-Learn.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        lr: float = 0.001,
        epochs: int = 25,
        batch_size: int = 256,
        dropout: float = 0.1,
        decision_threshold: float = 0.50,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.decision_threshold = decision_threshold
        self.random_state = random_state

        self.vectorizer = TableVectorizer()
        self.scaler = StandardScaler()
        self.net: AutoencoderNet | None = None
        self.feature_names_out_: list[str] = []
        self.mu_loss_: float = 0.0
        self.std_loss_: float = 1.0
        self.min_loss_: float = 0.0
        self.max_loss_: float = 1.0

    def _prepare_df(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Standardise les features tabulaires, composantes cycliques et spatiales."""
        return prepare_features(df_in, include_graph_ids=False)

    def _extract_clean_features(
        self, df_prepared: pd.DataFrame, is_train: bool = True
    ) -> np.ndarray:
        present_cols = [c for c in BASE_FEATURE_COLUMNS if c in df_prepared.columns]
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
            self.feature_names_out_ = list(self.vectorizer.get_feature_names_out())
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

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None):
        """
        Entraîne l'Auto-encodeur UNIQUEMENT sur les transactions saines (y == 0 ou toutes si non supervisé).
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        df_prep = self._prepare_df(X)

        if y is not None:
            y_arr = np.asarray(y)
            # Filtrage strict : transactions légitimes uniquement
            legit_mask = y_arr == 0
            df_legit = df_prep[legit_mask]
            if len(df_legit) == 0:
                df_legit = df_prep  # Fallback de sécurité
        else:
            df_legit = df_prep

        X_train_scaled = self._extract_clean_features(df_legit, is_train=True)
        in_dim = X_train_scaled.shape[1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AutoencoderNet(
            in_features=in_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(device)

        dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32))
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=1e-5
        )
        criterion = nn.MSELoss()

        self.net.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                reconstructed, _ = self.net(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

        # Calcul des statistiques d'erreur de reconstruction de référence sur la classe normale
        self.net.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
            reconstructed, _ = self.net(x_tensor)
            # Erreur MSE par transaction
            rec_losses = (
                torch.mean((x_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            )

            self.mu_loss_ = float(np.mean(rec_losses))
            self.std_loss_ = float(np.std(rec_losses) + 1e-6)
            self.min_loss_ = float(np.percentile(rec_losses, 1))
            self.max_loss_ = float(np.percentile(rec_losses, 99.5) + 1e-6)

        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calcule l'erreur de reconstruction MSE brute pour chaque transaction.
        Une erreur élevée indique une forte divergence par rapport à la distribution normale.
        """
        df_prep = self._prepare_df(X)
        X_scaled = self._extract_clean_features(df_prep, is_train=False)

        device = (
            next(self.net.parameters()).device
            if self.net is not None
            else torch.device("cpu")
        )
        self.net.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            reconstructed, _ = self.net(x_tensor)
            mse_errors = (
                torch.mean((x_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            )
            return mse_errors

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforme l'erreur de reconstruction en score probabiliste [P(légitime), P(fraude)].
        Utilise une fonction sigmoïde paramétrique calibrée sur l'écart z-score à la norme.
        """
        mse_errors = self.decision_function(X)

        # Z-score d'anomalie : à combien d'écarts-types sommes-nous de la moyenne normale
        z_scores = (mse_errors - self.mu_loss_) / self.std_loss_

        # Activation logistique centrée et adoucie : P(fraude) augmente avec l'erreur
        # z_scores = 0 (erreur normale moyenne) -> proba ~ 0.05
        # z_scores = 3 (anomalie 3 sigma) -> proba ~ 0.50
        # z_scores > 6 (très forte anomalie) -> proba ~ 0.90+
        p_fraud = 1.0 / (1.0 + np.exp(-(z_scores - 3.0) / 1.5))
        p_fraud = np.clip(p_fraud, 1e-6, 1.0 - 1e-6)

        p_legit = 1.0 - p_fraud
        return np.column_stack([p_legit, p_fraud])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prédit la classe binaire en fonction du seuil de décision calibré."""
        probas = self.predict_proba(X)[:, 1]
        thresh = getattr(self, "decision_threshold", 0.50)
        return (probas >= thresh).astype(int)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        return getattr(self, "feature_names_out_", [])


# Définition explicite du namespace canonique pour pickle
AutoencoderNet.__module__ = "src.training.autoencoder"
AutoencoderFraudDetector.__module__ = "src.training.autoencoder"


if __name__ == "__main__":
    from src.training.train_autoencoder import main

    main()
