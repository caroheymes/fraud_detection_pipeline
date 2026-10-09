# src/utils/threshold.py
"""
Module de calibration et d'optimisation du seuil de décision (Threshold Tuning).
Fournit des fonctions universelles pour calibrer le seuil de probabilité selon la métrique
cible (F2, F1, AUPRC / PR-AUC, matrice de coût, contrainte de rappel/précision) et évaluer
les métriques intrinsèques et au seuil.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_optimal_threshold(
    y_true: np.ndarray | pd.Series | list,
    y_probas: np.ndarray | pd.Series | list,
    metric_target: str = "f2",
    beta: float = 2.0,
    min_precision: float = 0.50,
    min_recall: float = 0.50,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
    min_thresh: float = 0.05,
    max_thresh: float = 0.95,
) -> tuple[float, float]:
    """
    Recherche le seuil de décision optimal maximisant la métrique choisie sur les probabilités prédites.

    Args:
        y_true: Vérité terrain binaire (0 ou 1).
        y_probas: Probabilités de la classe positive (1).
        metric_target: Métrique cible ('f2', 'f1', 'fbeta', 'auprc', 'recall_at_precision', 'cost_sensitive', 'balanced').
        beta: Paramètre beta pour F-beta (ex: 2.0 pour F2, 1.0 pour F1).
        min_precision: Précision minimale requise pour 'recall_at_precision'.
        min_recall: Rappel minimal requis pour 'precision_at_recall'.
        cost_fp: Coût unitaire d'un faux positif (pour 'cost_sensitive').
        cost_fn: Coût unitaire d'un faux négatif (pour 'cost_sensitive').
        min_thresh: Borne inférieure du seuil de décision (défaut: 0.05).
        max_thresh: Borne supérieure du seuil de décision (défaut: 0.95).

    Returns:
        Tuple (optimal_threshold: float, best_metric_score: float)
    """
    y_true_np = np.asarray(y_true).astype(int)
    y_probas_np = np.asarray(y_probas).astype(float)

    if len(np.unique(y_true_np)) < 2:
        return 0.50, 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_true_np, y_probas_np)

    # precision_recall_curve renvoie len(thresholds) = len(precisions) - 1
    p = precisions[:-1]
    r = recalls[:-1]
    eps = 1e-10

    metric = metric_target.lower()

    if metric in ["brier", "brier_score"]:
        bs = brier_score_loss(y_true_np, y_probas_np)
        return 0.50, round(float(bs), 4)

    if metric in ["f2", "f2_score"]:
        b = 2.0
        scores = (1 + b**2) * (p * r) / (b**2 * p + r + eps)
    elif metric in ["f1", "f1_score", "auprc", "pr_auc"]:
        # Pour AUPRC, le meilleur point opérationnel sur la courbe PR est souvent l'optimum F1 ou F-beta
        scores = 2 * (p * r) / (p + r + eps)
    elif metric == "fbeta":
        b2 = beta**2
        scores = (1 + b2) * (p * r) / (b2 * p + r + eps)
    elif metric == "recall_at_precision":
        # Maximise le rappel parmi les seuils respectant la précision minimale
        valid = p >= min_precision
        if np.any(valid):
            scores = np.where(valid, r, -1.0)
        else:
            scores = p * r  # Fallback
    elif metric == "precision_at_recall":
        # Maximise la précision parmi les seuils respectant le rappel minimal
        valid = r >= min_recall
        if np.any(valid):
            scores = np.where(valid, p, -1.0)
        else:
            scores = p * r  # Fallback
    elif metric == "cost_sensitive":
        # Minimise le coût total des erreurs : Cost = cost_fp * FP + cost_fn * FN
        # Normalisé en score à maximiser (score = -Cost)
        total_p = np.sum(y_true_np == 1)
        total_n = len(y_true_np) - total_p
        # FP = (1 - Precision) * TP / Precision, TP = Recall * Total_P
        tp = r * total_p
        fp = np.where(p > 0, (1 - p) * tp / (p + eps), total_n)
        fn = total_p - tp
        total_cost = cost_fp * fp + cost_fn * fn
        scores = -total_cost
    else:
        # F2 par défaut pour la détection de fraude
        scores = 5 * (p * r) / (4 * p + r + eps)

    # Filtrage par les bornes [min_thresh, max_thresh]
    valid_mask = (thresholds >= min_thresh) & (thresholds <= max_thresh)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        best_idx = int(np.argmax(scores))
        raw_thresh = float(thresholds[best_idx])
        optimal_thresh = max(min_thresh, min(max_thresh, raw_thresh))
        best_score = float(scores[best_idx])
    else:
        best_valid_idx = valid_indices[int(np.argmax(scores[valid_indices]))]
        optimal_thresh = float(thresholds[best_valid_idx])
        best_score = float(scores[best_valid_idx])

    return round(optimal_thresh, 4), round(best_score, 4)


def evaluate_predictions_and_curves(
    y_true: np.ndarray | pd.Series | list,
    y_probas: np.ndarray | pd.Series | list,
    threshold: float = 0.50,
) -> tuple[dict[str, float], dict[str, int]]:
    """
    Calcule à la fois :
      1. Les métriques globales indépendantes du seuil (PR-AUC / AUPRC, ROC-AUC)
      2. Les métriques de décision opérationnelles au seuil donné (Accuracy, F1, F2, Precision, Recall)
      3. La matrice de confusion (TN, FP, FN, TP)

    Args:
        y_true: Vérité terrain binaire (0 ou 1).
        y_probas: Probabilités de la classe 1 (array 1D).
        threshold: Seuil de décision pour la classification binaire.

    Returns:
        Tuple (metrics_dict: Dict[str, float], confusion_matrix_dict: Dict[str, int])
    """
    y_true_np = np.asarray(y_true).astype(int)
    y_probas_np = np.asarray(y_probas).astype(float)

    # 1. Métriques intrinsèques globales (classement)
    try:
        auprc = float(average_precision_score(y_true_np, y_probas_np))
    except Exception:
        auprc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true_np, y_probas_np))
    except Exception:
        roc_auc = 0.0

    try:
        brier = float(brier_score_loss(y_true_np, y_probas_np))
    except Exception:
        brier = 0.0

    # 2. Prédictions binaires au seuil calibré
    y_pred = (y_probas_np >= threshold).astype(int)

    prec_c1 = float(precision_score(y_true_np, y_pred, pos_label=1, zero_division=0))
    rec_c1 = float(recall_score(y_true_np, y_pred, pos_label=1, zero_division=0))
    f1_c1 = float(f1_score(y_true_np, y_pred, pos_label=1, zero_division=0))
    f2_c1 = float(
        fbeta_score(y_true_np, y_pred, beta=2.0, pos_label=1, zero_division=0)
    )
    f1_glob = float(f1_score(y_true_np, y_pred, average="macro", zero_division=0))
    rec_glob = float(recall_score(y_true_np, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true_np, y_pred))

    metrics = {
        "pr_auc": auprc,
        "auprc": auprc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "accuracy": acc,
        "prec_class_1": prec_c1,
        "rec_class_1": rec_c1,
        "f1_class_1": f1_c1,
        "f2_class_1": f2_c1,
        "F1_global": f1_glob,
        "recall_global": rec_glob,
        "decision_threshold": float(threshold),
    }

    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred, labels=[0, 1]).ravel()
    cm_dict = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    return metrics, cm_dict
