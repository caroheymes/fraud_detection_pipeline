# tests/test_threshold.py
import numpy as np

from src.utils.threshold import evaluate_predictions_and_curves, find_optimal_threshold


def test_find_optimal_threshold_f2_and_f1():
    """Vérifie la recherche de seuils optimaux pour F1 et F2 sur un dataset synthétique déséquilibré."""
    # Simulation de probabilités avec 10% de fraudes
    np.random.seed(42)
    y_true = np.zeros(1000, dtype=int)
    y_true[:100] = 1  # 100 fraudes

    # Les fraudes ont des probabilités tirées vers le haut, les non-fraudes vers le bas
    y_probas = np.random.uniform(0.01, 0.40, size=1000)
    y_probas[:100] = np.random.uniform(0.30, 0.95, size=100)

    # Test F2
    thresh_f2, score_f2 = find_optimal_threshold(y_true, y_probas, metric_target="f2")
    assert 0.05 <= thresh_f2 <= 0.95
    assert score_f2 > 0.50

    # Test F1
    thresh_f1, score_f1 = find_optimal_threshold(y_true, y_probas, metric_target="f1")
    assert 0.05 <= thresh_f1 <= 0.95
    assert score_f1 > 0.50

    # En règle générale, le seuil optimal pour F2 est plus bas que pour F1 car F2 favorise le Rappel
    assert thresh_f2 <= thresh_f1 + 0.15


def test_find_optimal_threshold_auprc():
    """Vérifie le fonctionnement pour la métrique AUPRC."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_probas = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.85, 0.4, 0.9])

    thresh, score = find_optimal_threshold(y_true, y_probas, metric_target="auprc")
    assert 0.05 <= thresh <= 0.95
    assert score > 0.60


def test_evaluate_predictions_and_curves():
    """Vérifie le calcul unifié des courbes PR-AUC, ROC-AUC et métriques de confusion."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_probas = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.85, 0.4, 0.9])

    metrics, cm = evaluate_predictions_and_curves(y_true, y_probas, threshold=0.50)

    assert "pr_auc" in metrics
    assert "auprc" in metrics
    assert "roc_auc" in metrics
    assert "f1_class_1" in metrics
    assert "f2_class_1" in metrics
    assert "decision_threshold" in metrics
    assert metrics["decision_threshold"] == 0.50
    assert metrics["auprc"] > 0.70
    assert metrics["roc_auc"] > 0.70

    assert cm["tp"] == 4
    assert cm["tn"] == 4
    assert cm["fp"] == 0
    assert cm["fn"] == 0
