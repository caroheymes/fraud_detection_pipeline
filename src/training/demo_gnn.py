# src/training/demo_gnn.py
"""
Pipeline Inductive Graph Representation Learning (Inductive GRL / HinSAGE + XGBoost).
Optimisation Bayésienne avancée Optuna avec espace de recherche recalibré et cache d'embeddings.

Intégration MLOps :
  - MLflow Expérience : 'fraud_detection'
  - Promotion automatique : Enregistrement 'fraud_detector' avec alias '@champion' via MLflowQualityGate.
  - Rechargement à chaud automatique de l'API de serving.
"""

from __future__ import annotations

import argparse
import gc
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
from xgboost import XGBClassifier

# Import du socle transverse MLOps
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.inductive_grl import (
    HinSAGERepresentationLearner,
    InductiveGRLPipeline,
)
from src.utils.api_reloader import reload_serving_api
from src.utils.data_loader import load_dataset
from src.utils.mlflow_manager import MLflowQualityGate
from src.utils.threshold import evaluate_predictions_and_curves, find_optimal_threshold

# Configuration MLflow
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("fraud_detection")


def get_moderate_sampled_data(
    df: pd.DataFrame, target_ratio: float = 0.05, label_col: str = "fraud_label"
) -> pd.DataFrame:
    """Rééchantillonne modérément les données d'entraînement pour équilibrer le GNN si demandé."""
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


def run_hpo_inductive_grl(
    df: pd.DataFrame,
    n_trials: int = 30,
    metric_target: str = "f2",
    sampling_ratio: float = 0.0,
) -> dict[str, Any]:
    """
    Exécute l'optimisation bayésienne d'hyperparamètres pour l'architecture Inductive GRL.
    Utilise un système de mise en cache des représentations HinSAGE pour accélérer drastiquement les trials.
    """
    target_metric_key = "f1_class_1" if metric_target == "f1" else "f2_class_1"
    target_metric_label = (
        "F1-Score Fraude" if metric_target == "f1" else "F2-Score Fraude"
    )

    df_clean = df.copy().reset_index(drop=True)
    cutoff = round(0.70 * len(df_clean))
    train_data = df_clean.iloc[:cutoff].copy().reset_index(drop=True)
    test_data = df_clean.iloc[cutoff:].copy().reset_index(drop=True)

    if sampling_ratio > 0.0:
        train_data = get_moderate_sampled_data(
            train_data, target_ratio=sampling_ratio, label_col="fraud_label"
        )

    y_train = train_data["fraud_label"].values
    y_test = test_data["fraud_label"].values

    print("=" * 70)
    print("  🚀 PIPELINE INDUCTIVE GRL (HinSAGE + XGBoost) RECALIBRÉ")
    print(f"  • Données Train : {len(train_data):,} lignes (Fraudes: {sum(y_train):,})")
    print(f"  • Données Test  : {len(test_data):,} lignes (Fraudes: {sum(y_test):,})")
    print(f"  • Métrique cible : {target_metric_label} ({target_metric_key})")
    print("=" * 70)

    # Cache des embeddings et features par embedding_size
    feature_cache: dict[
        int, tuple[HinSAGERepresentationLearner, np.ndarray, np.ndarray]
    ] = {}

    def get_or_compute_features(
        emb_size: int,
    ) -> tuple[HinSAGERepresentationLearner, np.ndarray, np.ndarray]:
        if emb_size in feature_cache:
            return feature_cache[emb_size]

        print(
            f"\n🧠 [HinSAGE GNN] Entraînement de l'encodeur de graphe (emb_dim={emb_size}, hidden_dim={emb_size * 2})...",
            flush=True,
        )
        learner = HinSAGERepresentationLearner(
            emb_dim=emb_size,
            hidden_dim=emb_size * 2,
            lr=0.005,
            epochs=8,
        )
        train_emb = learner.fit_transform(train_data, train_data["fraud_label"])
        test_emb = learner.transform(test_data)

        raw_train = learner._extract_clean_features(
            learner._prepare_df(train_data), is_train=False
        )
        raw_test = learner._extract_clean_features(
            learner._prepare_df(test_data), is_train=False
        )

        X_tr = np.hstack([train_emb, raw_train])
        X_te = np.hstack([test_emb, raw_test])

        feature_cache[emb_size] = (learner, X_tr, X_te)
        return feature_cache[emb_size]

    best_score_so_far = -1.0
    best_pipeline_bundle = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_score_so_far, best_pipeline_bundle

        # 1. Hyperparamètres HinSAGE
        embedding_size = trial.suggest_categorical("embedding_size", [12, 16, 24, 32])

        # 2. Hyperparamètres XGBoost Recalibrés
        max_depth = trial.suggest_int("max_depth", 6, 11)
        n_estimators = trial.suggest_int("n_estimators", 180, 320, step=20)
        learning_rate = trial.suggest_float("learning_rate", 0.012, 0.045, log=True)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 2.0, 7.5)
        gamma = trial.suggest_float("gamma", 3.0, 9.0)
        min_child_weight = trial.suggest_int("min_child_weight", 3, 8)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.70, 0.85)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-5, 0.1, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 0.1, 4.0, log=True)

        xgb_params = {
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "scale_pos_weight": scale_pos_weight,
            "gamma": gamma,
            "min_child_weight": min_child_weight,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": 1.0,
            "random_state": 42,
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": 2,
        }

        learner, X_train_comb, X_test_comb = get_or_compute_features(embedding_size)

        clf = XGBClassifier(**xgb_params)
        clf.fit(X_train_comb, y_train)
        preds_proba = clf.predict_proba(X_test_comb)[:, 1]

        optimal_thresh, _ = find_optimal_threshold(
            y_test, preds_proba, metric_target=metric_target
        )
        metrics, cm = evaluate_predictions_and_curves(
            y_test, preds_proba, threshold=optimal_thresh
        )

        current_score = metrics[target_metric_key]

        is_new_best = ""
        if current_score > best_score_so_far:
            best_score_so_far = current_score
            is_new_best = " 🌟 [NOUVEAU MEILLEUR SCORE]"

            # Sauvegarder le bundle complet du meilleur modèle
            best_pipeline_bundle = {
                "learner": learner,
                "clf": clf,
                "embedding_size": embedding_size,
                "xgb_params": xgb_params,
                "metrics": metrics,
                "confusion_matrix": cm,
                "calibrated_threshold": optimal_thresh,
                "predictions_proba": preds_proba,
            }

        print(
            f"┌── 🧪 [ESSAI {trial.number + 1:02d}/{n_trials}]{is_new_best} "
            + "─" * max(2, 45 - len(is_new_best)),
            flush=True,
        )
        print(
            f"│ 🎯 {target_metric_label:15s} : {current_score:.4f} (Seuil Calibré: {optimal_thresh:.4f})",
            flush=True,
        )
        print(
            f"│ 📈 Précision C1: {metrics['prec_class_1'] * 100:6.2f}% | Rappel C1: {metrics['rec_class_1'] * 100:6.2f}% | F1 C1: {metrics['f1_class_1']:.4f} | F2 C1: {metrics['f2_class_1']:.4f}",
            flush=True,
        )
        print(
            f"│ 📊 Matrice Confusion : TP={cm['tp']} | FP={cm['fp']} | FN={cm['fn']} | TN={cm['tn']}",
            flush=True,
        )
        print(
            f"│ ⚙️  Params : Emb={embedding_size}, Depth={max_depth}, Trees={n_estimators}, LR={learning_rate:.4f}, Gamma={gamma:.2f}",
            flush=True,
        )
        print("└" + "─" * 70, flush=True)

        try:
            with mlflow.start_run(
                run_name=f"Trial_{trial.number + 1:02d}_InductiveGRL_{metric_target.upper()}",
                nested=True,
            ):
                mlflow.log_params(
                    {
                        "embedding_size": embedding_size,
                        "target_metric": metric_target,
                        **xgb_params,
                    }
                )
                mlflow.log_metrics(metrics)
        except Exception as e:
            print(f"⚠️ [MLflow] Log du trial échoué : {e}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return current_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")

    parent_run_name = f"InductiveGRL_Study_{metric_target.upper()}_{datetime.now().strftime('%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=parent_run_name):
        mlflow.log_params(
            {
                "target_metric": metric_target.upper(),
                "n_trials": n_trials,
                "dataset_rows": len(df_clean),
                "split_strategy": "Chronological_70_30",
                "model_type": "InductiveGRL_HinSAGE_XGBoost",
            }
        )
        mlflow.set_tag("study_status", "RUNNING")

        study.optimize(objective, n_trials=n_trials)
        mlflow.set_tag("study_status", "FINISHED")

    print("\n" + "=" * 70)
    print(f"🏆 MEILLEUR ESSAI RETENU ({target_metric_label} : {study.best_value:.4f})")
    print(f"Hyperparamètres optimaux : {study.best_params}")
    print("=" * 70 + "\n")

    # Assemblage de l'objet de production InductiveGRLPipeline
    best_bundle = best_pipeline_bundle
    final_pipeline = InductiveGRLPipeline(
        embedding_size=best_bundle["embedding_size"],
        hidden_dim=best_bundle["embedding_size"] * 2,
        epochs=8,
        add_additional_data=True,
        xgb_params=best_bundle["xgb_params"],
    )
    final_pipeline.hinsage = best_bundle["learner"]
    final_pipeline.classifier = best_bundle["clf"]
    final_pipeline.decision_threshold = float(best_bundle["calibrated_threshold"])

    print(
        f"\n📊 RÉSULTATS FINAUX DU CHAMPION CANDIDAT (Optimisé sur {target_metric_label}) :"
    )
    for k, v in best_bundle["metrics"].items():
        print(f"  • {k:22s} : {v:.4f}")
    print("\nMatrice de confusion :")
    print(best_bundle["confusion_matrix"])

    # Évaluation et promotion via Quality Gate
    gate = MLflowQualityGate(
        model_name="fraud_detector",
        metric_target=metric_target,
    )
    promoted, model_version = gate.log_and_evaluate(
        model=final_pipeline,
        metrics=best_bundle["metrics"],
        params={
            **study.best_params,
            "dataset_rows": str(len(df_clean)),
            "split_strategy": "Chronological_70_30",
            "model_type": "InductiveGRL_HinSAGE_XGBoost",
            "decision_threshold": str(best_bundle["calibrated_threshold"]),
            "optimization_metric_target": metric_target.upper(),
        },
        tags={"model_type": "InductiveGRL_HinSAGE_XGBoost"},
        confusion_matrix_dict=best_bundle["confusion_matrix"],
        decision_threshold=best_bundle["calibrated_threshold"],
        X_test=test_data,
        y_test=y_test,
    )

    if promoted:
        reload_serving_api()

    # Mise à jour des métadonnées
    try:
        from update_experiment_metadata import main as update_metadata

        update_metadata()
    except Exception:
        pass

    # Export des métriques en JSON
    metrics_comp_path = os.path.join(
        project_root, "src/training/metrics_gnn_comparison.json"
    )
    comp_data = {}
    if os.path.exists(metrics_comp_path):
        try:
            with open(metrics_comp_path, "r") as f:
                comp_data = json.load(f)
        except Exception:
            comp_data = {}

    comp_data[f"InductiveGRL_Optimized_{metric_target.upper()}"] = {
        **best_bundle["metrics"],
        "confusion_matrix": best_bundle["confusion_matrix"],
        "timestamp": datetime.now().isoformat(),
        "best_params": study.best_params,
    }
    with open(metrics_comp_path, "w") as f:
        json.dump(comp_data, f, indent=4)
    print(f"\n✅ Métriques comparatives exportées dans : {metrics_comp_path}")

    return {
        "pipeline": final_pipeline,
        "study": study,
        "best_params": study.best_params,
        "metrics": best_bundle["metrics"],
        "confusion_matrix": best_bundle["confusion_matrix"],
        "promoted": promoted,
        "model_version": model_version,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Inductive GRL (HinSAGE + XGBoost) avec Espace Recalibré"
    )
    parser.add_argument(
        "--n-trials", type=int, default=30, help="Nombre de trials Optuna (défaut: 30)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=-1,
        help="Taille d'échantillon (-1 = complet)",
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
        default="f2",
        choices=["f1", "f2"],
        help="Métrique cible : 'f2' (recommandé) ou 'f1'",
    )
    parser.add_argument(
        "--sample-position",
        type=str,
        default="last",
        choices=["last", "first"],
        help="Position échantillon : 'last' ou 'first'",
    )
    args = parser.parse_args()

    df = load_dataset(
        sample_size=args.sample_size,
        sample_position=args.sample_position,
        include_graph_ids=True,
    )
    run_hpo_inductive_grl(
        df,
        n_trials=args.n_trials,
        metric_target=args.metric_target,
        sampling_ratio=args.sampling_ratio,
    )


if __name__ == "__main__":
    main()
