# src/training/optimize_xgb_iforest.py
"""
Optimisation des hyperparamètres XGBoost enrichi avec Meta-Feature Isolation Forest (Anomaly Stacking).
Intégration MLOps complète avec MLflow, Threshold Tuning et Quality Gate de promotion.
"""

import argparse
import gc
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import TimeSeriesSplit
from skrub import TableVectorizer
from xgboost import XGBClassifier

# Import du socle transverse MLOps
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.api_reloader import reload_serving_api
from src.utils.data_loader import load_dataset
from src.utils.features import BASE_FEATURE_COLUMNS
from src.utils.mlflow_manager import MLflowQualityGate
from src.utils.threshold import evaluate_predictions_and_curves, find_optimal_threshold


class IsolationForestXGBoostPipeline(BaseEstimator, ClassifierMixin):
    """Pipeline personnalisé combinant TableVectorizer, Isolation Forest Anomaly Stacking et XGBoost."""

    def __init__(
        self,
        n_estimators=110,
        max_depth=7,
        learning_rate=0.024,
        scale_pos_weight=6.0,
        subsample=1.0,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        iso_estimators=150,
        iso_contamination=0.01,
        iso_max_samples=0.8,
        random_state=42,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.iso_estimators = iso_estimators
        self.iso_contamination = iso_contamination
        self.iso_max_samples = iso_max_samples
        self.random_state = random_state
        self.kwargs = kwargs

        self.vectorizer = TableVectorizer()
        self.iso_forest = IsolationForest(
            n_estimators=self.iso_estimators,
            contamination=self.iso_contamination,
            max_samples=self.iso_max_samples,
            random_state=self.random_state,
            n_jobs=2,
        )
        self.xgb_model = None

    def fit(self, X, y):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_vec = self.vectorizer.fit_transform(X_df)
        if hasattr(X_vec, "toarray"):
            X_arr = X_vec.toarray().astype(np.float32)
        elif isinstance(X_vec, pd.DataFrame):
            X_arr = X_vec.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X_vec, dtype=np.float32)

        self.iso_forest.fit(X_arr)
        iso_score = (
            (-self.iso_forest.score_samples(X_arr)).reshape(-1, 1).astype(np.float32)
        )
        X_stacked = np.hstack([X_arr, iso_score])

        self.xgb_model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.scale_pos_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            random_state=self.random_state,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=2,
            **self.kwargs,
        )
        self.xgb_model.fit(X_stacked, y)
        self.classifier = self.xgb_model
        return self

    def _transform_input(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_vec = self.vectorizer.transform(X_df)
        if hasattr(X_vec, "toarray"):
            X_arr = X_vec.toarray().astype(np.float32)
        elif isinstance(X_vec, pd.DataFrame):
            X_arr = X_vec.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X_vec, dtype=np.float32)

        iso_score = (
            (-self.iso_forest.score_samples(X_arr)).reshape(-1, 1).astype(np.float32)
        )
        return np.hstack([X_arr, iso_score])

    def transform(self, X):
        return self._transform_input(X)

    def get_feature_names_out(self):
        try:
            base_names = list(self.vectorizer.get_feature_names_out())
        except Exception:
            base_names = [f"feat_{i}" for i in range(self.xgb_model.n_features_in_ - 1)]
        return base_names + ["iso_forest_anomaly_score"]

    def predict(self, X):
        X_stacked = self._transform_input(X)
        return self.xgb_model.predict(X_stacked)

    def predict_proba(self, X):
        X_stacked = self._transform_input(X)
        return self.xgb_model.predict_proba(X_stacked)


def main():
    parser = argparse.ArgumentParser(
        description="Optimisation XGBoost + Isolation Forest Anomaly Stacking"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Nombre d'essais d'optimisation Optuna",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=-1,
        help="Taille du jeu de données (-1 = toutes les données jusqu'à date max Postgres)",
    )
    parser.add_argument(
        "--metric-target",
        type=str,
        default="f1",
        choices=["f1", "f2", "auprc", "pr_auc", "recall", "brier"],
        help="Métrique cible pour Threshold Tuning et Quality Gate (défaut: f1)",
    )
    parser.add_argument(
        "--sample-position",
        type=str,
        default="last",
        choices=["last", "first", "random"],
        help="Mode d'échantillonnage : 'last' (défaut), 'first' ou 'random'",
    )
    args = parser.parse_args()

    # Configuration MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("fraud_detection")

    df = load_dataset(
        sample_size=args.sample_size,
        sample_position=args.sample_position,
        include_graph_ids=False,
    )

    features = [c for c in BASE_FEATURE_COLUMNS if c in df.columns]

    X = df[features]
    y = df["is_fraud"]

    split_idx = int(len(X) * 0.8)
    X_train_full = X.iloc[:split_idx].reset_index(drop=True)
    X_test_final = X.iloc[split_idx:].reset_index(drop=True)
    y_train_full = y.iloc[:split_idx].reset_index(drop=True)
    y_test_final = y.iloc[split_idx:].reset_index(drop=True)

    print(
        f"Validation croisée TimeSeriesSplit (5 Folds) sur {len(X_train_full):,} lignes Train / {len(X_test_final):,} lignes Test Out-of-Time."
    )
    print(
        f"Métrique cible : {args.metric_target.upper()} avec Isolation Forest Stacking."
    )

    is_minimize = args.metric_target.lower() in ["brier", "brier_score"]
    best_score_so_far = float("inf") if is_minimize else -1.0
    if args.metric_target.lower() in ["brier", "brier_score"]:
        target_metric_label = "Brier Score"
    elif args.metric_target.lower() == "f2":
        target_metric_label = "F2-Score Fraude"
    else:
        target_metric_label = "F1-Score Fraude"

    def objective(trial):
        nonlocal best_score_so_far
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 160, step=10),
            "max_depth": trial.suggest_int("max_depth", 8, 11),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.018, 0.032, log=True
            ),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 25, 60),
            "gamma": trial.suggest_float("gamma", 1.0, 6.0),  # Élagage anti-overfitting
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.5, 4.0
            ),  # Régularisation L2
            "subsample": trial.suggest_float("subsample", 0.75, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.75, 0.95),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 8),
            "iso_estimators": trial.suggest_int("iso_estimators", 50, 100, step=25),
            "iso_contamination": trial.suggest_float(
                "iso_contamination", 0.005, 0.025, log=True
            ),
            "iso_max_samples": trial.suggest_float("iso_max_samples", 0.7, 0.9),
            "random_state": 42,
        }

        tscv = TimeSeriesSplit(n_splits=4)
        scores = []
        val_threshs = []
        last_metrics = {}
        last_cm = {}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_full)):
            X_tr, y_tr = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
            X_val, y_val = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]

            if y_tr.sum() == 0 or y_val.sum() == 0:
                continue

            model = IsolationForestXGBoostPipeline(**params)
            model.fit(X_tr, y_tr)

            y_val_probas = model.predict_proba(X_val)[:, 1]
            t_thresh, score = find_optimal_threshold(
                y_val, y_val_probas, metric_target=args.metric_target
            )
            scores.append(score)
            val_threshs.append(t_thresh)
            m_f, cm_f = evaluate_predictions_and_curves(
                y_val, y_val_probas, threshold=t_thresh
            )
            last_metrics = m_f
            last_cm = cm_f
            del model, y_val_probas
            gc.collect()

        current_score = (
            float(np.mean(scores)) if scores else (float("inf") if is_minimize else 0.0)
        )
        avg_thresh = float(np.mean(val_threshs)) if val_threshs else 0.5

        is_new_best = ""
        if (is_minimize and current_score < best_score_so_far) or (
            not is_minimize and current_score > best_score_so_far
        ):
            best_score_so_far = current_score
            is_new_best = "  [NOUVEAU MEILLEUR SCORE]"

        print(
            f"\n┌──  [ESSAI {trial.number + 1:02d}/{args.n_trials}]{is_new_best} "
            + "─" * max(2, 45 - len(is_new_best)),
            flush=True,
        )
        print(
            f"│  {target_metric_label:15s} : {current_score:.4f} (Seuil Calibré Moyen: {avg_thresh:.4f})",
            flush=True,
        )
        if last_metrics:
            print(
                f"│  Précision C1: {last_metrics['prec_class_1'] * 100:6.2f}% | Rappel C1: {last_metrics['rec_class_1'] * 100:6.2f}% | F1 C1: {last_metrics['f1_class_1']:.4f} | F2 C1: {last_metrics['f2_class_1']:.4f}",
                flush=True,
            )
            print(
                f"│  Matrice Confusion (Fold final) : TP={last_cm['tp']} | FP={last_cm['fp']} | FN={last_cm['fn']} | TN={last_cm['tn']}",
                flush=True,
            )
        print(
            f"│   Params : Depth={params['max_depth']}, Trees={params['n_estimators']}, LR={params['learning_rate']:.4f}, SPW={params['scale_pos_weight']:.2f}, Iso_Est={params['iso_estimators']}",
            flush=True,
        )
        print("└" + "─" * 70, flush=True)

        return current_score

    parent_run_name = f"Optuna_XGB_IForest_{args.metric_target.upper()}_{datetime.now().strftime('%m%d_%H%M')}"
    with mlflow.start_run(run_name=parent_run_name):
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("optimization_metric", args.metric_target.upper())
        mlflow.log_param("model_architecture", "XGBoost_IsolationForest_Stacking")
        mlflow.log_param("dataset_rows", str(len(df)))

        study = optuna.create_study(direction="minimize" if is_minimize else "maximize")
        study.optimize(objective, n_trials=args.n_trials)

        print("\n" + "=" * 70)
        print(
            f" MEILLEUR ESSAI ({args.metric_target.upper()}) : {study.best_value:.4f}"
        )
        print(f"Hyperparamètres optimaux : {study.best_params}")
        print("=" * 70 + "\n")

        best_pipeline = IsolationForestXGBoostPipeline(**study.best_params)
        best_pipeline.fit(X_train_full, y_train_full)

        y_test_probas = best_pipeline.predict_proba(X_test_final)[:, 1]
        best_thresh, _best_score = find_optimal_threshold(
            y_test_final, y_test_probas, metric_target=args.metric_target
        )

        metrics, confusion_dict = evaluate_predictions_and_curves(
            y_test_final, y_test_probas, threshold=best_thresh
        )

        print(
            f"\n RÉSULTATS DU MODÈLE XGB + IFOREST (Seuil optimal: {best_thresh:.4f}) :"
        )
        for k, v in metrics.items():
            print(f"  • {k:22s} : {v:.4f}")

        print("\nMatrice de confusion :")
        print(confusion_dict)

        gate = MLflowQualityGate(
            model_name="fraud_detector",
            metric_target=args.metric_target,
        )
        promoted, _ = gate.log_and_evaluate(
            model=best_pipeline,
            metrics=metrics,
            params={
                **study.best_params,
                "dataset_rows": str(len(df)),
                "model_type": "XGBoost_IsolationForest_Stacking",
            },
            tags={"model_type": "XGBoost_IsolationForest_Stacking"},
            confusion_matrix_dict=confusion_dict,
            decision_threshold=best_thresh,
            X_test=X_test_final,
            y_test=y_test_final,
        )

        if promoted:
            reload_serving_api()

    # Mise à jour des métadonnées MLflow
    try:
        from update_experiment_metadata import main as update_metadata

        update_metadata()
    except Exception:
        pass


if __name__ == "__main__":
    main()
