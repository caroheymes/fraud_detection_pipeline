# src/training/optimize_xgb.py
# docker exec -t fraud-detection-ray-head python src/training/optimize_xgb.py --n-trials 50 --sample-size -1

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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer
from xgboost import XGBClassifier

# Import du socle transverse MLOps
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.api_reloader import reload_serving_api
from src.utils.data_loader import load_dataset
from src.utils.features import BASE_FEATURE_COLUMNS, get_moderate_sampled_data
from src.utils.mlflow_manager import MLflowQualityGate
from src.utils.threshold import evaluate_predictions_and_curves, find_optimal_threshold


def main():
    parser = argparse.ArgumentParser(
        description="Optimisation des hyperparamètres XGBoost avec Optuna"
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Nombre d'essais d'optimisation"
    )
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=0.05,
        help="Ratio d'échantillonnage de la fraude sur le train (ex: 0.05. 0.0 = pas de sampling)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30000,
        help="Taille du jeu de données pour l'optimisation",
    )
    parser.add_argument(
        "--metric-target",
        type=str,
        default="f2",
        choices=["f2", "f1", "auprc", "pr_auc", "recall", "brier", "cost_sensitive"],
        help="Métrique cible pour le Threshold Tuning et le Quality Gate (défaut: f2)",
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

    # Chargement et préparation des données via le chargeur universel
    df = load_dataset(
        sample_size=args.sample_size,
        sample_position=args.sample_position,
        include_graph_ids=False,
    )

    # Variables utilisées pour l'optimisation (socle standardisé unifié)
    features = [c for c in BASE_FEATURE_COLUMNS if c in df.columns]

    X = df[features]
    y = df["is_fraud"]

    # Séparation temporelle train global (80% plus anciens) / test final (20% plus récents)
    split_idx = int(len(X) * 0.8)
    X_train_full = X.iloc[:split_idx].reset_index(drop=True)
    X_test_final = X.iloc[split_idx:].reset_index(drop=True)
    y_train_full = y.iloc[:split_idx].reset_index(drop=True)
    y_test_final = y.iloc[split_idx:].reset_index(drop=True)

    print(
        f"Validation croisée temporelle TimeSeriesSplit (5 Folds Expanding Window) sur {len(X_train_full)} lignes (du passé vers le futur)."
    )
    print(
        f"Métrique cible d'optimisation : {args.metric_target.upper()} avec Threshold Tuning."
    )

    is_minimize = args.metric_target.lower() in ["brier", "brier_score"]
    best_score_so_far = float("inf") if is_minimize else -1.0

    # Définition de la fonction objectif d'Optuna
    def objective(trial):
        nonlocal best_score_so_far
        # Espace de recherche hyperparamètres recalibré sur la zone optimale
        params = {
            # 1. Profondeur augmentée & Apprentissage fin
            "max_depth": trial.suggest_int("max_depth", 7, 12),
            "n_estimators": trial.suggest_int("n_estimators", 180, 400, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            # 2. Régularisation forte (Gamma & L2)
            "gamma": trial.suggest_float("gamma", 3.0, 10.0, step=0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 5.0, log=True),
            # 3. Paramètres consensuels validés
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", 5.0, 10.0, step=0.5
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 8),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.70, 0.85, step=0.05
            ),
            "subsample": 1.0,
            "random_state": 42,
            "tree_method": "hist",
            "eval_metric": "logloss",
        }

        trial_start = datetime.now()
        print(
            f" [ESSAI {trial.number + 1}/{args.n_trials}] "
            f"n_est={params['n_estimators']}, depth={params['max_depth']}, lr={params['learning_rate']:.3f}, scale_pos={params['scale_pos_weight']:.1f}, gamma={params['gamma']:.2f}...",
            end="",
            flush=True,
        )

        # 5-Fold TimeSeriesSplit (Expanding Window)
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_full)):
            # Extraction des données du pli
            X_tr, y_tr = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
            X_val, y_val = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]

            # Sécurité pour les premiers plis sur classes rares
            if y_tr.sum() == 0 or y_val.sum() == 0:
                continue

            # Application du sampling uniquement sur le pli d'entraînement
            if args.sampling_ratio > 0.0:
                X_tr_sampled, y_tr_sampled = get_moderate_sampled_data(
                    X_tr, y_tr, target_ratio=args.sampling_ratio
                )
            else:
                X_tr_sampled, y_tr_sampled = X_tr, y_tr

            # Vectorisation avec TableVectorizer pour le pli
            vectorizer = TableVectorizer()
            X_tr_encoded = vectorizer.fit_transform(X_tr_sampled)
            X_val_encoded = vectorizer.transform(X_val)

            # Entraînement du modèle
            clf = XGBClassifier(**params)
            clf.fit(X_tr_encoded, y_tr_sampled)

            # Inférence probabiliste et calibration du seuil sur le pli de validation
            y_val_probas = clf.predict_proba(X_val_encoded)[:, 1]
            if is_minimize:
                val_metrics, _ = evaluate_predictions_and_curves(y_val, y_val_probas)
                score = val_metrics["brier_score"]
            else:
                _, score = find_optimal_threshold(
                    y_val, y_val_probas, metric_target=args.metric_target
                )
            scores.append(score)

            del clf, vectorizer, X_tr_encoded, X_val_encoded, y_val_probas
            gc.collect()

        mean_score = float(np.mean(scores)) if scores else 0.0
        elapsed = (datetime.now() - trial_start).total_seconds()

        is_new_best = ""
        is_improved = (
            (mean_score < best_score_so_far)
            if is_minimize
            else (mean_score > best_score_so_far)
        )
        if is_improved:
            best_score_so_far = mean_score
            is_new_best = "  [NOUVEAU MEILLEUR SCORE]"

        print(
            f" -> Score {args.metric_target.upper()} = {mean_score:.4f} ({elapsed:.1f}s){is_new_best}",
            flush=True,
        )
        return mean_score

    # Lancement du Run Parent dans MLflow
    parent_run_name = f"Optuna_XGBoost_{args.metric_target.upper()}_{datetime.now().strftime('%m%d_%H%M')}"
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        print(f"Enregistrement du run parent dans MLflow : {parent_run_name}")

        # Log des paramètres généraux de l'étude
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("sampling_ratio_train", args.sampling_ratio)
        mlflow.log_param("optimization_metric", args.metric_target.upper())
        mlflow.log_param(
            "validation_strategy",
            "TimeSeriesSplit_5Fold_ExpandingWindow_ThresholdTuned",
        )

        # Callback pour logger chaque essai dans MLflow comme un sous-run imbriqué
        def mlflow_trial_callback(study, trial):
            with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
                # Log des hyperparamètres testés
                mlflow.log_params(trial.params)
                mlflow.log_metric(f"mean_{args.metric_target}", trial.value)
                mlflow.set_tag("trial_status", str(trial.state))

        # Création et lancement de l'étude Optuna
        study = optuna.create_study(direction="minimize" if is_minimize else "maximize")
        study.optimize(
            objective, n_trials=args.n_trials, callbacks=[mlflow_trial_callback]
        )

        print("\nOptimisation terminée !")
        print(
            f"Meilleur score ({args.metric_target.upper()}) obtenu : {study.best_value:.4f}"
        )
        print("Meilleurs hyperparamètres :")
        for k, v in study.best_params.items():
            print(f"  {k} : {v}")

        # Entraînement final avec les meilleurs hyperparamètres sur tout le train
        print(
            "\nEntraînement final du meilleur modèle sur l'ensemble du jeu d'entraînement..."
        )
        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["tree_method"] = "hist"
        best_params["eval_metric"] = "logloss"

        # Sampling modéré sur l'ensemble du train
        if args.sampling_ratio > 0.0:
            X_train_full_sampled, y_train_full_sampled = get_moderate_sampled_data(
                X_train_full, y_train_full, target_ratio=args.sampling_ratio
            )
        else:
            X_train_full_sampled, y_train_full_sampled = X_train_full, y_train_full

        # Vectorisation finale
        vectorizer = TableVectorizer()
        pipeline = Pipeline(
            [("preprocessor", vectorizer), ("model", XGBClassifier(**best_params))]
        )

        # Entraînement du pipeline sur les données brutes échantillonnées
        pipeline.fit(X_train_full_sampled, y_train_full_sampled)

        # Évaluation finale probabiliste avec Threshold Tuning sur le test set brut
        y_test_probas = pipeline.predict_proba(X_test_final)[:, 1]
        best_thresh, best_score = find_optimal_threshold(
            y_test_final, y_test_probas, metric_target=args.metric_target
        )
        print(
            f"\n Seuil de décision optimal calibré sur {args.metric_target.upper()} : {best_thresh:.4f} (Score: {best_score:.4f})"
        )

        metrics, confusion_dict = evaluate_predictions_and_curves(
            y_test_final, y_test_probas, threshold=best_thresh
        )

        print("\n Métriques finales (avec AUPRC & seuil calibré) :")
        for k, v in metrics.items():
            print(f"  • {k:22s} : {v:.4f}")

        print("\nMatrice de confusion :")
        print(confusion_dict)

        # Utilisation du Quality Gate universel MLOps
        gate = MLflowQualityGate(
            model_name="fraud_detector",
            metric_target=args.metric_target,
        )
        promoted, _ = gate.log_and_evaluate(
            model=pipeline,
            metrics=metrics,
            params={
                **study.best_params,
                "dataset_rows": str(len(df)),
                "model_type": "XGBClassifier",
            },
            tags={"model_type": "XGBClassifier"},
            confusion_matrix_dict=confusion_dict,
            decision_threshold=best_thresh,
            X_test=X_test_final,
            y_test=y_test_final,
        )

        # Rechargement automatique à chaud de l'API si nouveau champion
        if promoted:
            reload_serving_api()
    # Sauvegarde du nouveau jeu de données comme référence d'observabilité
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ref_path = os.path.join(script_dir, "reference_data.csv")
        # df contient les données réelles avec toutes les colonnes calculées (hour_sin, distance_achat, etc.)
        df.to_csv(ref_path, index=False)
        print(f"Jeu de données de référence d'observabilité mis à jour : {ref_path}")
    except Exception as e:
        print(f"Avertissement : Échec de la mise à jour de reference_data.csv : {e}")

    # Mise à jour globale des métadonnées
    try:
        from update_experiment_metadata import main as update_metadata

        update_metadata()
    except Exception as e:
        print(f"Avertissement : Mise à jour des tags échouée : {e}")


if __name__ == "__main__":
    main()
