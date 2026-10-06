# src/training/optimize_autoencoder_xgb.py
"""
Script d'optimisation bayésienne et d'entraînement pour le pipeline hybride Auto-encodeur + XGBoost.
- L'Auto-encodeur extrait l'erreur de reconstruction (MSE / Anomaly z-score) et les embeddings latents.
- XGBoost combine les features tabulaires brutes et les indicateurs d'anomalie.
- Optimisation conjointe des hyperparamètres via Optuna.
- Threshold Tuning sur la métrique cible (F2, F1, AUPRC) et Quality Gate MLOps.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mlflow
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

# Import des modules transverses du projet
from src.training.autoencoder_xgb import (
    AutoencoderFeatureLearner,
    AutoencoderXGBoostPipeline,
)
from src.utils.api_reloader import reload_serving_api
from src.utils.data_loader import load_dataset
from src.utils.mlflow_manager import MLflowQualityGate
from src.utils.threshold import evaluate_predictions_and_curves, find_optimal_threshold


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimisation Bayésienne du Pipeline Hybride Auto-encodeur + XGBoost"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=25,
        help="Nombre d'essais Optuna (défaut: 25, si 1: entraînement direct)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=-1,
        help="Nombre de transactions (-1 pour tout l'historique disponible)",
    )
    parser.add_argument(
        "--metric-target",
        type=str,
        default="f2",
        choices=[
            "f2",
            "f1",
            "auprc",
            "pr_auc",
            "roc_auc",
            "recall",
            "precision",
            "cost_sensitive",
        ],
        help="Métrique cible pour l'optimisation et le Quality Gate (défaut: f2)",
    )
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=0.0,
        help="Ratio de sur-échantillonnage de la fraude (défaut: 0.0)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="fraud_detector",
        help="Nom du modèle enregistré dans MLflow",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("🚀 PIPELINE HYBRIDE AUTO-ENCODEUR + XGBOOST")
    print(
        f"   Configuration : n_trials={args.n_trials}, sample_size={args.sample_size}"
    )
    print(f"   Métrique cible : {args.metric_target.upper()}")
    print("=" * 80)

    # 1. Chargement des données unifiées
    df = load_dataset(sample_size=args.sample_size, sample_position="last")
    target_col = "is_fraud" if "is_fraud" in df.columns else "fraud_label"
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible introuvable ({df.columns})")

    # 2. Séparation chronologique Train (80%) / Test (20%)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    print(
        f"\n📊 Répartition des données : Train={len(train_df):,} ({y_train.sum():,} fraudes), Test={len(test_df):,} ({y_test.sum():,} fraudes)"
    )

    # 3. Pré-entraînement rapide de l'Auto-encodeur sur le Train pour extraction vectorisée
    print(
        "\n🧠 Phase 1 : Entraînement de l'Auto-encodeur de référence sur transactions saines..."
    )
    ae_extractor = AutoencoderFeatureLearner(
        hidden_dim=64,
        latent_dim=8,
        lr=0.001,
        epochs=15,
        batch_size=512,
        dropout=0.1,
    )
    X_train_enriched = ae_extractor.fit_transform(X_train, y_train)
    X_test_enriched = ae_extractor.transform(X_test)
    print(
        f"✅ Features enrichies générées : {X_train_enriched.shape[1]} dimensions (Tabulaires + MSE + Log-MSE + Z-Score + Embeddings Latents)."
    )

    best_xgb_params = {
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

    # 4. Optimisation Bayésienne Optuna sur les représentations enrichies
    if args.n_trials > 1:
        print(
            f"\n🎯 Phase 2 : Optimisation Bayésienne Optuna ({args.n_trials} essais)..."
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            xgb_p = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 250, step=30),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.02, 0.20, log=True
                ),
                "scale_pos_weight": trial.suggest_float(
                    "scale_pos_weight", 1.5, 12.0, step=0.5
                ),
                "subsample": trial.suggest_float("subsample", 0.70, 1.0, step=0.05),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.70, 1.0, step=0.05
                ),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
                "random_state": 42,
                "tree_method": "hist",
                "eval_metric": "logloss",
            }

            trial_start = datetime.now()
            print(
                f"⏳ [ESSAI {trial.number + 1}/{args.n_trials}] "
                f"n_est={xgb_p['n_estimators']}, depth={xgb_p['max_depth']}, lr={xgb_p['learning_rate']:.3f}, scale_pos={xgb_p['scale_pos_weight']}...",
                end="",
                flush=True,
            )

            # 3-Fold Stratified K-Fold CV sur les features enrichies
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            fold_scores = []

            from xgboost import XGBClassifier

            for trn_idx, val_idx in skf.split(X_train_enriched, y_train):
                X_tr, y_tr = X_train_enriched[trn_idx], y_train.iloc[trn_idx]
                X_va, y_va = X_train_enriched[val_idx], y_train.iloc[val_idx]

                clf = XGBClassifier(**xgb_p)
                clf.fit(X_tr, y_tr)

                y_va_probas = clf.predict_proba(X_va)[:, 1]
                _, s_val = find_optimal_threshold(
                    y_va, y_va_probas, metric_target=args.metric_target
                )
                fold_scores.append(s_val)

            mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
            elapsed = (datetime.now() - trial_start).total_seconds()
            print(
                f" -> Score {args.metric_target.upper()} = {mean_score:.4f} ({elapsed:.1f}s)"
            )
            return mean_score

        parent_run_name = f"Optuna_AE_XGBoost_{args.metric_target.upper()}_{datetime.now().strftime('%m%d_%H%M')}"
        with mlflow.start_run(run_name=parent_run_name):
            mlflow.log_param("n_trials", args.n_trials)
            mlflow.log_param("optimization_metric", args.metric_target.upper())
            mlflow.log_param("model_type", "Autoencoder_Enriched_XGBoost")

            def mlflow_callback(study, trial):
                with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
                    mlflow.log_params(trial.params)
                    mlflow.log_metric(f"mean_{args.metric_target}", trial.value)

            study = optuna.create_study(direction="maximize")
            study.optimize(
                objective, n_trials=args.n_trials, callbacks=[mlflow_callback]
            )

            print("\n" + "=" * 60)
            print(
                f"🏆 Meilleur Score Optuna ({args.metric_target.upper()}) : {study.best_value:.4f}"
            )
            print("🌟 Meilleurs Hyperparamètres :")
            for k, v in study.best_params.items():
                print(f"   • {k}: {v}")
            print("=" * 60)
            best_xgb_params.update(study.best_params)

    # 5. Calibration finale du seuil par CV
    print(
        f"\n🔍 Calibration finale du seuil optimal par CV (Métrique: {args.metric_target.upper()})..."
    )
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_thresholds = []
    from xgboost import XGBClassifier

    for trn_idx, val_idx in skf.split(X_train_enriched, y_train):
        X_tr, y_tr = X_train_enriched[trn_idx], y_train.iloc[trn_idx]
        X_va, y_va = X_train_enriched[val_idx], y_train.iloc[val_idx]

        clf = XGBClassifier(**best_xgb_params)
        clf.fit(X_tr, y_tr)
        val_probas = clf.predict_proba(X_va)[:, 1]
        t_opt, _ = find_optimal_threshold(
            y_va, val_probas, metric_target=args.metric_target
        )
        cv_thresholds.append(t_opt)

    opt_threshold = float(np.median(cv_thresholds))
    print(f"🎯 Seuil de décision optimal calibré : {opt_threshold:.4f}")

    # 6. Construction et entraînement du pipeline complet Scikit-Learn
    print("\n📦 Construction du pipeline hybride sérialisable...")
    pipeline = AutoencoderXGBoostPipeline(
        hidden_dim=64,
        latent_dim=8,
        ae_lr=0.001,
        ae_epochs=15,
        ae_batch_size=512,
        decision_threshold=opt_threshold,
        xgb_params=best_xgb_params,
    )
    pipeline.fit(X_train, y_train)

    # 7. Évaluation sur le jeu de test holdout
    print("\n📈 Évaluation sur le jeu de test holdout...")
    y_test_probas = pipeline.predict_proba(X_test)[:, 1]
    metrics, cm = evaluate_predictions_and_curves(
        y_test, y_test_probas, threshold=opt_threshold
    )

    print("\n" + "-" * 60)
    print("📋 RÉSULTATS SUR LE JEU DE TEST (HOLD-OUT) :")
    print(f"   • Seuil Opérationnel (decision_threshold) : {opt_threshold:.4f}")
    print(f"   • AUPRC / PR-AUC                          : {metrics['auprc']:.4f}")
    print(f"   • ROC-AUC                                 : {metrics['roc_auc']:.4f}")
    print(f"   • F2 Fraude (f2_class_1)                  : {metrics['f2_class_1']:.4f}")
    print(f"   • F1 Fraude (f1_class_1)                  : {metrics['f1_class_1']:.4f}")
    print(
        f"   • Rappel Fraude (rec_class_1)             : {metrics['rec_class_1']:.4f}"
    )
    print(
        f"   • Précision Fraude (prec_class_1)         : {metrics['prec_class_1']:.4f}"
    )
    print(
        f"   • Matrice de confusion                    : TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}"
    )
    print("-" * 60)

    # 8. Contrôle Qualité MLOps & Promotion Champion
    gate = MLflowQualityGate(
        model_name=args.model_name,
        metric_target=args.metric_target,
    )

    params = {
        "model_type": "Autoencoder_Enriched_XGBoost",
        "n_trials": args.n_trials,
        "decision_threshold": str(round(opt_threshold, 4)),
        "optimization_metric_target": args.metric_target.upper(),
        **best_xgb_params,
    }

    tags = {
        "architecture": "Autoencoder (PyTorch) + XGBoost Ensemble",
        "paradigme": "Hybrid Semi-Supervised Anomaly + Gradient Boosting",
        "metric_target": args.metric_target.upper(),
    }

    promoted, _target_version = gate.log_and_evaluate(
        model=pipeline,
        metrics=metrics,
        params=params,
        tags=tags,
        confusion_matrix_dict=cm,
        decision_threshold=opt_threshold,
        X_test=X_test,
        y_test=y_test,
    )

    if promoted:
        print(
            "\n👑 Modèle Hybride Auto-encodeur + XGBoost promu Champion ! Rechargement de l'API..."
        )
        reload_serving_api()

    print("\n🏁 Processus terminé avec succès.")


if __name__ == "__main__":
    main()
