# src/training/train_autoencoder.py
"""
Script d'entraînement et d'optimisation bayésienne (Optuna) de l'Auto-encodeur Semi-supervisé.
- Optimisation des hyperparamètres (latent_dim, hidden_dim, lr, batch_size, epochs) avec Optuna (--n-trials).
- Entraînement semi-supervisé exclusivement sur les transactions légitimes (classe 0).
- Calibration fine du seuil optimal d'anomalie (Threshold Tuning) sur la métrique cible (F2, F1, AUPRC, etc.).
- Enregistrement MLflow avec suivi hiérarchique des trials et Quality Gate automatique pour le Champion.
- Signal de rechargement automatique de l'API FastAPI.
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
from src.training.autoencoder import AutoencoderFraudDetector
from src.utils.api_reloader import trigger_api_reload
from src.utils.data_loader import load_dataset
from src.utils.mlflow_manager import MLflowQualityGate
from src.utils.threshold import evaluate_predictions_and_curves, find_optimal_threshold


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimisation Bayésienne & Entraînement Auto-encodeur Semi-supervisé"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Nombre d'essais pour l'optimisation bayésienne Optuna (défaut: 20, si 1: entraînement direct)",
    )
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=0.0,
        help="Ratio d'échantillonnage de la fraude (défaut: 0.0)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=-1,
        help="Nombre de transactions à utiliser (-1 pour tout l'historique disponible)",
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
            "recall_at_precision",
            "cost_sensitive",
        ],
        help="Métrique cible pour le Threshold Tuning, Optuna et le Quality Gate (défaut: f2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Nombre d'époques pour l'entraînement direct ou de référence (défaut: 25)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Taille des mini-batches PyTorch (défaut: 256)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=8,
        help="Dimension de l'espace latent (défaut: 8)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Dimension des couches cachées (défaut: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Taux d'apprentissage AdamW (défaut: 0.001)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="fraud_detector",
        help="Nom du modèle enregistré dans MLflow Model Registry",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("🧠 PIPELINE AUTO-ENCODEUR SEMI-SUPERVISÉ (DÉTECTION D'ANOMALIES)")
    print(
        f"   Configuration : n_trials={args.n_trials}, sample_size={args.sample_size}"
    )
    print(f"   Métrique cible : {args.metric_target.upper()}")
    print("=" * 80)

    # 1. Chargement des données unifiées
    df = load_dataset(sample_size=args.sample_size, sample_position="last")
    target_col = "is_fraud" if "is_fraud" in df.columns else "fraud_label"
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible introuvable dans le DataFrame ({df.columns})")

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

    best_params = {
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dropout": 0.1,
    }

    # 3. Optimisation Bayésienne Optuna si n_trials > 1
    if args.n_trials > 1:
        print(
            f"\n🎯 Lancement de l'optimisation bayésienne Optuna ({args.n_trials} essais)..."
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            latent_dim = trial.suggest_categorical("latent_dim", [4, 8, 12, 16, 24])
            hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.25, step=0.05)
            batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
            epochs = trial.suggest_int("epochs", 15, 35, step=5)

            trial_start = datetime.now()
            print(
                f"⏳ [ESSAI {trial.number + 1}/{args.n_trials}] "
                f"latent_dim={latent_dim}, hidden_dim={hidden_dim}, lr={lr:.5f}, batch_size={batch_size}, epochs={epochs}...",
                end="",
                flush=True,
            )

            # 3-Fold Stratified K-Fold CV pour évaluer la capacité de reconstruction anormale
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            fold_scores = []

            for trn_idx, val_idx in skf.split(X_train, y_train):
                X_f_trn, y_f_trn = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
                X_f_val, y_f_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

                ae = AutoencoderFraudDetector(
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    dropout=dropout,
                )
                ae.fit(X_f_trn, y_f_trn)

                val_probas = ae.predict_proba(X_f_val)[:, 1]
                _, score = find_optimal_threshold(
                    y_f_val, val_probas, metric_target=args.metric_target
                )
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
            elapsed = (datetime.now() - trial_start).total_seconds()
            print(
                f" -> Score {args.metric_target.upper()} = {mean_score:.4f} ({elapsed:.1f}s)"
            )
            return mean_score

        parent_run_name = f"Optuna_Autoencoder_{args.metric_target.upper()}_{datetime.now().strftime('%m%d_%H%M')}"
        with mlflow.start_run(run_name=parent_run_name):
            mlflow.log_param("n_trials", args.n_trials)
            mlflow.log_param("optimization_metric", args.metric_target.upper())
            mlflow.log_param("model_type", "Autoencoder_SemiSupervised")

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
            best_params.update(study.best_params)

    # 4. Calibration fine du seuil optimal par validation croisée avec les meilleurs hyperparamètres
    print(
        f"\n🔍 Calibration finale du seuil d'anomalie par CV (Métrique: {args.metric_target.upper()})..."
    )
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_thresholds = []
    cv_scores = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_f_trn, y_f_trn = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        X_f_val, y_f_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        fold_ae = AutoencoderFraudDetector(
            hidden_dim=best_params["hidden_dim"],
            latent_dim=best_params["latent_dim"],
            lr=best_params["lr"],
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            dropout=best_params.get("dropout", 0.1),
        )
        fold_ae.fit(X_f_trn, y_f_trn)

        val_probas = fold_ae.predict_proba(X_f_val)[:, 1]
        t_opt, s_opt = find_optimal_threshold(
            y_f_val, val_probas, metric_target=args.metric_target
        )
        cv_thresholds.append(t_opt)
        cv_scores.append(s_opt)
        print(
            f"   • Fold {fold}/3 : Seuil optimal = {t_opt:.4f} -> Score = {s_opt:.4f}"
        )

    opt_threshold = float(np.median(cv_thresholds))
    print(
        f"\n🎯 Seuil de décision optimal calibré : {opt_threshold:.4f} (Score moyen validation: {np.mean(cv_scores):.4f})"
    )

    # 5. Entraînement final de l'Auto-encodeur sur l'ensemble du jeu d'entraînement sain
    print(
        f"\n🚀 Entraînement final de l'Auto-encodeur ({best_params['epochs']} époques sur transactions saines)..."
    )
    detector = AutoencoderFraudDetector(
        hidden_dim=best_params["hidden_dim"],
        latent_dim=best_params["latent_dim"],
        lr=best_params["lr"],
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        dropout=best_params.get("dropout", 0.1),
        decision_threshold=opt_threshold,
    )
    detector.fit(X_train, y_train)

    # 6. Évaluation complète sur le jeu de test holdout
    print("\n📈 Évaluation sur le jeu de test holdout...")
    y_test_probas = detector.predict_proba(X_test)[:, 1]
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

    # 7. Enregistrement MLflow & Contrôle Qualité MLOps (Champion vs Challenger)
    gate = MLflowQualityGate(
        model_name=args.model_name,
        metric_target=args.metric_target,
    )

    params = {
        "model_type": "Autoencoder_SemiSupervised",
        "n_trials": args.n_trials,
        "epochs": best_params["epochs"],
        "batch_size": best_params["batch_size"],
        "latent_dim": best_params["latent_dim"],
        "hidden_dim": best_params["hidden_dim"],
        "lr": best_params["lr"],
        "dropout": best_params.get("dropout", 0.1),
        "optimization_metric_target": args.metric_target.upper(),
        "decision_threshold": str(round(opt_threshold, 4)),
        "mu_loss_normal": str(round(detector.mu_loss_, 6)),
        "std_loss_normal": str(round(detector.std_loss_, 6)),
    }

    tags = {
        "architecture": "PyTorch Autoencoder Anomaly Detector",
        "paradigme": "Semi-Supervised / Anomaly Detection",
        "metric_target": args.metric_target.upper(),
    }

    promoted, _target_version = gate.log_and_evaluate(
        model=detector,
        metrics=metrics,
        params=params,
        tags=tags,
        confusion_matrix_dict=cm,
        decision_threshold=opt_threshold,
        X_test=X_test,
        y_test=y_test,
    )

    # 8. Hot-reload de l'API si le modèle est promu Champion
    if promoted:
        print(
            "\n🚀 Modèle Auto-encodeur promu Champion ! Déclenchement du Hot-Reload API..."
        )
        trigger_api_reload()

    print("\n🏁 Processus terminé avec succès.")


if __name__ == "__main__":
    main()
