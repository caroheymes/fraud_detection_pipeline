# src/utils/mlflow_manager.py
"""
Gestionnaire universel MLflow & Quality Gate MLOps.
Permet d'évaluer, d'enregistrer et de promouvoir automatiquement n'importe quel modèle (Arbres, GNN, etc.)
avec l'alias '@champion' uniquement si ses performances surpassent le modèle actif en production.
"""

from __future__ import annotations

import json
import os
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


class MLflowQualityGate:
    """
    Contrôleur de qualité et de gouvernance MLOps pour le cycle de vie des modèles.
    """

    def __init__(
        self,
        model_name: str = "fraud_detector",
        metric_target: str = "f2",
        experiment_name: str = "fraud_detection",
        tracking_uri: str | None = None,
        min_precision: float = 0.50,
        max_brier: float = 0.0050,
        min_recall: float = 0.50,
    ):
        self.model_name = model_name
        self.metric_target = metric_target.lower()
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://mlflow:5000"
        )
        self.min_precision = min_precision
        self.max_brier = max_brier
        self.min_recall = min_recall

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def get_target_metric_key(self, available_metrics: dict[str, float]) -> str:
        """Détermine la clé exacte de la métrique cible parmi les métriques calculées."""
        target = self.metric_target.lower()
        if target in ["pr_auc", "auprc"]:
            candidates = ["pr_auc", "auprc", "average_precision", "PR_AUC", "AUPRC"]
        elif target in ["roc_auc", "auc"]:
            candidates = ["roc_auc", "auc", "ROC_AUC"]
        elif target in ["f2", "f2_score"]:
            candidates = ["f2_class_1", "trial_f2_c1", "f2", "f2_score"]
        elif target in ["f1", "f1_score"]:
            candidates = ["f1_class_1", "F1_global", "trial_f1_c1", "f1", "f1_score"]
        elif target in ["recall", "rec"]:
            candidates = [
                "rec_class_1",
                "recall_class_1",
                "recall_global",
                "recall",
                "recall_score",
            ]
        elif target in ["brier", "brier_score"]:
            candidates = ["brier_score", "brier_score_loss", "brier", "Brier_Score"]
        elif target in ["precision", "prec"]:
            candidates = [
                "prec_class_1",
                "precision_class_1",
                "precision",
                "precision_score",
            ]
        else:
            candidates = [target]

        for c in candidates:
            if c in available_metrics:
                return c
        return next(iter(available_metrics.keys()))

    def get_active_champion_info(self) -> dict[str, Any] | None:
        """Interroge le Model Registry pour extraire les métadonnées et le score du Champion actif."""
        try:
            champ = self.client.get_model_version_by_alias(self.model_name, "champion")
            if not champ:
                return None
            run = self.client.get_run(champ.run_id)
            return {
                "version": str(champ.version),
                "run_id": champ.run_id,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "model_type": run.data.params.get("model_type", "Unknown"),
                "decision_threshold": float(
                    run.data.params.get(
                        "decision_threshold",
                        run.data.metrics.get("decision_threshold", 0.50),
                    )
                ),
            }
        except Exception:
            return None

    def evaluate_model_on_dataset(
        self,
        model: Any,
        X_test: Any,
        y_test: Any,
        decision_threshold: float = 0.50,
    ) -> tuple[dict[str, float], dict[str, int]] | None:
        """Évalue un modèle sur un jeu de test donné avec calcul complet des métriques et matrice de confusion."""
        from src.utils.threshold import evaluate_predictions_and_curves

        try:
            import numpy as np
            import pandas as pd

            # Prédiction probabiliste
            if hasattr(model, "predict_proba"):
                try:
                    y_probas = model.predict_proba(X_test)[:, 1]
                except Exception:
                    from src.utils.features import BASE_FEATURE_COLUMNS

                    if isinstance(X_test, pd.DataFrame):
                        avail = [c for c in BASE_FEATURE_COLUMNS if c in X_test.columns]
                        y_probas = model.predict_proba(X_test[avail])[:, 1]
                    else:
                        raise
            elif hasattr(model, "predict"):
                y_probas = model.predict(X_test)
            else:
                return None

            y_true_np = np.asarray(y_test).astype(int)
            metrics, cm = evaluate_predictions_and_curves(
                y_true_np, y_probas, threshold=decision_threshold
            )
            return metrics, cm
        except Exception as e:
            print(
                f"[INFO] [QualityGate Side-by-Side] Impossible d'evaluer le modele sur le test set : {e}"
            )
            return None

    def log_and_evaluate(
        self,
        model: Any,
        metrics: dict[str, float],
        params: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        confusion_matrix_dict: dict[str, Any] | None = None,
        decision_threshold: float | None = None,
        X_test: Any = None,
        y_test: Any = None,
    ) -> tuple[bool, str]:
        """
        Enregistre le run MLflow, publie le modele dans le registre et applique le Quality Gate.
        Effectue une comparaison Side-by-Side si X_test et y_test sont fournis.

        Retourne :
            (should_promote: bool, target_version: str)
        """
        # 0. Gestion explicite du Run MLflow actif ou creation d'un Run nomme explicite
        from datetime import datetime

        active_run = mlflow.active_run()
        created_local_run = False
        if active_run is None:
            model_type_str = (
                params.get("model_type")
                or (tags.get("model_type") if tags else None)
                or self.model_name
            )
            explicit_name = f"Champion_Candidate_{model_type_str}_{self.metric_target.upper()}_{datetime.now().strftime('%m%d_%H%M%S')}"
            mlflow.start_run(run_name=explicit_name)
            created_local_run = True
        else:
            if tags and "model_type" in tags:
                mlflow.set_tag("model_type", tags["model_type"])

        # 1. Logging des parametres, metriques et tags
        if params is None:
            params = {}
        if decision_threshold is not None:
            params["decision_threshold"] = str(round(float(decision_threshold), 4))
            metrics["decision_threshold"] = float(decision_threshold)
        elif "decision_threshold" in metrics and "decision_threshold" not in params:
            params["decision_threshold"] = str(
                round(float(metrics["decision_threshold"]), 4)
            )

        if params:
            mlflow.log_params(params)
        mlflow.log_param("optimization_metric_target", self.metric_target.upper())

        # Nettoyage et securisation des metriques (elimination des NaN/Inf pour PostgreSQL)
        import numpy as np

        clean_metrics = {}
        for k, v in metrics.items():
            try:
                val = float(v)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                clean_metrics[k] = val
            except (ValueError, TypeError):
                pass
        mlflow.log_metrics(clean_metrics)

        if tags:
            mlflow.set_tags(tags)

        # 2. Sauvegarde de la matrice de confusion en artefact
        if confusion_matrix_dict:
            temp_json = f"confusion_matrix_{self.metric_target}.json"
            with open(temp_json, "w") as f:
                json.dump(confusion_matrix_dict, f, indent=4)
            mlflow.log_artifact(temp_json)
            if os.path.exists(temp_json):
                os.remove(temp_json)

        # 3. Sauvegarde du modele dans le Model Registry
        print(f"\n[INFO] Enregistrement du modele dans MLflow ('{self.model_name}')...")
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            serialization_format="pickle",
            registered_model_name=self.model_name,
        )

        # 4. Identification de la nouvelle version creee
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        target_version = str(max(versions, key=lambda v: int(v.version)).version)

        # 5. Controle Qualite MLOps (Challenger vs Champion - Side-by-Side ou Historique)
        target_key = self.get_target_metric_key(metrics)
        candidate_score = metrics.get(target_key, 0.0)

        champion_info = self.get_active_champion_info()
        champion_side_by_side_score = None
        champion_hist_score = 0.0

        if champion_info:
            champion_hist_score = champion_info["metrics"].get(target_key, 0.0)
            # Evaluation comparative Side-by-Side si le jeu de test est fourni
            if X_test is not None and y_test is not None:
                try:
                    loaded_champ, _, champ_thresh = load_champion_model(
                        model_name=self.model_name,
                        alias="champion",
                        tracking_uri=self.tracking_uri,
                    )
                    if loaded_champ is not None:
                        champ_eval = self.evaluate_model_on_dataset(
                            model=loaded_champ,
                            X_test=X_test,
                            y_test=y_test,
                            decision_threshold=champ_thresh,
                        )
                        if champ_eval is not None:
                            champ_metrics, _ = champ_eval
                            champ_key = self.get_target_metric_key(champ_metrics)
                            champion_side_by_side_score = champ_metrics.get(
                                champ_key, 0.0
                            )
                except Exception as eval_err:
                    print(
                        f"[INFO] [Quality Gate] Evaluation Side-by-Side indisponible ({eval_err}), repli sur le score historique."
                    )

        is_minimize = self.metric_target.lower() in [
            "brier",
            "brier_score",
            "loss",
            "log_loss",
            "mse",
            "mae",
        ]

        if champion_side_by_side_score is not None:
            comparison_mode = "Side-by-Side (Meme jeu de test recent)"
            reference_score = champion_side_by_side_score
        elif champion_info:
            comparison_mode = "Score Historique MLflow (Fallback)"
            reference_score = champion_hist_score
        else:
            comparison_mode = "Premier Deploiement"
            reference_score = float("inf") if is_minimize else 0.0

        print("\n" + "=" * 75)
        print("[QUALITY GATE MLOPS] EVALUATION COMPARATIVE ET REGLE DE PROMOTION")
        print(f"  - Mode de comparaison   : {comparison_mode}")
        print(
            f"  - Metrique cible        : {self.metric_target.upper()} ({target_key})"
        )
        if champion_info:
            if champion_side_by_side_score is not None:
                print(
                    f"  - Champion Actuel (V{champion_info['version']}) sur ce Test Set : {self.metric_target.upper()} = {champion_side_by_side_score:.4f} (Score Hist: {champion_hist_score:.4f})"
                )
            else:
                print(
                    f"  - Champion Actuel (V{champion_info['version']}) Score Historique  : {self.metric_target.upper()} = {champion_hist_score:.4f}"
                )
        else:
            print("  - Champion Actuel        : Aucun champion actif enregistre")
        print(
            f"  - Nouveau Candidat (V{target_version}) sur ce Test Set   : {self.metric_target.upper()} = {candidate_score:.4f}"
        )
        print("=" * 75)

        # 5.1 Verification des garde-fous bancaires d'admissibilite (Hard Constraints)
        cand_prec = float(
            metrics.get("prec_class_1", metrics.get("precision_class_1", 1.0))
        )
        cand_rec = float(metrics.get("rec_class_1", metrics.get("recall_class_1", 1.0)))
        cand_brier = float(
            metrics.get("brier_score", metrics.get("brier_score_loss", 0.0))
        )

        guardrail_failures = []
        if self.min_precision > 0.0 and cand_prec < self.min_precision:
            guardrail_failures.append(
                f"Precision insuffisante ({cand_prec:.2%} < {self.min_precision:.2%})"
            )
        if self.min_recall > 0.0 and cand_rec < self.min_recall:
            guardrail_failures.append(
                f"Rappel insuffisant ({cand_rec:.2%} < {self.min_recall:.2%})"
            )
        if self.max_brier > 0.0 and cand_brier > self.max_brier:
            guardrail_failures.append(
                f"Brier Score trop eleve / calibration divergente ({cand_brier:.4f} > {self.max_brier:.4f})"
            )

        if is_minimize:
            score_improves = (champion_info is None) or (
                candidate_score < reference_score
            )
            comp_sym = "<"
            comp_sym_inv = ">="
        else:
            score_improves = (champion_info is None) or (
                candidate_score > reference_score
            )
            comp_sym = ">"
            comp_sym_inv = "<="

        # Promotion si et seulement si l'objectif principal s'ameliore ET tous les garde-fous sont valides
        should_promote = score_improves and (len(guardrail_failures) == 0)

        if should_promote:
            self.client.set_registered_model_alias(
                name=self.model_name, alias="champion", version=target_version
            )
            print(
                f"\n[PROMOTION REUSSIE] Version {target_version} ({candidate_score:.4f} {comp_sym} {reference_score:.4f}) promue avec l'alias '@champion' !"
            )
            print(
                f"   [GARDE-FOUS] Respectes : Precision={cand_prec:.2%}, Rappel={cand_rec:.2%}, Brier={cand_brier:.4f}."
            )
        else:
            if not score_improves:
                print(
                    f"\n[MODELE NON PROMU] Score candidat ({candidate_score:.4f}) {comp_sym_inv} Score de reference ({reference_score:.4f})."
                )
            else:
                print(
                    f"\n[MODELE NON PROMU] Le score cible s'ameliore ({candidate_score:.4f} {comp_sym} {reference_score:.4f}) mais des garde-fous bancaires sont enfreints :"
                )
                for f_msg in guardrail_failures:
                    print(f"   [ALERTE] {f_msg}")
            if champion_info:
                print(
                    f"   [INFO] L'alias '@champion' est maintenu sur la Version {champion_info['version']}."
                )

        if created_local_run:
            mlflow.end_run()

        return should_promote, target_version

    def list_registered_versions(self) -> list[Any]:
        """Retourne la liste des versions enregistrees pour ce modele."""
        try:
            return self.client.search_model_versions(f"name='{self.model_name}'")
        except Exception as e:
            print(
                f"[ERREUR] Erreur lors de la recherche des versions du modele '{self.model_name}' : {e}"
            )
            return []

    def set_champion_alias(self, version: str) -> bool:
        """Attribue manuellement l'alias @champion a une version specifique (Rollback / Promotion manuelle)."""
        try:
            self.client.set_registered_model_alias(
                self.model_name, "champion", str(version)
            )
            print(
                f"[SUCCES] Alias '@champion' reassigne avec succes a la Version {version} du modele '{self.model_name}' !"
            )
            return True
        except Exception as e:
            print(
                f"[ERREUR] Impossible d'assigner l'alias @champion a la version {version} : {e}"
            )
            return False

    def print_status_table(self):
        """Affiche un tableau clair de toutes les versions enregistrees et du champion actuel."""
        print(f"[INFO] Connexion a MLflow : {self.tracking_uri}")
        versions = self.list_registered_versions()
        if not versions:
            print(
                f"[ALERTE] Aucune version trouvee pour le modele '{self.model_name}'."
            )
            return

        print(
            f"\n Versions enregistrées pour '{self.model_name}' ({len(versions)} trouvées) :"
        )
        print("-" * 80)
        print(f"{'Version':<10} | {'Statut':<12} | {'Aliases':<15} | {'Run ID':<35}")
        print("-" * 80)
        for v in sorted(versions, key=lambda x: int(x.version)):
            aliases_str = ", ".join(v.aliases) if v.aliases else "-"
            print(
                f"V{v.version:<9} | {v.status:<12} | {aliases_str:<15} | {v.run_id:<35}"
            )
        print("-" * 80)

        champ_info = self.get_active_champion_info()
        if champ_info:
            print(
                f"\n Modèle Champion Actuel : Version {champ_info['version']} (Run ID: {champ_info['run_id']})"
            )
            metrics = champ_info.get("metrics", {})
            print(
                f"   • F1 C1      : {metrics.get('f1_class_1', metrics.get('f1_score', 'N/A'))}"
            )
            print(
                f"   • F2 C1      : {metrics.get('f2_class_1', metrics.get('f2_score', 'N/A'))}"
            )
            print(
                f"   • Rappel C1  : {metrics.get('rec_class_1', metrics.get('recall_class_1', 'N/A'))}"
            )
            print(
                f"   • Précision  : {metrics.get('prec_class_1', metrics.get('precision_class_1', 'N/A'))}"
            )
        else:
            print(
                f"\n Aucun modèle n'a actuellement l'alias '@champion' pour '{self.model_name}'."
            )

    def load_champion(
        self, fallback_uri: str | None = None
    ) -> tuple[Any, str | None, float]:
        """Charge le modèle Champion actif associé à cette instance de QualityGate."""
        return load_champion_model(
            model_name=self.model_name,
            alias="champion",
            fallback_uri=fallback_uri,
            tracking_uri=self.tracking_uri,
        )


def load_champion_model(
    model_name: str = "fraud_detector",
    alias: str = "champion",
    fallback_uri: str | None = None,
    tracking_uri: str | None = None,
) -> tuple[Any, str | None, float]:
    """
    Charge le modèle Champion actif depuis MLflow Model Registry de manière agnostique.

    Retourne :
        (model_pipeline, active_model_version_id, decision_threshold: float)
    """
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Bridge de compatibilité pickle pour les modèles sauvegardés sous __main__
    import builtins
    import sys

    main_mod = sys.modules.get("__main__")

    for mod_name, class_names in [
        (
            "src.training.inductive_grl",
            [
                "InductiveGRLPipeline",
                "HinSAGEPyTorchNet",
                "HinSAGERepresentationLearner",
                "FocalLoss",
            ],
        ),
        ("src.training.autoencoder", ["AutoencoderFraudDetector", "AutoencoderNet"]),
        (
            "src.training.autoencoder_xgb",
            ["AutoencoderXGBoostPipeline", "AutoencoderFeatureLearner"],
        ),
        (
            "src.training.optimize_xgb_iforest",
            ["IsolationForestXGBoostPipeline"],
        ),
    ]:
        try:
            mod = __import__(mod_name, fromlist=class_names)
            for cls_name in class_names:
                if hasattr(mod, cls_name):
                    cls_obj = getattr(mod, cls_name)
                    if main_mod is not None:
                        setattr(main_mod, cls_name, cls_obj)
                    setattr(builtins, cls_name, cls_obj)
        except ImportError:
            pass

    try:
        model_uri = f"models:/{model_name}@{alias}"
        loaded = mlflow.sklearn.load_model(model_uri)
        decision_threshold = 0.50
        try:
            version_details = client.get_model_version_by_alias(model_name, alias)
            model_version_id = f"{model_name}_v{version_details.version}"
            if version_details.run_id:
                run_data = client.get_run(version_details.run_id).data
                decision_threshold = float(
                    run_data.params.get(
                        "decision_threshold",
                        run_data.metrics.get("decision_threshold", 0.50),
                    )
                )
        except Exception:
            model_version_id = f"{model_name}@{alias}"

        print(
            f"[MLOps Champion]  Modèle Champion '{model_uri}' chargé avec succès en mémoire : '{model_version_id}' (Seuil calibré: {decision_threshold:.4f})."
        )
        return loaded, model_version_id, decision_threshold
    except Exception as e:
        print(
            f"[MLOps Champion]  Échec chargement champion MLflow ('{model_name}@{alias}') : {e}"
        )
        if fallback_uri:
            try:
                fallback_model = mlflow.sklearn.load_model(fallback_uri)
                fallback_id = (
                    fallback_uri.split("/")[-2]
                    if "runs:/" in fallback_uri
                    else "fallback_model"
                )
                print(f"[MLOps Champion] Modèle de secours chargé ({fallback_id}).")
                return fallback_model, fallback_id, 0.50
            except Exception as fb_err:
                print(f"[MLOps Champion] Erreur critique secours : {fb_err}")
        return None, None, 0.50


def main():
    """Point d'entrée CLI pour inspection et gouvernance du Model Registry."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspection et gestion de l'alias @champion dans MLflow"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="fraud_detector",
        help="Nom du modèle enregistré dans MLflow (défaut: fraud_detector)",
    )
    parser.add_argument(
        "--set-version",
        type=str,
        default=None,
        help="Numéro de version à promouvoir comme @champion (ex: 15)",
    )
    args = parser.parse_args()

    gate = MLflowQualityGate(model_name=args.model_name)
    if args.set_version:
        print(
            f"\n Réassignation de l'alias '@champion' vers la Version {args.set_version}..."
        )
        gate.set_champion_alias(args.set_version)
    gate.print_status_table()


if __name__ == "__main__":
    main()
