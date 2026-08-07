# src/training/update_experiment_metadata.py

import os

from mlflow.tracking import MlflowClient


def main():
    print("--- MISE À JOUR DES DESCRIPTIONS ET TAGS DES EXPÉRIENCES ---")

    # Configuration du client MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Récupération de toutes les expériences
    experiments = client.search_experiments()
    print(f"Trouvé {len(experiments)} expériences dans MLflow.")

    for exp in experiments:
        exp_id = exp.experiment_id
        exp_name = exp.name
        print(f"\nTraitement de l'expérience : '{exp_name}' (ID: {exp_id})")

        # 2. Recherche de toutes les exécutions (runs) de cette expérience
        runs = client.search_runs(experiment_ids=[exp_id])
        if not runs:
            print("  -> Aucun run trouvé.")
            # Description par défaut si vide
            client.set_experiment_tag(
                exp_id, "mlflow.note.content", "Aucun run enregistré pour le moment."
            )
            continue

        print(f"  -> {len(runs)} run(s) trouvé(s). Recherche du meilleur run...")

        # 3. Recherche du meilleur run basé sur le F2-score de la classe 1
        best_run = None
        best_f2 = -1.0
        best_metrics = {}

        for run in runs:
            metrics = run.data.metrics
            f2 = metrics.get("f2_class_1", metrics.get("f2_score", 0.0))

            # Critère de sélection : meilleur F2-score de la classe 1
            if f2 > best_f2:
                best_f2 = f2
                best_run = run
                best_metrics = metrics
            elif f2 == best_f2 and best_run is None:
                best_run = run
                best_metrics = metrics

        if best_run:
            run_name = best_run.info.run_name or best_run.info.run_id[:8]
            f1_val = best_metrics.get(
                "F1_global",
                best_metrics.get("f1_global", best_metrics.get("f1_score", 0.0)),
            )
            rec_glob = best_metrics.get(
                "recall_global", best_metrics.get("recall", 0.0)
            )
            f1_c1 = best_metrics.get("f1_class_1", best_metrics.get("f1_score", 0.0))
            rec_c1 = best_metrics.get(
                "rec_class_1",
                best_metrics.get("recall_class_1", best_metrics.get("recall", 0.0)),
            )
            f2_c1 = best_metrics.get("f2_class_1", best_metrics.get("f2_score", 0.0))
            prec_c1 = best_metrics.get(
                "prec_class_1",
                best_metrics.get(
                    "precision_class_1", best_metrics.get("precision", 0.0)
                ),
            )

            # Formatage de la description (Description column)
            desc_text = (
                f"Meilleur Run : '{run_name}' | "
                f"F1 Global: {f1_val:.4f} | "
                f"Rappel Global: {rec_glob:.4f} | "
                f"F1 C1: {f1_c1:.4f} | "
                f"Rappel C1: {rec_c1:.4f} | "
                f"Précision C1: {prec_c1:.4f} | "
                f"F2 C1: {f2_c1:.4f}"
            )

            # Enregistrement de la description via le tag réservé
            client.set_experiment_tag(exp_id, "mlflow.note.content", desc_text)
            print(f"  -> Description mise à jour : {desc_text}")

            # Enregistrement des 2 tags de synthèse simplifiés demandés par l'utilisateur
            client.set_experiment_tag(exp_id, "F1_global", f"{f1_val:.4f}")
            client.set_experiment_tag(exp_id, "F1_c1", f"{f1_c1:.4f}")
            client.set_experiment_tag(
                exp_id, "F2_c1", f"{f2_c1:.4f}"
            )  # Enregistre le tag F2_c1
            print(
                f"  -> Tags mis à jour : F1_global={f1_val:.4f}, F1_c1={f1_c1:.4f}, F2_c1={f2_c1:.4f}"
            )

    print("\n--- TOUTES LES EXPÉRIENCES ONT ÉTÉ MISES À JOUR ---")


if __name__ == "__main__":
    main()
