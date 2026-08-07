# src/training/train.py
import os

import mlflow
import ray

# 1. Connexion au cluster Ray local
# En indiquant "auto", Ray se connecte au cluster existant démarré par Docker
ray.init(address="auto", ignore_reinit_error=True)


# 2. Définition de la tâche d'entraînement qui va s'exécuter sur le GPU
@ray.remote(num_gpus=1)  # Indique à Ray de planifier cette tâche sur le worker avec GPU
def train_model_on_gpu(params):
    # Initialisation de MLflow au sein du Worker Ray
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    # Le nom de l'experiment correspond au nom du modèle testé
    model_name = params["model_type"]
    mlflow.set_experiment(model_name)

    # Démarrage du run dans l'expérience du modèle
    with mlflow.start_run(
        run_name=f"Run_{model_name}_depth_{params.get('max_depth', 'default')}"
    ):
        print(f"Début de l'entraînement sur GPU pour {model_name}...")

        # Log des paramètres
        mlflow.log_params(params)

        # Simulation d'entraînement et métriques
        # (À remplacer par votre vrai code d'entraînement)
        accuracy = 0.975 if model_name == "XGBoost" else 0.958
        precision = 0.942 if model_name == "XGBoost" else 0.925

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)

        # Simulation d'enregistrement du code
        mlflow.log_artifact(__file__, artifact_path="model_code")

        return f"Entraînement réussi pour {model_name} (Accuracy: {accuracy})"


def main():
    # Tester différents modèles et paramètres
    configs = [
        {"model_type": "XGBoost", "max_depth": 6, "learning_rate": 0.1},
        {"model_type": "RandomForest", "n_estimators": 100},
        {"model_type": "XGBoost", "max_depth": 10, "learning_rate": 0.05},
    ]

    # Lancement asynchrone sur le cluster Ray
    futures = [train_model_on_gpu.remote(cfg) for cfg in configs]

    # Récupération et affichage des résultats
    results = ray.get(futures)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
