# src/explain/generate_report.py
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from shapash import SmartExplainer
from sklearn.pipeline import Pipeline

# ==========================================================
# 1. CONFIGURATION DE L'URI DE TRACKING MLFLOW
# ==========================================================
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

print("--- CHARGEMENT DU MODÈLE CHAMPION DEPUIS MLFLOW ---")
model = None

try:
    # On cible en priorité l'alias @champion de notre modèle générique
    model_uri = "models:/fraud_detector@champion"
    model = mlflow.sklearn.load_model(model_uri)
    print("Modèle 'fraud_detector@champion' chargé avec succès !")
except Exception as e:
    print(f"[AVERTISSEMENT] Impossible de charger via l'alias : {e}")

    # REPLI DYNAMIQUE : On cherche le dernier run de l'expérience 'fraud_detection'
    print(
        "\n--- REPLI : RECHERCHE DU DERNIER RUN DE L'EXPÉRIENCE 'fraud_detection' ---"
    )
    try:
        experiment = mlflow.get_experiment_by_name("fraud_detection")
        if experiment is not None:
            # On cherche tous les runs de cette expérience, triés du plus récent au plus ancien
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
            )
            if not runs.empty:
                # On prend le run le plus récent
                latest_run_id = runs.iloc[0].run_id
                print(
                    f"Dernier run détecté : {latest_run_id} (statut : {runs.iloc[0].status})"
                )

                model_uri = f"runs:/{latest_run_id}/model"
                model = mlflow.sklearn.load_model(model_uri)
                print(
                    f"Modèle chargé avec succès depuis le dernier run : {latest_run_id}"
                )
            else:
                print("Aucun run trouvé dans l'expérience 'fraud_detection'.")
        else:
            print("Expérience 'fraud_detection' introuvable dans MLflow.")
    except Exception as search_err:
        print(f"[ERREUR CRITIQUE] Échec de la recherche du dernier run : {search_err}")

# ==========================================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES DE RÉFÉRENCE
# ==========================================================
X_sample, y_sample = None, None
X_encoded = None

if model is not None:
    print("\n--- CHARGEMENT DES DONNÉES DE RÉFÉRENCE ---")
    try:
        # Lecture des données de référence (qui contiennent déjà nos colonnes calculées)
        df_ref = pd.read_csv("src/training/reference_data.csv")

        # Liste des caractéristiques attendues par notre pipeline XGBoost
        features = [
            "category",
            "amt",
            "gender",
            "distance_achat",
            "age",
            "city_pop",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "month_sin",
            "month_cos",
        ]

        # Extraction des données et labels
        X = df_ref[features]
        y = df_ref["is_fraud"]

        # Échantillonnage équilibré pour optimiser le calcul SHAP :
        # On prend 800 transactions normales (0) et 200 frauduleuses (1)
        df_normal = df_ref[df_ref["is_fraud"] == 0].sample(n=800, random_state=42)
        df_fraud = df_ref[df_ref["is_fraud"] == 1].sample(n=200, random_state=42)

        # Concaténation et mélange (shuffling) des lignes
        df_sample = pd.concat([df_normal, df_fraud]).sample(frac=1.0, random_state=42)

        X_sample = df_sample[features]
        y_sample = df_sample["is_fraud"]

        print("Données de référence chargées et échantillonnées.")
        print(
            f"Taille de l'échantillon : {X_sample.shape[0]} lignes ({y_sample.sum()} fraudes détectées)."
        )

    except Exception as e:
        print(f"[ERREUR] Impossible de charger les données de référence : {e}")

# ==========================================================
# 3. CALCUL DES CONTRIBUTIONS SHAPASH & EXTRACTION
# ==========================================================
if X_sample is not None and model is not None:
    print("\n--- CALCUL DES CONTRIBUTIONS SHAPASH ---")
    try:
        # Si le modèle est un Pipeline Scikit-Learn, on extrait le préprocesseur et le modèle XGBoost
        if isinstance(model, Pipeline):
            print(
                "Détection d'un Pipeline Scikit-Learn. Extraction du préprocesseur..."
            )
            preprocessor = model.named_steps["preprocessor"]
            predictor = model.named_steps["model"]

            # Transformation des données en variables purement numériques
            X_encoded = preprocessor.transform(X_sample)

            # Initialisation avec le classifieur final et les données encodées
            # (Grâce au patch appliqué sur SHAP, TreeExplainer va fonctionner nativement !)
            xpl = SmartExplainer(model=predictor)
            xpl.compile(x=X_encoded, y_target=y_sample)
        else:
            # Cas standard si ce n'est pas un pipeline
            X_encoded = X_sample.copy()
            xpl = SmartExplainer(model=model)
            xpl.compile(x=X_sample, y_target=y_sample)
        print("Valeurs SHAP calculées avec succès !")

        # ==========================================================
        # EXTRACTION DES SEUILS AND RÈGLES DEPUIS SHAPASH (TOUTES VARIABLES)
        # ==========================================================
        print("\n--- EXTRACTION DES SEUILS ET RÈGLES (SHAPASH - QUANTILE 10%) ---")

        # DataFrame des contributions SHAP calculées par Shapash
        if isinstance(xpl.contributions, list):
            shap_contribs = (
                xpl.contributions[1]
                if len(xpl.contributions) > 1
                else xpl.contributions[0]
            )
        else:
            shap_contribs = xpl.contributions

        rules_config = {}

        for col in X_encoded.columns:
            actual_values = X_encoded[col]
            shap_values = shap_contribs[col]

            # FILTRE DE SIGNIFICATIVITÉ : Écart-type des contributions
            significant_threshold = shap_values.std()
            if significant_threshold < 0.01:
                significant_threshold = 0.05

            suspicious_cases = actual_values[shap_values > significant_threshold]

            if not suspicious_cases.empty:
                # Vérifier si la variable est binaire
                unique_vals = actual_values.dropna().unique()
                is_binary = len(unique_vals) <= 2 and all(
                    v in [0, 1] for v in unique_vals
                )

                if is_binary:
                    # Règle binaire
                    if 1 in suspicious_cases.values:
                        rules_config[f"has_{col}"] = True
                        print(
                            f"  [RÈGLE BINAIRE] La présence de '{col}' augmente significativement le risque de fraude."
                        )
                else:
                    # Variable continue : on prend le 10ème percentile (quantile 0.10)
                    threshold = float(suspicious_cases.quantile(0.10))
                    rules_config[f"{col}_threshold"] = round(threshold, 4)
                    print(
                        f"  [SEUIL CONTINU] '{col}' suspect de manière significative à partir de : {rules_config[f'{col}_threshold']}"
                    )
            else:
                print(f"  [-] Pas d'impact significatif suspect détecté pour '{col}'.")

        # ==========================================================
        # 4. EXPORTATION DU RAPPORT HTML
        # ==========================================================
        print("\n--- GÉNÉRATION DU RAPPORT HTML ---")
        report_path = "perso/shapash_report.html"
        project_info_path = "src/explain/project_info.yml"

        # Pour éviter un bug interne de Shapash avec la clé 'metrics'
        metrics_config = [
            {"path": "sklearn.metrics.accuracy_score", "name": "Accuracy"}
        ]

        # Génération du rapport
        xpl.generate_report(
            output_file=report_path,
            project_info_file=project_info_path,
            x_train=X_encoded,
            y_train=y_sample,
            y_test=y_sample,
            metrics=metrics_config,
        )
        print(
            f"🎉 SUCCÈS : Rapport Shapash généré avec succès à l'adresse : {report_path}"
        )

    except Exception as e:
        print(f"[ERREUR] Échec de la génération Shapash : {e}")
else:
    print(
        "\n[ERREUR] Impossible de lancer Shapash car le modèle ou les données sont manquants."
    )
