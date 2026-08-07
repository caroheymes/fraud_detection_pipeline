# src/explain/export_rules.py
import json
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import redis
from shapash import SmartExplainer

# 1. Connexion à MLflow et Redis
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

print("--- CONNEXION À REDIS ---")
try:
    r = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
    print("Connexion à Redis réussie.")
except Exception as re:
    print(f"Erreur de connexion à Redis : {re}")
    r = None

# ==========================================================
# 2. CHARGEMENT DU MODÈLE ET DES DONNÉES
# ==========================================================
print("\n--- CHARGEMENT DU MODÈLE CHAMPION ---")
model = None
try:
    model_uri = "models:/fraud_detector@champion"
    model = mlflow.sklearn.load_model(model_uri)
    print("Modèle champion chargé.")
except Exception as e:
    print(f"Impossible de charger via l'alias champion : {e}")
    # Repli sur le dernier run
    try:
        experiment = mlflow.get_experiment_by_name("fraud_detection")
        if experiment is not None:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
            )
            if not runs.empty:
                latest_run_id = runs.iloc[0].run_id
                model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")
                print(f"Modèle chargé depuis le dernier run : {latest_run_id}")
    except Exception as search_err:
        print(f"Échec de la recherche du dernier run : {search_err}")

if model is None:
    print("Erreur : Aucun modèle disponible. Arrêt.")
    exit(1)

print("\n--- CHARGEMENT DES DONNÉES DE RÉFÉRENCE ---")
df_ref = pd.read_csv("src/training/reference_data.csv")
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

# Échantillonnage équilibré
df_normal = df_ref[df_ref["is_fraud"] == 0].sample(n=800, random_state=42)
df_fraud = df_ref[df_ref["is_fraud"] == 1].sample(n=200, random_state=42)
df_sample = pd.concat([df_normal, df_fraud]).sample(frac=1.0, random_state=42)

X_sample = df_sample[features]
y_sample = df_sample["is_fraud"]

# Extraction du préprocesseur et transformation
preprocessor = model.named_steps["preprocessor"]
predictor = model.named_steps["model"]
X_encoded = preprocessor.transform(X_sample)

# ==========================================================
# 3. CALCUL SHAPASH
# ==========================================================
print("\n--- CALCUL DES CONTRIBUTIONS SHAP ---")
xpl = SmartExplainer(model=predictor)

xpl.compile(x=X_encoded, y_target=y_sample)
shap_contribs = (
    xpl.contributions[1] if isinstance(xpl.contributions, list) else xpl.contributions
)

# ==========================================================
# 4. EXTRACTION ET RETRANSFORMATION DES RÈGLES
# ==========================================================
print("\n--- EXTRACTION DES RÈGLES & RETRANSFORMATION ---")
rules_config = {
    "thresholds": {},
    "suspicious_categories": [],
    "suspicious_hours": [],
    "suspicious_weekdays": [],
    "suspicious_months": [],
}

# A. Extraction des variables continues simples
for col in ["amt", "distance_achat", "age", "city_pop"]:
    if col in X_encoded.columns:
        actual_values = X_encoded[col]
        shap_values = shap_contribs[col]

        # Filtre de significativité (écart-type)
        sig_threshold = shap_values.std()
        suspicious_cases = actual_values[shap_values > sig_threshold]

        if len(suspicious_cases) > 0:
            threshold = float(suspicious_cases.quantile(0.10))
            rules_config["thresholds"][f"{col}_max"] = round(threshold, 2)
            print(
                f"  [Seuil] {col} maximum toléré : {rules_config['thresholds'][f'{col}_max']}"
            )

# B. Extraction des catégories binaires
for col in X_encoded.columns:
    if col.startswith("category_") or col.startswith("gender_"):
        actual_values = X_encoded[col]
        shap_values = shap_contribs[col]

        sig_threshold = shap_values.std() if shap_values.std() > 0.01 else 0.05
        suspicious_cases = actual_values[shap_values > sig_threshold]

        if len(suspicious_cases) > 0 and 1 in suspicious_cases.values:
            # On extrait le nom de la catégorie propre
            clean_name = col.replace("category_", "").replace("gender_", "")
            if col.startswith("category_"):
                rules_config["suspicious_categories"].append(clean_name)
                print(f"  [Catégorie Suspecte] {clean_name}")

# C. Retransformation des variables cycliques (Heures, Jours, Mois)
# Formule inverse de l'encodage cyclique : hour = (arctan2(sin, cos) % 2pi) * 24 / 2pi

# 1) Heures
if "hour_sin" in X_encoded.columns and "hour_cos" in X_encoded.columns:
    sin_vals = X_encoded["hour_sin"]
    cos_vals = X_encoded["hour_cos"]

    # Calcul de l'heure d'origine décimale
    angles = np.arctan2(sin_vals, cos_vals) % (2 * np.pi)
    reconstructed_hours = np.round(angles * 12.0 / np.pi).astype(int) % 24

    # Contribution SHAP cumulée du facteur heure
    total_hour_shap = shap_contribs["hour_sin"] + shap_contribs["hour_cos"]

    # Sélection des heures où la contribution dépasse l'écart-type cumulé
    sig_threshold = total_hour_shap.std()
    suspicious_indices = reconstructed_hours[total_hour_shap > sig_threshold]

    if len(suspicious_indices) > 0:
        # On garde les heures uniques triées
        rules_config["suspicious_hours"] = sorted(
            list(map(int, np.unique(suspicious_indices)))
        )
        print(f"  [Heures Suspectes] : {rules_config['suspicious_hours']}")

# 2) Jour de la semaine (0 = Lundi, 6 = Dimanche)
if "weekday_sin" in X_encoded.columns and "weekday_cos" in X_encoded.columns:
    sin_vals = X_encoded["weekday_sin"]
    cos_vals = X_encoded["weekday_cos"]

    angles = np.arctan2(sin_vals, cos_vals) % (2 * np.pi)
    reconstructed_weekdays = np.round(angles * 3.5 / np.pi).astype(int) % 7

    total_weekday_shap = shap_contribs["weekday_sin"] + shap_contribs["weekday_cos"]
    sig_threshold = total_weekday_shap.std()
    suspicious_indices = reconstructed_weekdays[total_weekday_shap > sig_threshold]

    if len(suspicious_indices) > 0:
        rules_config["suspicious_weekdays"] = sorted(
            list(map(int, np.unique(suspicious_indices)))
        )
        print(
            f"  [Jours Suspectes (0=Lundi, 6=Dim)] : {rules_config['suspicious_weekdays']}"
        )

# 3) Mois (1 = Janvier, 12 = Décembre)
if "month_sin" in X_encoded.columns and "month_cos" in X_encoded.columns:
    sin_vals = X_encoded["month_sin"]
    cos_vals = X_encoded["month_cos"]

    angles = np.arctan2(sin_vals, cos_vals) % (2 * np.pi)
    reconstructed_months = np.round(angles * 6.0 / np.pi).astype(int)
    reconstructed_months = np.where(reconstructed_months == 0, 12, reconstructed_months)

    total_month_shap = shap_contribs["month_sin"] + shap_contribs["month_cos"]
    sig_threshold = total_month_shap.std()
    suspicious_indices = reconstructed_months[total_month_shap > sig_threshold]

    if len(suspicious_indices) > 0:
        rules_config["suspicious_months"] = sorted(
            list(map(int, np.unique(suspicious_indices)))
        )
        print(f"  [Mois Suspects] : {rules_config['suspicious_months']}")

# ==========================================================
# 5. ÉCRITURE DANS REDIS ET FICHIER JSON LOCAL
# ==========================================================
# A. Sauvegarde dans le fichier JSON local dans le dossier perso/
try:
    json_path = "perso/fraud_rules_config.json"
    # S'assurer que le répertoire parent existe (même s'il existe déjà)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(rules_config, f, indent=2)
    print(f"\n[Fichier Local] Configuration sauvegardée dans : {json_path}")
except Exception as je:
    print(f"\n[ERREUR] Échec de la sauvegarde du fichier JSON local : {je}")

# B. Sauvegarde dans Redis
if r is not None:
    print("\n--- ÉCRITURE DANS REDIS ---")
    redis_key = "fraud_rules:config"
    r.set(redis_key, json.dumps(rules_config))
    print(
        f"Seuils et règles enregistrés dans Redis avec succès sous la clé '{redis_key}' !"
    )

    # Lecture de vérification
    val = r.get(redis_key)
    print("Vérification Redis :", val)
else:
    print("\n[ERREUR] Impossible d'écrire dans Redis car la connexion a échoué.")
