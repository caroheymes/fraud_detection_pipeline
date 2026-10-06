# src/explain/export_rules.py
import json
import os
import sys

# Assurer l'accès aux modules du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from shapash import SmartExplainer

from src.utils.db import get_redis_client
from src.utils.mlflow_manager import load_champion_model

# 1. Connexion à Redis
print("--- CONNEXION À REDIS ---")
r = get_redis_client()
if r is not None:
    print("Connexion à Redis réussie.")
else:
    print("Avertissement : Redis indisponible.")

# ==========================================================
# 2. CHARGEMENT DU MODÈLE ET DES DONNÉES
# ==========================================================
print("\n--- CHARGEMENT DU MODÈLE CHAMPION ---")
model, model_version_id, decision_threshold = load_champion_model(
    model_name="fraud_detector", alias="champion"
)

if model is None:
    print("Erreur : Aucun modèle disponible. Arrêt.")
    sys.exit(1)

print("\n--- CHARGEMENT DES DONNÉES DE RÉFÉRENCE ---")
ref_path = os.path.join(project_root, "src/training/reference_data.csv")
if not os.path.exists(ref_path):
    ref_path = "src/training/reference_data.csv"
df_ref = pd.read_csv(ref_path)

# Échantillonnage représentatif pour l'explicabilité
df_normal = df_ref[df_ref["is_fraud"] == 0].sample(
    n=min(200, len(df_ref[df_ref["is_fraud"] == 0])), random_state=42
)
df_fraud = df_ref[df_ref["is_fraud"] == 1].sample(
    n=min(50, len(df_ref[df_ref["is_fraud"] == 1])), random_state=42
)
df_sample = (
    pd.concat([df_normal, df_fraud])
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)

y_sample = df_sample["is_fraud"]

# Extraction des features et du prédicteur selon le type de modèle
if hasattr(model, "named_steps"):
    # Pipeline classique Scikit-Learn (ex: XGBoost baseline)
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
    X_sample = df_sample[features]
    preprocessor = model.named_steps["preprocessor"]
    predictor = model.named_steps["model"]
    X_trans = preprocessor.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    try:
        col_names = preprocessor.get_feature_names_out()
    except Exception:
        col_names = [f"feat_{i}" for i in range(X_trans.shape[1])]
    X_encoded = pd.DataFrame(X_trans, columns=col_names, index=df_sample.index)
elif hasattr(model, "ae_extractor"):
    # Pipeline Hybride Auto-encodeur + XGBoost
    X_features_arr = model.ae_extractor.transform(df_sample)
    predictor = model.classifier
    try:
        col_names = model.get_feature_names_out()
    except Exception:
        col_names = [f"feat_{i}" for i in range(X_features_arr.shape[1])]
    X_encoded = pd.DataFrame(X_features_arr, columns=col_names, index=df_sample.index)
elif hasattr(model, "hinsage"):
    # Pipeline Inductive GRL (HinSAGE + XGBoost)
    test_embeddings = model.hinsage.transform(df_sample)
    X_features_arr = model._prepare_features(df_sample, test_embeddings)
    predictor = model.classifier
    try:
        col_names = model.get_feature_names_out()
    except Exception:
        col_names = [f"feat_{i}" for i in range(X_features_arr.shape[1])]
    X_encoded = pd.DataFrame(X_features_arr, columns=col_names, index=df_sample.index)
else:
    # Modèle générique avec classifier interne ou direct
    if hasattr(model, "classifier"):
        predictor = model.classifier
    else:
        predictor = model
    try:
        X_trans = model.transform(df_sample)
        col_names = [f"feat_{i}" for i in range(X_trans.shape[1])]
        X_encoded = pd.DataFrame(X_trans, columns=col_names, index=df_sample.index)
    except Exception:
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
        X_encoded = df_sample[[c for c in features if c in df_sample.columns]]

# ==========================================================
# 3. CALCUL SHAPASH
# ==========================================================
print(
    f"\n--- CALCUL DES CONTRIBUTIONS SHAP (Échantillon : {len(df_sample)} transactions) ---"
)
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

# A. Extraction des variables continues simples (valeurs réelles non standardisées)
for col in ["amt", "distance_achat", "age", "city_pop"]:
    if col in X_encoded.columns and col in df_sample.columns:
        actual_values = df_sample[col]
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

# B. Extraction des catégories suspectes
for col in X_encoded.columns:
    if col.startswith("category_"):
        shap_values = shap_contribs[col]
        sig_threshold = shap_values.std() if shap_values.std() > 0.01 else 0.05
        is_active = X_encoded[col] > 0
        if is_active.any():
            category_shap = shap_values[is_active]
            if category_shap.mean() > 0.0:
                clean_name = col.replace("category_", "")
                if clean_name not in rules_config["suspicious_categories"]:
                    rules_config["suspicious_categories"].append(clean_name)
                    print(f"  [Catégorie Suspecte] {clean_name}")

# C. Retransformation des variables cycliques (Heures, Jours, Mois)
# 1) Heures
if "hour_sin" in X_encoded.columns and "hour_cos" in X_encoded.columns:
    sin_vals = X_encoded["hour_sin"]
    cos_vals = X_encoded["hour_cos"]

    angles = np.arctan2(sin_vals, cos_vals) % (2 * np.pi)
    reconstructed_hours = np.round(angles * 12.0 / np.pi).astype(int) % 24

    total_hour_shap = shap_contribs["hour_sin"] + shap_contribs["hour_cos"]
    sig_threshold = total_hour_shap.std()
    suspicious_indices = reconstructed_hours[total_hour_shap > sig_threshold]

    if len(suspicious_indices) > 0:
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
            f"  [Jours Suspects (0=Lundi, 6=Dim)] : {rules_config['suspicious_weekdays']}"
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
# A. Sauvegarde dans le fichier JSON local
try:
    json_path = os.path.join(project_root, "perso/fraud_rules_config.json")
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
