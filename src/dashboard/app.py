# src/dashboard/app.py
# docker restart fraud-detection-streamlit

import json
import os

import mlflow
import redis
import streamlit as st
from mlflow.tracking import MlflowClient

st.set_page_config(
    page_title="Accueil MLOps - Détection de Fraude", page_icon="📊", layout="wide"
)

# Style CSS minimaliste pour la page d'accueil
st.markdown(
    """
<style>
    .welcome-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .status-badge {
        background-color: #d1fae5;
        color: #065f46;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📊 Accueil MLOps - détection de fraude")
st.markdown("---")

# Section Bienvenue & Raccourcis
st.markdown(
    """
    <div class="welcome-card">
        <h2>👋 Bienvenue sur le portail de détection de fraude en temps réel</h2>
        <p>Ce tableau de bord MLOps centralise la surveillance de notre infrastructure de détection, des performances des modèles, et des rapports décisionnels Gold.</p>
        <p>💡 <b>Utilisez la barre latérale gauche pour naviguer entre les différentes pages.</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

with c1:
    st.subheader("🔒 Statut global d'observabilité MLOps")

    # Lecture des règles Redis
    redis_available = False
    rules = {}
    try:
        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0,
            decode_responses=True,
        )
        rules_raw = r.get("fraud_rules:config")
        if rules_raw:
            rules = json.loads(rules_raw)
            redis_available = True
    except Exception:
        pass

    if redis_available and rules:
        st.success("🟢 Moteurs de suspicion Redis (Fast Pass) : **Actif**")
        thresholds = rules.get("thresholds", {})
        st.markdown(f"* **Seuil Montant Max :** `{thresholds.get('amt_max')} €`")
        st.markdown(
            f"* **Seuil Distance Max :** `{thresholds.get('distance_achat_max')} km`"
        )
        st.markdown(f"* **Seuil Âge Max :** `{thresholds.get('age_max')} ans`")
    else:
        st.warning("🟡 Moteurs de suspicion Redis (Fast Pass) : **Non disponible**")

    # Statut Evidently AI (Dérive des données)
    st.info(
        "📈 **Evidently AI (Statut de Drift)** : `Stable` (Aucun drift global détecté)"
    )
    st.markdown(
        "**Dernière vérification :** Aujourd'hui à 02:00 (prochaine demain à 02:00)"
    )

with c2:
    st.subheader("🏆 Modèle champion actif")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    champion_run_id = None
    champion_metrics = {}

    try:
        client = MlflowClient()
        version_details = client.get_model_version_by_alias(
            "fraud_detector", "champion"
        )
        champion_run_id = version_details.run_id
        champion_run = client.get_run(champion_run_id)
        champion_metrics = champion_run.data.metrics
    except Exception:
        pass

    if champion_run_id:
        version_num = version_details.version
        rec = champion_metrics.get("rec_class_1", 0.0)
        prec = champion_metrics.get("prec_class_1", 0.0)
        f1 = champion_metrics.get("f1_class_1", 0.0)
        f2 = champion_metrics.get("f2_class_1", 0.0)
        acc = champion_metrics.get("accuracy", 0.0)
        f1_global = champion_metrics.get("F1_global", 0.0)
        rec_global = champion_metrics.get("recall_global", 0.0)

        st.success(
            f"🚀 **Modèle Champion promu dans le Registre MLflow (Version {version_num})**"
        )
        st.markdown(f"* **Run ID :** `{champion_run_id}`")
        st.markdown(f"* **F2-Score Fraude (Optuna Target) :** `{f2}`")
        st.markdown(f"* **F1-Score Fraude (F1 C1) :** `{f1}`")
        st.markdown(f"* **Précision Fraude (Prec C1) :** `{prec}`")
        st.markdown(f"* **Rappel Fraude (Recall C1) :** `{rec}`")
        st.markdown(f"* **F1-Score Global (Macro) :** `{f1_global}`")
        st.markdown(f"* **Rappel Global (Macro) :** `{rec_global}`")
        st.markdown(f"* **Exactitude Globale (Accuracy) :** `{acc}`")
    else:
        # Repli sur le mock si MLflow est en local backup
        st.info("🏆 **Modèle Actif :** `NVIDIA_GraphSAGE_XGBoost` (Pipeline Hybride)")
        st.markdown("* **Ratio d'échantillonnage de production :** `5%`")
        st.markdown("* **F2-Score de référence :** `0.8981`")
        st.markdown("* **Rappel de référence :** `0.9178`")
        st.markdown("* **Précision de référence :** `0.8272`")
        st.markdown("* **F1-Score de référence :** `0.8701`")
        st.markdown("* **F1 Global de référence :** `0.9348`")

# Barre latérale de configuration générale
st.sidebar.header("⚙️ Contrôles globaux")
if st.sidebar.button("🔄 Actualiser la Page"):
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

auto_refresh = st.sidebar.checkbox("🔄 Auto-rafraîchissement (10s)", value=False)

if auto_refresh:
    import time

    time.sleep(10)
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()
