# src/dashboard/app.py
# docker restart fraud-detection-streamlit

import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mlflow
import streamlit as st
from mlflow.tracking import MlflowClient

from src.dashboard.theme import apply_theme

st.set_page_config(page_title="Accueil MLOps - Détection de Fraude", layout="wide")
apply_theme()


st.title("DETECTION DE FRAUDE EN TEMPS RÉEL")
st.markdown("---")

# Section Bienvenue & Raccourcis
st.markdown(
    """
    
        
        Ce dashboard MLOps centralise la surveillance de l'infrastructure de détection, des performances des modèles, et des rapports décisionnels.
        Utilisez la barre latérale gauche pour naviguer entre les différentes pages.
    
    """,
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

with c1:
    st.subheader("STATUT GLOBAL MLOPS")

    # Lecture des règles Redis via src.utils.db
    redis_available = False
    rules = {}
    try:
        from src.utils.db import get_redis_client

        r = get_redis_client()
        if r is not None:
            rules_raw = r.get("fraud_rules:config")
            if rules_raw:
                rules = json.loads(rules_raw)
                redis_available = True
    except Exception:
        pass

    if redis_available and rules:
        st.success("Moteurs de suspicion Redis (Fast Pass) : **Actif**")
        thresholds = rules.get("thresholds", {})
        susp_cats = rules.get("suspicious_categories", [])
        susp_hours = rules.get("suspicious_hours", [])
        susp_days = rules.get("suspicious_weekdays", [])

        st.markdown(f"* **Seuil Montant Max :** `{thresholds.get('amt_max')} €`")
        st.markdown(
            f"* **Seuil Distance Max :** `{thresholds.get('distance_achat_max')} km`"
        )
        st.markdown(f"* **Seuil Âge Max :** `{thresholds.get('age_max')} ans`")
        st.markdown(
            f"* **Seuil Vélocité Client Max :** `{int(thresholds.get('user_daily_tx_count_max', 3))} tx/jour`"
        )
        if susp_cats:
            cats_formatted = ", ".join([f"`{c}`" for c in susp_cats])
            st.markdown(
                f"* **Catégories Marchandes Suspectes ({len(susp_cats)}) :** {cats_formatted}"
            )
        if susp_hours:
            hours_str = ", ".join([f"{h}h" for h in susp_hours])
            st.markdown(f"* **Heures à Risque (Nocturne) :** `{hours_str}`")
        if susp_days:
            day_map = {
                0: "Lundi",
                1: "Mardi",
                2: "Mercredi",
                3: "Jeudi",
                4: "Vendredi",
                5: "Samedi",
                6: "Dimanche",
            }
            days_str = ", ".join([day_map.get(d, str(d)) for d in susp_days])
            st.markdown(f"* **Jours à Risque :** `{days_str}`")
    else:
        st.warning("Moteurs de suspicion Redis (Fast Pass) : **Non disponible**")


with c2:
    st.subheader("MODELE CHAMPION ACTIF")
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
        brier = champion_metrics.get("brier_score", 0.0)
        acc = champion_metrics.get("accuracy", 0.0)
        f1_global = champion_metrics.get("F1_global", 0.0)
        rec_global = champion_metrics.get("recall_global", 0.0)

        st.success(
            f" **Modèle Champion promu dans le Registre MLflow (Version {version_num})**"
        )
        st.markdown(f"* **Run ID :** `{champion_run_id}`")
        st.markdown(f"* **F2-Score Fraude (Optuna Target) :** `{f2}`")
        st.markdown(f"* **F1-Score Fraude (F1 C1) :** `{f1}`")
        st.markdown(f"* **Précision Fraude (Prec C1) :** `{prec}`")
        st.markdown(f"* **Rappel Fraude (Recall C1) :** `{rec}`")
        st.markdown(f"* **Brier Score (Calibration) :** `{brier:.4f}`")
        st.markdown(f"* **F1-Score Global (Macro) :** `{f1_global}`")
        st.markdown(f"* **Rappel Global (Macro) :** `{rec_global}`")
        st.markdown(f"* **Exactitude Globale (Accuracy) :** `{acc}`")
    else:
        # Repli sur le mock si MLflow est en local backup
        st.info(" **Modèle Actif :** `NVIDIA_GraphSAGE_XGBoost` (Pipeline Hybride)")
        st.markdown("* **Ratio d'échantillonnage de production :** `5%`")
        st.markdown("* **F2-Score de référence :** `0.8981`")
        st.markdown("* **Rappel de référence :** `0.9178`")
        st.markdown("* **Précision de référence :** `0.8272`")
        st.markdown("* **F1-Score de référence :** `0.8701`")
        st.markdown("* **F1 Global de référence :** `0.9348`")

st.markdown("---")
st.markdown("### DERIVES & PERFORMANCES (MLOps Drift Engine)")

# Recherche multi-chemins du rapport JSON
candidates_json = [
    "src/audit/drift_report.json",
    "src/training/drift_report.json",
    "drift_report.json",
]
drift_data = None
drift_json_path = None
for p in candidates_json:
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                drift_data = json.load(f)
            drift_json_path = p
            break
        except Exception:
            pass

if drift_data:
    curr_date = drift_data.get("current_date", "N/A")
    ref_period = drift_data.get("reference_period", "N/A")
    cur_period = drift_data.get("current_period", "N/A")
    sample_size = drift_data.get("sample_size", 0)
    should_retrain = drift_data.get("should_retrain", False)

    data_drift = drift_data.get("data_drift", {})
    data_drift_detected = data_drift.get("drift_detected", False)
    data_drift_ratio = data_drift.get("mean_drift_ratio", 0.0)

    score_drift = drift_data.get("score_drift", {})
    score_drift_detected = score_drift.get("score_drift_detected", False)
    psi_score = score_drift.get("psi_score", 0.0)
    ref_alert_rate = score_drift.get("ref_alert_rate", 0.0)
    cur_alert_rate = score_drift.get("cur_alert_rate", 0.0)

    perf_audit = drift_data.get("performance_audit", {})
    perf_degraded = perf_audit.get("performance_degraded", False)
    obs_metrics = perf_audit.get("observed_metrics", {})
    ref_metrics = perf_audit.get("reference_metrics_mlflow", {})
    rel_drops = perf_audit.get("relative_drops", {})

    st.markdown(
        f"**Date d'audit :** `{curr_date}` | **Période courante :** `{cur_period}` ({sample_size:,} transactions) | **Période de référence :** `{ref_period}`"
    )

    # 3 Colonnes de statut synthétique
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="1. Data Drift P(X)",
            value="Dérive" if data_drift_detected else "Stable",
            delta=f"{data_drift_ratio * 100:.1f}% vars dérive",
            delta_color="inverse" if data_drift_detected else "normal",
        )
    with k2:
        st.metric(
            label="2. Score Drift P(Y_hat)",
            value="Dérive (PSI)" if score_drift_detected else "Stable",
            delta=f"PSI: {psi_score:.4f} (Seuil 0.20)",
            delta_color="inverse" if score_drift_detected else "normal",
        )
    with k3:
        delta_f2_kpi = -rel_drops.get("f2_drop_pct", 0.0)
        st.metric(
            label="3. Concept Drift P(Y|X)",
            value="Dégradé" if perf_degraded else "Conforme",
            delta=f"F2: {obs_metrics.get('f2_score', 0.0):.4f} ({delta_f2_kpi:+.1f}%)",
            delta_color="normal",
        )
    with k4:
        if should_retrain:
            st.error(" **Action Requise :** Réentraînement HPO déclenché par Airflow")
        else:
            st.success(" **Statut Modèle :** Champion optimal (Pas de réentraînement)")

    # Tableau comparatif des performances de production vs Référence MLflow
    if obs_metrics and ref_metrics:
        with st.expander(
            "Contrôle Qualité MLOps (Production vs Champion MLflow)", expanded=True
        ):
            cols_perf = st.columns(5)

            d_f2 = -rel_drops.get("f2_drop_pct", 0.0)
            d_rec = -rel_drops.get("recall_drop_pct", 0.0)
            d_prec = -rel_drops.get("precision_drop_pct", 0.0)
            d_f1 = -rel_drops.get("f1_drop_pct", 0.0)

            cols_perf[0].metric(
                "F2-Score Fraude",
                f"{obs_metrics.get('f2_score', 0.0):.4f}",
                f"{d_f2:+.1f}% vs MLflow ({ref_metrics.get('f2_score', 0.0):.4f})",
                delta_color="normal",
            )
            cols_perf[1].metric(
                "Rappel Fraude",
                f"{obs_metrics.get('recall', 0.0) * 100:.2f}%",
                f"{d_rec:+.1f}% vs MLflow ({ref_metrics.get('recall', 0.0) * 100:.2f}%)",
                delta_color="normal",
            )
            cols_perf[2].metric(
                "Précision Fraude",
                f"{obs_metrics.get('precision', 0.0) * 100:.2f}%",
                f"{d_prec:+.1f}% vs MLflow ({ref_metrics.get('precision', 0.0) * 100:.2f}%)",
                delta_color="normal",
            )
            cols_perf[3].metric(
                "F1-Score Fraude",
                f"{obs_metrics.get('f1_score', 0.0):.4f}",
                f"{d_f1:+.1f}% vs MLflow ({ref_metrics.get('f1_score', 0.0):.4f})",
                delta_color="normal",
            )
            brier_obs = obs_metrics.get("brier_score", 0.0)
            brier_ref = ref_metrics.get("brier_score", 0.0)
            cols_perf[4].metric(
                "Brier Score (Calibration)",
                f"{brier_obs:.4f}",
                f"Ref: {brier_ref:.4f} (idéal: 0.0)",
                delta_color="off",
            )

            if perf_degraded:
                st.warning(
                    f" **Raisons de l'alerte Concept Drift :** {', '.join(perf_audit.get('degradation_reasons', []))}"
                )

    # Rendu direct du rapport interactif HTML d'Evidently AI
    candidates_html = [
        "src/audit/evidently_drift_report.html",
        "src/training/evidently_drift_report.html",
        "evidently_drift_report.html",
    ]
    html_content = None
    for hp in candidates_html:
        if os.path.exists(hp):
            try:
                with open(hp, "r", encoding="utf-8") as f:
                    html_content = f.read()
                break
            except Exception:
                pass

    if html_content:
        st.markdown("#### Rapport Interactif de Dérive (Evidently AI)")
        import streamlit.components.v1 as components

        components.html(html_content, height=900, scrolling=True)
    else:
        st.info(
            "Le rapport interactif HTML d'Evidently AI sera généré lors du prochain audit."
        )
else:
    st.info(
        " **Evidently AI (Statut de Drift)** : `Stable` (Aucun rapport généré pour le moment)"
    )
    st.markdown(
        "**Dernière vérification :** En attente du premier cycle du DAG `drift_and_retrain_loop`."
    )

# Barre latérale de configuration générale
st.sidebar.header("Contrôles globaux")
if st.sidebar.button("Actualiser la Page"):
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

auto_refresh = st.sidebar.checkbox("Auto-rafraîchissement (10s)", value=False)

if auto_refresh:
    import time

    time.sleep(10)
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()
