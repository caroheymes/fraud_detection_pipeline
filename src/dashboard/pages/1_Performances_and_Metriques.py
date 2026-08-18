# src/dashboard/pages/1_Performances_and_Metriques.py

import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import pytz
import streamlit as st
import plotly.express as px
from mlflow.tracking import MlflowClient
import mlflow

st.set_page_config(
    page_title="Performances & Métriques", page_icon="📈", layout="wide"
)

st.title("📈 Performances & métriques de production")
st.markdown("---")

# Utilitaires de base de données
def query_db(query):
    import psycopg2
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            database=os.getenv("POSTGRES_DB", "fraud-detection"),
            user=os.getenv("POSTGRES_USER", "fraud-detection"),
            password=os.getenv("POSTGRES_PASSWORD", "fraud-detection_password"),
            port=os.getenv("POSTGRES_PORT", "5432")
        )
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)

# Chargement de la configuration MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
runs_df = pd.DataFrame()
local_backup_loaded = False
champion_run_id = None

script_dir = os.path.dirname(os.path.abspath(__file__))
json_backup_path = os.path.abspath(
    os.path.join(script_dir, "../../training/metrics_all_models.json")
)

try:
    client = MlflowClient()
    try:
        version_details = client.get_model_version_by_alias("fraud_detector", "champion")
        champion_run_id = version_details.run_id
    except Exception:
        pass
    experiments = client.search_experiments()
    exp_ids = [e.experiment_id for e in experiments] if experiments else []
    if exp_ids:
        runs = client.search_runs(experiment_ids=exp_ids)
        if len(runs) > 0:
            rows = []
            for r_run in runs:
                m = r_run.data.metrics
                p = r_run.data.params
                run_name = r_run.info.run_name
                if "prec_class_1" in m:
                    rec = m.get("rec_class_1", 0.0)
                    prec = m.get("prec_class_1", 0.0)
                    f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
                    rows.append({
                        "Model Run": run_name,
                        "Model Type": p.get("model_type", run_name.split("_")[0]),
                        "Ratio (%)": str(int(float(p.get("target_ratio", 0)) * 100)) if p.get("target_ratio") else "N/A",
                        "rec_class_1 (Rappel C1)": rec,
                        "prec_class_1 (Précision C1)": prec,
                        "f1_class_1 (F1 C1)": m.get("f1_class_1", 0.0),
                        "f2_class_1 (F2 C1)": f2,
                        "F1_global (F1 Macro)": m.get("F1_global", 0.0),
                        "recall_global (Rappel Macro)": m.get("recall_global", 0.0),
                        "run_id": r_run.info.run_id,
                    })
            if len(rows) > 0:
                runs_df = pd.DataFrame(rows)
except Exception as e:
    st.sidebar.warning(f"MLflow indisponible (backup local actif) : {e}")

if runs_df.empty and os.path.exists(json_backup_path):
    try:
        with open(json_backup_path, "r") as f:
            local_data = json.load(f)
        rows = []
        for name, m in local_data.items():
            parts = name.split("_")
            model_type = parts[0]
            ratio_str = parts[-1].replace("pc", "") if "pc" in parts[-1] else "N/A"
            rec = m.get("rec_class_1", 0.0)
            prec = m.get("prec_class_1", 0.0)
            f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
            rows.append({
                "Model Run": name,
                "Model Type": model_type,
                "Ratio (%)": ratio_str,
                "rec_class_1 (Rappel C1)": rec,
                "prec_class_1 (Précision C1)": prec,
                "f1_class_1 (F1 C1)": m.get("f1_class_1", 0.0),
                "f2_class_1 (F2 C1)": f2,
                "F1_global (F1 Macro)": m.get("F1_global", 0.0),
                "recall_global (Rappel Macro)": m.get("recall_global", 0.0),
                "confusion_matrix": m.get("confusion_matrix", None),
            })
        runs_df = pd.DataFrame(rows)
        local_backup_loaded = True
    except Exception as e:
        st.error(f"Erreur de lecture du backup JSON local : {e}")

# ==========================================================
# AFFICHAGE DES MÉTRIQUES EN DIRECT
# ==========================================================
total_tx = 0
total_fraud = 0
fraud_rate = 0.0

stats_df, stats_err = query_db("SELECT COUNT(*), SUM(is_fraud::int) FROM silver.rawdata")
if stats_df is not None and not stats_df.empty:
    total_tx = int(stats_df.iloc[0, 0])
    total_fraud = int(stats_df.iloc[0, 1]) if stats_df.iloc[0, 1] is not None else 0
    if total_tx > 0:
        fraud_rate = (total_fraud / total_tx) * 100.0

col_ratio = "Sans"
if not runs_df.empty:
    champion_rows = runs_df[
        runs_df.apply(lambda r: (champion_run_id and r.get("run_id") == champion_run_id) or (local_backup_loaded and "NVIDIA_GraphSAGE_XGBoost" in r.get("Model Run", "")), axis=1)
    ]
    if not champion_rows.empty:
        raw_ratio = champion_rows.iloc[0].get("Ratio (%)", "N/A")
        if raw_ratio != "N/A" and raw_ratio != "None":
            col_ratio = f"{raw_ratio}%"

st.subheader("📊 Métriques globales (historique complet)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Transactions (Base)", f"{total_tx:,}")
with col2:
    st.metric("Transactions Suspectes (Fraudes)", f"{total_fraud:,}")
with col3:
    st.metric("Taux de Fraude Réel", f"{fraud_rate:.4f}%")
with col4:
    st.metric("Échantillonnage Champion", col_ratio, "Ratio en Production")

# ==========================================================
# MÉTRIQUES DU JOUR
# ==========================================================
today_tx = 0
today_fraud = 0
today_rate = 0.0
yesterday_tx = 0
yesterday_fraud = 0
yesterday_rate = 0.0
last_date_str = "N/A"

date_df, _ = query_db("SELECT MAX(trans_date_trans_time::date) FROM silver.rawdata")
if date_df is not None and not date_df.empty and date_df.iloc[0, 0] is not None:
    from datetime import date, timedelta
    last_date = date_df.iloc[0, 0]
    if isinstance(last_date, str):
        from datetime import datetime
        last_date = datetime.strptime(last_date, "%Y-%m-%d").date()
    last_date_str = last_date.strftime("%d/%m/%Y")
    
    yesterday_date = last_date - timedelta(days=1)
    
    daily_query = f"""
        SELECT 
            trans_date_trans_time::date as dt,
            COUNT(*) as tx_count,
            COALESCE(SUM(is_fraud::int), 0) as fraud_count
        FROM silver.rawdata
        WHERE trans_date_trans_time::date IN ('{last_date}', '{yesterday_date}')
        GROUP BY 1
    """
    daily_df, _ = query_db(daily_query)
    if daily_df is not None and not daily_df.empty:
        daily_df["dt"] = pd.to_datetime(daily_df["dt"]).dt.date
        today_row = daily_df[daily_df["dt"] == last_date]
        if not today_row.empty:
            today_tx = int(today_row.iloc[0]["tx_count"])
            today_fraud = int(today_row.iloc[0]["fraud_count"])
            if today_tx > 0:
                today_rate = (today_fraud / today_tx) * 100.0
        
        yesterday_row = daily_df[daily_df["dt"] == yesterday_date]
        if not yesterday_row.empty:
            yesterday_tx = int(yesterday_row.iloc[0]["tx_count"])
            yesterday_fraud = int(yesterday_row.iloc[0]["fraud_count"])
            if yesterday_tx > 0:
                yesterday_rate = (yesterday_fraud / yesterday_tx) * 100.0

st.markdown("---")
st.subheader(f"📅 Métriques du jour : {last_date_str} (dernier jour disponible vs veille)")

col_day1, col_day2, col_day3, col_day4 = st.columns(4)
with col_day1:
    tx_delta = today_tx - yesterday_tx
    st.metric(
        "Transactions (Jour)", 
        f"{today_tx:,}", 
        delta=f"{tx_delta:+,} vs hier" if yesterday_tx > 0 else None
    )
with col_day2:
    fraud_delta = today_fraud - yesterday_fraud
    st.metric(
        "Fraudes Bloquées (Jour)", 
        f"{today_fraud:,}", 
        delta=f"{fraud_delta:+,} vs hier" if yesterday_tx > 0 else None,
        delta_color="inverse"
    )
with col_day3:
    rate_delta = today_rate - yesterday_rate
    st.metric(
        "Taux de Fraude (Jour)", 
        f"{today_rate:.4f}%", 
        delta=f"{rate_delta:+.4f}% vs hier" if yesterday_tx > 0 else None,
        delta_color="inverse"
    )
with col_day4:
    st.metric("Transactions (Veille)", f"{yesterday_tx:,}")

# ==========================================================
# MATRICE DE CONFUSION LIVE
# ==========================================================
st.markdown("---")
st.subheader("📊 Matrice de confusion de production (temps réel)")
st.write("Cette matrice de confusion montre les performances réelles du modèle champion en production basées sur toutes les transactions traitées.")

cm_df, _ = query_db("""
    SELECT 
        COUNT(CASE WHEN is_fraud::int = 0 AND prediction::int = 0 THEN 1 END) as tn,
        COUNT(CASE WHEN is_fraud::int = 0 AND prediction::int = 1 THEN 1 END) as fp,
        COUNT(CASE WHEN is_fraud::int = 1 AND prediction::int = 0 THEN 1 END) as fn,
        COUNT(CASE WHEN is_fraud::int = 1 AND prediction::int = 1 THEN 1 END) as tp
    FROM silver.rawdata
    WHERE prediction IS NOT NULL
""")
if cm_df is not None and not cm_df.empty:
    cm_live = cm_df.iloc[0].to_dict()
    
    c_cm_live1, c_cm_live2 = st.columns([1, 2])
    with c_cm_live1:
        st.markdown("#### Métriques Réelles en Direct")
        live_total = sum(cm_live.values())
        st.write(f"**Total évalué :** {live_total:,} transactions")
        
        live_tp = cm_live["tp"]
        live_fp = cm_live["fp"]
        live_fn = cm_live["fn"]
        live_tn = cm_live["tn"]
        
        live_prec = live_tp / (live_tp + live_fp) if (live_tp + live_fp) > 0 else 0.0
        live_rec = live_tp / (live_tp + live_fn) if (live_tp + live_fn) > 0 else 0.0
        live_f1 = 2 * (live_prec * live_rec) / (live_prec + live_rec) if (live_prec + live_rec) > 0 else 0.0
        live_f2 = 5 * (live_prec * live_rec) / (4 * live_prec + live_rec) if (4 * live_prec + live_rec) > 0 else 0.0
        
        st.write(f"🎯 **Précision en direct :** `{live_prec:.4%}`")
        st.write(f"📈 **Rappel (Recall) en direct :** `{live_rec:.4%}`")
        st.write(f"🏆 **F2-Score en direct :** `{live_f2:.4f}`")
        st.write(f"⚖️ **F1-Score en direct :** `{live_f1:.4f}`")
        
    with c_cm_live2:
        z_live = [[cm_live["tn"], cm_live["fp"]], [cm_live["fn"], cm_live["tp"]]]
        fig_live = px.imshow(
            z_live,
            x=["Sains Prédits (0)", "Fraudes Prédites (1)"],
            y=["Sains Réels (0)", "Fraudes Réelles (1)"],
            color_continuous_scale="Reds",
            text_auto=True,
            title="Matrice de Confusion de Production (En Direct)"
        )
        fig_live.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_live, use_container_width=True)
