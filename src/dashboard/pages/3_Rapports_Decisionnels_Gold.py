# src/dashboard/pages/3_Rapports_Decisionnels_Gold.py

import os

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from mlflow.tracking import MlflowClient
from shapash import SmartExplainer

st.set_page_config(
    page_title="Rapports Décisionnels Gold (dbt)", page_icon="🥇", layout="wide"
)

st.title("🥇 Couche décisionnelle - rapports Gold dbt")
st.write(
    "Ces rapports analytiques sont générés à partir des schémas Gold de dbt dans PostgreSQL. Ils fournissent un outil complet de suivi commercial, opérationnel et de risques."
)
st.markdown("---")


def query_db(query):
    import psycopg2

    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            database=os.getenv("POSTGRES_DB", "fraud-detection"),
            user=os.getenv("POSTGRES_USER", "fraud-detection"),
            password=os.getenv("POSTGRES_PASSWORD", "fraud-detection_password"),
            port=os.getenv("POSTGRES_PORT", "5432"),
        )
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_champion_explainer_assets():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    try:
        client = MlflowClient()
        version_details = client.get_model_version_by_alias(
            "fraud_detector", "champion"
        )
        run_id = version_details.run_id
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        return model.named_steps["preprocessor"], model.named_steps["model"]
    except Exception:
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name("Default")
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
            )
            if len(runs) > 0:
                model = mlflow.sklearn.load_model(f"runs:/{runs[0].info.run_id}/model")
                return model.named_steps["preprocessor"], model.named_steps["model"]
        except Exception:
            pass
    return None, None


def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


# ==========================================================
# A. CHARGEMENT PRÉALABLE DES DONNÉES ET LISTES DE FILTRES
# ==========================================================
# 1. Requête du nombre total de marchands et liste
total_merchants_df, _ = query_db(
    "select distinct merchant_name from gold.mart_merchant_daily_metrics order by merchant_name"
)
merchant_list = ["Aucun (Afficher tous)"]
if total_merchants_df is not None and not total_merchants_df.empty:
    merchant_list += total_merchants_df["merchant_name"].tolist()
    total_merchants_count = len(total_merchants_df)
else:
    total_merchants_count = 117

# 2. Liste des catégories
total_categories_df, _ = query_db(
    "select distinct transaction_category from gold.mart_merchant_blocked_transactions order by transaction_category"
)
category_list = ["Toutes"]
if total_categories_df is not None and not total_categories_df.empty:
    category_list += total_categories_df["transaction_category"].tolist()

# 3. Requête Pareto (Top 80% Fraude)
pareto_query = """
with merchant_fraud as (
    select
        merchant_name,
        sum(blocked_fraud_volume) as fraud_volume
    from gold.mart_merchant_daily_metrics
    group by 1
),
cumulative_fraud as (
    select
        merchant_name,
        fraud_volume,
        sum(fraud_volume) over (order by fraud_volume desc) as running_cum,
        coalesce(sum(fraud_volume) over (), 0) as total_fraud_volume
    from merchant_fraud
),
pareto_calc as (
    select
        merchant_name,
        fraud_volume,
        round(
            case when total_fraud_volume > 0 then (running_cum / total_fraud_volume) * 100 else 0 end, 
            2
        ) as cum_percentage,
        round(
            case when total_fraud_volume > 0 then (lag(running_cum, 1, 0::numeric) over (order by fraud_volume desc) / total_fraud_volume) * 100 else 0 end, 
            2
        ) as prev_cum_percentage
    from cumulative_fraud
)
select
    merchant_name,
    fraud_volume,
    cum_percentage
from pareto_calc
where prev_cum_percentage < 80.0 and fraud_volume > 0
order by fraud_volume desc
"""
df_pareto, pareto_err = query_db(pareto_query)
pareto_merchants = set()
if df_pareto is not None and not df_pareto.empty:
    pareto_merchants = set(df_pareto["merchant_name"].tolist())

# ==========================================================
# B. CRÉATION DU PANNEAU DE FILTRES DYNAMIQUES
# ==========================================================
st.markdown("### 🔍 Panneau de filtrage général")
c_f1, c_f2, c_f3 = st.columns(3)

with c_f1:
    use_pareto = st.checkbox(
        "🎯 Limiter le périmètre au Top 80% Pareto (Marchands critiques)", value=True
    )
with c_f2:
    selected_merchant = st.selectbox(
        "🔍 Sélectionner un marchand spécifique (Désactive Pareto) :",
        merchant_list,
        index=0,
    )
with c_f3:
    selected_category = st.selectbox(
        "🛍️ Filtrer par catégorie (Transactions bloquées uniquement) :",
        category_list,
        index=0,
    )

# Résolution de la liste des marchands actifs
if selected_merchant != "Aucun (Afficher tous)":
    active_merchants = [selected_merchant]
    filter_description = f"Filtre actif : Marchand spécifique **{selected_merchant}**"
elif use_pareto and len(pareto_merchants) > 0:
    active_merchants = list(pareto_merchants)
    filter_description = f"Filtre actif : **Périmètre Pareto** ({len(active_merchants)} marchands critiques responsables de 80% de la fraude)"
else:
    active_merchants = []
    filter_description = "Filtre actif : **Tous les marchands** (Aucune restriction)"

st.info(f"💡 {filter_description}")
st.markdown("---")

# ----------------- SECTION 1 : PARETO ANALYSIS -----------------
st.markdown("### ⚠️ 1. Analyse de Pareto (Concentration des Risques)")
if pareto_err:
    st.error(f"Impossible de calculer le périmètre de Pareto : {pareto_err}")
else:
    if df_pareto is not None and not df_pareto.empty:
        df_pareto["fraud_volume"] = df_pareto["fraud_volume"].astype(float)
        df_pareto["cum_percentage"] = df_pareto["cum_percentage"].astype(float)

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.metric(
                "Marchands critiques (Cibles)",
                f"{len(df_pareto)} / {total_merchants_count}",
                f"soit {len(df_pareto) / total_merchants_count * 100:.1f}% des marchands",
            )
        with c_p2:
            st.metric(
                "Volume de Fraude Couvert",
                f"{df_pareto['cum_percentage'].max()}%",
                "Objectif de ciblage : > 80%",
            )

        with st.expander("👁️ Voir la liste des marchands critiques (Pareto)"):
            st.dataframe(df_pareto, use_container_width=True)

        fig_p = px.bar(
            df_pareto,
            x="merchant_name",
            y="fraud_volume",
            text="cum_percentage",
            title="Concentration de la Fraude par Marchand Critique (% Cumulé)",
            labels={
                "fraud_volume": "Volume de Fraude (€)",
                "merchant_name": "Marchand",
            },
            color="fraud_volume",
            color_continuous_scale="Oranges",
        )
        fig_p.update_traces(textposition="outside")
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.warning("Aucune donnée de fraude n'est disponible pour l'analyse de Pareto.")

st.markdown("---")

# ----------------- SECTION 2 : PERFORMANCE COMMERCIALE -----------------
st.markdown("### 📈 2. Chiffre d'affaires & taux de fraude")
df_metrics, err_metrics = query_db(
    "SELECT * FROM gold.mart_merchant_daily_metrics ORDER BY transaction_date DESC, total_volume DESC"
)
if err_metrics:
    st.error(
        f"La table gold.mart_merchant_daily_metrics n'est pas disponible : {err_metrics}"
    )
else:
    if df_metrics is not None and not df_metrics.empty:
        if active_merchants:
            df_metrics = df_metrics[df_metrics["merchant_name"].isin(active_merchants)]

        if df_metrics.empty:
            st.warning(
                "Aucun résultat ne correspond aux filtres de marchand sélectionnés pour ce rapport."
            )
        else:
            df_metrics["clean_volume"] = df_metrics["clean_volume"].astype(float)
            df_metrics["blocked_fraud_volume"] = df_metrics[
                "blocked_fraud_volume"
            ].astype(float)
            df_metrics["total_volume"] = df_metrics["total_volume"].astype(float)

            df_grouped = (
                df_metrics.groupby("merchant_name")
                .agg(
                    {
                        "total_transactions": "sum",
                        "total_volume": "sum",
                        "clean_volume": "sum",
                        "blocked_fraud_volume": "sum",
                        "fraud_rate_percentage": "mean",
                    }
                )
                .reset_index()
                .sort_values(by="total_volume", ascending=False)
            )

            st.dataframe(df_grouped, use_container_width=True)

            fig_m = px.bar(
                df_grouped,
                x="merchant_name",
                y=["clean_volume", "blocked_fraud_volume"],
                title="Volume d'affaires Sain vs Fraude Bloquée (Périmètre filtré)",
                labels={"value": "Volume (€)", "merchant_name": "Marchand"},
                barmode="group",
                color_discrete_map={
                    "clean_volume": "#2ecc71",
                    "blocked_fraud_volume": "#e74c3c",
                },
            )
            st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.warning("La table de métriques journalières marchands est vide.")

st.markdown("---")

# ----------------- SECTION 2b : EXPLICABILITÉ PAR TRANSACTION PAR MARCHAND -----------------
st.markdown("### 🔍 2b. Expliquabilité individuelle par marchand")
st.write(
    "Visualisez en cascade (Waterfall) les contributions SHAP pour les transactions suspectes d'un marchand particulier."
)

target_merchant = None
if selected_merchant != "Aucun (Afficher tous)":
    target_merchant = selected_merchant
    st.write(f"Marchand analysé : **{target_merchant}**")
elif df_pareto is not None and not df_pareto.empty:
    target_merchant = df_pareto.iloc[0]["merchant_name"]
    st.write(
        f"Aucun marchand spécifique sélectionné dans les filtres. Analyse par défaut du marchand le plus frauduleux (Pareto) : **{target_merchant}**"
    )

if target_merchant:
    tx_query = f"""
        SELECT * FROM silver.rawdata
        WHERE merchant = '{target_merchant}' AND prediction = 1
        ORDER BY trans_date_trans_time DESC
        LIMIT 50
    """
    df_tx_list, tx_err = query_db(tx_query)

    if tx_err:
        st.error(f"Impossible de récupérer les transactions du marchand : {tx_err}")
    elif df_tx_list is not None and not df_tx_list.empty:
        tx_options = []
        for idx, row in df_tx_list.iterrows():
            tx_options.append(
                f"{row['trans_date_trans_time']} | {row['trans_num']} | {row['amt']} €"
            )

        selected_option = st.selectbox(
            "Sélectionnez une transaction suspecte à expliquer :", tx_options
        )
        selected_idx = tx_options.index(selected_option)
        tx_row = df_tx_list.iloc[[selected_idx]].copy()
        tx_id = tx_row["trans_num"].iloc[0]
        tx_row.index = [tx_id]

        preprocessor, predictor = load_champion_explainer_assets()

        if preprocessor is not None and predictor is not None:
            tx_row = tx_row.copy()
            tx_row["trans_date_trans_time"] = pd.to_datetime(
                tx_row["trans_date_trans_time"]
            )
            tx_row["dob"] = pd.to_datetime(tx_row["dob"])
            tx_row["age"] = (
                tx_row["trans_date_trans_time"].dt.year - tx_row["dob"].dt.year
            )
            tx_row["distance_achat"] = haversine_vectorized(
                float(tx_row["lat"].iloc[0]),
                float(tx_row["long"].iloc[0]),
                float(tx_row["merch_lat"].iloc[0]),
                float(tx_row["merch_long"].iloc[0]),
            )
            dt_col = tx_row["trans_date_trans_time"]
            tx_row["hour_sin"] = np.sin(2 * np.pi * dt_col.dt.hour / 24.0)
            tx_row["hour_cos"] = np.cos(2 * np.pi * dt_col.dt.hour / 24.0)
            tx_row["weekday_sin"] = np.sin(2 * np.pi * dt_col.dt.dayofweek / 7.0)
            tx_row["weekday_cos"] = np.cos(2 * np.pi * dt_col.dt.dayofweek / 7.0)
            tx_row["month_sin"] = np.sin(2 * np.pi * dt_col.dt.month / 12.0)
            tx_row["month_cos"] = np.cos(2 * np.pi * dt_col.dt.month / 12.0)

            features_list = [
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
            X_single = tx_row[features_list]
            y_single = tx_row["is_fraud"]

            X_enc_single = preprocessor.transform(X_single)
            if hasattr(preprocessor, "get_feature_names_out"):
                cols = [c.split("__")[-1] for c in preprocessor.get_feature_names_out()]
            else:
                cols = X_single.columns.tolist()

            if not isinstance(X_enc_single, pd.DataFrame):
                X_enc_single = pd.DataFrame(X_enc_single, columns=cols)
            else:
                X_enc_single.columns = [c.split("__")[-1] for c in X_enc_single.columns]

            features_groups = {
                "Heure": ["hour_sin", "hour_cos"],
                "Jour de la semaine": ["weekday_sin", "weekday_cos"],
                "Mois de l'année": ["month_sin", "month_cos"],
            }
            features_dict = {
                "amt": "Montant (€)",
                "distance_achat": "Distance d'achat (km)",
                "age": "Âge du client",
                "city_pop": "Population de la ville",
                "category": "Catégorie d'achat",
                "gender": "Genre",
            }

            xpl = SmartExplainer(
                model=predictor,
                features_groups=features_groups,
                features_dict=features_dict,
            )

            def dummy_get_interaction_values(selection=None, n_samples_max=None):
                return np.zeros((1, len(cols), len(cols)))

            xpl.get_interaction_values = dummy_get_interaction_values

            with st.spinner("Calcul de la contribution locale..."):
                xpl.compile(x=X_enc_single, y_target=y_single)
                fig_local = xpl.plot.local_plot(index=tx_id)

            c_d1, c_d2 = st.columns([1, 2])
            with c_d1:
                st.markdown("##### Détails de la transaction")
                st.write(f"🆔 **ID Transaction :** `{tx_row['trans_num'].iloc[0]}`")
                import hashlib

                cc_hash = hashlib.sha256(
                    str(tx_row["cc_num"].iloc[0]).encode()
                ).hexdigest()
                st.write(f"💳 **Numéro carte (SHA-256) :** `{cc_hash[:16]}...`")
                st.write(f"🛍️ **Catégorie :** `{tx_row['category'].iloc[0]}`")
                st.write(f"💰 **Montant :** `{tx_row['amt'].iloc[0]} €`")
                st.write(
                    f"👤 **Âge & Genre :** `{tx_row['age'].iloc[0]} ans` (`{tx_row['gender'].iloc[0]}`)"
                )
                st.write(
                    f"🗺️ **Distance d'achat :** `{tx_row['distance_achat'].iloc[0]:.2f} km`"
                )
                st.write(
                    f"🏙️ **Population ville :** `{tx_row['city_pop'].iloc[0]} hab.`"
                )
                st.write(
                    f"📈 **Probabilité de fraude :** **`{float(tx_row['prediction_proba'].iloc[0]):.4%}`**"
                )
            with c_d2:
                st.plotly_chart(fig_local, use_container_width=True)
        else:
            st.error(
                "Impossible de charger le modèle champion ou son préprocesseur depuis MLflow."
            )
    else:
        st.info(
            f"Aucune transaction suspecte récente (`prediction = 1`) enregistrée dans PostgreSQL pour le marchand **{target_merchant}**."
        )
else:
    st.info(
        "Sélectionnez un marchand spécifique pour afficher ses explications de transactions."
    )

st.markdown("---")

# ----------------- SECTION 3 : PIC HORAIRE DE FRAUDE -----------------
st.markdown("### 🕒 3. Pics de fraude & analyse temporelle")
df_hourly, err_hourly = query_db(
    "SELECT * FROM gold.mart_merchant_hourly_fraud ORDER BY merchant_name, transaction_hour"
)
if err_hourly:
    st.error(
        f"La table gold.mart_merchant_hourly_fraud n'est pas disponible : {err_hourly}"
    )
else:
    if df_hourly is not None and not df_hourly.empty:
        if active_merchants:
            df_hourly = df_hourly[df_hourly["merchant_name"].isin(active_merchants)]

        if df_hourly.empty:
            st.warning(
                "Aucun résultat horaire ne correspond aux filtres de marchand sélectionnés."
            )
        else:
            df_hourly["fraud_volume_loss"] = df_hourly["fraud_volume_loss"].astype(
                float
            )

            df_hourly_grouped = (
                df_hourly.groupby("transaction_hour")
                .agg(
                    {
                        "total_transactions": "sum",
                        "fraud_transactions_count": "sum",
                        "fraud_volume_loss": "sum",
                    }
                )
                .reset_index()
            )

            fig_h = px.line(
                df_hourly_grouped,
                x="transaction_hour",
                y="fraud_transactions_count",
                title="Pics du nombre de fraudes par heure de la journée (Périmètre filtré)",
                labels={
                    "transaction_hour": "Heure (0h - 23h)",
                    "fraud_transactions_count": "Nombre de Fraudes",
                },
            )
            fig_h.update_traces(
                mode="lines+markers", line=dict(color="#e74c3c", width=3)
            )
            st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.warning("La table de répartition horaire est vide.")

st.markdown("---")

# ----------------- SECTION 4 : DETAIL TRANSACTIONS BLOQUEES -----------------
st.markdown("### 🛑 4. Détail des transactions bloquées (périmètre filtré)")
df_blocked, err_blocked = query_db(
    "SELECT * FROM gold.mart_merchant_blocked_transactions ORDER BY transaction_timestamp DESC"
)
if err_blocked:
    st.error(
        f"La table gold.mart_merchant_blocked_transactions n'est pas disponible : {err_blocked}"
    )
else:
    if df_blocked is not None and not df_blocked.empty:
        if active_merchants:
            df_blocked = df_blocked[df_blocked["merchant_name"].isin(active_merchants)]
        if selected_category != "Toutes":
            df_blocked = df_blocked[
                df_blocked["transaction_category"] == selected_category
            ]

        if df_blocked.empty:
            st.warning(
                "Aucune transaction bloquée ne correspond aux filtres sélectionnés."
            )
        else:
            st.write(
                f"Affichage des {min(100, len(df_blocked))} dernières transactions bloquées :"
            )
            st.dataframe(df_blocked.head(100), use_container_width=True)
    else:
        st.warning("La table de détails des transactions bloquées est vide.")

st.markdown("---")

# ----------------- SECTION 5 : EXPLICABILITÉ SHAP MARCHANDS -----------------
st.markdown("### 🧠 5. Profils d'explicabilité SHAP par marchand")
df_shap, err_shap = query_db("SELECT * FROM gold.mart_merchant_shap_importance")
if err_shap:
    st.error(
        f"La table gold.mart_merchant_shap_importance n'est pas disponible : {err_shap}"
    )
else:
    if df_shap is not None and not df_shap.empty:
        if active_merchants:
            df_shap = df_shap[df_shap["merchant_name"].isin(active_merchants)]

        if df_shap.empty:
            st.warning(
                "Aucune explication SHAP ne correspond aux marchands sélectionnés."
            )
        else:
            for col in [
                "avg_amt_impact",
                "avg_distance_impact",
                "avg_age_impact",
                "avg_city_pop_impact",
                "avg_time_impact",
            ]:
                df_shap[col] = df_shap[col].astype(float)

            avg_amt = df_shap["avg_amt_impact"].mean()
            avg_dist = df_shap["avg_distance_impact"].mean()
            avg_age = df_shap["avg_age_impact"].mean()
            avg_pop = df_shap["avg_city_pop_impact"].mean()
            avg_time = df_shap["avg_time_impact"].mean()

            shap_summary_df = pd.DataFrame(
                {
                    "Variable": [
                        "Montant (amt)",
                        "Distance Achat",
                        "Âge Client",
                        "Population Ville",
                        "Facteur Temps",
                    ],
                    "Impact SHAP Moyen (Absolu)": [
                        avg_amt,
                        avg_dist,
                        avg_age,
                        avg_pop,
                        avg_time,
                    ],
                }
            ).sort_values(by="Impact SHAP Moyen (Absolu)", ascending=True)

            title_shap = "Importance moyenne des facteurs de risques de fraude"
            if selected_merchant != "Aucun (Afficher tous)":
                title_shap += f" chez {selected_merchant}"
            else:
                title_shap += " sur le périmètre filtré"

            fig_s = px.bar(
                shap_summary_df,
                x="Impact SHAP Moyen (Absolu)",
                y="Variable",
                orientation="h",
                title=title_shap,
                color="Impact SHAP Moyen (Absolu)",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.warning("La table d'explicabilité SHAP marchands est vide.")

st.markdown("---")

# ----------------- SECTION 6 : METRIQUES OPERATIONNELLES SLA -----------------
st.markdown("### ⚡ 6. Performance opérationnelle & disponibilité (SLA)")
df_sla, err_sla = query_db(
    "SELECT * FROM gold.mart_operational_sla ORDER BY check_date DESC"
)
if err_sla:
    st.error(f"La table gold.mart_operational_sla n'est pas disponible : {err_sla}")
else:
    if df_sla is not None and not df_sla.empty:
        df_sla["avg_latency_ms"] = df_sla["avg_latency_ms"].astype(float)
        df_sla["max_latency_ms"] = df_sla["max_latency_ms"].astype(float)
        df_sla["sla_compliance_percentage"] = df_sla[
            "sla_compliance_percentage"
        ].astype(float)

        c_sla1, c_sla2 = st.columns(2)
        with c_sla1:
            fig_sla1 = px.line(
                df_sla,
                x="check_date",
                y="avg_latency_ms",
                title="Vitesse d'inférence moyenne de l'API (ms)",
                labels={"check_date": "Date", "avg_latency_ms": "Latence Moyenne (ms)"},
            )
            fig_sla1.update_traces(
                mode="lines+markers", line=dict(color="#3498db", width=3)
            )
            st.plotly_chart(fig_sla1, use_container_width=True)
        with c_sla2:
            fig_sla2 = px.line(
                df_sla,
                x="check_date",
                y="sla_compliance_percentage",
                title="Taux de respect du SLA (<20ms) %",
                labels={
                    "check_date": "Date",
                    "sla_compliance_percentage": "Respect SLA (%)",
                },
            )
            fig_sla2.update_traces(
                mode="lines+markers", line=dict(color="#2ecc71", width=3)
            )
            fig_sla2.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_sla2, use_container_width=True)
    else:
        st.warning("La table opérationnelle SLA est vide.")
