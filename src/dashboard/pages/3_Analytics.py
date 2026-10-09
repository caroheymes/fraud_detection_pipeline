import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from shapash import SmartExplainer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dashboard.theme import apply_theme
from src.utils.db import get_postgres_engine
from src.utils.features import (
    BASE_FEATURE_COLUMNS,
    haversine_vectorized,
    prepare_features,
)
from src.utils.mlflow_manager import load_champion_model

st.set_page_config(page_title="Rapports Décisionnels Gold (dbt)", layout="wide")
apply_theme()

st.title("ANALYTICS")
st.write(
    "Ces rapports sont générés à partir des schémas Gold de dbt dans PostgreSQL. Ils permettent un  suivi complet des performances commerciales, de la fraude et des impacts financiers réels du modèle Champion sur les marchands critiques."
)
st.markdown("---")


def query_db(query, params=None):
    try:
        from sqlalchemy import text

        engine = get_postgres_engine()
        with engine.connect() as conn:
            if isinstance(query, str):
                query = text(query)
            df = pd.read_sql_query(query, conn, params=params)
        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_champion_explainer_assets():
    try:
        model, version_id, _threshold = load_champion_model()
        return model, version_id
    except Exception as e:
        st.warning(f"Erreur de chargement du champion : {e}")
        return None, None


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

# 3. Bornes temporelles réelles dans la base PostgreSQL (dbt Gold)
date_bounds_df, _ = query_db(
    "SELECT MIN(transaction_date) as min_dt, MAX(transaction_date) as max_dt FROM gold.mart_financial_roi_daily"
)
if (
    date_bounds_df is not None
    and not date_bounds_df.empty
    and date_bounds_df.iloc[0]["min_dt"] is not None
):
    db_min_date = pd.to_datetime(date_bounds_df.iloc[0]["min_dt"]).date()
    db_max_date = pd.to_datetime(date_bounds_df.iloc[0]["max_dt"]).date()
else:
    db_min_date = datetime(2020, 1, 1).date()
    db_max_date = datetime(2020, 12, 31).date()

# ==========================================================
# B. CRÉATION DU PANNEAU DE FILTRES DYNAMIQUES
# ==========================================================
st.markdown("### Période d'analyse")
c_date1, c_date2 = st.columns(2)
with c_date1:
    selected_start_date = st.date_input(
        "Date de début d'analyse :",
        value=db_min_date,
        min_value=db_min_date,
        max_value=db_max_date,
        key="gold_global_start_date",
    )
with c_date2:
    selected_end_date = st.date_input(
        "Date de fin d'analyse :",
        value=db_max_date,
        min_value=db_min_date,
        max_value=db_max_date,
        key="gold_global_end_date",
    )

if selected_start_date > selected_end_date:
    st.error("La date de début ne peut pas être postérieure à la date de fin.")
    selected_start_date, selected_end_date = db_min_date, db_max_date

c_f1, c_f2, c_f3 = st.columns(3)

with c_f1:
    use_pareto = st.checkbox(
        "Limiter le périmètre au Top 80% Pareto (Marchands critiques)", value=True
    )
with c_f2:
    selected_merchant = st.selectbox(
        "Sélectionner un marchand spécifique (Désactive Pareto) :",
        merchant_list,
        index=0,
    )
with c_f3:
    selected_category = st.selectbox(
        "Filtrer par catégorie :",
        category_list,
        index=0,
    )

# 4. Requête Pareto dynamique sur la période sélectionnée (Top 80% Fraude)
pareto_query = f"""
with merchant_fraud as (
    select
        merchant_name,
        sum(blocked_fraud_volume) as fraud_amount
    from gold.mart_merchant_daily_metrics
    where transaction_date between '{selected_start_date}' and '{selected_end_date}'
    group by 1
),
cumulative_fraud as (
    select
        merchant_name,
        fraud_amount,
        sum(fraud_amount) over (order by fraud_amount desc) as running_cum,
        coalesce(sum(fraud_amount) over (), 0) as total_fraud_amount
    from merchant_fraud
),
pareto_calc as (
    select
        merchant_name,
        fraud_amount,
        round(
            case when total_fraud_amount > 0 then (running_cum / total_fraud_amount) * 100 else 0 end, 
            2
        ) as cum_percentage,
        round(
            case when total_fraud_amount > 0 then (lag(running_cum, 1, 0::numeric) over (order by fraud_amount desc) / total_fraud_amount) * 100 else 0 end, 
            2
        ) as prev_cum_percentage
    from cumulative_fraud
)
select
    merchant_name,
    fraud_amount,
    cum_percentage
from pareto_calc
where prev_cum_percentage < 80.0 and fraud_amount > 0
order by fraud_amount desc
"""
df_pareto, pareto_err = query_db(pareto_query)
pareto_merchants = set()
if df_pareto is not None and not df_pareto.empty:
    pareto_merchants = set(df_pareto["merchant_name"].tolist())

# Nombre de marchands actifs sur la période filtrée
merchants_in_period_df, _ = query_db(
    f"select count(distinct merchant_name) as cnt from gold.mart_merchant_daily_metrics where transaction_date between '{selected_start_date}'and '{selected_end_date}'"
)
period_merchants_count = (
    int(merchants_in_period_df.iloc[0]["cnt"])
    if merchants_in_period_df is not None and not merchants_in_period_df.empty
    else total_merchants_count
)

# Résolution de la liste des marchands actifs
if selected_merchant != "Aucun (Afficher tous)":
    active_merchants = [selected_merchant]
    filter_description = f"Filtre actif : Marchand spécifique **{selected_merchant}** | Période du **{selected_start_date.strftime('%d/%m/%Y')}** au **{selected_end_date.strftime('%d/%m/%Y')}**"
elif use_pareto and len(pareto_merchants) > 0:
    active_merchants = list(pareto_merchants)
    filter_description = f"Filtre actif : **Périmètre Pareto** ({len(active_merchants)} marchands critiques) | Période du **{selected_start_date.strftime('%d/%m/%Y')}** au **{selected_end_date.strftime('%d/%m/%Y')}**"
else:
    active_merchants = []
    filter_description = f"Filtre actif : **Tous les marchands** | Période du **{selected_start_date.strftime('%d/%m/%Y')}** au **{selected_end_date.strftime('%d/%m/%Y')}**"

st.info(f"{filter_description}")
st.markdown("---")

# ==========================================================
# C. CHARGEMENT CENTRALISÉ DES DATA MARTS (dbt Gold)
# ==========================================================
roi_query = f"""
    SELECT 
        transaction_date,
        merchant_name,
        transaction_category,
        total_transactions,
        total_volume,
        tp_count,
        tp_volume,
        fn_count,
        fn_volume,
        fp_count,
        fp_volume,
        tn_count,
        tn_volume
    FROM gold.mart_financial_roi_daily
    WHERE transaction_date BETWEEN '{selected_start_date}' AND '{selected_end_date}'
    ORDER BY transaction_date ASC;
"""
df_roi_raw, err_roi = query_db(roi_query)

metrics_query = f"""
    SELECT * 
    FROM gold.mart_merchant_daily_metrics 
    WHERE transaction_date BETWEEN '{selected_start_date}' AND '{selected_end_date}'
    ORDER BY transaction_date DESC, blocked_fraud_volume DESC;
"""
df_metrics_raw, err_metrics = query_db(metrics_query)

hourly_query = f"""
    SELECT 
        merchant as merchant_name,
        category as transaction_category,
        extract(hour from trans_date_trans_time)::int as transaction_hour,
        count(*) as total_transactions,
        sum(case when prediction = 1 then 1 else 0 end) as fraud_transactions_count,
        sum(case when prediction = 1 then amt else 0 end) as fraud_amount_loss
    FROM silver.rawdata
    WHERE trans_date_trans_time::date BETWEEN '{selected_start_date}' AND '{selected_end_date}'
    GROUP BY 1, 2, 3
    ORDER BY 3 ASC;
"""
df_hourly_raw, err_hourly = query_db(hourly_query)

st.markdown("### 1. KPI Finance")

st.markdown(
    """
    Simulateur de chargeback réel et d'impact financier du modèle Champion sur les marchands critiques.
    Calcule le **retour sur investissement (ROI) financier réel** généré par le modèle Champion. 
    Source: data mart **`gold.mart_financial_roi_daily`** matérialisé par dbt dans PostgreSQL. 
    Intègre les montants bruts de fraude sauvés, les **frais fixes de dossier de chargeback** et les **pénalités de surveillance de réseau (Visa VFMP / Mastercard ECP)**.
    Moduler les coûts unitaires pour estimer l'impact financier réel sur les marchands critiques et la valeur nette générée par le modèle Champion.)
    """
)

with st.expander("Structure des coûts réels de la fraude bancaire", expanded=False):
    st.markdown(
        """
        * **1. Montant de la transaction ($AMT$) :** somme intégrale dérobée remboursée à la victime.
        * **2. Frais fixes de dossier de rétrofacturation (*chargeback fee*) :** facturés par les banques acquéreuses / PSP (Stripe, Adyen, Worldline) pour instruire le litige (généralement **15 € à 25 € / litige**).
        * **3. Pénalités de surveillance réseau (*Visa VFMP / Mastercard ECP*) :** facturées forfaitairement par les réseaux de cartes dès lors qu'un marchand dépasse les ratios réglementaires de fraude (généralement **25 € à 50 € / chargeback** supplémentaire).
        * **4. Coût de traitement d'un faux positif ($FP$) :** Coût d'authentification renforcée (SMS OTP 3DS), support client et friction d'achat (**~2 € à 5 € / fausse alerte**).
        """
    )

# Paramétrage interactif des coûts unitaires
c_cost1, c_cost2, c_cost3 = st.columns(3)
with c_cost1:
    fee_chargeback = st.slider(
        "Frais fixes de chargeback (EUR / litige)",
        min_value=0.0,
        max_value=50.0,
        value=20.0,
        step=1.0,
        help="Frais de traitement facturés par le PSP / acquéreur pour chaque contestation client.",
        key="gold_slider_cb",
    )
with c_cost2:
    fee_network_penalty = st.slider(
        "Pénalité réseau Visa/Mastercard (EUR / chargeback)",
        min_value=0.0,
        max_value=100.0,
        value=35.0,
        step=5.0,
        help="Pénalité réglementaire des programmes de surveillance de fraude (VFMP / ECP).",
        key="gold_slider_net",
    )
with c_cost3:
    cost_fp_unit = st.slider(
        "Coût de friction par Faux Positif (EUR)",
        min_value=0.0,
        max_value=15.0,
        value=3.0,
        step=0.5,
        help="Coût opérationnel d'investigation, envoi de SMS OTP 3DS et support client.",
        key="gold_slider_fp",
    )

if err_roi:
    st.error(f"La table gold.mart_financial_roi_daily n'est pas disponible : {err_roi}")
elif df_roi_raw is not None and not df_roi_raw.empty:
    df_roi = df_roi_raw.copy()
    if active_merchants:
        df_roi = df_roi[df_roi["merchant_name"].isin(active_merchants)]
    if selected_category != "Toutes":
        df_roi = df_roi[df_roi["transaction_category"] == selected_category]

    if df_roi.empty:
        st.warning(
            "Aucune donnée financière ne correspond aux filtres de marchand / catégorie sur la période sélectionnée."
        )
    else:
        unit_chargeback_cost = fee_chargeback + fee_network_penalty

        tp_cnt = int(df_roi["tp_count"].sum())
        tp_val = float(df_roi["tp_volume"].sum())
        fn_cnt = int(df_roi["fn_count"].sum())
        fn_val = float(df_roi["fn_volume"].sum())
        fp_cnt = int(df_roi["fp_count"].sum())
        fp_val = float(df_roi["fp_volume"].sum())
        tn_cnt = int(df_roi["tn_count"].sum())
        total_tx = int(df_roi["total_transactions"].sum())

        # 1. Économies réalisées sur la fraude bloquée (TP)
        fraud_saved_amt = tp_val
        chargeback_saved = tp_cnt * unit_chargeback_cost
        total_fraud_saved = fraud_saved_amt + chargeback_saved

        # 2. Pertes subies sur la fraude non détectée (FN)
        fraud_lost_amt = fn_val
        chargeback_lost = fn_cnt * unit_chargeback_cost
        total_fraud_lost = fraud_lost_amt + chargeback_lost

        # 3. Coût opérationnel de la friction (FP)
        total_fp_friction_cost = fp_cnt * cost_fp_unit

        # 4. Scénario passif (Sans IA) vs Scénario Actuel
        baseline_passive_loss = (tp_val + fn_val) + (
            (tp_cnt + fn_cnt) * unit_chargeback_cost
        )
        net_financial_benefit = total_fraud_saved - total_fp_friction_cost
        savings_ratio = (
            (net_financial_benefit / baseline_passive_loss * 100.0)
            if baseline_passive_loss > 0
            else 0.0
        )

        # Affichage des 4 grands KPIs Financiers
        # st.markdown("#### KPI Finance")
        kpi_b1, kpi_b2, kpi_b3, kpi_b4 = st.columns(4)
        with kpi_b1:
            st.metric(
                "Bénéfice net réalisé",
                f"{net_financial_benefit / 1000:,.1f} K€".replace(",", " "),
                delta=f"{savings_ratio:.1f}% d'économies nettes",
                delta_color="normal",
            )
        with kpi_b2:
            st.metric(
                "Fraude & pénalités évitées (TP)",
                f"{total_fraud_saved / 1000:,.2f} K€".replace(",", " "),
                delta=f"{tp_cnt:,} fraudes bloquées",
                delta_color="normal",
            )
        with kpi_b3:
            st.metric(
                "Pertes résiduelles subies(FN)",
                f"{total_fraud_lost / 1000:,.2f} K€".replace(",", " "),
                delta=f"{fn_cnt:,} fraudes manquées",
                delta_color="inverse",
            )
        with kpi_b4:
            st.metric(
                "Coût friction faux positifs (FP)",
                f"{total_fp_friction_cost / 1000:,.2f} K€".replace(",", " "),
                delta=f"{fp_cnt:,} fausses alertes",
                delta_color="inverse",
            )
        st.markdown("")
        # Graphiques d'impact financier
        c_g1, c_g2 = st.columns(2)
        with c_g1:
            # Comparatif Barres : Scénario Sans IA vs Avec IA
            fig_comp = go.Figure()
            fig_comp.add_trace(
                go.Bar(
                    name="Pertes sans IA (baseline)",
                    x=["Scénario passif (sans IA)", "scénario Actuel (IA Champion)"],
                    y=[
                        baseline_passive_loss,
                        total_fraud_lost + total_fp_friction_cost,
                    ],
                    marker_color=["#EF4444", "#F59E0B"],
                    text=[
                        f"Pertes totales : {baseline_passive_loss:,.0f} EUR".replace(
                            ",", " "
                        ),
                        f"Coût résiduel : {(total_fraud_lost + total_fp_friction_cost):,.0f} EUR".replace(
                            ",", " "
                        ),
                    ],
                    textposition="auto",
                )
            )
            fig_comp.add_trace(
                go.Bar(
                    name="Économies nettes générées (€)",
                    x=["Scénario actuel (model en production)"],
                    y=[net_financial_benefit],
                    marker_color=["#10B981"],
                    text=[
                        f"Gain net : +{net_financial_benefit:,.0f} €".replace(",", " ")
                    ],
                    textposition="auto",
                )
            )
            fig_comp.update_layout(
                barmode="group",
                title="Bilan financier",
                yaxis_title="Montant en Euros (€)",
                height=400,
                margin=dict(l=20, r=20, t=60, b=40),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        with c_g2:
            # Répartition des montants sauvés
            labels_pie = [
                "Montants bruts de fraude sauvés",
                "Frais fixes de chargeback évités",
                "Pénalités réseau Visa/Mastercard évitées",
            ]
            values_pie = [
                fraud_saved_amt,
                tp_cnt * fee_chargeback,
                tp_cnt * fee_network_penalty,
            ]
            fig_pie = px.pie(
                values=values_pie,
                names=labels_pie,
                title="Décomposition de la valeur financière protégée",
                color_discrete_sequence=["#10B981", "#3B82F6", "#8B5CF6"],
                hole=0.5,
            )
            fig_pie.update_traces(
                textinfo="percent",
                hoverinfo="label+value+percent",
                textfont_size=13,
            )
            fig_pie.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=60, b=40),
                legend=dict(
                    orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5
                ),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Évolution Temporelle des Économies Cumulées sur la plage filtrée
        df_roi_daily = (
            df_roi.groupby("transaction_date")
            .agg(
                {
                    "tp_count": "sum",
                    "tp_volume": "sum",
                    "fp_count": "sum",
                }
            )
            .reset_index()
        )
        df_roi_daily["transaction_date"] = pd.to_datetime(
            df_roi_daily["transaction_date"]
        )
        df_roi_daily["saved_day"] = (
            df_roi_daily["tp_volume"].astype(float)
            + df_roi_daily["tp_count"].astype(float) * unit_chargeback_cost
            - df_roi_daily["fp_count"].astype(float) * cost_fp_unit
        )
        df_roi_daily["saved_cum"] = df_roi_daily["saved_day"].cumsum()

        fig_cum = px.area(
            df_roi_daily,
            x="transaction_date",
            y="saved_cum",
            title=f"Évolution des économies nettes cumulées du {selected_start_date.strftime('%d/%m/%Y')} au {selected_end_date.strftime('%d/%m/%Y')}",
            labels={"transaction_date": "Date", "saved_cum": "Économies Cumulées (€)"},
            color_discrete_sequence=["#10B981"],
        )
        fig_cum.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=60, b=40),
            yaxis=dict(gridcolor="#E5E7EB"),
            xaxis=dict(gridcolor="#E5E7EB"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # Évolution temporelle du taux de fraude sur la période filtrée
        if df_metrics_raw is not None and not df_metrics_raw.empty:
            df_metrics_temp = df_metrics_raw.copy()
            if active_merchants:
                df_metrics_temp = df_metrics_temp[
                    df_metrics_temp["merchant_name"].isin(active_merchants)
                ]
            df_metrics_temp["transaction_date"] = pd.to_datetime(
                df_metrics_temp["transaction_date"]
            )
            duration_days = (selected_end_date - selected_start_date).days
            if duration_days <= 45:
                df_metrics_temp["time_bucket"] = df_metrics_temp[
                    "transaction_date"
                ].dt.strftime("%Y-%m-%d")
                time_label = "Date"
            else:
                df_metrics_temp["time_bucket"] = df_metrics_temp[
                    "transaction_date"
                ].dt.strftime("%Y-%m")
                time_label = "Mois"

            df_time_grouped = (
                df_metrics_temp.groupby("time_bucket")
                .agg(
                    {
                        "total_transactions": "sum",
                        "fraud_transactions_count": "sum",
                        "total_volume": "sum",
                        "blocked_fraud_volume": "sum",
                    }
                )
                .reset_index()
                .sort_values(by="time_bucket")
            )
            df_time_grouped["fraud_rate_percentage"] = round(
                (
                    df_time_grouped["fraud_transactions_count"]
                    / df_time_grouped["total_transactions"].replace(0, 1)
                )
                * 100,
                2,
            )

            fig_time_rate = px.line(
                df_time_grouped,
                x="time_bucket",
                y="fraud_rate_percentage",
                title=f"Évolution temporelle du taux de fraude (%) ({time_label} - Périmètre filtré)",
                labels={
                    "time_bucket": time_label,
                    "fraud_rate_percentage": "Taux de Fraude (%)",
                },
                markers=True,
            )
            fig_time_rate.update_traces(line=dict(color="#e74c3c", width=3))
            fig_time_rate.update_layout(
                height=420,
                margin=dict(l=20, r=20, t=60, b=40),
                yaxis=dict(gridcolor="#E5E7EB"),
                xaxis=dict(gridcolor="#E5E7EB"),
                hovermode="x unified",
            )
            st.plotly_chart(fig_time_rate, use_container_width=True)

        # Pics de fraude & analyse horaire sur la période filtrée
        if df_hourly_raw is not None and not df_hourly_raw.empty:
            df_hourly = df_hourly_raw.copy()
            if active_merchants:
                df_hourly = df_hourly[df_hourly["merchant_name"].isin(active_merchants)]
            if selected_category != "Toutes":
                df_hourly = df_hourly[
                    df_hourly["transaction_category"] == selected_category
                ]

            if not df_hourly.empty:
                df_hourly_grouped = (
                    df_hourly.groupby("transaction_hour")
                    .agg(
                        {
                            "total_transactions": "sum",
                            "fraud_transactions_count": "sum",
                            "fraud_amount_loss": "sum",
                        }
                    )
                    .reset_index()
                )

                fig_h = px.line(
                    df_hourly_grouped,
                    x="transaction_hour",
                    y="fraud_transactions_count",
                    title=f"Pics du nombre de fraudes par heure de la journée (Périmètre filtré du {selected_start_date.strftime('%d/%m/%Y')} au {selected_end_date.strftime('%d/%m/%Y')})",
                    labels={
                        "transaction_hour": "Heure de la transaction (0h - 23h)",
                        "fraud_transactions_count": "Nombre de Fraudes Détectées (TP)",
                    },
                    markers=True,
                )
                fig_h.update_traces(line=dict(color="#f59e0b", width=3))
                fig_h.update_layout(
                    height=420,
                    margin=dict(l=20, r=20, t=60, b=40),
                    yaxis=dict(gridcolor="#E5E7EB"),
                    xaxis=dict(gridcolor="#E5E7EB", dtick=1),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_h, use_container_width=True)

        # Top 10 Marchands les plus protégés (€)
        with st.expander(
            "Classement des marchands les plus protégés (€ d'économies générées)",
            expanded=False,
        ):
            df_merchant_roi = (
                df_roi.groupby("merchant_name")
                .agg(
                    {
                        "total_transactions": "sum",
                        "tp_count": "sum",
                        "tp_volume": "sum",
                        "fp_count": "sum",
                        "fn_count": "sum",
                        "fn_volume": "sum",
                    }
                )
                .reset_index()
            )
            df_merchant_roi["fraud_saved_total"] = (
                df_merchant_roi["tp_volume"].astype(float)
                + df_merchant_roi["tp_count"].astype(float) * unit_chargeback_cost
            )
            df_merchant_roi["fp_friction_cost"] = (
                df_merchant_roi["fp_count"].astype(float) * cost_fp_unit
            )
            df_merchant_roi["net_gain"] = (
                df_merchant_roi["fraud_saved_total"]
                - df_merchant_roi["fp_friction_cost"]
            )
            df_merchant_roi = df_merchant_roi.sort_values(
                by="net_gain", ascending=False
            ).head(15)

            df_merchant_roi_display = pd.DataFrame(
                {
                    "Marchand": df_merchant_roi["merchant_name"],
                    "Transactions": df_merchant_roi["total_transactions"],
                    "Fraudes bloquées (TP)": df_merchant_roi["tp_count"],
                    "Montant sauvé (€)": df_merchant_roi["tp_volume"].map(
                        "{:,.2f} €".format
                    ),
                    "Fraude & pénalités Évitées (€)": df_merchant_roi[
                        "fraud_saved_total"
                    ].map("{:,.2f} €".format),
                    "Faux positifs (FP)": df_merchant_roi["fp_count"],
                    "Gain net (EUR)": df_merchant_roi["net_gain"].map(
                        "{:,.2f} €".format
                    ),
                }
            )
            st.dataframe(df_merchant_roi_display, use_container_width=True)
else:
    st.info("La table dbt Gold mart_financial_roi_daily ne contient aucune donnée.")

st.markdown("---")


# ----------------- SECTION 1 : PARETO ANALYSIS -----------------
st.markdown("### 2. Analyse de Pareto (20/80) des marchands critiques")
if pareto_err:
    st.error(f"Impossible de calculer le périmètre de Pareto : {pareto_err}")
else:
    if df_pareto is not None and not df_pareto.empty:
        df_pareto["fraud_amount"] = df_pareto["fraud_amount"].astype(float)
        df_pareto["cum_percentage"] = df_pareto["cum_percentage"].astype(float)

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.metric(
                "Marchands critiques (Cibles)",
                f"{len(df_pareto)} / {period_merchants_count}",
                f"soit {len(df_pareto) / max(1, period_merchants_count) * 100:.1f}% des marchands actifs",
            )
        with c_p2:
            st.metric(
                "Montant de Fraude Couvert",
                f"{df_pareto['cum_percentage'].max()}%",
                "Objectif de ciblage : > 80%",
            )

        with st.expander("Voir la liste des marchands critiques (Pareto)"):
            st.dataframe(
                df_pareto,
                use_container_width=True,
                column_config={
                    "merchant_name": st.column_config.TextColumn("Marchand"),
                    "fraud_amount": st.column_config.NumberColumn(
                        "Montant Fraude (€)", format="%.2f €"
                    ),
                    "cum_percentage": st.column_config.NumberColumn(
                        "Part Cumulée (%)", format="%.2f %%"
                    ),
                },
                hide_index=True,
            )

        fig_p = px.bar(
            df_pareto,
            x="merchant_name",
            y="fraud_amount",
            text="cum_percentage",
            title=f"Concentration de la Fraude par Marchand Critique (% Cumulé) - du {selected_start_date.strftime('%d/%m/%Y')} au {selected_end_date.strftime('%d/%m/%Y')}",
            labels={
                "fraud_amount": "Montant de Fraude (€)",
                "merchant_name": "Marchand",
            },
            color="fraud_amount",
            color_continuous_scale="Oranges",
        )
        fig_p.update_traces(textposition="outside")
        fig_p.update_layout(height=600, margin=dict(l=40, r=20, t=60, b=40))
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.warning(
            "Aucune donnée de fraude n'est disponible pour l'analyse de Pareto sur la période sélectionnée."
        )

st.markdown("---")


# ----------------- SECTION 3 : PERFORMANCE COMMERCIALE -----------------
st.markdown("### 3. Chiffre d'affaires & taux de fraude")
if err_metrics:
    st.error(
        f"La table gold.mart_merchant_daily_metrics n'est pas disponible : {err_metrics}"
    )
elif df_metrics_raw is not None and not df_metrics_raw.empty:
    df_metrics = df_metrics_raw.copy()
    if active_merchants:
        df_metrics = df_metrics[df_metrics["merchant_name"].isin(active_merchants)]

    if df_metrics.empty:
        st.warning(
            "Aucun résultat ne correspond aux filtres de marchand sélectionnés pour ce rapport sur la période."
        )
    else:
        df_metrics["clean_amount"] = df_metrics["clean_volume"].astype(float)
        df_metrics["blocked_fraud_amount"] = df_metrics["blocked_fraud_volume"].astype(
            float
        )
        df_metrics["total_amount"] = df_metrics["total_volume"].astype(float)

        df_grouped = (
            df_metrics.groupby("merchant_name")
            .agg(
                {
                    "total_transactions": "sum",
                    "fraud_transactions_count": "sum",
                    "total_amount": "sum",
                    "clean_amount": "sum",
                    "blocked_fraud_amount": "sum",
                }
            )
            .reset_index()
        )
        df_grouped["fraud_rate_percentage"] = (
            (
                df_grouped["fraud_transactions_count"]
                / df_grouped["total_transactions"].replace(0, 1)
            )
            * 100
        ).round(2)
        df_grouped = df_grouped.sort_values(by="blocked_fraud_amount", ascending=False)

        st.dataframe(
            df_grouped,
            use_container_width=True,
            column_config={
                "merchant_name": st.column_config.TextColumn("Marchand"),
                "total_transactions": st.column_config.NumberColumn(
                    "Transactions Totales", format="%d"
                ),
                "fraud_transactions_count": st.column_config.NumberColumn(
                    "Fraudes Bloquées (Nb)", format="%d"
                ),
                "total_amount": st.column_config.NumberColumn(
                    "Chiffre d'Affaires Global (€)", format="%.2f €"
                ),
                "clean_amount": st.column_config.NumberColumn(
                    "Chiffre d'Affaires Sain (€)", format="%.2f €"
                ),
                "blocked_fraud_amount": st.column_config.NumberColumn(
                    "Fraude Bloquée (€)", format="%.2f €"
                ),
                "fraud_rate_percentage": st.column_config.NumberColumn(
                    "Taux de Fraude (%)", format="%.2f %%"
                ),
            },
            hide_index=True,
        )

        fig_m = px.bar(
            df_grouped,
            x="merchant_name",
            y=["clean_amount", "blocked_fraud_amount"],
            title=f"Chiffre d'Affaires Sain vs Fraude Bloquée (Périmètre filtré du {selected_start_date.strftime('%d/%m/%Y')} au {selected_end_date.strftime('%d/%m/%Y')})",
            labels={"value": "Montant (€)", "merchant_name": "Marchand"},
            barmode="group",
            color_discrete_map={
                "clean_amount": "#2ecc71",
                "blocked_fraud_amount": "#e74c3c",
            },
        )
        fig_m.for_each_trace(
            lambda t: t.update(
                name="Chiffre d'Affaires Sain (€)"
                if t.name == "clean_amount"
                else "Fraude Bloquée (€)"
            )
        )
        st.plotly_chart(fig_m, use_container_width=True)
else:
    st.warning(
        "La table de métriques journalières marchands est vide pour la période sélectionnée."
    )

st.markdown("---")


# ----------------- SECTION 4 : EXPLICABILITÉ PAR TRANSACTION PAR MARCHAND -----------------
st.markdown("### 4. Expliquabilité individuelle par marchand")
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
        f"Aucun marchand spécifique sélectionné dans les filtres. Analyse par défaut du marchand le plus frauduleux (Pareto sur la période) : **{target_merchant}**"
    )

if target_merchant:
    tx_query = """
        SELECT * FROM silver.rawdata
        WHERE merchant = :merchant 
          AND prediction = 1
          AND trans_date_trans_time::date BETWEEN :start_dt AND :end_dt
        ORDER BY trans_date_trans_time DESC
        LIMIT 50
    """
    df_tx_list, tx_err = query_db(
        tx_query,
        params={
            "merchant": target_merchant,
            "start_dt": selected_start_date,
            "end_dt": selected_end_date,
        },
    )

    if (df_tx_list is None or df_tx_list.empty) and not tx_err:
        # Fallback to all-time suspect transactions for this merchant if none in the exact filtered window
        tx_fallback_query = """
            SELECT * FROM silver.rawdata
            WHERE merchant = :merchant AND prediction = 1
            ORDER BY trans_date_trans_time DESC
            LIMIT 50
        """
        df_tx_list, tx_err = query_db(
            tx_fallback_query, params={"merchant": target_merchant}
        )
        if df_tx_list is not None and not df_tx_list.empty:
            st.info(
                f"Aucune transaction suspecte trouvée pour {target_merchant} sur la période filtrée. Affichage des dernières transactions suspectes enregistrées."
            )

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

        model_res = load_champion_explainer_assets()
        champion_model = model_res[0] if model_res else None

        if champion_model is not None:
            df_tx_list_proc = df_tx_list.copy()
            df_tx_list_proc.index = df_tx_list_proc["trans_num"].tolist()

            df_tx_list_proc["trans_date_trans_time"] = pd.to_datetime(
                df_tx_list_proc["trans_date_trans_time"]
            )
            df_tx_list_proc["dob"] = pd.to_datetime(df_tx_list_proc["dob"])
            df_tx_list_proc["age"] = (
                df_tx_list_proc["trans_date_trans_time"].dt.year
                - df_tx_list_proc["dob"].dt.year
            )
            df_tx_list_proc["distance_achat"] = haversine_vectorized(
                df_tx_list_proc["lat"].astype(float),
                df_tx_list_proc["long"].astype(float),
                df_tx_list_proc["merch_lat"].astype(float),
                df_tx_list_proc["merch_long"].astype(float),
            )
            dt_cols = df_tx_list_proc["trans_date_trans_time"]
            df_tx_list_proc["hour_sin"] = np.sin(2 * np.pi * dt_cols.dt.hour / 24.0)
            df_tx_list_proc["hour_cos"] = np.cos(2 * np.pi * dt_cols.dt.hour / 24.0)
            df_tx_list_proc["weekday_sin"] = np.sin(
                2 * np.pi * dt_cols.dt.dayofweek / 7.0
            )
            df_tx_list_proc["weekday_cos"] = np.cos(
                2 * np.pi * dt_cols.dt.dayofweek / 7.0
            )
            df_tx_list_proc["month_sin"] = np.sin(2 * np.pi * dt_cols.dt.month / 12.0)
            df_tx_list_proc["month_cos"] = np.cos(2 * np.pi * dt_cols.dt.month / 12.0)

            if "client_node" not in df_tx_list_proc.columns:
                df_tx_list_proc["client_node"] = df_tx_list_proc["cc_num"].astype(str)
            if "merchant_node" not in df_tx_list_proc.columns:
                df_tx_list_proc["merchant_node"] = df_tx_list_proc["merchant"].astype(
                    str
                )

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
            X_all = df_tx_list_proc[features_list]
            y_all = df_tx_list_proc["is_fraud"]

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

            if hasattr(champion_model, "named_steps"):
                preprocessor = champion_model.named_steps["preprocessor"]
                predictor = champion_model.named_steps["model"]
                X_enc_all = preprocessor.transform(X_all)
                if hasattr(preprocessor, "get_feature_names_out"):
                    cols = [
                        c.split("__")[-1] for c in preprocessor.get_feature_names_out()
                    ]
                else:
                    cols = X_all.columns.tolist()

                if not isinstance(X_enc_all, pd.DataFrame):
                    X_enc_all = pd.DataFrame(
                        X_enc_all, columns=cols, index=df_tx_list_proc.index
                    )
                else:
                    X_enc_all.columns = [c.split("__")[-1] for c in X_enc_all.columns]
                    X_enc_all.index = df_tx_list_proc.index
            elif hasattr(champion_model, "ae_extractor") and hasattr(
                champion_model, "classifier"
            ):
                X_enc_arr = champion_model.ae_extractor.transform(df_tx_list_proc)
                try:
                    cols = list(champion_model.get_feature_names_out())
                except Exception:
                    cols = [f"feat_{i}" for i in range(X_enc_arr.shape[1])]
                X_enc_all = pd.DataFrame(
                    X_enc_arr, columns=cols, index=df_tx_list_proc.index
                )
                features_groups["Autoencodeur Anomalie & Latent"] = [
                    c for c in cols if "ae_" in c
                ]
                predictor = champion_model.classifier
            elif hasattr(champion_model, "hinsage") and hasattr(
                champion_model, "classifier"
            ):
                test_embeddings = champion_model.hinsage.transform(df_tx_list_proc)
                raw_scaled = champion_model.hinsage._extract_clean_features(
                    df_tx_list_proc, is_train=False
                )
                X_enc_arr = np.hstack([test_embeddings, raw_scaled])
                emb_cols = [
                    f"Embedding GRL {i + 1}" for i in range(test_embeddings.shape[1])
                ]
                vec_cols = (
                    [
                        c.split("__")[-1]
                        for c in champion_model.hinsage.vectorizer.get_feature_names_out()
                    ]
                    if hasattr(
                        champion_model.hinsage.vectorizer, "get_feature_names_out"
                    )
                    else [f"feat_{i}" for i in range(raw_scaled.shape[1])]
                )
                cols = emb_cols + vec_cols
                X_enc_all = pd.DataFrame(
                    X_enc_arr, columns=cols, index=df_tx_list_proc.index
                )
                features_groups["Embeddings Réseau Graphe (HinSAGE)"] = emb_cols
                predictor = champion_model.classifier
            elif hasattr(champion_model, "iso_forest"):
                df_prep = prepare_features(df_tx_list_proc, include_graph_ids=False)
                present_cols = [c for c in BASE_FEATURE_COLUMNS if c in df_prep.columns]
                X_in = df_prep[present_cols]
                X_enc_arr = champion_model.transform(X_in)
                try:
                    cols = list(champion_model.get_feature_names_out())
                except Exception:
                    cols = [f"feat_{i}" for i in range(X_enc_arr.shape[1])]
                X_enc_all = pd.DataFrame(
                    X_enc_arr, columns=cols, index=df_tx_list_proc.index
                )
                features_groups["Détection d'Anomalies (Isolation Forest)"] = [
                    c for c in cols if "iso_forest" in c
                ]
                predictor = getattr(
                    champion_model,
                    "classifier",
                    getattr(champion_model, "xgb_model", champion_model),
                )
            else:
                df_prep = prepare_features(df_tx_list_proc, include_graph_ids=False)
                present_cols = [c for c in BASE_FEATURE_COLUMNS if c in df_prep.columns]
                if hasattr(champion_model, "transform"):
                    X_enc_arr = champion_model.transform(df_prep[present_cols])
                    try:
                        cols = list(champion_model.get_feature_names_out())
                    except Exception:
                        cols = [f"feat_{i}" for i in range(X_enc_arr.shape[1])]
                    X_enc_all = pd.DataFrame(
                        X_enc_arr, columns=cols, index=df_tx_list_proc.index
                    )
                elif hasattr(champion_model, "vectorizer") and hasattr(
                    champion_model.vectorizer, "transform"
                ):
                    X_enc_arr = champion_model.vectorizer.transform(
                        df_prep[present_cols]
                    )
                    try:
                        cols = list(champion_model.vectorizer.get_feature_names_out())
                    except Exception:
                        cols = [f"feat_{i}" for i in range(X_enc_arr.shape[1])]
                    X_enc_all = pd.DataFrame(
                        X_enc_arr, columns=cols, index=df_tx_list_proc.index
                    )
                else:
                    X_enc_all = pd.get_dummies(df_prep[present_cols], drop_first=True)
                    X_enc_all.index = df_tx_list_proc.index
                    cols = X_enc_all.columns.tolist()
                predictor = getattr(
                    champion_model,
                    "classifier",
                    getattr(champion_model, "xgb_model", champion_model),
                )

            # Extraction de la ligne spécifique pour l'affichage des détails
            tx_row = df_tx_list_proc.loc[[tx_id]]

            xpl = SmartExplainer(
                model=predictor,
                features_groups=features_groups,
                features_dict=features_dict,
            )

            def dummy_get_interaction_values(selection=None, n_samples_max=None):
                return np.zeros((1, len(cols), len(cols)))

            xpl.get_interaction_values = dummy_get_interaction_values

            with st.spinner("Calcul de la contribution locale SHAP..."):
                xpl.compile(x=X_enc_all, y_target=y_all)
                fig_local = xpl.plot.local_plot(index=tx_id)

            c_d1, c_d2 = st.columns([1, 2])
            with c_d1:
                st.markdown("##### Détails de la transaction")
                st.write(f" **ID Transaction :** `{tx_row['trans_num'].iloc[0]}`")
                import hashlib

                cc_hash = hashlib.sha256(
                    str(tx_row["cc_num"].iloc[0]).encode()
                ).hexdigest()
                st.write(f" **Numéro carte (SHA-256) :** `{cc_hash[:16]}...`")
                st.write(f" **Catégorie :** `{tx_row['category'].iloc[0]}`")
                st.write(f" **Montant :** `{tx_row['amt'].iloc[0]} €`")
                st.write(
                    f" **Âge & Genre :** `{tx_row['age'].iloc[0]} ans` (`{tx_row['gender'].iloc[0]}`)"
                )
                st.write(
                    f" **Distance d'achat :** `{tx_row['distance_achat'].iloc[0]:.2f} km`"
                )
                st.write(f" **Population ville :** `{tx_row['city_pop'].iloc[0]} hab.`")
                st.write(
                    f" **Probabilité de fraude :** **`{float(tx_row['prediction_proba'].iloc[0]):.4%}`**"
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

# ----------------- SECTION 5 : DETAIL TRANSACTIONS BLOQUEES -----------------
st.markdown("### 5. Détail des transactions bloquées (périmètre filtré)")
st.caption(
    "Gouvernance & Conformité RGPD / PCI-DSS : Les identifiants de cartes bancaires sont strictement pseudonymisés via empreinte cryptographique irréversible SHA-256 et masquage standardisé."
)
df_blocked, err_blocked = query_db(
    f"""
    SELECT * FROM gold.mart_merchant_blocked_transactions 
    WHERE transaction_timestamp::date BETWEEN '{selected_start_date}' AND '{selected_end_date}'
    ORDER BY transaction_timestamp DESC
    """
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
                "Aucune transaction bloquée ne correspond aux filtres sélectionnés sur cette période."
            )
        else:
            st.write(
                f"Affichage des {min(100, len(df_blocked))} dernières transactions bloquées :"
            )
            display_blocked = df_blocked.head(100).copy()
            if "credit_card_hash" in display_blocked.columns:
                display_blocked["credit_card_hash_short"] = display_blocked[
                    "credit_card_hash"
                ].apply(lambda h: f"{str(h)[:16]}..." if pd.notna(h) else "")

            column_order = [
                "transaction_id",
                "transaction_timestamp",
                "merchant_name",
                "credit_card_masked",
                "credit_card_hash_short",
                "transaction_amount",
                "transaction_category",
                "customer_age",
                "customer_gender",
                "distance_achat",
                "prediction_probability",
                "fast_pass_suspicion",
                "model_version",
            ]
            cols_to_use = [c for c in column_order if c in display_blocked.columns]

            display_blocked_renamed = display_blocked[cols_to_use].rename(
                columns={
                    "transaction_id": "ID Transaction",
                    "transaction_timestamp": "Date & Heure",
                    "merchant_name": "Marchand",
                    "credit_card_masked": "Carte Bancaire (Masquée RGPD)",
                    "credit_card_hash_short": "Empreinte SHA-256",
                    "transaction_amount": "Montant (€)",
                    "transaction_category": "Catégorie",
                    "customer_age": "Âge",
                    "customer_gender": "Genre",
                    "distance_achat": "Distance (km)",
                    "prediction_probability": "Score Risque Modèle",
                    "fast_pass_suspicion": "Fast Pass Alerte",
                    "model_version": "Version Modèle",
                }
            )
            st.dataframe(display_blocked_renamed, use_container_width=True)
    else:
        st.warning("Aucune transaction bloquée trouvée pour la période sélectionnée.")

st.markdown("---")

# ----------------- SECTION 6 : EXPLICABILITÉ SHAP MARCHANDS -----------------
st.markdown("### 6. Profils d'explicabilité SHAP par marchand")
st.write(
    "Visualisez l'impact moyen des facteurs de risque de fraude (valeurs SHAP agrégées au niveau Gold dbt) par variable explicative pour chaque marchand ou pour l'ensemble du périmètre."
)

df_shap_raw, err_shap = query_db(
    "SELECT * FROM gold.mart_merchant_shap_importance ORDER BY merchant_name ASC"
)
if err_shap:
    st.error(
        f"La table gold.mart_merchant_shap_importance n'est pas disponible : {err_shap}"
    )
else:
    if df_shap_raw is not None and not df_shap_raw.empty:
        # Conversion des colonnes numériques
        for col in [
            "total_fraud_cases",
            "avg_amt_impact",
            "avg_distance_impact",
            "avg_age_impact",
            "avg_city_pop_impact",
            "avg_hour_impact",
            "avg_weekday_impact",
            "avg_month_impact",
        ]:
            df_shap_raw[col] = pd.to_numeric(df_shap_raw[col], errors="coerce").fillna(
                0.0
            )

        # Filtrage initial selon les marchands actifs du filtre global
        if active_merchants:
            df_shap_filtered = df_shap_raw[
                df_shap_raw["merchant_name"].isin(active_merchants)
            ].copy()
        else:
            df_shap_filtered = df_shap_raw.copy()

        available_shap_merchants = sorted(
            df_shap_filtered["merchant_name"].unique().tolist()
        )
        if not available_shap_merchants:
            available_shap_merchants = sorted(
                df_shap_raw["merchant_name"].unique().tolist()
            )
            df_shap_filtered = df_shap_raw.copy()

        shap_select_options = ["Moyenne du périmètre filtré"] + available_shap_merchants

        default_shap_idx = 0
        if selected_merchant in available_shap_merchants:
            default_shap_idx = shap_select_options.index(selected_merchant)

        c_s1, c_s2 = st.columns([2, 1])
        with c_s1:
            chosen_shap_merchant = st.selectbox(
                "Sélectionner un marchand à analyser :",
                shap_select_options,
                index=default_shap_idx,
                key="shap_section_merchant_select",
            )

        feature_mapping = [
            ("avg_amt_impact", "Montant (amt)"),
            ("avg_distance_impact", "Distance Achat"),
            ("avg_age_impact", "Âge Client"),
            ("avg_city_pop_impact", "Population Ville"),
            ("avg_hour_impact", "Heure de la journée"),
            ("avg_weekday_impact", "Jour de la semaine"),
            ("avg_month_impact", "Mois de l'année"),
        ]

        if chosen_shap_merchant == "Moyenne du périmètre filtré":
            with c_s2:
                st.metric(
                    "Marchands dans le périmètre",
                    f"{len(df_shap_filtered)}",
                    f"{int(df_shap_filtered['total_fraud_cases'].sum())} fraudes analysées",
                )

            shap_summary_df = pd.DataFrame(
                {
                    "Variable": [label for _, label in feature_mapping],
                    "Impact SHAP Moyen (Absolu)": [
                        df_shap_filtered[col].mean() for col, _ in feature_mapping
                    ],
                }
            ).sort_values(by="Impact SHAP Moyen (Absolu)", ascending=True)

            title_shap = "Importance moyenne des facteurs de risques de fraude (Moyenne du périmètre filtré)"
        else:
            merchant_row = df_shap_raw[
                df_shap_raw["merchant_name"] == chosen_shap_merchant
            ].iloc[0]
            with c_s2:
                st.metric(
                    f"Fraudes détectées : {chosen_shap_merchant}",
                    f"{int(merchant_row['total_fraud_cases'])} cas",
                )

            shap_summary_df = pd.DataFrame(
                {
                    "Variable": [label for _, label in feature_mapping],
                    "Impact SHAP Moyen (Absolu)": [
                        float(merchant_row[col]) for col, _ in feature_mapping
                    ],
                }
            ).sort_values(by="Impact SHAP Moyen (Absolu)", ascending=True)

            title_shap = (
                f"Profil de risque SHAP pour le marchand : {chosen_shap_merchant}"
            )

        fig_s = px.bar(
            shap_summary_df,
            x="Impact SHAP Moyen (Absolu)",
            y="Variable",
            orientation="h",
            title=title_shap,
            color="Impact SHAP Moyen (Absolu)",
            color_continuous_scale="Reds",
        )
        fig_s.update_layout(height=480, margin=dict(l=40, r=20, t=60, b=40))
        st.plotly_chart(fig_s, use_container_width=True)

        with st.expander(
            "Consulter le tableau complet des profils SHAP marchands", expanded=False
        ):
            display_shap_table = (
                df_shap_filtered[
                    [
                        "merchant_name",
                        "total_fraud_cases",
                        "avg_amt_impact",
                        "avg_distance_impact",
                        "avg_age_impact",
                        "avg_city_pop_impact",
                        "avg_hour_impact",
                        "avg_weekday_impact",
                        "avg_month_impact",
                    ]
                ]
                .rename(
                    columns={
                        "merchant_name": "Marchand",
                        "total_fraud_cases": "Total Fraudes",
                        "avg_amt_impact": "Impact Montant",
                        "avg_distance_impact": "Impact Distance",
                        "avg_age_impact": "Impact Âge",
                        "avg_city_pop_impact": "Impact Population",
                        "avg_hour_impact": "Impact Heure",
                        "avg_weekday_impact": "Impact Jour",
                        "avg_month_impact": "Impact Mois",
                    }
                )
                .sort_values(by="Total Fraudes", ascending=False)
            )
            st.dataframe(display_shap_table, use_container_width=True)
    else:
        st.warning("La table d'explicabilité SHAP marchands est vide.")

st.markdown("---")

# ----------------- SECTION 7 : METRIQUES OPERATIONNELLES SLA -----------------
st.markdown("### 7. Performance opérationnelle & disponibilité (SLA)")
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
            fig_sla1.update_layout(height=600, margin=dict(l=40, r=20, t=60, b=40))
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
            fig_sla2.update_layout(height=600, margin=dict(l=40, r=20, t=60, b=40))
            st.plotly_chart(fig_sla2, use_container_width=True)
    else:
        st.warning("La table opérationnelle SLA est vide.")
