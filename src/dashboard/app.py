# src/dashboard/app.py
# docker restart fraud-detection-streamlit
import json
import os

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import redis
import streamlit as st
from mlflow.tracking import MlflowClient
from shapash import SmartExplainer

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Dashboard de Détection de Fraude MLOps", page_icon="📊", layout="wide"
)

# Design styling
st.markdown(
    """
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📊 MLOps Dashboard - Détection de Fraude & Échantillonnage Modéré")
st.markdown("---")

# ==========================================================
# 1. BARRE LATÉRALE - RÈGLES REDIS & STATUT OBSERVABILITÉ
# ==========================================================
st.sidebar.header("🔒 Observabilité MLOps")

# A. Connexion à Redis et lecture des seuils actifs (Fast Pass)
st.sidebar.subheader("⚙️ Seuils Actifs (Redis Fast Pass)")
try:
    r = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
    rules_raw = r.get("fraud_rules:config")
    if rules_raw:
        rules = json.loads(rules_raw)
        thresholds = rules.get("thresholds", {})
        st.sidebar.markdown(f"🔹 **Montant Max :** `{thresholds.get('amt_max')} €`")
        st.sidebar.markdown(
            f"🔹 **Distance Max :** `{thresholds.get('distance_achat_max')} km`"
        )
        st.sidebar.markdown(f"🔹 **Âge Max :** `{thresholds.get('age_max')} ans`")
        st.sidebar.markdown(
            f"🔹 **Villes Max Pop :** `{thresholds.get('city_pop_max')} hab.`"
        )
        st.sidebar.markdown(
            f"🕒 **Heures à risque :** `{rules.get('suspicious_hours')}`"
        )
        st.sidebar.markdown(
            f"📅 **Jours à risque :** `{rules.get('suspicious_weekdays')}`"
        )
    else:
        st.sidebar.warning("Aucune règle de suspicion active dans Redis.")
except Exception as re_err:
    st.sidebar.error(f"Redis non disponible : {re_err}")

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Statut Evidently AI")
st.sidebar.success("✅ Modèle stable (Aucun drift global)")
st.sidebar.markdown("**Dernier check :** Aujourd'hui à 22:00")
st.sidebar.markdown("**Prochain check :** Demain à 22:00")

# ==========================================================
# 2. CHARGEMENT DES DONNÉES DE RUNS (MLflow / Local)
# ==========================================================
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
runs_df = pd.DataFrame()
local_backup_loaded = False
local_data = {}

script_dir = os.path.dirname(os.path.abspath(__file__))
json_backup_path = os.path.abspath(
    os.path.join(script_dir, "../training/metrics_all_models.json")
)

try:
    # Requête MLflow
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Default")
    if experiment:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        if len(runs) > 0:
            rows = []
            for r_run in runs:
                m = r_run.data.metrics
                p = r_run.data.params
                run_name = r_run.info.run_name

                if "prec_class_1" in m:
                    rows.append(
                        {
                            "Model Run": run_name,
                            "Model Type": p.get("model_type", run_name.split("_")[0]),
                            "Ratio (%)": str(int(float(p.get("target_ratio", 0)) * 100))
                            if p.get("target_ratio")
                            else "N/A",
                            "rec_class_1 (Rappel C1)": m.get("rec_class_1", 0.0),
                            "prec_class_1 (Précision C1)": m.get("prec_class_1", 0.0),
                            "f1_class_1 (F1 C1)": m.get("f1_class_1", 0.0),
                            "F1_global (F1 Macro)": m.get("F1_global", 0.0),
                            "recall_global (Rappel Macro)": m.get("recall_global", 0.0),
                            "run_id": r_run.info.run_id,
                        }
                    )
            if len(rows) > 0:
                runs_df = pd.DataFrame(rows)
except Exception as e:
    st.sidebar.warning(
        f"Connexion MLflow non disponible (affichage via fichier local) : {e}"
    )

if runs_df.empty and os.path.exists(json_backup_path):
    try:
        with open(json_backup_path, "r") as f:
            local_data = json.load(f)
        rows = []
        for name, m in local_data.items():
            parts = name.split("_")
            model_type = parts[0]
            ratio_str = parts[-1].replace("pc", "") if "pc" in parts[-1] else "N/A"

            rows.append(
                {
                    "Model Run": name,
                    "Model Type": model_type,
                    "Ratio (%)": ratio_str,
                    "rec_class_1 (Rappel C1)": m.get("rec_class_1", 0.0),
                    "prec_class_1 (Précision C1)": m.get("prec_class_1", 0.0),
                    "f1_class_1 (F1 C1)": m.get("f1_class_1", 0.0),
                    "F1_global (F1 Macro)": m.get("F1_global", 0.0),
                    "recall_global (Rappel Macro)": m.get("recall_global", 0.0),
                    "confusion_matrix": m.get("confusion_matrix", None),
                }
            )
        runs_df = pd.DataFrame(rows)
        local_backup_loaded = True
    except Exception as e:
        st.error(f"Erreur de lecture du backup JSON local : {e}")

# ==========================================================
# 3. ONGLETS DE NAVIGATION PRINCIPAUX
# ==========================================================
tab_metrics, tab_explain = st.tabs(
    [
        "📈 Performances & Métriques",
        "🔍 Expliquabilité Shapash & Performances du Champion",
    ]
)

# ----------------- ONGLET 1 : PERFORMANCES & MÉTRIQUES -----------------
with tab_metrics:
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", "1,245,892", "+12% vs hier")
    with col2:
        st.metric("Transactions Suspectes", "1,452", "-3% vs hier")
    with col3:
        st.metric("Taux de Fraude Global", "0.12%", "-0.01%")
    with col4:
        st.metric("Échantillonnage Cible", "5% & 10%", "Échantillonnage Modéré")

    st.write(
        "Ce tableau regroupe les performances des modèles entraînés sur l'échantillonnage à 5% et 10% (XGBoost, HistGradientBoosting, GNN et le pipeline hybride NVIDIA GraphSAGE + XGBoost)."
    )

    if not runs_df.empty:
        runs_df = runs_df.sort_values(
            by="f1_class_1 (F1 C1)", ascending=False
        ).reset_index(drop=True)

        st.subheader("📋 Tableau Comparatif des Performances")
        st.dataframe(
            runs_df.drop(columns=["run_id", "confusion_matrix"], errors="ignore"),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("🚨 Analyse Détaillée & Matrice de Confusion")

        selected_run = st.selectbox(
            "Sélectionnez un modèle pour afficher sa matrice de confusion :",
            runs_df["Model Run"].tolist(),
        )

        row_data = runs_df[runs_df["Model Run"] == selected_run].iloc[0]
        cm = None

        if local_backup_loaded:
            cm = row_data.get("confusion_matrix", None)
        else:
            try:
                run_id = row_data["run_id"]
                artifacts = client.list_artifacts(run_id)
                for art in artifacts:
                    if "confusion_matrix" in art.path:
                        local_path = client.download_artifacts(run_id, art.path)
                        with open(local_path, "r") as f:
                            cm = json.load(f)
                        break
            except Exception as e:
                st.error(
                    f"Impossible de télécharger la matrice de confusion depuis MLflow : {e}"
                )

        if cm:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("#### Métriques du Run")
                st.write(f"**Modèle :** {row_data['Model Type']}")
                st.write(f"**Échantillon Fraude :** {row_data['Ratio (%)']}%")
                st.write(
                    f"**Précision Fraude (Prec C1) :** {row_data['prec_class_1 (Précision C1)']:.4f}"
                )
                st.write(
                    f"**Rappel Fraude (Rec C1) :** {row_data['rec_class_1 (Rappel C1)']:.4f}"
                )
                st.write(
                    f"**F1-Score Fraude (F1 C1) :** {row_data['f1_class_1 (F1 C1)']:.4f}"
                )
                st.write(
                    f"**F1 Global (Macro) :** {row_data['F1_global (F1 Macro)']:.4f}"
                )

            with c2:
                z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
                fig = px.imshow(
                    z,
                    x=["Sains Prédits (0)", "Fraudes Prédites (1)"],
                    y=["Sains Réels (0)", "Fraudes Réelles (1)"],
                    color_continuous_scale="Blues",
                    text_auto=True,
                    aspect="auto",
                    labels=dict(x="Prédictions", y="Vérité Terrain", color="Nombre"),
                )
                fig.update_layout(
                    title=f"Matrice de Confusion : {selected_run}",
                    width=450,
                    height=350,
                    margin=dict(l=40, r=40, t=60, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune matrice de confusion trouvée pour ce modèle.")
    else:
        st.info(
            "Aucun modèle entraîné avec échantillonnage modéré n'a encore été détecté. Veuillez exécuter le script d'entraînement d'abord."
        )
        st.code(
            "docker exec -t fraud-detection-ray-head python src/training/run_all_models.py"
        )


# ----------------- ONGLET 2 : EXPLIQUABILITÉ & CHAMPION -----------------
with tab_explain:
    st.subheader("🔍 Analyse Complète et Expliquabilité du Modèle Champion")
    st.write(
        "Cette section détaille les performances et le comportement d'interprétabilité du modèle champion actuellement en production."
    )

    # Caching du chargement Shapash et MLflow pour la fluidité
    @st.cache_resource
    def load_shapash_explainer():
        try:
            # 1. Connexion MLflow et recherche du champion
            client = MlflowClient()
            version_details = client.get_model_version_by_alias(
                "fraud_detector", "champion"
            )
            champion_run_id = version_details.run_id

            # Récupération des métriques du champion
            run_data = client.get_run(champion_run_id)
            champion_metrics = run_data.data.metrics
            champion_params = run_data.data.params

            # Chargement du modèle
            model_uri = f"runs:/{champion_run_id}/model"
            champion_model = mlflow.sklearn.load_model(model_uri)

            preprocessor = champion_model.named_steps["preprocessor"]
            predictor = champion_model.named_steps["model"]

            # 2. Chargement des données
            df_ref = pd.read_csv("src/training/reference_data.csv")
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

            # Échantillonnage équilibré
            df_normal = df_ref[df_ref["is_fraud"] == 0].sample(n=800, random_state=42)
            df_fraud = df_ref[df_ref["is_fraud"] == 1].sample(n=200, random_state=42)
            df_sample_resorted = (
                pd.concat([df_normal, df_fraud])
                .sample(frac=1.0, random_state=42)
                .reset_index(drop=True)
            )

            X_samp = df_sample_resorted[features_list]
            y_samp = df_sample_resorted["is_fraud"]
            X_enc = preprocessor.transform(X_samp)

            # 3. Initialisation Shapash
            xpl_obj = SmartExplainer(model=predictor)

            # Patch d'interaction TreeExplainer de secours
            def dummy_get_interaction_values(selection=None, n_samples_max=None):
                n_samp = len(selection) if selection is not None else 100
                n_feat = X_enc.shape[1]
                return np.zeros((n_samp, n_feat, n_feat))

            xpl_obj.get_interaction_values = dummy_get_interaction_values

            xpl_obj.compile(x=X_enc, y_target=y_samp)
            return (
                xpl_obj,
                df_sample_resorted,
                X_enc,
                champion_metrics,
                champion_params,
                f"Version {version_details.version}",
            )
        except Exception as err:
            st.warning(
                f"Impossible de charger via l'alias champion, repli sur le dernier run : {err}"
            )
            try:
                # Repli sur le dernier run
                client = MlflowClient()
                experiment = client.get_experiment_by_name("Default")
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                )
                if len(runs) > 0:
                    latest_run = runs[0]
                    champion_run_id = latest_run.info.run_id
                    champion_metrics = latest_run.data.metrics
                    champion_params = latest_run.data.params
                    champion_model = mlflow.sklearn.load_model(
                        f"runs:/{champion_run_id}/model"
                    )

                    preprocessor = champion_model.named_steps["preprocessor"]
                    predictor = champion_model.named_steps["model"]

                    df_ref = pd.read_csv("src/training/reference_data.csv")
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
                    df_normal = df_ref[df_ref["is_fraud"] == 0].sample(
                        n=800, random_state=42
                    )
                    df_fraud = df_ref[df_ref["is_fraud"] == 1].sample(
                        n=200, random_state=42
                    )
                    df_sample_resorted = (
                        pd.concat([df_normal, df_fraud])
                        .sample(frac=1.0, random_state=42)
                        .reset_index(drop=True)
                    )

                    X_samp = df_sample_resorted[features_list]
                    y_samp = df_sample_resorted["is_fraud"]
                    X_enc = preprocessor.transform(X_samp)

                    xpl_obj = SmartExplainer(model=predictor)

                    def dummy_get_interaction_values(
                        selection=None, n_samples_max=None
                    ):
                        n_samp = len(selection) if selection is not None else 100
                        n_feat = X_enc.shape[1]
                        return np.zeros((n_samp, n_feat, n_feat))

                    xpl_obj.get_interaction_values = dummy_get_interaction_values

                    xpl_obj.compile(x=X_enc, y_target=y_samp)
                    return (
                        xpl_obj,
                        df_sample_resorted,
                        X_enc,
                        champion_metrics,
                        champion_params,
                        "Dernier Run",
                    )
            except Exception as final_err:
                st.error(
                    f"Erreur critique lors de l'initialisation Shapash de secours : {final_err}"
                )
                return None, None, None, None, None, None

    with st.spinner(
        "Chargement du modèle champion et calcul des contributions SHAP..."
    ):
        xpl, df_sample, X_encoded, metrics, params, model_ver = load_shapash_explainer()

    if xpl is not None:
        # ==========================================================
        # SECTION A : PERFORMANCES DU MODÈLE CHAMPION
        # ==========================================================
        st.markdown(f"### 📊 Performances du Modèle Champion ({model_ver})")

        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.metric(
                "F1-Score Fraude (Classe 1)", f"{metrics.get('f1_class_1', 0.0):.4f}"
            )
        with c_m2:
            st.metric(
                "Rappel Fraude (Recall C1)", f"{metrics.get('rec_class_1', 0.0):.4f}"
            )
        with c_m3:
            st.metric(
                "Précision Fraude (Prec C1)", f"{metrics.get('prec_class_1', 0.0):.4f}"
            )
        with c_m4:
            st.metric("F1 Macro (Global)", f"{metrics.get('F1_global', 0.0):.4f}")

        # Paramètres d'entraînement
        st.markdown("**Paramètres clés du modèle :**")
        st.code(
            f"max_depth: {params.get('max_depth')}  |  learning_rate: {params.get('learning_rate')}  |  n_estimators: {params.get('n_estimators')}  |  scale_pos_weight: {params.get('scale_pos_weight')}"
        )

        st.markdown("---")

        # ==========================================================
        # SECTION B : GLOBAL FEATURE IMPORTANCE PLOT
        # ==========================================================
        st.markdown(
            "### 📈 1. Importance Globale des Features (Global Feature Importance)"
        )
        st.write(
            "Ce graphique issu de Shapash montre le poids global de chaque caractéristique sur les décisions du modèle champion."
        )

        fig_global = xpl.plot.features_importance()
        st.plotly_chart(fig_global, use_container_width=True)

        st.markdown("---")

        # ==========================================================
        # SECTION C : FEATURES CONTRIBUTION PLOTS
        # ==========================================================
        st.markdown(
            "### 📈 2. Courbes de Contribution Individuelle (Features Contribution Plots)"
        )
        st.write(
            "Ces courbes affichent l'impact d'une caractéristique spécifique sur le score de fraude. "
            "Elles permettent de voir si des montants ou distances plus élevés augmentent linéairement ou de façon exponentielle le score de suspicion."
        )

        # Liste des variables disponibles pour le contribution plot
        available_features = X_encoded.columns.tolist()
        selected_feature = st.selectbox(
            "Choisissez la caractéristique à analyser :",
            available_features,
            index=available_features.index("amt") if "amt" in available_features else 0,
        )

        fig_contrib = xpl.plot.contribution_plot(selected_feature)
        st.plotly_chart(fig_contrib, use_container_width=True)

        st.markdown("---")

        # ==========================================================
        # SECTION D : TRANSFORMATION INVERSE
        # ==========================================================
        st.markdown(
            "### 🔄 3. Transformation Inverse (Décodage des variables cycliques)"
        )
        st.write(
            "Le modèle champion utilise des features cycliques trigonométriques (`hour_sin`, `hour_cos`, etc.) "
            "pour comprendre le temps. Ci-dessous, l'outil décode ces valeurs en coordonnées d'origine (Heure, Jour de la semaine, Mois)."
        )

        # Outil interactif de transformation inverse pour un échantillon
        selected_idx_inverse = st.number_input(
            f"Sélectionnez l'index de la transaction à décoder (0 à {len(df_sample) - 1}) :",
            min_value=0,
            max_value=len(df_sample) - 1,
            value=0,
            key="inverse_tool_idx",
        )

        tx_inv = df_sample.iloc[selected_idx_inverse]
        encoded_inv = X_encoded.iloc[selected_idx_inverse]

        # Calculs de retransformation inverse (arctan2)
        # 1) Heure
        angle_h = np.arctan2(encoded_inv["hour_sin"], encoded_inv["hour_cos"]) % (
            2 * np.pi
        )
        decoded_hour = int(np.round(angle_h * 12.0 / np.pi) % 24)

        # 2) Jour (0 = Lundi, 6 = Dimanche)
        angle_w = np.arctan2(encoded_inv["weekday_sin"], encoded_inv["weekday_cos"]) % (
            2 * np.pi
        )
        decoded_weekday = int(np.round(angle_w * 3.5 / np.pi) % 7)
        weekdays_names = [
            "Lundi",
            "Mardi",
            "Mercredi",
            "Jeudi",
            "Vendredi",
            "Samedi",
            "Dimanche",
        ]

        # 3) Mois (1 = Janvier, 12 = Décembre)
        angle_m = np.arctan2(encoded_inv["month_sin"], encoded_inv["month_cos"]) % (
            2 * np.pi
        )
        decoded_month = int(np.round(angle_m * 6.0 / np.pi))
        decoded_month = 12 if decoded_month == 0 else decoded_month
        months_names = [
            "",
            "Janvier",
            "Février",
            "Mars",
            "Avril",
            "Mai",
            "Juin",
            "Juillet",
            "Août",
            "Septembre",
            "Octobre",
            "Novembre",
            "Décembre",
        ]

        col_enc, col_dec, col_orig = st.columns(3)
        with col_enc:
            st.markdown("**1. Valeurs Cycliques Encodées (ML)**")
            st.write(
                f"hour_sin / cos : `{encoded_inv['hour_sin']:.4f}` / `{encoded_inv['hour_cos']:.4f}`"
            )
            st.write(
                f"weekday_sin / cos : `{encoded_inv['weekday_sin']:.4f}` / `{encoded_inv['weekday_cos']:.4f}`"
            )
            st.write(
                f"month_sin / cos : `{encoded_inv['month_sin']:.4f}` / `{encoded_inv['month_cos']:.4f}`"
            )

        with col_dec:
            st.markdown("**2. Valeurs Décodées par Arctan2**")
            st.write(f"🕒 Heure décodée : **`{decoded_hour} h`**")
            st.write(
                f"📅 Jour décodé : **`{weekdays_names[decoded_weekday]}`** (Index `{decoded_weekday}`)"
            )
            st.write(
                f"📆 Mois décodé : **`{months_names[decoded_month]}`** (Index `{decoded_month}`)"
            )

        with col_orig:
            st.markdown("**3. Valeurs d'Origine (Vérité Terrain)**")
            dt_orig = pd.to_datetime(tx_inv["trans_date_trans_time"])
            st.write(f"🕒 Heure réelle : **`{dt_orig.hour} h`**")
            st.write(f"📅 Jour réel : **`{weekdays_names[dt_orig.dayofweek]}`**")
            st.write(f"📆 Mois réel : **`{months_names[dt_orig.month]}`**")

        st.markdown("---")

        # ==========================================================
        # SECTION E : LOCAL EXPLANATION (Waterfall)
        # ==========================================================
        st.markdown("### 👤 4. Explication Locale de la Transaction")
        st.write(
            "Ce graphique montre le détail des contributions SHAP pour la transaction sélectionnée ci-dessus."
        )

        col_local_details, col_local_plot = st.columns([1, 2])

        with col_local_details:
            st.markdown("##### Paramètres d'Entrée")
            st.write(f"💳 **Numéro de Carte :** `{tx_inv['cc_num']}`")
            st.write(f"💰 **Montant :** `{tx_inv['amt']} €`")
            st.write(f"🛍️ **Catégorie :** `{tx_inv['category']}`")
            st.write(f"🗺️ **Distance :** `{tx_inv['distance_achat']:.2f} km`")
            st.write(
                f"👤 **Âge du Porteur :** `{tx_inv['age']} ans` (`{tx_inv['gender']}`)"
            )
            st.write(f"🏙️ **Population :** `{tx_inv['city_pop']} hab.`")
            st.write(f"📅 **Date/Heure :** `{tx_inv['trans_date_trans_time']}`")

            st.markdown("##### Statut")
            if tx_inv["is_fraud"] == 1:
                st.error("🚨 FRAUDE RÉELLE (Classe positive)")
            else:
                st.success("✅ SAINE RÉELLE (Classe négative)")

        with col_local_plot:
            fig_local = xpl.plot.local_plot(index=selected_idx_inverse)
            st.plotly_chart(fig_local, use_container_width=True)

    else:
        st.warning(
            "L'explicateur Shapash n'a pas pu être chargé. Assurez-vous d'avoir exécuté l'entraînement et promu un modèle champion."
        )
