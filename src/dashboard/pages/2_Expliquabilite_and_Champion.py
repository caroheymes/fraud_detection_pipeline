# src/dashboard/pages/2_Expliquabilite_and_Champion.py

import os

import mlflow
import numpy as np
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient
from shapash import SmartExplainer

st.set_page_config(
    page_title="Expliquabilité Shapash & Performances du Champion",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Expliquabilité Shapash & performances du champion")
st.markdown("---")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))


def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


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
        return df
    except Exception:
        return None


def get_hybrid_explain_sample():
    # 1. Charger l'historique de référence
    df_ref = pd.read_csv("src/training/reference_data.csv")
    df_ref["trans_date_trans_time"] = pd.to_datetime(df_ref["trans_date_trans_time"])

    # 2. Récupérer les données réelles des 30 derniers jours de PostgreSQL
    db_query = """
        SELECT * FROM silver.rawdata
        WHERE trans_date_trans_time >= (SELECT COALESCE(MAX(trans_date_trans_time), NOW()) - INTERVAL '30 days' FROM silver.rawdata)
    """
    df_prod = query_db(db_query)

    if df_prod is not None and not df_prod.empty:
        # Formater les colonnes temporelles
        df_prod["trans_date_trans_time"] = pd.to_datetime(
            df_prod["trans_date_trans_time"]
        )
        df_prod["dob"] = pd.to_datetime(df_prod["dob"])

        # Calculer à la volée les variables requises par le modèle
        df_prod["age"] = (
            df_prod["trans_date_trans_time"].dt.year - df_prod["dob"].dt.year
        )
        df_prod["distance_achat"] = haversine_vectorized(
            df_prod["lat"].astype(float),
            df_prod["long"].astype(float),
            df_prod["merch_lat"].astype(float),
            df_prod["merch_long"].astype(float),
        )
        dt_col = df_prod["trans_date_trans_time"]
        df_prod["hour_sin"] = np.sin(2 * np.pi * dt_col.dt.hour / 24.0)
        df_prod["hour_cos"] = np.cos(2 * np.pi * dt_col.dt.hour / 24.0)
        df_prod["weekday_sin"] = np.sin(2 * np.pi * dt_col.dt.dayofweek / 7.0)
        df_prod["weekday_cos"] = np.cos(2 * np.pi * dt_col.dt.dayofweek / 7.0)
        df_prod["month_sin"] = np.sin(2 * np.pi * dt_col.dt.month / 12.0)
        df_prod["month_cos"] = np.cos(2 * np.pi * dt_col.dt.month / 12.0)

        prod_normal = df_prod[df_prod["is_fraud"] == 0]
        prod_fraud = df_prod[df_prod["is_fraud"] == 1]
    else:
        prod_normal = pd.DataFrame()
        prod_fraud = pd.DataFrame()

    ref_normal = df_ref[df_ref["is_fraud"] == 0]
    ref_fraud = df_ref[df_ref["is_fraud"] == 1]

    # Échantillonnage de 800 transactions saines
    n_prod_normal = len(prod_normal)
    if n_prod_normal >= 800:
        sample_normal = prod_normal.sample(n=800, random_state=42)
    else:
        n_needed = 800 - n_prod_normal
        sample_ref_normal = ref_normal.sample(n=n_needed, random_state=42)
        sample_normal = pd.concat([prod_normal, sample_ref_normal])

    # Échantillonnage de 200 transactions frauduleuses
    n_prod_fraud = len(prod_fraud)
    if n_prod_fraud >= 200:
        sample_fraud = prod_fraud.sample(n=200, random_state=42)
    else:
        n_needed = 200 - n_prod_fraud
        sample_ref_fraud = ref_fraud.sample(n=n_needed, random_state=42)
        sample_fraud = pd.concat([prod_fraud, sample_ref_fraud])

    # Combinaison et mélange
    df_sample_resorted = (
        pd.concat([sample_normal, sample_fraud])
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )
    return df_sample_resorted


# Cache pour le chargement de l'explicateur Shapash
@st.cache_resource
def load_shapash_explainer():
    champion_run_id = None
    champion_metrics = {}
    champion_params = {}

    try:
        client = MlflowClient()
        version_details = client.get_model_version_by_alias(
            "fraud_detector", "champion"
        )
        champion_run_id = version_details.run_id

        champion_run = client.get_run(champion_run_id)
        champion_metrics = champion_run.data.metrics
        champion_params = champion_run.data.params

        champion_model = mlflow.sklearn.load_model(f"runs:/{champion_run_id}/model")
        preprocessor = champion_model.named_steps["preprocessor"]
        predictor = champion_model.named_steps["model"]

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
        df_sample_resorted = get_hybrid_explain_sample()

        X_samp = df_sample_resorted[features_list]
        y_samp = df_sample_resorted["is_fraud"]
        X_enc = preprocessor.transform(X_samp)

        # Garantir que X_enc est un DataFrame avec des noms de colonnes valides
        if hasattr(preprocessor, "get_feature_names_out"):
            cols = [c.split("__")[-1] for c in preprocessor.get_feature_names_out()]
        else:
            cols = X_samp.columns.tolist()

        if not isinstance(X_enc, pd.DataFrame):
            X_enc = pd.DataFrame(X_enc, columns=cols)
        else:
            X_enc.columns = [c.split("__")[-1] for c in X_enc.columns]
        X_enc.index = df_sample_resorted["trans_num"].tolist()

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
        xpl_obj = SmartExplainer(
            model=predictor,
            features_groups=features_groups,
            features_dict=features_dict,
        )

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
            client = MlflowClient()
            experiment = client.get_experiment_by_name("Default")
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
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
                df_sample_resorted = get_hybrid_explain_sample()

                X_samp = df_sample_resorted[features_list]
                y_samp = df_sample_resorted["is_fraud"]
                X_enc = preprocessor.transform(X_samp)

                # Garantir que X_enc est un DataFrame avec des noms de colonnes valides
                if hasattr(preprocessor, "get_feature_names_out"):
                    cols = [
                        c.split("__")[-1] for c in preprocessor.get_feature_names_out()
                    ]
                else:
                    cols = X_samp.columns.tolist()

                if not isinstance(X_enc, pd.DataFrame):
                    X_enc = pd.DataFrame(X_enc, columns=cols)
                else:
                    X_enc.columns = [c.split("__")[-1] for c in X_enc.columns]
                X_enc.index = df_sample_resorted["trans_num"].tolist()

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
                xpl_obj = SmartExplainer(
                    model=predictor,
                    features_groups=features_groups,
                    features_dict=features_dict,
                )

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
                    "Dernier Run",
                )
        except Exception as final_err:
            st.error(
                f"Erreur critique lors de l'initialisation Shapash de secours : {final_err}"
            )
            return None, None, None, None, None, None


with st.spinner("Chargement du modèle champion et calcul des contributions SHAP..."):
    xpl, df_sample, X_encoded, metrics, params, model_ver = load_shapash_explainer()

if xpl is not None:
    # Les variables cycliques (sin/cos) sont regroupées nativement par Shapash grâce à l'argument features_groups
    available_features = [
        col for col in X_encoded.columns if not any(x in col for x in ["sin", "cos"])
    ]
    available_features += ["Heure", "Jour de la semaine", "Mois de l'année"]
    # SECTION A : PERFORMANCES DU MODÈLE CHAMPION
    st.markdown(f"### 📊 Performances du modèle champion ({model_ver})")

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("F1-Score Fraude (Classe 1)", f"{metrics.get('f1_class_1', 0.0):.4f}")
    with c_m2:
        st.metric("Rappel Fraude (Recall C1)", f"{metrics.get('rec_class_1', 0.0):.4f}")
    with c_m3:
        st.metric(
            "Précision Fraude (Prec C1)", f"{metrics.get('prec_class_1', 0.0):.4f}"
        )
    with c_m4:
        st.metric("F1 Macro (Global)", f"{metrics.get('F1_global', 0.0):.4f}")

    st.markdown("**Paramètres clés du modèle :**")
    st.code(
        f"max_depth: {params.get('max_depth')}  |  learning_rate: {params.get('learning_rate')}  |  n_estimators: {params.get('n_estimators')}  |  scale_pos_weight: {params.get('scale_pos_weight')}"
    )

    st.markdown("---")

    # SECTION B : GLOBAL FEATURE IMPORTANCE PLOT
    st.markdown(
        "### 📈 1. Importance globale des caractéristiques (global feature importance)"
    )

    st.markdown(
        r"""
        > 💡 **Note de lisibilité sur les caractéristiques cycliques (temps) :**
        > Afin de permettre au modèle ML de comprendre la continuité temporelle (par exemple, le fait que 23h et 00h soient consécutifs), les variables temporelles ont été encodées en deux indicateurs cycliques : sinus ($\sin$) et cosinus ($\cos$).
        > 
        > Pour rendre les graphiques interprétables par un humain, nous appliquons la **transformation inverse** (décodage) à l'aide de la fonction **arc tangente à deux variables ($\operatorname{arctan2}$)** pour reconstruire la valeur d'origine :
        > 
        > $$\theta = \operatorname{arctan2}(\sin(x), \cos(x)) \pmod{2\pi}$$
        > 
        > Cette valeur angulaire $\theta$ (exprimée en radians entre $0$ et $2\pi$) est ensuite convertie dans son unité d'origine :
        > * 🕒 **Heure** : $\text{heure} = \text{round}\left(\theta \times \frac{24}{2\pi}\right) \pmod{24}$
        > * 📅 **Jour de la semaine** : $\text{jour} = \text{round}\left(\theta \times \frac{7}{2\pi}\right) \pmod{7}$ (Lundi = 0, Dimanche = 6)
        > * 📆 **Mois de l'année** : $\text{mois} = \text{round}\left(\theta \times \frac{12}{2\pi}\right)$ (Janvier = 1, Décembre = 12)
        """
    )
    st.write(
        "Le graphique ci-dessous affiche l'importance globale de chaque caractéristique sur les prédictions du modèle champion. Les variables cycliques y sont renommées pour plus de clarté."
    )
    fig_global = xpl.plot.features_importance()
    st.plotly_chart(fig_global, use_container_width=True)

    st.markdown("---")

    # SECTION C : FEATURES CONTRIBUTION PLOTS
    st.markdown(
        "### 📈 2. Courbes de contribution individuelle (features contribution plots)"
    )
    st.write(
        "Ces courbes affichent l'impact d'une caractéristique spécifique sur le score de fraude. Elles permettent de voir si des montants ou distances plus élevés augmentent le score de suspicion."
    )

    selected_feature = st.selectbox(
        "Choisissez la caractéristique à analyser :",
        available_features,
        index=available_features.index("amt") if "amt" in available_features else 0,
    )
    fig_contrib = xpl.plot.contribution_plot(selected_feature)
    st.plotly_chart(fig_contrib, use_container_width=True)

    st.markdown("---")

    # SECTION D : TRANSFORMATION INVERSE
    st.markdown("### 🔄 3. Transformation inverse (décodage des variables cycliques)")
    st.write(
        "Le modèle champion utilise des features cycliques trigonométriques pour comprendre le temps. Ci-dessous, l'outil décode ces valeurs en coordonnées d'origine (Heure, Jour de la semaine, Mois)."
    )

    selected_idx_inverse = st.number_input(
        f"Sélectionnez l'index de la transaction à décoder (0 à {len(df_sample) - 1}) :",
        min_value=0,
        max_value=len(df_sample) - 1,
        value=0,
        key="inverse_tool_idx",
    )

    tx_inv = df_sample.iloc[selected_idx_inverse]
    encoded_inv = X_encoded.iloc[selected_idx_inverse]

    angle_h = np.arctan2(tx_inv["hour_sin"], tx_inv["hour_cos"]) % (2 * np.pi)
    decoded_hour = int(np.round(angle_h * 12.0 / np.pi) % 24)

    angle_w = np.arctan2(tx_inv["weekday_sin"], tx_inv["weekday_cos"]) % (2 * np.pi)
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

    angle_m = np.arctan2(tx_inv["month_sin"], tx_inv["month_cos"]) % (2 * np.pi)
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
        st.markdown("**1. Valeurs Encodées**")
        st.write(
            f"hour_sin/cos : `{tx_inv['hour_sin']:.4f}` / `{tx_inv['hour_cos']:.4f}`"
        )
        st.write(
            f"weekday_sin/cos : `{tx_inv['weekday_sin']:.4f}` / `{tx_inv['weekday_cos']:.4f}`"
        )
        st.write(
            f"month_sin/cos : `{tx_inv['month_sin']:.4f}` / `{tx_inv['month_cos']:.4f}`"
        )
    with col_dec:
        st.markdown("**2. Valeurs Décodées**")
        st.write(f"Heure : **`{decoded_hour} h`**")
        st.write(f"Jour : **`{weekdays_names[decoded_weekday]}`**")
        st.write(f"Mois : **`{months_names[decoded_month]}`**")
    with col_orig:
        st.markdown("**3. Valeurs d'Origine**")
        dt_orig = pd.to_datetime(tx_inv["trans_date_trans_time"])
        st.write(f"Heure : **`{dt_orig.hour} h`**")
        st.write(f"Jour : **`{weekdays_names[dt_orig.dayofweek]}`**")
        st.write(f"Mois : **`{months_names[dt_orig.month]}`**")

    st.markdown("---")

    # SECTION E : LOCAL EXPLANATION (Waterfall)
    st.markdown("### 👤 4. Explication locale de la transaction")
    st.write(
        "Ce graphique montre le détail des contributions SHAP pour la transaction sélectionnée ci-dessus."
    )

    col_local_details, col_local_plot = st.columns([1, 2])
    with col_local_details:
        st.markdown("##### Paramètres d'Entrée")
        st.write(f"🆔 **ID Transaction :** `{tx_inv['trans_num']}`")
        import hashlib

        cc_hash = hashlib.sha256(str(tx_inv["cc_num"]).encode()).hexdigest()
        st.write(f"💳 **Carte (SHA-256) :** `{cc_hash[:16]}...`")
        st.write(f"💰 **Montant :** `{tx_inv['amt']} €`")
        st.write(f"🛍️ **Catégorie :** `{tx_inv['category']}`")
        st.write(f"🗺️ **Distance :** `{tx_inv['distance_achat']:.2f} km`")
        st.write(f"👤 **Âge/Genre :** `{tx_inv['age']} ans` (`{tx_inv['gender']}`)")
        st.write(f"🏙️ **Population :** `{tx_inv['city_pop']} hab.`")

        if tx_inv["is_fraud"] == 1:
            st.error("🚨 FRAUDE RÉELLE")
        else:
            st.success("✅ SAINE RÉELLE")

    with col_local_plot:
        tx_id = tx_inv["trans_num"]
        fig_local = xpl.plot.local_plot(index=tx_id)
        st.plotly_chart(fig_local, use_container_width=True)
else:
    st.warning("L'explicateur Shapash n'a pas pu être chargé.")
