# src/dashboard/pages/2_Explicabilite_et_champion.py

import os

import mlflow
import numpy as np
import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient
from shapash import SmartExplainer

st.set_page_config(
    page_title="Explicabilité Shapash & performances du champion",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Explicabilité Shapash & performances du champion")
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


def get_current_champion_version_key() -> str:
    """Récupère l'identifiant unique de la version champion pour invalider automatiquement le cache Streamlit."""
    try:
        client = MlflowClient()
        v = client.get_model_version_by_alias("fraud_detector", "champion")
        return f"v{v.version}_{v.run_id}"
    except Exception:
        return "fallback_run"


# Cache dynamique pour le chargement de l'explicateur Shapash
@st.cache_resource
def load_shapash_explainer(version_key: str):
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
        y_samp.index = X_enc.index

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

        predictor_class = getattr(predictor, "__class__", type(predictor)).__name__
        preprocessor_class = (
            getattr(preprocessor, "__class__", type(preprocessor)).__name__
            if preprocessor is not None
            else "Standard"
        )
        run_name = getattr(champion_run.info, "run_name", "") or ""
        run_source = champion_run.data.tags.get("mlflow.source.name", "") or ""
        is_gnn = (
            ("embedding_size" in champion_params)
            or ("GRL" in run_name)
            or ("HinSAGE" in run_name)
            or ("gnn" in run_source.lower())
            or ("gnn" in predictor_class.lower())
        )

        model_meta = {
            "version": f"Version {version_details.version}",
            "raw_version": str(version_details.version),
            "run_id": champion_run_id,
            "run_name": run_name,
            "source": run_source,
            "is_gnn": is_gnn,
            "predictor_class": predictor_class,
            "preprocessor_class": preprocessor_class,
            "tags": champion_run.data.tags,
        }

        return (
            xpl_obj,
            df_sample_resorted,
            X_enc,
            champion_metrics,
            champion_params,
            model_meta,
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
                y_samp.index = X_enc.index

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

                pred_cls = getattr(predictor, "__class__", type(predictor)).__name__
                prep_cls = (
                    getattr(preprocessor, "__class__", type(preprocessor)).__name__
                    if preprocessor is not None
                    else "Standard"
                )
                model_meta = {
                    "version": "Dernier Run",
                    "raw_version": "N/A",
                    "run_id": champion_run_id,
                    "run_name": latest_run.info.run_name if latest_run else "Inconnu",
                    "source": "Fallback",
                    "is_gnn": False,
                    "predictor_class": pred_cls,
                    "preprocessor_class": prep_cls,
                    "tags": latest_run.data.tags if latest_run else {},
                }
                return (
                    xpl_obj,
                    df_sample_resorted,
                    X_enc,
                    champion_metrics,
                    champion_params,
                    model_meta,
                )
        except Exception as final_err:
            st.error(
                f"Erreur critique lors de l'initialisation Shapash de secours : {final_err}"
            )
            return None, None, None, None, None, None


with st.spinner("Chargement du modèle champion et calcul des contributions SHAP..."):
    current_champion_key = get_current_champion_version_key()
    xpl, df_sample, X_encoded, metrics, params, model_meta = load_shapash_explainer(
        current_champion_key
    )

if xpl is not None:
    # Extraction dynamique des métadonnées
    predictor_cls = (
        model_meta.get("predictor_class", "XGBClassifier")
        if isinstance(model_meta, dict)
        else "XGBClassifier"
    )
    preprocessor_cls = (
        model_meta.get("preprocessor_class", "TableVectorizer")
        if isinstance(model_meta, dict)
        else "TableVectorizer"
    )
    is_gnn = model_meta.get("is_gnn", False) if isinstance(model_meta, dict) else False
    ver_label = (
        model_meta.get("version", "Champion")
        if isinstance(model_meta, dict)
        else str(model_meta)
    )
    run_name = (
        model_meta.get("run_name", "Run MLflow")
        if isinstance(model_meta, dict)
        else "Inconnu"
    )
    run_source = (
        model_meta.get("source", "Pipeline ML")
        if isinstance(model_meta, dict)
        else "Script"
    )

    # Détection de la famille d'algorithme
    if is_gnn:
        family_title = (
            f"Inductive Graph Representation Learning (HinSAGE + {predictor_cls})"
        )
        family_icon = "🧠"
        family_desc = "Modélisation sur graphe tripartite hétérogène (*Clients ↔ Transactions ↔ Marchands*) avec agrégation de voisinage 2-hop et classification aval avec Focal Loss."
    elif any(
        k in predictor_cls for k in ["XGB", "Gradient", "LGBM", "CatBoost", "Hist"]
    ):
        family_title = f"Gradient Boosted Decision Trees ({predictor_cls})"
        family_icon = "🌲"
        family_desc = "Ensemble d'arbres de décision boostés séquentiellement, optimisant la fonction de perte avec régularisation et gestion des classes déséquilibrées."
    elif any(k in predictor_cls for k in ["Forest", "Tree", "ExtraTrees"]):
        family_title = f"Ensemble d'Arbres Aléatoires ({predictor_cls})"
        family_icon = "🌳"
        family_desc = "Forêt d'arbres de décision indépendants avec agrégation par vote majoritaire et pondération de classes."
    elif any(k in predictor_cls for k in ["Logistic", "Linear", "SGD", "Ridge"]):
        family_title = f"Modèle Linéaire Supervisé ({predictor_cls})"
        family_icon = "📐"
        family_desc = "Modèle linéaire probabiliste avec pénalité de régularisation et calibration de seuil de décision."
    else:
        family_title = f"Classifieur Supervisé ({predictor_cls})"
        family_icon = "⚙️"
        family_desc = f"Modèle supervisé Scikit-Learn avec pipeline de prétraitement {preprocessor_cls}."

    # Les variables cycliques (sin/cos) sont regroupées nativement par Shapash grâce à l'argument features_groups
    available_features = [
        col for col in X_encoded.columns if not any(x in col for x in ["sin", "cos"])
    ]
    available_features += ["Heure", "Jour de la semaine", "Mois de l'année"]

    # SECTION A : PERFORMANCES & CARACTÉRISTIQUES DU MODÈLE CHAMPION
    st.markdown(
        f"### 📊 Performances & Caractéristiques du modèle champion ({ver_label})"
    )

    # Bannière adaptative
    if is_gnn:
        st.success(f"{family_icon} **Architecture : {family_title}**  \n{family_desc}")
    else:
        st.info(f"{family_icon} **Architecture : {family_title}**  \n{family_desc}")

    # Cartouches des 5 métriques clés
    c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
    with c_m1:
        st.metric("F1-Score Fraude (C1)", f"{metrics.get('f1_class_1', 0.0):.4f}")
    with c_m2:
        f2_val = metrics.get("f2_class_1", 0.0)
        if f2_val == 0.0:
            p_val = metrics.get("prec_class_1", 0.0)
            r_val = metrics.get("rec_class_1", 0.0)
            f2_val = (
                (5 * p_val * r_val) / (4 * p_val + r_val)
                if (4 * p_val + r_val) > 0
                else 0.0
            )
        st.metric("F2-Score (Cible Rappel)", f"{f2_val:.4f}")
    with c_m3:
        st.metric("Rappel Fraude (Recall C1)", f"{metrics.get('rec_class_1', 0.0):.4f}")
    with c_m4:
        st.metric(
            "Précision Fraude (Prec C1)", f"{metrics.get('prec_class_1', 0.0):.4f}"
        )
    with c_m5:
        st.metric("F1 Macro (Global)", f"{metrics.get('F1_global', 0.0):.4f}")

    # Spécifications détaillées et adaptatives du modèle
    with st.expander(
        "⚙️ Fiche technique détaillée & Hyperparamètres du champion", expanded=True
    ):
        c_spec1, c_spec2 = st.columns(2)
        with c_spec1:
            st.markdown("#### 🏗️ Pipeline & Encodage")
            st.write(f"• **Algorithme Principal :** `{predictor_cls}`")
            st.write(f"• **Préprocesseur :** `{preprocessor_cls}`")
            if is_gnn:
                st.write(
                    "• **Topologie du graphe :** Tripartite (*Clients ↔ Transactions ↔ Marchands*)"
                )
                emb_dim = params.get("embedding_size", 16)
                st.write(f"• **Dimension des embeddings GNN :** `{emb_dim}` dimensions")
                st.write(
                    "• **Agrégation de voisinage :** HinSAGE 2-Hop inductive pooling"
                )
                st.write("• **Fonction de coût GNN :** Focal Loss")
            else:
                st.write("• **Mode d'apprentissage :** Supervisé Tabulaire")
                st.write(
                    "• **Features d'entrée :** 12 variables (temporelles, spatiales, montants)"
                )

            src_file = os.path.basename(run_source) if run_source else "demo_gnn.py"
            st.write(f"• **Script source :** `{src_file}`")
            st.write(f"• **Nom du Run :** `{run_name}`")

        with c_spec2:
            st.markdown("#### 🎛️ Hyperparamètres Enregistrés")
            if params:
                for param_k, param_v in sorted(params.items()):
                    try:
                        v_flt = float(param_v)
                        if "." in str(param_v) and len(str(param_v).split(".")[1]) > 4:
                            display_str = f"{v_flt:.4f}"
                        else:
                            display_str = str(param_v)
                    except Exception:
                        display_str = str(param_v)
                    st.write(f"• **`{param_k}` :** {display_str}")
            else:
                st.write("• *Aucun paramètre spécifique consigné.*")

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
