# src/dashboard/pages/2_Expliquabilite_and_Champion.py

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from shapash import SmartExplainer
from mlflow.tracking import MlflowClient
import mlflow

st.set_page_config(
    page_title="Expliquabilité Shapash & Performances du Champion", page_icon="🔍", layout="wide"
)

st.title("🔍 Expliquabilité Shapash & performances du champion")
st.markdown("---")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

# Cache pour le chargement de l'explicateur Shapash
@st.cache_resource
def load_shapash_explainer():
    champion_run_id = None
    champion_metrics = {}
    champion_params = {}
    
    try:
        client = MlflowClient()
        version_details = client.get_model_version_by_alias("fraud_detector", "champion")
        champion_run_id = version_details.run_id
        
        champion_run = client.get_run(champion_run_id)
        champion_metrics = champion_run.data.metrics
        champion_params = champion_run.data.params
        
        champion_model = mlflow.sklearn.load_model(f"runs:/{champion_run_id}/model")
        preprocessor = champion_model.named_steps["preprocessor"]
        predictor = champion_model.named_steps["model"]
        
        df_ref = pd.read_csv("src/training/reference_data.csv")
        features_list = [
            "category", "amt", "gender", "distance_achat", "age", "city_pop",
            "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos"
        ]
        
        df_normal = df_ref[df_ref["is_fraud"] == 0].sample(n=800, random_state=42)
        df_fraud = df_ref[df_ref["is_fraud"] == 1].sample(n=200, random_state=42)
        df_sample_resorted = pd.concat([df_normal, df_fraud]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        X_samp = df_sample_resorted[features_list]
        y_samp = df_sample_resorted["is_fraud"]
        X_enc = preprocessor.transform(X_samp)
        
        # Traduction des variables pour la lisibilité
        features_dict = {
            "amt": "Montant (€)",
            "distance_achat": "Distance d'achat (km)",
            "age": "Âge du client",
            "city_pop": "Population de la ville",
            "category": "Catégorie d'achat",
            "gender": "Genre",
            "hour_sin": "Heure (trigonométrique sinus)",
            "hour_cos": "Heure (trigonométrique cosinus)",
            "weekday_sin": "Jour de la semaine (sinus)",
            "weekday_cos": "Jour de la semaine (cosinus)",
            "month_sin": "Mois de l'année (sinus)",
            "month_cos": "Mois de l'année (cosinus)"
        }

        xpl_obj = SmartExplainer(model=predictor, features_dict=features_dict)
        
        def dummy_get_interaction_values(selection=None, n_samples_max=None):
            n_samp = len(selection) if selection is not None else 100
            n_feat = X_enc.shape[1]
            return np.zeros((n_samp, n_feat, n_feat))
            
        xpl_obj.get_interaction_values = dummy_get_interaction_values
        xpl_obj.compile(x=X_enc, y_target=y_samp)
        return xpl_obj, df_sample_resorted, X_enc, champion_metrics, champion_params, f"Version {version_details.version}"
        
    except Exception as err:
        st.warning(f"Impossible de charger via l'alias champion, repli sur le dernier run : {err}")
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name("Default")
            runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
            if len(runs) > 0:
                latest_run = runs[0]
                champion_run_id = latest_run.info.run_id
                champion_metrics = latest_run.data.metrics
                champion_params = latest_run.data.params
                champion_model = mlflow.sklearn.load_model(f"runs:/{champion_run_id}/model")
                
                preprocessor = champion_model.named_steps["preprocessor"]
                predictor = champion_model.named_steps["model"]
                
                df_ref = pd.read_csv("src/training/reference_data.csv")
                features_list = [
                    "category", "amt", "gender", "distance_achat", "age", "city_pop",
                    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos"
                ]
                df_normal = df_ref[df_ref["is_fraud"] == 0].sample(n=800, random_state=42)
                df_fraud = df_ref[df_ref["is_fraud"] == 1].sample(n=200, random_state=42)
                df_sample_resorted = pd.concat([df_normal, df_fraud]).sample(frac=1.0, random_state=42).reset_index(drop=True)
                
                X_samp = df_sample_resorted[features_list]
                y_samp = df_sample_resorted["is_fraud"]
                X_enc = preprocessor.transform(X_samp)
                
                # Traduction des variables pour la lisibilité
                features_dict = {
                    "amt": "Montant (€)",
                    "distance_achat": "Distance d'achat (km)",
                    "age": "Âge du client",
                    "city_pop": "Population de la ville",
                    "category": "Catégorie d'achat",
                    "gender": "Genre",
                    "hour_sin": "Heure (trigonométrique sinus)",
                    "hour_cos": "Heure (trigonométrique cosinus)",
                    "weekday_sin": "Jour de la semaine (sinus)",
                    "weekday_cos": "Jour de la semaine (cosinus)",
                    "month_sin": "Mois de l'année (sinus)",
                    "month_cos": "Mois de l'année (cosinus)"
                }

                xpl_obj = SmartExplainer(model=predictor, features_dict=features_dict)
                
                def dummy_get_interaction_values(selection=None, n_samples_max=None):
                    n_samp = len(selection) if selection is not None else 100
                    n_feat = X_enc.shape[1]
                    return np.zeros((n_samp, n_feat, n_feat))
                    
                xpl_obj.get_interaction_values = dummy_get_interaction_values
                xpl_obj.compile(x=X_enc, y_target=y_samp)
                return xpl_obj, df_sample_resorted, X_enc, champion_metrics, champion_params, "Dernier Run"
        except Exception as final_err:
            st.error(f"Erreur critique lors de l'initialisation Shapash de secours : {final_err}")
            return None, None, None, None, None, None

with st.spinner("Chargement du modèle champion et calcul des contributions SHAP..."):
    xpl, df_sample, X_encoded, metrics, params, model_ver = load_shapash_explainer()

if xpl is not None:
    # --- GROUPEMENT DES VARIABLES CYCLIQUES ---
    import numpy as np
    
    # 1. Recalcul des valeurs décodées humaines
    angle_h = np.arctan2(xpl.x_init["hour_sin"], xpl.x_init["hour_cos"]) % (2 * np.pi)
    xpl.x_init["Heure"] = np.round(angle_h * 12.0 / np.pi) % 24
    
    angle_w = np.arctan2(xpl.x_init["weekday_sin"], xpl.x_init["weekday_cos"]) % (2 * np.pi)
    xpl.x_init["Jour de la semaine"] = np.round(angle_w * 3.5 / np.pi) % 7
    
    angle_m = np.arctan2(xpl.x_init["month_sin"], xpl.x_init["month_cos"]) % (2 * np.pi)
    decoded_m = np.round(angle_m * 6.0 / np.pi)
    decoded_m = np.where(decoded_m == 0, 12, decoded_m)
    xpl.x_init["Mois de l'année"] = decoded_m
    
    # 2. Somme des contributions SHAP (gestion si c'est une liste de DataFrames pour classification binaire/multi)
    to_drop = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos"]
    if isinstance(xpl.contributions, list):
        for c_df in xpl.contributions:
            c_df["Heure"] = c_df["hour_sin"] + c_df["hour_cos"]
            c_df["Jour de la semaine"] = c_df["weekday_sin"] + c_df["weekday_cos"]
            c_df["Mois de l'année"] = c_df["month_sin"] + c_df["month_cos"]
            c_df.drop(columns=to_drop, inplace=True)
    else:
        xpl.contributions["Heure"] = xpl.contributions["hour_sin"] + xpl.contributions["hour_cos"]
        xpl.contributions["Jour de la semaine"] = xpl.contributions["weekday_sin"] + xpl.contributions["weekday_cos"]
        xpl.contributions["Mois de l'année"] = xpl.contributions["month_sin"] + xpl.contributions["month_cos"]
        xpl.contributions = xpl.contributions.drop(columns=to_drop)
        
    # 3. Suppression des variables sin/cos d'origine de x_init
    xpl.x_init = xpl.x_init.drop(columns=to_drop)
    
    # Mise à jour de X_encoded (utilisé pour les sélecteurs de features)
    X_encoded = X_encoded.drop(columns=to_drop)
    X_encoded["Heure"] = xpl.x_init["Heure"]
    X_encoded["Jour de la semaine"] = xpl.x_init["Jour de la semaine"]
    X_encoded["Mois de l'année"] = xpl.x_init["Mois de l'année"]
    
    # 4. Alignement des dictionnaires Shapash
    for col in to_drop:
        if col in xpl.features_dict:
            del xpl.features_dict[col]
            
    xpl.features_dict["Heure"] = "Heure de la transaction"
    xpl.features_dict["Jour de la semaine"] = "Jour de la semaine"
    xpl.features_dict["Mois de l'année"] = "Mois de l'année"
    
    xpl.columns_dict = {col: col for col in xpl.x_init.columns}
    # ------------------------------------------
    # SECTION A : PERFORMANCES DU MODÈLE CHAMPION
    st.markdown(f"### 📊 Performances du modèle champion ({model_ver})")
    
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("F1-Score Fraude (Classe 1)", f"{metrics.get('f1_class_1', 0.0):.4f}")
    with c_m2:
        st.metric("Rappel Fraude (Recall C1)", f"{metrics.get('rec_class_1', 0.0):.4f}")
    with c_m3:
        st.metric("Précision Fraude (Prec C1)", f"{metrics.get('prec_class_1', 0.0):.4f}")
    with c_m4:
        st.metric("F1 Macro (Global)", f"{metrics.get('F1_global', 0.0):.4f}")
        
    st.markdown("**Paramètres clés du modèle :**")
    st.code(
        f"max_depth: {params.get('max_depth')}  |  learning_rate: {params.get('learning_rate')}  |  n_estimators: {params.get('n_estimators')}  |  scale_pos_weight: {params.get('scale_pos_weight')}"
    )
    
    st.markdown("---")
    
    # SECTION B : GLOBAL FEATURE IMPORTANCE PLOT
    st.markdown("### 📈 1. Importance globale des caractéristiques (global feature importance)")
    
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
    st.markdown("### 📈 2. Courbes de contribution individuelle (features contribution plots)")
    st.write("Ces courbes affichent l'impact d'une caractéristique spécifique sur le score de fraude. Elles permettent de voir si des montants ou distances plus élevés augmentent le score de suspicion.")
    
    available_features = X_encoded.columns.tolist()
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
    st.write("Le modèle champion utilise des features cycliques trigonométriques pour comprendre le temps. Ci-dessous, l'outil décode ces valeurs en coordonnées d'origine (Heure, Jour de la semaine, Mois).")
    
    selected_idx_inverse = st.number_input(
        f"Sélectionnez l'index de la transaction à décoder (0 à {len(df_sample) - 1}) :",
        min_value=0, max_value=len(df_sample) - 1, value=0, key="inverse_tool_idx"
    )
    
    tx_inv = df_sample.iloc[selected_idx_inverse]
    encoded_inv = X_encoded.iloc[selected_idx_inverse]
    
    angle_h = np.arctan2(tx_inv["hour_sin"], tx_inv["hour_cos"]) % (2 * np.pi)
    decoded_hour = int(np.round(angle_h * 12.0 / np.pi) % 24)
    
    angle_w = np.arctan2(tx_inv["weekday_sin"], tx_inv["weekday_cos"]) % (2 * np.pi)
    decoded_weekday = int(np.round(angle_w * 3.5 / np.pi) % 7)
    weekdays_names = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    
    angle_m = np.arctan2(tx_inv["month_sin"], tx_inv["month_cos"]) % (2 * np.pi)
    decoded_month = int(np.round(angle_m * 6.0 / np.pi))
    decoded_month = 12 if decoded_month == 0 else decoded_month
    months_names = ["", "Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
    
    col_enc, col_dec, col_orig = st.columns(3)
    with col_enc:
        st.markdown("**1. Valeurs Encodées**")
        st.write(f"hour_sin/cos : `{tx_inv['hour_sin']:.4f}` / `{tx_inv['hour_cos']:.4f}`")
        st.write(f"weekday_sin/cos : `{tx_inv['weekday_sin']:.4f}` / `{tx_inv['weekday_cos']:.4f}`")
        st.write(f"month_sin/cos : `{tx_inv['month_sin']:.4f}` / `{tx_inv['month_cos']:.4f}`")
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
    st.write("Ce graphique montre le détail des contributions SHAP pour la transaction sélectionnée ci-dessus.")
    
    col_local_details, col_local_plot = st.columns([1, 2])
    with col_local_details:
        st.markdown("##### Paramètres d'Entrée")
        st.write(f"💳 **Carte :** `{tx_inv['cc_num']}`")
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
        fig_local = xpl.plot.local_plot(index=selected_idx_inverse)
        st.plotly_chart(fig_local, use_container_width=True)
else:
    st.warning("L'explicateur Shapash n'a pas pu être chargé.")
