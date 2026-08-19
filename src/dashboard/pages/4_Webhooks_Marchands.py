# src/dashboard/pages/4_Webhooks_Marchands.py

import json
import os

import pandas as pd
import redis
import streamlit as st

st.set_page_config(
    page_title="Webhooks Marchands (Live)", page_icon="🔔", layout="wide"
)

st.title("🔔 Webhooks marchands en direct")
st.write(
    "Historique en temps réel des alertes de fraudes envoyées aux serveurs des marchands partenaires (toutes les alertes des 24 dernières heures glissantes sont conservées dans Redis)."
)
st.markdown("---")

# --- Section de simulation & cURL ---
with st.expander("🛠️ Espace de test & Intégration API Marchand (cURL)"):
    st.write("Les marchands partenaires reçoivent des notifications webhooks en temps réel lors de suspicions de fraude. Vous pouvez simuler manuellement l'arrivée d'une alerte en appelant l'endpoint de réception :")
    st.code("https://fraud-detection.ngrok.app/mock-merchant-webhook", language="text")
    
    st.write("Voici un exemple de commande **cURL** incluant un **Auth-Token de sécurité (X-Merchant-Token)** et le **numéro de carte chiffré par SHA-256** :")
    
    import hashlib
    cc_sha = hashlib.sha256(b"423578912345").hexdigest()
    token_sha = hashlib.sha256(b"merchant_secret_key_2026").hexdigest()
    
    mock_request = {
        "transaction_id": "simulated_tx_web_999"
    }
    request_str = json.dumps(mock_request, indent=2)
    
    st.code(f"""curl -X POST "https://fraud-detection.ngrok.app/mock-merchant-webhook" \\
  -H "Content-Type: application/json" \\
  -H "X-Merchant-Token: {token_sha}" \\
  -d '{request_str}'""", language="bash")
    
    st.write("💡 **Fonctionnement interne** : Lorsque vous envoyez cette commande, le serveur API interroge sa base de données pour charger toutes les caractéristiques réelles de la transaction, chiffre le numéro de carte bancaire par SHA-256 et renvoie la notification webhook enrichie complète en réponse (que vous verrez s'afficher ci-dessous après rafraîchissement).")
    
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


# 1. Lecture du cache des alertes temps réel depuis Redis (Sorted Set 24h)
alerts_list = []
try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True,
    )
    import time

    now = time.time()
    if r.type("merchant_webhook_alerts") == "list":
        r.delete("merchant_webhook_alerts")
    r.zremrangebyscore("merchant_webhook_alerts", "-inf", now - 86400)
    alerts_raw = r.zrevrange("merchant_webhook_alerts", 0, -1)

    for raw in alerts_raw:
        try:
            alert = json.loads(raw)
            data = alert.get("data", {})
            alerts_list.append(
                {
                    "Date & Heure": alert.get("timestamp", "N/A"),
                    "Événement": alert.get("event", "N/A"),
                    "ID Transaction": data.get("transaction_id", "N/A"),
                    "Marchand": data.get("merchant", "N/A"),
                    "Montant (€)": data.get("amount", 0.0),
                    "Catégorie": data.get("category", "N/A"),
                    "Prédiction Modèle": "🚨 Suspect"
                    if int(data.get("prediction", 0)) == 1
                    else "✅ Sain",
                    "Probabilité Fraude": f"{float(data.get('prediction_proba', 0.0)):.4%}",
                }
            )
        except Exception:
            pass
except Exception as redis_err:
    st.sidebar.warning(f"Connexion Redis indisponible : {redis_err}")

# 2. Récupération de secours depuis PostgreSQL (toutes les fraudes des 24 dernières heures glissantes du flux)
db_query = """
    SELECT 
        trans_date_trans_time::text as "Date & Heure",
        'transaction.suspecte' as "Événement",
        trans_num as "ID Transaction",
        merchant as "Marchand",
        amt as "Montant (€)",
        category as "Catégorie",
        '🚨 Suspect' as "Prédiction Modèle",
        ROUND((prediction_proba * 100.0)::numeric, 4)::text || '%' as "Probabilité Fraude"
    FROM silver.rawdata
    WHERE prediction = 1
      AND trans_date_trans_time >= (SELECT COALESCE(MAX(trans_date_trans_time), NOW()) - INTERVAL '24 hours' FROM silver.rawdata)
    ORDER BY trans_date_trans_time DESC
"""
db_df, db_err = query_db(db_query)
if db_df is not None and not db_df.empty:
    db_list = db_df.to_dict(orient="records")
    seen_ids = set()
    combined_list = []

    # Intégrer Redis en priorité (temps réel)
    for alert in alerts_list:
        tx_id = alert["ID Transaction"]
        if tx_id not in seen_ids:
            seen_ids.add(tx_id)
            combined_list.append(alert)

    # Compléter avec l'historique de PostgreSQL
    for alert in db_list:
        tx_id = alert["ID Transaction"]
        if tx_id not in seen_ids:
            seen_ids.add(tx_id)
            combined_list.append(alert)

    alerts_list = combined_list

# 3. Affichage du tableau de bord et export CSV
if alerts_list:
    df_alerts = pd.DataFrame(alerts_list)

    # Filtre par marchand via un menu déroulant
    st.subheader("🔍 Filtrer les alertes par marchand")
    merchant_options = ["Tous"] + sorted(list(df_alerts["Marchand"].unique()))
    selected_merchant = st.selectbox("Sélectionnez le marchand :", merchant_options)

    if selected_merchant != "Tous":
        df_filtered = df_alerts[df_alerts["Marchand"] == selected_merchant]
    else:
        df_filtered = df_alerts

    st.subheader("📋 Tableau de suivi des webhooks marchands")
    st.dataframe(df_filtered, use_container_width=True)

    # Bouton d'export CSV
    csv_data = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Télécharger les alertes filtrées au format CSV",
        data=csv_data,
        file_name=f"alertes_fraude_{selected_merchant.replace(' ', '_')}.csv",
        mime="text/csv",
    )

    # Petit widget de métrique
    total_alerts = len(df_filtered)
    st.metric(
        f"Alertes affichées pour {selected_merchant}",
        f"{total_alerts} / {len(df_alerts)}",
    )
else:
    st.info(
        "Aucune alerte de fraude suspecte identifiée sur les dernières 24 heures glissantes (Redis et base de données vides)."
    )
