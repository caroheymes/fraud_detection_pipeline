# src/api/main.py
import time

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="API de Détection de Fraude - MLOps",
    description="Inférence temps réel et détection de fraude avec sauvegarde asynchrone.",
    version="1.0.0",
)


# Schéma d'entrée de transaction
class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    ip_address: str
    country: str


# Simulation de chargement de modèle
class ModelLoader:
    def __init__(self):
        self.model = "Modele_Chargé_En_Mémoire"

    def predict(self, transaction: Transaction) -> float:
        # Algorithme de détection de fraude simulé
        if transaction.amount > 5000:
            return 0.95  # Probabilité de fraude élevée
        return 0.05


model_loader = ModelLoader()


def log_to_postgres(transaction: Transaction, score: float):
    # Simulation d'écriture dans Postgres
    print(
        f"[Postgres Log] Sauvegarde de la transaction {transaction.transaction_id} (Score: {score})"
    )
    # Implémentation réelle : Connexion SQLAlchemy/psycopg2 et insertion


def send_alert_email(transaction: Transaction, score: float):
    # Simulation d'envoi d'e-mail via notifier.py
    print(
        f"[Alerte Mail] Notification marchand pour la transaction suspecte {transaction.transaction_id} (Score: {score})"
    )


@app.post("/predict", response_model=dict)
def predict_fraud(transaction: Transaction, background_tasks: BackgroundTasks):
    start_time = time.time()

    # 1. Inférence synchrone immédiate
    fraud_score = model_loader.predict(transaction)
    is_fraud = fraud_score > 0.8

    # 2. Ingestion asynchrone (Postgres) et Notification
    background_tasks.add_task(log_to_postgres, transaction, fraud_score)
    if is_fraud:
        background_tasks.add_task(send_alert_email, transaction, fraud_score)

    latency_ms = (time.time() - start_time) * 1000

    return {
        "transaction_id": transaction.transaction_id,
        "is_fraud": is_fraud,
        "fraud_score": fraud_score,
        "latency_ms": round(latency_ms, 2),
    }


@app.post("/reload-model")
def reload_model():
    # Route appelée par Airflow pour recharger le modèle après réentraînement
    global model_loader
    model_loader = ModelLoader()
    return {
        "status": "success",
        "message": "Nouveau modèle chargé en mémoire avec succès.",
    }
