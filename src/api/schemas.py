# src/api/schemas.py
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    transaction_id: str = Field(..., example="tx_987654")
    user_id: str = Field(..., example="usr_112233")
    amount: float = Field(..., gt=0, example=150.50)
    ip_address: str = Field(..., example="192.168.1.50")
    country: str = Field(..., min_length=2, max_length=3, example="FRA")


class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_score: float
    latency_ms: float
