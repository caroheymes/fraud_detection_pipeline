# src/api/schemas.py
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    trans_date_trans_time: str
    cc_num: int
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: int
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    trans_num: str
    unix_time: int
    merch_lat: float
    merch_long: float
    is_fraud: int = 0


class TransactionBatch(BaseModel):
    transactions: list[TransactionInput]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transactions": [
                        {
                            "trans_date_trans_time": "2020-07-22 14:05:00",
                            "cc_num": 423578912345,
                            "merchant": "fraud_gas_station",
                            "category": "gas_transport",
                            "amt": 85.50,
                            "first": "Caro",
                            "last": "MS",
                            "gender": "F",
                            "street": "12 rue de la Paix",
                            "city": "Lyon",
                            "state": "Rhone",
                            "zip": 69000,
                            "lat": 45.764043,
                            "long": 4.835659,
                            "city_pop": 513000,
                            "job": "Data Ingé",
                            "dob": "1985-04-12",
                            "trans_num": "test_tx_001",
                            "unix_time": 1595426700,
                            "merch_lat": 45.768000,
                            "merch_long": 4.840000,
                            "is_fraud": 0,
                        }
                    ]
                }
            ]
        }
    }


class WebhookData(BaseModel):
    transaction_id: str
    cc_num_sha256: str
    amount: float
    category: str
    merchant: str
    prediction: int
    prediction_proba: float
    explications_shap: dict[str, float] = Field(default_factory=dict)


class WebhookPayload(BaseModel):
    event: str
    timestamp: str
    data: WebhookData


class WebhookResponse(BaseModel):
    status: str
    message: str


class WebhookRequest(BaseModel):
    transaction_id: str
