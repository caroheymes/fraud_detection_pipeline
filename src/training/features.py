# src/training/features.py
def get_historical_features():
    # Simulation de requêtes SQL analytiques sur Postgres OLAP (tables dbt)
    print("Extraction des features historiques depuis Postgres (Public.dbt_marts)...")

    query = """
    SELECT 
        user_id,
        count_transactions_1h,
        sum_amount_24h,
        avg_amount_7d,
        is_fraud
    FROM public.marts_transactions_features;
    """
    return query
