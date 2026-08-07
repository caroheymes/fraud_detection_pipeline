with transactions as (
    select * from {{ ref('stg_transactions') }}
),
performance as (
    select
        date_trunc('day', transaction_timestamp)::date as transaction_date,
        count(transaction_id) as total_transactions,
        -- Detections par le Fast Pass
        sum(case when fast_pass_suspicion = 1 then 1 else 0 end) as fast_pass_detections,
        sum(case when fast_pass_suspicion = 1 then transaction_amount else 0 end) as fast_pass_blocked_amount,
        -- Detections par le modèle XGBoost uniquement (non suspectées par le Fast Pass mais prédites fraude par ML)
        sum(case when fast_pass_suspicion = 0 and is_predicted_fraud = 1 then 1 else 0 end) as xgboost_only_detections,
        sum(case when fast_pass_suspicion = 0 and is_predicted_fraud = 1 then transaction_amount else 0 end) as xgboost_only_blocked_amount,
        -- Total détecté
        sum(case when is_predicted_fraud = 1 or fast_pass_suspicion = 1 then 1 else 0 end) as total_detections,
        sum(case when is_predicted_fraud = 1 or fast_pass_suspicion = 1 then transaction_amount else 0 end) as total_blocked_amount
    from transactions
    group by 1
)
select * from performance
