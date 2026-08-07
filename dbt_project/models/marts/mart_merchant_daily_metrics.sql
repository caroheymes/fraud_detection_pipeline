with transactions as (
    select * from {{ ref('stg_transactions') }}
),
daily as (
    select
        date_trunc('day', transaction_timestamp)::date as transaction_date,
        merchant_name,
        count(transaction_id) as total_transactions,
        sum(transaction_amount) as total_volume,
        -- Volume/nombre sains
        sum(case when is_predicted_fraud = 0 then transaction_amount else 0 end) as clean_volume,
        sum(case when is_predicted_fraud = 0 then 1 else 0 end) as clean_transactions_count,
        -- Volume/nombre fraude (perte potentielle)
        sum(case when is_predicted_fraud = 1 then transaction_amount else 0 end) as fraud_volume_loss,
        sum(case when is_predicted_fraud = 1 then 1 else 0 end) as fraud_transactions_count,
        -- Taux de fraude
        round(
            (sum(case when is_predicted_fraud = 1 then 1 else 0 end)::numeric / count(transaction_id)::numeric) * 100, 
            2
        ) as fraud_rate_percentage
    from transactions
    group by 1, 2
)
select * from daily
