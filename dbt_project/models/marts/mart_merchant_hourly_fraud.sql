with transactions as (
    select * from {{ ref('stg_transactions') }}
),
hourly as (
    select
        merchant_name,
        extract(hour from transaction_timestamp)::int as transaction_hour,
        count(transaction_id) as total_transactions,
        sum(case when is_predicted_fraud = 1 then 1 else 0 end) as fraud_transactions_count,
        sum(case when is_predicted_fraud = 1 then transaction_amount else 0 end) as fraud_volume_loss,
        sum(case when fast_pass_suspicion = 1 then 1 else 0 end) as fast_pass_suspicion_count
    from transactions
    group by 1, 2
)
select * from hourly
