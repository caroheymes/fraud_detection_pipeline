with transactions as (
    select * from {{ ref('stg_transactions') }}
),
blocked as (
    select
        transaction_id,
        transaction_timestamp,
        merchant_name,
        credit_card_number,
        transaction_amount,
        transaction_category,
        customer_age,
        customer_gender,
        distance_achat,
        fast_pass_suspicion,
        fast_pass_score,
        is_predicted_fraud,
        prediction_probability,
        model_version
    from transactions
    where is_predicted_fraud = 1 or fast_pass_suspicion = 1
)
select * from blocked
