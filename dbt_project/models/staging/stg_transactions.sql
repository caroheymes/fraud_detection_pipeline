with source as (
    select * from {{ source('silver_raw', 'rawdata') }}
),
renamed as (
    select
        trans_num as transaction_id,
        trans_date_trans_time as transaction_timestamp,
        cc_num as credit_card_number,
        merchant as merchant_name,
        category as transaction_category,
        amt as transaction_amount,
        gender as customer_gender,
        age as customer_age,
        city_pop as city_population,
        distance_achat,
        prediction as is_predicted_fraud,
        prediction_proba as prediction_probability,
        model_version,
        fast_pass_suspicion,
        fast_pass_score,
        logged_at
    from source
)
select * from renamed
