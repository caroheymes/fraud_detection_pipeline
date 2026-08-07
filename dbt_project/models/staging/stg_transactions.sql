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
        -- Calcul de l'âge du client par rapport à l'année du dataset (2020)
        (2020 - extract(year from dob))::int as customer_age,
        city_pop as city_population,
        -- Calcul de la distance d'achat Haversine en SQL Postgres
        (6371.0 * 2.0 * asin(
            sqrt(
                power(sin(radians(merch_lat - lat) / 2.0), 2) +
                cos(radians(lat)) * cos(radians(merch_lat)) *
                power(sin(radians(merch_long - long) / 2.0), 2)
            )
        ))::numeric(10, 2) as distance_achat,
        prediction as is_predicted_fraud,
        prediction_proba as prediction_probability,
        model_version,
        fast_pass_suspicion,
        fast_pass_score,
        prediction_latency_ms,
        shap_values,
        logged_at
    from source
)
select * from renamed
