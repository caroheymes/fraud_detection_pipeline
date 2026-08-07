with transactions as (
    -- Extraction des valeurs SHAP du JSONB dans le staging Silver
    select
        merchant_name,
        is_predicted_fraud,
        cast(shap_values->>'amt' as numeric) as amt_shap,
        cast(shap_values->>'distance_achat' as numeric) as distance_shap,
        cast(shap_values->>'age' as numeric) as age_shap,
        cast(shap_values->>'city_pop' as numeric) as city_pop_shap,
        -- Somme cumulée du facteur temps
        (cast(shap_values->>'hour_sin' as numeric) + cast(shap_values->>'hour_cos' as numeric)) as time_shap
    from {{ ref('stg_transactions') }}
    where is_predicted_fraud = 1 -- On se concentre uniquement sur les fraudes
),
merchant_shap as (
    select
        merchant_name,
        count(*) as total_fraud_cases,
        -- Moyenne de l'impact absolu de chaque variable pour ce marchand
        round(avg(abs(amt_shap)), 3) as avg_amt_impact,
        round(avg(abs(distance_shap)), 3) as avg_distance_impact,
        round(avg(abs(age_shap)), 3) as avg_age_impact,
        round(avg(abs(city_pop_shap)), 3) as avg_city_pop_impact,
        round(avg(abs(time_shap)), 3) as avg_time_impact
    from transactions
    group by 1
)
select * from merchant_shap
