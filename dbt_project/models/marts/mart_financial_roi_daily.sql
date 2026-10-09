-- dbt_project/models/marts/mart_financial_roi_daily.sql
with transactions as (
    select * from {{ ref('stg_transactions') }}
),
daily as (
    select
        date_trunc('day', transaction_timestamp)::date as transaction_date,
        merchant_name,
        transaction_category,
        model_version,
        count(transaction_id) as total_transactions,
        sum(transaction_amount) as total_volume,
        -- Vrais Positifs (TP) : Fraudes réelles interceptées par le modèle
        count(case when is_actual_fraud = 1 and is_predicted_fraud = 1 then 1 end) as tp_count,
        coalesce(sum(case when is_actual_fraud = 1 and is_predicted_fraud = 1 then transaction_amount else 0 end), 0) as tp_volume,
        -- Faux Négatifs (FN) : Fraudes réelles manquées (pertes subies)
        count(case when is_actual_fraud = 1 and is_predicted_fraud = 0 then 1 end) as fn_count,
        coalesce(sum(case when is_actual_fraud = 1 and is_predicted_fraud = 0 then transaction_amount else 0 end), 0) as fn_volume,
        -- Faux Positifs (FP) : Transactions saines bloquées à tort (friction client)
        count(case when is_actual_fraud = 0 and is_predicted_fraud = 1 then 1 end) as fp_count,
        coalesce(sum(case when is_actual_fraud = 0 and is_predicted_fraud = 1 then transaction_amount else 0 end), 0) as fp_volume,
        -- Vrais Négatifs (TN) : Transactions saines approuvées
        count(case when is_actual_fraud = 0 and is_predicted_fraud = 0 then 1 end) as tn_count,
        coalesce(sum(case when is_actual_fraud = 0 and is_predicted_fraud = 0 then transaction_amount else 0 end), 0) as tn_volume
    from transactions
    where is_actual_fraud is not null and is_predicted_fraud is not null
    group by 1, 2, 3, 4
)
select * from daily
