with transactions as (
    select * from {{ ref('stg_transactions') }}
),
sla_metrics as (
    select
        date_trunc('day', transaction_timestamp)::date as check_date,
        count(transaction_id) as total_api_calls,
        -- Latence opérationnelle
        round(avg(prediction_latency_ms)::numeric, 2) as avg_latency_ms,
        round(max(prediction_latency_ms)::numeric, 2) as max_latency_ms,
        -- Respect du SLA (< 20 ms pour l'inférence temps réel d'un paiement en caisse)
        round(
            (sum(case when prediction_latency_ms < 20.0 then 1 else 0 end)::numeric / count(transaction_id)::numeric) * 100,
            2
        ) as sla_compliance_percentage
    from transactions
    group by 1
)
select * from sla_metrics
