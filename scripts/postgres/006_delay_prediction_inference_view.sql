CREATE SCHEMA IF NOT EXISTS delay_prediction_evidence;

CREATE OR REPLACE VIEW delay_prediction_evidence.vw_delay_prediction_inference_orders AS
WITH filtered_orders AS (
    SELECT DISTINCT
        co.order_id,
        co.product_id,
        co.order_quantity,
        co.order_date,
        co.due_date
    FROM customer_orders co
    WHERE COALESCE(UPPER(co.order_status::text), '') NOT IN ('COMPLETE', 'COMPLETED', 'CANCELLED')
),
plan_features AS (
    SELECT
        pp.order_id,
        COUNT(DISTINCT pp.line_id) AS line_count,
        COUNT(*) AS total_plan_count,
        COUNT(*) FILTER (
            WHERE pp.planned_end_at > (fo.due_date::timestamp without time zone + '23:59:59'::interval)::timestamp with time zone
        ) AS overdue_plan_count,
        COALESCE(SUM(pp.estimated_duration_hr), 0)::numeric AS estimated_duration_hr_sum,
        COALESCE(MAX(pp.plan_sequence), 0)::integer AS plan_sequence_max,
        COALESCE(
            MAX(
                GREATEST(
                    EXTRACT(
                        epoch FROM pp.planned_end_at - (fo.due_date::timestamp without time zone + '23:59:59'::interval)::timestamp with time zone
                    ) / 86400.0,
                    0::numeric
                )
            ),
            0::numeric
        ) AS plan_overdue_days_max
    FROM production_plans pp
    JOIN filtered_orders fo
      ON fo.order_id = pp.order_id
    GROUP BY pp.order_id
),
order_lines AS (
    SELECT DISTINCT
        pp.order_id,
        pp.line_id
    FROM production_plans pp
    JOIN filtered_orders fo
      ON fo.order_id = pp.order_id
),
line_features AS (
    SELECT
        ol.order_id,
        COALESCE(AVG(ls.waiting_time_hr), 0)::numeric AS waiting_time_hr_avg,
        COALESCE(AVG(ls.utilization_rate), 0)::numeric AS utilization_rate_avg
    FROM order_lines ol
    LEFT JOIN line_status ls
      ON ls.line_id = ol.line_id
    GROUP BY ol.order_id
),
material_features AS (
    SELECT
        pp.order_id,
        CASE
            WHEN COALESCE(SUM(ppm.required_quantity), 0) = 0::numeric THEN 0::numeric
            ELSE COALESCE(SUM(ppm.shortage_quantity), 0)::numeric
                 / NULLIF(SUM(ppm.required_quantity), 0)::numeric
        END AS shortage_ratio
    FROM production_plans pp
    JOIN filtered_orders fo
      ON fo.order_id = pp.order_id
    LEFT JOIN production_plan_materials ppm
      ON ppm.plan_id = pp.plan_id
    GROUP BY pp.order_id
)
SELECT
    fo.order_id,
    fo.order_quantity,
    (fo.due_date - fo.order_date)::integer AS due_gap_days,
    fo.product_id,
    COALESCE(p.average_yield_rate, 0)::numeric AS average_yield_rate,
    COALESCE(pf.line_count, 0)::bigint AS line_count,
    COALESCE(pf.estimated_duration_hr_sum, 0)::numeric AS estimated_duration_hr_sum,
    COALESCE(pf.plan_sequence_max, 0)::integer AS plan_sequence_max,
    COALESCE(pf.plan_overdue_days_max, 0)::numeric AS plan_overdue_days_max,
    COALESCE(lf.waiting_time_hr_avg, 0)::numeric AS waiting_time_hr_avg,
    COALESCE(lf.utilization_rate_avg, 0)::numeric AS utilization_rate_avg,
    COALESCE(mf.shortage_ratio, 0)::numeric AS shortage_ratio,
    NULL::numeric AS actual_delay_hr_sum,
    CASE
        WHEN (fo.due_date - fo.order_date) <= 0 THEN NULL::numeric
        ELSE COALESCE(pf.estimated_duration_hr_sum, 0)::numeric
             / NULLIF((fo.due_date - fo.order_date)::numeric * 24.0, 0::numeric)
    END AS due_load_ratio,
    COALESCE(pf.overdue_plan_count, 0)::bigint AS overdue_plan_count,
    CASE
        WHEN COALESCE(pf.total_plan_count, 0) = 0 THEN 0::numeric
        ELSE COALESCE(pf.overdue_plan_count, 0)::numeric / pf.total_plan_count::numeric
    END AS overdue_plan_ratio
FROM filtered_orders fo
JOIN products p
  ON p.product_id = fo.product_id
LEFT JOIN plan_features pf
  ON pf.order_id = fo.order_id
LEFT JOIN line_features lf
  ON lf.order_id = fo.order_id
LEFT JOIN material_features mf
  ON mf.order_id = fo.order_id;
