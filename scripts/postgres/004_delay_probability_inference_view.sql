CREATE SCHEMA IF NOT EXISTS delay_prediction_evidence;

CREATE OR REPLACE VIEW delay_prediction_evidence.vw_delay_probability_inference_orders AS
WITH plan_base AS (
    SELECT
        pp.plan_id,
        pp.order_id,
        pp.product_id,
        pp.line_id,
        pp.planned_start_at,
        pp.planned_end_at,
        pp.estimated_duration_hr,
        pp.planned_quantity,
        pp.plan_sequence,
        pp.plan_status
    FROM public.production_plans pp
    WHERE pp.plan_status::text <> 'CANCELLED'
),

plan_agg AS (
    SELECT
        pb.order_id,
        COUNT(*)::int AS plan_count,
        COUNT(DISTINCT pb.line_id)::int AS assigned_line_count,
        SUM(pb.planned_quantity)::numeric AS total_planned_quantity,
        SUM(pb.estimated_duration_hr)::numeric AS total_estimated_duration_hr,
        AVG(pb.estimated_duration_hr)::numeric AS avg_estimated_duration_hr,
        MIN(pb.planned_start_at) AS first_planned_start_at,
        MAX(pb.planned_end_at) AS last_planned_end_at
    FROM plan_base pb
    GROUP BY pb.order_id
),

primary_plan AS (
    SELECT DISTINCT ON (pb.order_id)
        pb.order_id,
        pb.plan_id,
        pb.line_id
    FROM plan_base pb
    ORDER BY
        pb.order_id,
        pb.planned_start_at ASC,
        pb.plan_sequence ASC,
        pb.plan_id ASC
),

capability_by_planned_lines AS (
    SELECT
        pb.order_id,
        AVG(plc.standard_production_time_hr)::numeric AS avg_standard_production_time_hr,
        AVG(plc.capacity_per_day)::numeric AS avg_capacity_per_day,
        AVG(plc.standard_yield_rate)::numeric AS avg_standard_yield_rate
    FROM plan_base pb
    JOIN public.product_line_capabilities plc
      ON plc.product_id = pb.product_id
     AND plc.line_id = pb.line_id
    GROUP BY pb.order_id
),

capability_by_product AS (
    SELECT
        plc.product_id,
        AVG(plc.standard_production_time_hr)::numeric AS avg_standard_production_time_hr,
        AVG(plc.capacity_per_day)::numeric AS avg_capacity_per_day,
        AVG(plc.standard_yield_rate)::numeric AS avg_standard_yield_rate
    FROM public.product_line_capabilities plc
    GROUP BY plc.product_id
),

default_line_by_product AS (
    SELECT DISTINCT ON (plc.product_id)
        plc.product_id,
        plc.line_id
    FROM public.product_line_capabilities plc
    ORDER BY
        plc.product_id,
        plc.priority_rank ASC NULLS LAST,
        plc.line_id ASC
),

bom_agg AS (
    SELECT
        b.product_id,
        COUNT(DISTINCT b.material_id)::int AS distinct_material_count
    FROM public.boms b
    GROUP BY b.product_id
),

material_agg AS (
    SELECT
        pb.order_id,
        COUNT(DISTINCT ppm.material_id)::int AS plan_material_count,

        MAX(
            CASE
                WHEN ppm.shortage_quantity > 0
                  OR ppm.material_plan_status::text = 'SHORTAGE'
                THEN 1
                ELSE 0
            END
        )::int AS ever_had_material_shortage,

        COALESCE(
            MAX(
                GREATEST(
                    EXTRACT(EPOCH FROM (mi.expected_inbound_at - pb.planned_start_at)) / 86400.0,
                    0
                )
            ),
            0
        )::numeric AS inbound_delay_days,

        CASE
            WHEN SUM(
                CASE
                    WHEN ppm.shortage_quantity > 0
                      OR ppm.material_plan_status::text = 'SHORTAGE'
                      OR (
                            mi.expected_inbound_at IS NOT NULL
                        AND mi.expected_inbound_at > pb.planned_start_at
                      )
                    THEN 1
                    ELSE 0
                END
            ) > 0
            THEN 0
            ELSE 1
        END::int AS material_ready_before_start

    FROM plan_base pb
    LEFT JOIN public.production_plan_materials ppm
      ON ppm.plan_id = pb.plan_id
    LEFT JOIN public.material_inventories mi
      ON mi.material_id = ppm.material_id
    GROUP BY pb.order_id
)

SELECT
    co.order_id,
    co.product_id,

    -- 주문 단위 예측이므로 plan_id는 예측 기준으로 사용하지 않습니다.
    NULL::bigint AS plan_id,

    COALESCE(pp.line_id, dlp.line_id) AS line_id,

    p.product_code,
    COALESCE(pp.line_id, dlp.line_id) AS primary_line_id,

    co.order_quantity,

    EXTRACT(MONTH FROM co.order_date)::int AS order_month,
    (EXTRACT(ISODOW FROM co.order_date)::int - 1) AS order_dayofweek,

    EXTRACT(MONTH FROM co.due_date)::int AS due_month,
    (EXTRACT(ISODOW FROM co.due_date)::int - 1) AS due_dayofweek,

    COALESCE(
        (pa.first_planned_start_at::date - co.order_date)::numeric,
        0
    ) AS order_to_plan_start_days,

    CASE
        WHEN COALESCE(pa.plan_count, 0) > 1 THEN 1
        ELSE 0
    END AS is_multi_plan,

    CASE
        WHEN COALESCE(pa.assigned_line_count, 0) > 1 THEN 1
        ELSE 0
    END AS is_multi_line,

    COALESCE(pa.avg_estimated_duration_hr, 0)::numeric AS avg_estimated_duration_hr,

    COALESCE(p.min_production_quantity, 0)::numeric AS min_production_quantity,

    COALESCE(
        cbpl.avg_standard_production_time_hr,
        cbp.avg_standard_production_time_hr,
        0
    )::numeric AS avg_standard_production_time_hr,

    COALESCE(
        cbpl.avg_capacity_per_day,
        cbp.avg_capacity_per_day,
        0
    )::numeric AS avg_capacity_per_day,

    COALESCE(
        cbpl.avg_standard_yield_rate,
        cbp.avg_standard_yield_rate,
        p.average_yield_rate,
        0
    )::numeric AS avg_standard_yield_rate,

    COALESCE(
        ba.distinct_material_count,
        ma.plan_material_count,
        0
    )::int AS distinct_material_count,

    COALESCE(ma.ever_had_material_shortage, 0)::int AS ever_had_material_shortage,

    COALESCE(ma.inbound_delay_days, 0)::numeric AS inbound_delay_days,

    COALESCE(ma.material_ready_before_start, 1)::int AS material_ready_before_start,

    CASE
        WHEN co.order_quantity > 0
        THEN COALESCE(pa.total_planned_quantity, 0)::numeric / co.order_quantity::numeric
        ELSE 0
    END AS planned_quantity_ratio,

    GREATEST((co.due_date - co.order_date), 0)::numeric AS order_lead_time_days,

    COALESCE(
        EXTRACT(
            EPOCH FROM (
                (co.due_date::timestamp + INTERVAL '1 day')
                - COALESCE(pa.last_planned_end_at, co.order_date::timestamp)
            )
        ) / 3600.0,
        0
    )::numeric AS due_margin_hr,

    COALESCE(pa.total_estimated_duration_hr, 0)::numeric AS total_estimated_duration_hr

FROM public.customer_orders co
JOIN public.products p
  ON p.product_id = co.product_id
LEFT JOIN plan_agg pa
  ON pa.order_id = co.order_id
LEFT JOIN primary_plan pp
  ON pp.order_id = co.order_id
LEFT JOIN default_line_by_product dlp
  ON dlp.product_id = co.product_id
LEFT JOIN capability_by_planned_lines cbpl
  ON cbpl.order_id = co.order_id
LEFT JOIN capability_by_product cbp
  ON cbp.product_id = co.product_id
LEFT JOIN bom_agg ba
  ON ba.product_id = co.product_id
LEFT JOIN material_agg ma
  ON ma.order_id = co.order_id
WHERE co.order_status::text <> 'CANCELLED';