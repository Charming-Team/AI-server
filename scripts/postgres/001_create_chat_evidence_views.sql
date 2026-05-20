begin;

create schema if not exists chat_evidence;

create or replace view chat_evidence.chat_material_shortage_evidence_view as
select
    ppm.plan_material_id,
    ppm.plan_id,
    pp.order_id,
    co.order_no,
    pp.product_id,
    p.product_code,
    p.product_name,
    pp.line_id,
    pl.line_code,
    pl.line_name,
    ppm.material_id,
    m.material_code,
    m.material_name,
    m.material_type,
    ppm.required_quantity,
    ppm.reserved_quantity,
    ppm.consumed_quantity,
    ppm.shortage_quantity,
    m.unit,
    ppm.material_plan_status,
    mi.current_quantity,
    mi.available_quantity,
    mi.reserved_quantity as inventory_reserved_quantity,
    mi.safety_stock_quantity,
    mi.expected_inbound_at,
    mi.expected_inbound_quantity,
    mi.inventory_status,
    pp.planned_start_at,
    pp.planned_end_at,
    pp.plan_status
from public.production_plan_materials ppm
join public.materials m on m.material_id = ppm.material_id
left join public.material_inventories mi on mi.material_id = ppm.material_id
left join public.production_plans pp on pp.plan_id = ppm.plan_id
left join public.customer_orders co on co.order_id = pp.order_id
left join public.products p on p.product_id = pp.product_id
left join public.production_lines pl on pl.line_id = pp.line_id
where ppm.material_plan_status in ('SHORTAGE', 'PARTIAL_RESERVED')
   or ppm.shortage_quantity > 0;

create or replace view chat_evidence.chat_delivery_risk_evidence_view as
select
    apr.prediction_id,
    apr.order_id,
    co.order_no,
    apr.product_id,
    p.product_code,
    p.product_name,
    apr.line_id,
    pl.line_code,
    pl.line_name,
    apr.plan_id,
    co.customer_name,
    co.due_date,
    co.order_status,
    apr.delay_probability,
    apr.risk_level,
    apr.predicted_delay_days,
    null::text as main_cause_type,
    apr.cause_detail,
    apr.analysis_summary,
    apr.recommended_action,
    apr.is_notified,
    apr.is_checked,
    apr.predicted_at
from public.ai_prediction_results apr
join public.customer_orders co on co.order_id = apr.order_id
join public.products p on p.product_id = apr.product_id
left join public.production_lines pl on pl.line_id = apr.line_id
where apr.risk_level in ('CAUTION', 'WARNING', 'CRITICAL');

create or replace view chat_evidence.chat_production_plan_evidence_view as
select
    pp.plan_id,
    pp.order_id,
    co.order_no,
    pp.product_id,
    p.product_code,
    p.product_name,
    p.product_category,
    pp.line_id,
    pl.line_code,
    pl.line_name,
    pl.line_type,
    pp.operator_id,
    u.name as operator_name,
    pp.planned_start_at,
    pp.planned_end_at,
    pp.estimated_duration_hr,
    pp.planned_quantity,
    pp.plan_sequence,
    pp.plan_status,
    co.due_date,
    co.order_quantity,
    co.order_status
from public.production_plans pp
join public.customer_orders co on co.order_id = pp.order_id
join public.products p on p.product_id = pp.product_id
join public.production_lines pl on pl.line_id = pp.line_id
left join public.users u on u.id = pp.operator_id
where pp.plan_status in ('SCHEDULED', 'IN_PROGRESS', 'DELAYED');

create or replace view chat_evidence.chat_line_bottleneck_evidence_view as
select
    ls.line_status_id,
    ls.line_id,
    pl.line_code,
    pl.line_name,
    pl.line_type,
    ls.product_id,
    p.product_code,
    p.product_name,
    ls.plan_id,
    ls.operator_id,
    u.name as operator_name,
    ls.recorded_at,
    ls.operation_status,
    ls.throughput_rate,
    ls.current_yield_rate,
    ls.waiting_quantity,
    ls.waiting_time_hr,
    ls.processed_quantity,
    ls.defect_quantity,
    ls.utilization_rate,
    ls.progress_rate,
    coalesce(machine_summary.abnormal_machine_count, 0) as abnormal_machine_count,
    machine_summary.abnormal_machine_codes,
    machine_summary.latest_machine_status_note
from public.line_status ls
join public.production_lines pl on pl.line_id = ls.line_id
left join public.products p on p.product_id = ls.product_id
left join public.users u on u.id = ls.operator_id
left join lateral (
    select
        count(*) filter (
            where ms.operation_status in ('STOPPED', 'ERROR', 'MAINTENANCE')
        ) as abnormal_machine_count,
        array_remove(array_agg(pm.machine_code) filter (
            where ms.operation_status in ('STOPPED', 'ERROR', 'MAINTENANCE')
        ), null) as abnormal_machine_codes,
        max(ms.status_note) filter (
            where ms.operation_status in ('STOPPED', 'ERROR', 'MAINTENANCE')
        ) as latest_machine_status_note
    from public.machine_statuses ms
    left join public.production_machines pm on pm.machine_id = ms.machine_id
    where ms.line_id = ls.line_id
      and (
          ms.plan_id = ls.plan_id
          or ls.plan_id is null
          or ms.recorded_at >= ls.recorded_at - interval '1 hour'
      )
) machine_summary on true
where ls.operation_status in ('STOPPED', 'ERROR', 'MAINTENANCE')
   or coalesce(ls.waiting_quantity, 0) > 0
   or coalesce(ls.waiting_time_hr, 0) > 0
   or coalesce(ls.throughput_rate, 1) < 1
   or coalesce(machine_summary.abnormal_machine_count, 0) > 0;

create or replace view chat_evidence.chat_urgent_order_impact_evidence_view as
select
    ssr.simulation_id,
    ssd.simulation_detail_id,
    ssr.simulation_group_id,
    ssr.simulation_name,
    ssr.simulation_type,
    ssr.prediction_id,
    ssd.order_id,
    co.order_no,
    ssd.plan_id,
    pp.product_id,
    p.product_code,
    p.product_name,
    ssd.before_line_id,
    before_line.line_code as before_line_code,
    ssd.after_line_id,
    after_line.line_code as after_line_code,
    ssd.before_sequence,
    ssd.after_sequence,
    ssd.before_start_at,
    ssd.before_end_at,
    ssd.after_start_at,
    ssd.after_end_at,
    ssd.expected_completion_date,
    ssd.after_is_delayed,
    ssd.before_quantity,
    ssd.after_quantity,
    ssd.change_reason,
    ssr.action_result,
    ssr.before_total_delay_hr,
    ssr.after_total_delay_hr,
    ssr.delay_reduction_hr,
    ssr.recommendation_grade,
    ssr.created_at
from public.schedule_simulation_details ssd
join public.schedule_simulation_results ssr on ssr.simulation_id = ssd.simulation_id
join public.customer_orders co on co.order_id = ssd.order_id
join public.production_plans pp on pp.plan_id = ssd.plan_id
join public.products p on p.product_id = pp.product_id
left join public.production_lines before_line on before_line.line_id = ssd.before_line_id
left join public.production_lines after_line on after_line.line_id = ssd.after_line_id;

create or replace view chat_evidence.chat_work_priority_evidence_view as
with material_shortage as (
    select
        plan_id,
        count(*) filter (
            where material_plan_status in ('SHORTAGE', 'PARTIAL_RESERVED')
               or shortage_quantity > 0
        ) as shortage_count
    from public.production_plan_materials
    group by plan_id
),
latest_line_status as (
    select distinct on (line_id)
        line_id,
        waiting_time_hr
    from public.line_status
    order by line_id, recorded_at desc
)
select
    row_number() over (
        order by
            case apr.risk_level
                when 'CRITICAL' then 1
                when 'WARNING' then 2
                when 'CAUTION' then 3
                else 4
            end,
            co.due_date asc,
            pp.planned_start_at asc
    ) as priority_rank,
    pp.plan_id,
    pp.order_id,
    co.order_no,
    pp.product_id,
    p.product_code,
    p.product_name,
    pp.line_id,
    pl.line_code,
    pl.line_name,
    pp.operator_id,
    u.name as operator_name,
    pp.planned_start_at,
    pp.planned_end_at,
    pp.planned_quantity,
    pp.plan_sequence,
    pp.plan_status,
    co.due_date,
    apr.prediction_id,
    apr.risk_level,
    apr.delay_probability,
    null::text as main_cause_type,
    coalesce(ms.shortage_count, 0) as shortage_count,
    lls.waiting_time_hr as line_waiting_time_hr,
    apr.recommended_action
from public.production_plans pp
join public.customer_orders co on co.order_id = pp.order_id
join public.products p on p.product_id = pp.product_id
join public.production_lines pl on pl.line_id = pp.line_id
left join public.users u on u.id = pp.operator_id
left join public.ai_prediction_results apr on apr.plan_id = pp.plan_id
left join material_shortage ms on ms.plan_id = pp.plan_id
left join latest_line_status lls on lls.line_id = pp.line_id
where pp.plan_status in ('SCHEDULED', 'IN_PROGRESS', 'DELAYED');

create or replace view chat_evidence.chat_report_metadata_evidence_view as
select
    r.report_id,
    r.report_title,
    r.report_type,
    r.author_id,
    u.name as author_name,
    r.target_start_date,
    r.target_end_date,
    r.included_items,
    r.report_evidence,
    r.related_simulation_id,
    r.created_at,
    r.updated_at
from public.reports r
left join public.users u on u.id = r.author_id;

comment on schema chat_evidence is
    'Read-only evidence views used by the FastAPI chatbot.';

commit;
