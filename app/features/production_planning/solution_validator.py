from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from app.features.production_planning.config import SolverConfig
from app.features.production_planning.preprocessing import (
    calculate_required_material_quantity_scaled,
    get_inbound_material_quantity_scaled,
    get_inbound_material_time,
    get_initial_available_material_quantity_scaled,
    to_minute_offset,
)
from app.features.production_planning.schemas import (
    NormalizedPlanningData,
    PlanResult,
    ProductionPlanningResult,
    ScheduleItem,
)

SOLVED_STATUSES = {"OPTIMAL", "FEASIBLE"}


@dataclass(frozen=True)
class ValidationReport:
    constraint_violations: list[str] = field(default_factory=list)
    metric_mismatches: list[str] = field(default_factory=list)
    business_rule_violations: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """
        Parameters:
            - None.

        Methodology:
            - Treat any hard constraint, metric, or business rule issue as validation failure.

        Output:
            - True when no validation category contains issues.
        """
        return not (
            self.constraint_violations
            or self.metric_mismatches
            or self.business_rule_violations
        )


def validate_solution(
    result: ProductionPlanningResult,
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> ValidationReport:
    """
    Parameters:
        - result: Production planning result containing due-date and amount plans.
        - data: Normalized planning data used to re-derive constraints.
        - config: Solver configuration containing business policies.

    Methodology:
        - Recompute hard constraints, metric consistency, and business rules from extracted
          schedule output only.
        - Keep the validator pure by returning a report without logging or mutating inputs.

    Output:
        - ValidationReport with grouped violations and is_valid status.
    """
    return ValidationReport(
        constraint_violations=verify_hard_constraints(result, data),
        metric_mismatches=verify_metrics(result, data),
        business_rule_violations=verify_business_rules(result, data, config),
    )


def verify_hard_constraints(
    result: ProductionPlanningResult,
    data: NormalizedPlanningData,
) -> list[str]:
    """
    Parameters:
        - result: Production planning result to verify.
        - data: Normalized planning data with line availability, materials, and BOM.

    Methodology:
        - Recalculate no-overlap per line, line availability bounds, and material usage from
          schedule items.
        - Skip plans without a feasible solution because they have no concrete schedule.

    Output:
        - List of hard constraint violation messages.
    """
    violations: list[str] = []
    for plan in _iter_solved_plans(result):
        tag = plan.plan_variant_code
        violations.extend(_verify_no_overlap(plan.schedule_items, data, tag))
        violations.extend(_verify_line_availability(plan.schedule_items, data, tag))
        violations.extend(_verify_daily_capacity(plan.schedule_items, data, tag))
        violations.extend(_verify_no_changeover_lines(plan.schedule_items, data, tag))
        violations.extend(_verify_material_usage(plan.schedule_items, data, tag))
        violations.extend(_verify_unit_compatibility(data, tag))
    return violations


def verify_metrics(
    result: ProductionPlanningResult,
    data: NormalizedPlanningData,
) -> list[str]:
    """
    Parameters:
        - result: Production planning result with reported plan metrics.
        - data: Normalized planning data with due-date offsets.

    Methodology:
        - Recompute total tardiness and delayed order count from schedule item end times.
        - Compare recalculated values with reported PlanMetrics fields.

    Output:
        - List of metric mismatch messages.
    """
    mismatches: list[str] = []
    for plan in _iter_solved_plans(result):
        tag = plan.plan_variant_code
        recalculated_tardiness = 0
        recalculated_delayed = 0
        for item in plan.schedule_items:
            order = data.orders.get(item.order_id)
            if order is None:
                continue
            end_offset = to_minute_offset(item.end_time, data.planning_start)
            tardiness = max(0, end_offset - order.due_date_offset)
            recalculated_tardiness += tardiness
            if tardiness > 0:
                recalculated_delayed += 1

        if recalculated_tardiness != plan.metrics.total_tardiness_minutes:
            mismatches.append(
                f"[{tag}] total_tardiness_minutes: "
                f"reported={plan.metrics.total_tardiness_minutes}, "
                f"recalculated={recalculated_tardiness}"
            )
        if recalculated_delayed != plan.metrics.delayed_order_count:
            mismatches.append(
                f"[{tag}] delayed_order_count: "
                f"reported={plan.metrics.delayed_order_count}, "
                f"recalculated={recalculated_delayed}"
            )
    return mismatches


def verify_business_rules(
    result: ProductionPlanningResult,
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> list[str]:
    """
    Parameters:
        - result: Production planning result with extracted plan schedules.
        - data: Normalized planning data with order due dates.
        - config: Solver configuration containing due-date and unscheduled policies.

    Methodology:
        - Verify HARD due-date policy and mandatory scheduling policy against each solved plan.
        - Use extracted plan data rather than solver variables.

    Output:
        - List of business rule violation messages.
    """
    violations: list[str] = []
    for plan in _iter_solved_plans(result):
        tag = plan.plan_variant_code
        scheduled_ids = {item.order_id for item in plan.schedule_items}

        if config.due_date_policy == "HARD":
            violations.extend(_verify_hard_due_dates(plan.schedule_items, data, tag))

        if not config.allow_unscheduled_orders:
            for order_id in data.orders:
                if order_id not in scheduled_ids:
                    violations.append(
                        f"[{tag}] order {order_id} unscheduled (config disallows it)"
                    )
    return violations


def _verify_no_overlap(
    schedule_items: list[ScheduleItem],
    data: NormalizedPlanningData,
    tag: str,
) -> list[str]:
    violations: list[str] = []
    by_line: dict[str, list[tuple[int, int, str]]] = {}
    for item in schedule_items:
        start = to_minute_offset(item.start_time, data.planning_start)
        end = to_minute_offset(item.end_time, data.planning_start)
        if start >= end:
            violations.append(
                f"[{tag}] order {item.order_id} has invalid interval: start={start}, end={end}"
            )
        by_line.setdefault(item.line_id, []).append((start, end, item.order_id))

    for line_id, intervals in by_line.items():
        intervals.sort()
        for index in range(len(intervals) - 1):
            current = intervals[index]
            next_item = intervals[index + 1]
            if current[1] > next_item[0]:
                violations.append(
                    f"[{tag}] overlap on line {line_id}: "
                    f"order {current[2]} ends {current[1]}, "
                    f"order {next_item[2]} starts {next_item[0]}"
                )
    return violations


def _verify_line_availability(
    schedule_items: list[ScheduleItem],
    data: NormalizedPlanningData,
    tag: str,
) -> list[str]:
    violations: list[str] = []
    for item in schedule_items:
        line = data.lines.get(item.line_id)
        if line is None:
            violations.append(f"[{tag}] order {item.order_id} uses unknown line {item.line_id}")
            continue
        start = to_minute_offset(item.start_time, data.planning_start)
        end = to_minute_offset(item.end_time, data.planning_start)
        if start < line.available_from_offset:
            violations.append(
                f"[{tag}] order {item.order_id} starts before line {item.line_id} available"
            )
        if end > line.available_to_offset:
            violations.append(
                f"[{tag}] order {item.order_id} ends after line {item.line_id} available"
            )
    return violations


def _verify_material_usage(
    schedule_items: list[ScheduleItem],
    data: NormalizedPlanningData,
    tag: str,
) -> list[str]:
    violations: list[str] = []
    material_usage = _calculate_material_usage_by_item(schedule_items, data, tag, violations)
    for material_id, usage_items in material_usage.items():
        material = data.materials.get(material_id)
        if material is None:
            violations.append(f"[{tag}] material {material_id} usage has no material record")
            continue
        used = sum(item_usage[1] for item_usage in usage_items)
        initial_available = get_initial_available_material_quantity_scaled(material)
        inbound_quantity = get_inbound_material_quantity_scaled(material)
        inbound_time = get_inbound_material_time(material)
        if inbound_time is None:
            available = initial_available
        else:
            inbound_offset = to_minute_offset(inbound_time, data.planning_start)
            if inbound_offset <= 0:
                available = initial_available + inbound_quantity
            elif inbound_offset > data.horizon_minutes:
                available = initial_available
            else:
                before_inbound_used = sum(
                    quantity for start, quantity in usage_items if start < inbound_offset
                )
                if before_inbound_used > initial_available:
                    violations.append(
                        f"[{tag}] material {material_id} over-used before inbound: "
                        f"used={before_inbound_used}, available={initial_available}"
                    )
                available = initial_available + inbound_quantity
        if used > available:
            violations.append(
                f"[{tag}] material {material_id} over-used: used={used}, available={available}"
            )
    return violations


def _calculate_material_usage_by_item(
    schedule_items: list[ScheduleItem],
    data: NormalizedPlanningData,
    tag: str,
    violations: list[str],
) -> dict[str, list[tuple[int, int]]]:
    material_usage: dict[str, list[tuple[int, int]]] = {}
    for item in schedule_items:
        order = data.orders.get(item.order_id)
        if order is None:
            violations.append(f"[{tag}] schedule references unknown order {item.order_id}")
            continue
        start = to_minute_offset(item.start_time, data.planning_start)
        product = data.products[order.order.product_id]
        for bom_item in data.bom_items_by_product_id.get(order.order.product_id, []):
            quantity = calculate_required_material_quantity_scaled(order.order, product, bom_item)
            material_usage.setdefault(bom_item.material_id, []).append((start, quantity))
    return material_usage


def _verify_daily_capacity(
    schedule_items: list[ScheduleItem],
    data: NormalizedPlanningData,
    tag: str,
) -> list[str]:
    violations: list[str] = []
    line_bucket_usage: dict[tuple[str, int], int] = {}
    capability_bucket_usage: dict[tuple[str, str, int], int] = {}
    for item in schedule_items:
        end_offset = to_minute_offset(item.end_time, data.planning_start)
        bucket_index = end_offset // (24 * 60)
        planned_quantity = item.planned_production_quantity
        line_bucket_usage[(item.line_id, bucket_index)] = (
            line_bucket_usage.get((item.line_id, bucket_index), 0) + planned_quantity
        )
        capability_bucket_usage[(item.product_id, item.line_id, bucket_index)] = (
            capability_bucket_usage.get((item.product_id, item.line_id, bucket_index), 0)
            + planned_quantity
        )

    for (line_id, bucket_index), used in line_bucket_usage.items():
        line = data.lines.get(line_id)
        if line is None or line.line.max_capacity_per_day is None:
            continue
        if used > line.line.max_capacity_per_day:
            violations.append(
                f"[{tag}] line {line_id} capacity exceeded in bucket {bucket_index}: "
                f"used={used}, capacity={line.line.max_capacity_per_day}"
            )

    for (product_id, line_id, bucket_index), used in capability_bucket_usage.items():
        capability = data.capabilities.get((product_id, line_id))
        if capability is None or capability.capacity_per_day is None:
            continue
        if used > capability.capacity_per_day:
            violations.append(
                f"[{tag}] product-line capacity exceeded for {product_id}/{line_id} "
                f"in bucket {bucket_index}: used={used}, capacity={capability.capacity_per_day}"
            )
    return violations


def _verify_no_changeover_lines(
    schedule_items: list[ScheduleItem],
    data: NormalizedPlanningData,
    tag: str,
) -> list[str]:
    violations: list[str] = []
    for line_id, line in data.lines.items():
        if line.line.supports_changeover:
            continue
        products = {
            item.product_id for item in schedule_items if item.line_id == line_id
        }
        products.update(
            block.schedule.product_id
            for block in data.existing_schedule_blocks_by_line_id.get(line_id, [])
            if block.schedule.product_id
        )
        if len(products) > 1:
            violations.append(
                f"[{tag}] line {line_id} has multiple products while changeover is disabled"
            )
    return violations


def _verify_unit_compatibility(data: NormalizedPlanningData, tag: str) -> list[str]:
    violations: list[str] = []
    for product_id, line_id in data.capabilities:
        product = data.products[product_id]
        line = data.lines.get(line_id)
        if line is None or product.unit is None or line.line.capacity_unit is None:
            continue
        if product.unit.strip().upper() != line.line.capacity_unit.strip().upper():
            violations.append(f"[{tag}] product {product_id} unit does not match line {line_id}")

    for bom_items in data.bom_items_by_product_id.values():
        for bom_item in bom_items:
            material = data.materials.get(bom_item.material_id)
            if material is None or bom_item.unit is None or material.material_unit is None:
                continue
            if bom_item.unit.strip().upper() != material.material_unit.strip().upper():
                violations.append(
                    f"[{tag}] BOM unit does not match material {bom_item.material_id}"
                )
    return violations


def _verify_hard_due_dates(
    schedule_items: list[ScheduleItem],
    data: NormalizedPlanningData,
    tag: str,
) -> list[str]:
    violations: list[str] = []
    for item in schedule_items:
        order = data.orders.get(item.order_id)
        if order is None:
            continue
        end_offset = to_minute_offset(item.end_time, data.planning_start)
        if end_offset > order.due_date_offset:
            violations.append(
                f"[{tag}] HARD due date violated: order {item.order_id} "
                f"ends at {end_offset}, due at {order.due_date_offset}"
            )
    return violations


def _iter_solved_plans(result: ProductionPlanningResult) -> Iterable[PlanResult]:
    return (plan for plan in result.plan_results if plan.status in SOLVED_STATUSES)
