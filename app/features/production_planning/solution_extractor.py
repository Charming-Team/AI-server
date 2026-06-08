from __future__ import annotations

from app.features.production_planning.config import SolverConfig
from app.features.production_planning.cpsat_model_builder import CpsatModelBundle
from app.features.production_planning.exceptions import SolutionExtractionError
from app.features.production_planning.preprocessing import (
    MATERIAL_QUANTITY_SCALE,
    calculate_planned_production_quantity,
    calculate_required_material_quantity_scaled,
    from_minute_offset,
    get_changeover_cost,
    get_changeover_minutes,
    get_inbound_material_quantity_scaled,
    get_inbound_material_time,
    get_initial_available_material_quantity_scaled,
    get_order_quantity,
    to_minute_offset,
)
from app.features.production_planning.schemas import (
    PLAN_VARIANT_FAMILIES,
    PLAN_VARIANT_NAMES,
    NormalizedPlanningData,
    PlanMetrics,
    PlanResult,
    RawSolverResult,
    ScheduleItem,
)

SOLVED_STATUSES = {"OPTIMAL", "FEASIBLE"}


def extract_plan_result(
    raw_result: RawSolverResult,
    bundle: CpsatModelBundle,
    variant_code: str,
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> PlanResult:
    """
    Parameters:
        - raw_result: Solver output and CpSolver instance.
        - bundle: Model bundle containing variables to read.
        - variant_code: Plan variant code used for objective selection.
        - data: Normalized planning data for datetime conversion and business values.
        - config: Solver configuration with shortage cost policy.

    Methodology:
        - Return a deterministic empty plan for infeasible or unresolved statuses.
        - For feasible solutions, read selected assignment variables, convert offsets to
          datetimes, calculate tardiness and plan metrics, and sort schedule items.

    Output:
        - PlanResult with schedule items, unscheduled orders, metrics, warnings, and metadata.
    """
    try:
        if raw_result.status not in SOLVED_STATUSES:
            warnings = list(bundle.warnings)
            if raw_result.status == "INFEASIBLE":
                warnings.append("CP-SAT model is infeasible with current hard constraints.")
            elif raw_result.status == "UNKNOWN":
                warnings.append("CP-SAT solver did not return a feasible solution.")
            return PlanResult(
                plan_family=PLAN_VARIANT_FAMILIES[variant_code],
                plan_variant_code=variant_code,
                plan_variant_name=PLAN_VARIANT_NAMES[variant_code],
                status=raw_result.status,
                schedule_items=[],
                unscheduled_orders=sorted(data.orders),
                metrics=_empty_metrics(data),
                warnings=warnings,
                solver_metadata=_solver_metadata(raw_result),
            )

        schedule_items = extract_schedule_items(raw_result, bundle, data)
        unscheduled_orders = extract_unscheduled_orders(raw_result, bundle, data)
        metrics = calculate_plan_metrics(schedule_items, unscheduled_orders, data, config)
        return PlanResult(
            plan_family=PLAN_VARIANT_FAMILIES[variant_code],
            plan_variant_code=variant_code,
            plan_variant_name=PLAN_VARIANT_NAMES[variant_code],
            status=raw_result.status,
            schedule_items=schedule_items,
            unscheduled_orders=unscheduled_orders,
            metrics=metrics,
            warnings=list(bundle.warnings),
            solver_metadata=_solver_metadata(raw_result),
        )
    except (KeyError, ValueError, RuntimeError) as exc:
        raise SolutionExtractionError(f"Failed to extract {variant_code} result: {exc}") from exc


def extract_schedule_items(
    raw_result: RawSolverResult,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> list[ScheduleItem]:
    """
    Parameters:
        - raw_result: Feasible solver result.
        - bundle: Model bundle containing assignment and timing variables.
        - data: Normalized planning data for order values and datetime conversion.

    Methodology:
        - For each scheduled order, find the single selected capable line and read start/end
          offsets from CP-SAT variables.
        - Compute business status from tardiness and sort deterministically.

    Output:
        - Sorted schedule item list.
    """
    solver = raw_result.solver
    items: list[ScheduleItem] = []
    for order_id in sorted(data.orders):
        order_vars = bundle.order_vars[order_id]
        if solver.Value(order_vars["scheduled"]) != 1:
            continue
        selected_line_id = None
        for line_id in sorted(order_vars["assigned"]):
            if solver.Value(order_vars["assigned"][line_id]) == 1:
                selected_line_id = line_id
                break
        if selected_line_id is None:
            continue

        normalized_order = data.orders[order_id]
        order = normalized_order.order
        start_offset = solver.Value(order_vars["start"][selected_line_id])
        end_offset = solver.Value(order_vars["end"][selected_line_id])
        tardiness = max(0, end_offset - normalized_order.due_date_offset)
        status = "ON_TIME" if tardiness == 0 else "DELAYED"
        order_quantity = get_order_quantity(order)
        planned_quantity = calculate_planned_production_quantity(
            order,
            data.products[order.product_id],
        )
        start_time, end_time = _extract_schedule_times(
            order,
            selected_line_id,
            start_offset,
            end_offset,
            data,
        )
        items.append(
            ScheduleItem(
                order_id=order.order_id,
                product_id=order.product_id,
                line_id=selected_line_id,
                start_time=start_time,
                end_time=end_time,
                quantity=planned_quantity,
                order_quantity=order_quantity,
                planned_production_quantity=planned_quantity,
                order_amount=order.order_amount,
                tardiness_minutes=tardiness,
                status=status,
            )
        )
    return sorted(items, key=lambda item: (item.start_time, item.line_id, item.order_id))


def _extract_schedule_times(
    order,
    selected_line_id: str,
    start_offset: int,
    end_offset: int,
    data: NormalizedPlanningData,
):
    """
    Parameters:
        - order: Source order input for the extracted schedule item.
        - selected_line_id: Line selected by CP-SAT.
        - start_offset: Solver start offset in minutes.
        - end_offset: Solver end offset in minutes.
        - data: Normalized planning data with the planning start timestamp.

    Methodology:
        - CP-SAT uses integer-minute offsets, while locked edits can originate from DB
          timestamps that include seconds.
        - For locked orders fixed to their requested line, preserve the original DB/request
          timestamps in the response while keeping solver constraints minute-based.

    Output:
        - Tuple of start and end datetimes for the schedule item.
    """
    if order.is_locked and order.locked_plan and order.locked_plan.line_id == selected_line_id:
        return order.locked_plan.planned_start_at, order.locked_plan.planned_end_at
    return (
        from_minute_offset(start_offset, data.planning_start),
        from_minute_offset(end_offset, data.planning_start),
    )


def extract_unscheduled_orders(
    raw_result: RawSolverResult,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> list[str]:
    """
    Parameters:
        - raw_result: Feasible solver result.
        - bundle: Model bundle containing scheduled variables.
        - data: Normalized planning data with order IDs.

    Methodology:
        - Read scheduled booleans and return orders not selected into the plan.

    Output:
        - Sorted list of unscheduled order IDs.
    """
    solver = raw_result.solver
    return sorted(
        order_id
        for order_id in data.orders
        if solver.Value(bundle.order_vars[order_id]["scheduled"]) != 1
    )


def calculate_plan_metrics(
    schedule_items: list[ScheduleItem],
    unscheduled_orders: list[str],
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> PlanMetrics:
    """
    Parameters:
        - schedule_items: Extracted scheduled order items.
        - unscheduled_orders: Order IDs not scheduled.
        - data: Normalized planning data for original order amounts and changeover rules.
        - config: Solver configuration with material shortage penalty weight.

    Methodology:
        - Aggregate count, tardiness, amount, makespan, and selected transition metrics from
          deterministic extracted schedule items.

    Output:
        - PlanMetrics summarizing the plan.
    """
    delayed_items = [item for item in schedule_items if item.status == "DELAYED"]
    on_time_items = [item for item in schedule_items if item.status == "ON_TIME"]
    unscheduled_amount = sum(
        data.orders[order_id].order.order_amount for order_id in unscheduled_orders
    )
    total_tardiness = sum(item.tardiness_minutes for item in schedule_items)
    total_late_penalty = sum(
        data.orders[item.order_id].order.late_penalty_amount for item in delayed_items
    )
    total_changeover_cost = calculate_total_changeover_cost(schedule_items, data)
    material_shortage_quantity = calculate_total_material_shortage_quantity(
        schedule_items,
        data,
    )
    return PlanMetrics(
        scheduled_count=len(schedule_items),
        unscheduled_count=len(unscheduled_orders),
        delayed_order_count=len(delayed_items),
        on_time_order_count=len(on_time_items),
        total_tardiness_minutes=total_tardiness,
        max_tardiness_minutes=max((item.tardiness_minutes for item in schedule_items), default=0),
        total_scheduled_amount=sum(item.order_amount for item in schedule_items),
        on_time_amount=sum(item.order_amount for item in on_time_items),
        delayed_amount=sum(item.order_amount for item in delayed_items),
        unscheduled_amount=unscheduled_amount,
        makespan_minutes=calculate_makespan_minutes(schedule_items, data),
        total_changeover_minutes=calculate_total_changeover_minutes(schedule_items, data),
        total_late_penalty_amount=total_late_penalty,
        total_changeover_cost=total_changeover_cost,
        total_material_shortage_quantity=material_shortage_quantity,
        total_material_shortage_penalty_amount=0,
        estimated_total_cost=total_late_penalty + total_changeover_cost,
    )


def calculate_total_material_shortage_quantity(
    schedule_items: list[ScheduleItem],
    data: NormalizedPlanningData,
) -> int:
    """
    Parameters:
        - schedule_items: Extracted schedule items.
        - data: Normalized planning data with materials and BOM rules.

    Methodology:
        - Recalculate scheduled material usage from BOM by product_id.
        - Add total shortage plus shortage before expected inbound so early starts carry
          a material-risk cost even when enough stock arrives later.

    Output:
        - Material shortage quantity rounded up to integer material units.
    """
    usage_by_material = _calculate_material_usage_units_by_item(schedule_items, data)
    total_shortage = 0
    for material_id, usage_items in usage_by_material.items():
        material = data.materials.get(material_id)
        if material is None:
            continue
        used = sum(quantity for _, quantity in usage_items)
        initial_available = _scaled_quantity_to_units(
            get_initial_available_material_quantity_scaled(material)
        )
        inbound_quantity = _scaled_quantity_to_units(
            get_inbound_material_quantity_scaled(material)
        )
        total_shortage += max(0, used - initial_available - inbound_quantity)

        inbound_time = get_inbound_material_time(material)
        if inbound_time is None:
            continue
        inbound_offset = to_minute_offset(inbound_time, data.planning_start)
        if inbound_offset <= 0 or inbound_offset > data.horizon_minutes:
            continue
        before_inbound_used = sum(
            quantity for start, quantity in usage_items if start < inbound_offset
        )
        total_shortage += max(0, before_inbound_used - initial_available)
    return total_shortage


def _calculate_material_usage_units_by_item(
    schedule_items: list[ScheduleItem],
    data: NormalizedPlanningData,
) -> dict[str, list[tuple[int, int]]]:
    material_usage: dict[str, list[tuple[int, int]]] = {}
    for item in schedule_items:
        normalized_order = data.orders.get(item.order_id)
        if normalized_order is None:
            continue
        start = to_minute_offset(item.start_time, data.planning_start)
        product = data.products[normalized_order.order.product_id]
        for bom_item in data.bom_items_by_product_id.get(
            normalized_order.order.product_id,
            [],
        ):
            required_scaled = calculate_required_material_quantity_scaled(
                normalized_order.order,
                product,
                bom_item,
            )
            quantity = _scaled_quantity_to_units(required_scaled)
            material_usage.setdefault(bom_item.material_id, []).append((start, quantity))
    return material_usage


def _scaled_quantity_to_units(quantity_scaled: int) -> int:
    return (quantity_scaled + MATERIAL_QUANTITY_SCALE - 1) // MATERIAL_QUANTITY_SCALE


def calculate_total_changeover_minutes(
    schedule_items: list[ScheduleItem],
    data: NormalizedPlanningData,
) -> int:
    """
    Parameters:
        - schedule_items: Extracted schedule items.
        - data: Normalized planning data with changeover rules.

    Methodology:
        - Sort scheduled items per line and sum transition setup minutes between consecutive
          products using the same rule priority as model construction.

    Output:
        - Total selected changeover minutes.
    """
    total = 0
    for line_id in sorted({item.line_id for item in schedule_items}):
        line_items = sorted(
            [item for item in schedule_items if item.line_id == line_id],
            key=lambda item: (item.start_time, item.order_id),
        )
        for previous, current in zip(line_items, line_items[1:], strict=False):
            total += get_changeover_minutes(
                previous.product_id,
                current.product_id,
                line_id,
                data.changeover_rules,
                0,
            )
    return total


def calculate_makespan_minutes(
    schedule_items: list[ScheduleItem],
    data: NormalizedPlanningData,
) -> int:
    """
    Parameters:
        - schedule_items: Extracted schedule items.
        - data: Normalized planning data with planning start.

    Methodology:
        - Use the latest scheduled end time and convert it back to an offset.

    Output:
        - Makespan in minutes from planning start.
    """
    if not schedule_items:
        return 0
    latest_end = max(item.end_time for item in schedule_items)
    return int((latest_end - data.planning_start).total_seconds() // 60)


def calculate_total_changeover_cost(
    schedule_items: list[ScheduleItem],
    data: NormalizedPlanningData,
) -> int:
    """
    Parameters:
        - schedule_items: Extracted schedule items sorted by start time.
        - data: Normalized planning data with changeover rules.

    Methodology:
        - Sort scheduled items per line and sum transition costs between consecutive products
          using the same rule priority as model construction.

    Output:
        - Total selected changeover cost in integer cost units.
    """
    total = 0
    for line_id in sorted({item.line_id for item in schedule_items}):
        line_items = sorted(
            [item for item in schedule_items if item.line_id == line_id],
            key=lambda item: (item.start_time, item.order_id),
        )
        for previous, current in zip(line_items, line_items[1:], strict=False):
            total += get_changeover_cost(
                previous.product_id,
                current.product_id,
                line_id,
                data.changeover_rules,
                0,
            )
    return total


def _empty_metrics(data: NormalizedPlanningData) -> PlanMetrics:
    return PlanMetrics(
        scheduled_count=0,
        unscheduled_count=len(data.orders),
        delayed_order_count=0,
        on_time_order_count=0,
        total_tardiness_minutes=0,
        max_tardiness_minutes=0,
        total_scheduled_amount=0,
        on_time_amount=0,
        delayed_amount=0,
        unscheduled_amount=sum(order.order.order_amount for order in data.orders.values()),
        makespan_minutes=0,
        total_changeover_minutes=0,
        total_late_penalty_amount=0,
        total_changeover_cost=0,
        estimated_total_cost=0,
    )


def _solver_metadata(raw_result: RawSolverResult) -> dict:
    return {
        "plan_variant_code": raw_result.plan_variant_code,
        "status": raw_result.status,
        "objective_value": raw_result.objective_value,
        "wall_time_seconds": raw_result.wall_time_seconds,
        "conflicts": raw_result.conflicts,
        "branches": raw_result.branches,
        "response_stats": raw_result.response_stats,
    }
