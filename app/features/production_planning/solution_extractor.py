from __future__ import annotations

from app.features.production_planning.cpsat_model_builder import CpsatModelBundle
from app.features.production_planning.exceptions import SolutionExtractionError
from app.features.production_planning.preprocessing import (
    calculate_planned_production_quantity,
    from_minute_offset,
    get_changeover_cost,
    get_changeover_minutes,
    get_order_quantity,
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
) -> PlanResult:
    """
    Parameters:
        - raw_result: Solver output and CpSolver instance.
        - bundle: Model bundle containing variables to read.
        - variant_code: Plan variant code used for objective selection.
        - data: Normalized planning data for datetime conversion and business values.

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
        metrics = calculate_plan_metrics(schedule_items, unscheduled_orders, data)
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
        items.append(
            ScheduleItem(
                order_id=order.order_id,
                product_id=order.product_id,
                line_id=selected_line_id,
                start_time=from_minute_offset(start_offset, data.planning_start),
                end_time=from_minute_offset(end_offset, data.planning_start),
                quantity=planned_quantity,
                order_quantity=order_quantity,
                planned_production_quantity=planned_quantity,
                order_amount=order.order_amount,
                tardiness_minutes=tardiness,
                status=status,
            )
        )
    return sorted(items, key=lambda item: (item.start_time, item.line_id, item.order_id))


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
) -> PlanMetrics:
    """
    Parameters:
        - schedule_items: Extracted scheduled order items.
        - unscheduled_orders: Order IDs not scheduled.
        - data: Normalized planning data for original order amounts and changeover rules.

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
        estimated_total_cost=total_late_penalty + total_changeover_cost,
    )


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
        "plan_type": raw_result.plan_type,
        "status": raw_result.status,
        "objective_value": raw_result.objective_value,
        "wall_time_seconds": raw_result.wall_time_seconds,
        "conflicts": raw_result.conflicts,
        "branches": raw_result.branches,
        "response_stats": raw_result.response_stats,
    }
