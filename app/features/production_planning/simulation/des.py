from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from random import Random

from app.features.production_planning.config import SimulationConfig, SolverConfig
from app.features.production_planning.preprocessing import (
    MATERIAL_QUANTITY_SCALE,
    calculate_required_material_quantity_scaled,
    get_changeover_cost,
    get_changeover_minutes,
    get_inbound_material_quantity_scaled,
    get_initial_available_material_quantity_scaled,
)
from app.features.production_planning.schemas import (
    NormalizedPlanningData,
    PlanResult,
    ScheduleItem,
)
from app.features.production_planning.simulation.sampling import (
    SamplingDistributions,
    sample_cause,
    sample_changeover_minutes,
    sample_defect_rate,
    sample_duration_minutes,
    sample_line_change_delay_minutes,
    sample_machine_breakdown_minutes,
    sample_material_inbound_delay_minutes,
    sample_operational_delay_minutes,
    sample_setup_delay_minutes,
    sample_yield_rate,
)

EVENT_MATERIAL_SHORTAGE = "MATERIAL_SHORTAGE"
EVENT_MACHINE_BREAKDOWN = "MACHINE_BREAKDOWN"
EVENT_SETUP_DELAY = "SETUP_DELAY"
EVENT_LINE_CHANGE_DELAY = "LINE_CHANGE_DELAY"
EVENT_QUALITY_DEFECT = "QUALITY_DEFECT"
EVENT_LATE_COMPLETION = "LATE_COMPLETION"
EVENT_ORDER_COMPLETED = "ORDER_COMPLETED"
EVENT_UNSCHEDULED = "UNSCHEDULED"

EVENT_BY_DELAY_CAUSE = {
    "MATERIAL_SHORTAGE": EVENT_MATERIAL_SHORTAGE,
    "MATERIAL_DELAY": EVENT_MATERIAL_SHORTAGE,
    "LOW_YIELD": EVENT_QUALITY_DEFECT,
    "MACHINE_ABNORMAL": EVENT_MACHINE_BREAKDOWN,
    "LINE_ABNORMAL": EVENT_LINE_CHANGE_DELAY,
}


@dataclass(frozen=True)
class ScenarioResult:
    plan_variant_code: str
    total_tardiness_minutes: int
    delayed_order_count: int
    late_penalty_amount: int
    changeover_cost: int
    material_shortage_quantity: int
    material_shortage_penalty_amount: int
    total_risk_cost: int
    material_shortage_count: int
    yield_rate_sum: float
    yield_sample_count: int
    defect_quantity: float
    delay_causes: dict[str, int] = field(default_factory=dict)
    event_counts: dict[str, int] = field(default_factory=dict)
    event_log: list[dict] = field(default_factory=list)


@dataclass
class SimulationState:
    line_available_at: dict[str, datetime]
    line_last_product_id: dict[str, str | None]
    inventory_by_material_id: dict[str, int]
    inbound_events: list[tuple[datetime, str, int]]
    next_inbound_index: int = 0


def run_discrete_event_simulation(
    plan: PlanResult,
    data: NormalizedPlanningData,
    solver_config: SolverConfig,
    simulation_config: SimulationConfig,
    distributions: SamplingDistributions,
    rng: Random,
) -> ScenarioResult:
    """
    Parameters:
        - plan: One CP-SAT plan candidate to simulate.
        - data: Normalized planning data containing orders, BOM, materials, and rules.
        - solver_config: Solver configuration containing default changeover values.
        - simulation_config: Simulation configuration containing shortage delay policy.
        - distributions: Sampling distributions used for stochastic event outcomes.
        - rng: Iteration-local random generator.

    Methodology:
        - Initialize line, product, and inventory state from planning inputs.
        - Replay scheduled items as production events while sampling duration, yield, defects,
          changeover gaps, material shortages, and operational delays.
        - Treat unscheduled CP-SAT orders as undelivered work in the scenario metrics.

    Output:
        - ScenarioResult with simulated tardiness, cost, shortage, and quality metrics.
    """
    state = initialize_simulation_state(data, distributions, rng)
    totals = _initial_unscheduled_metrics(plan, data)

    for item in sorted(plan.schedule_items, key=lambda value: (value.start_time, value.line_id)):
        _process_inbounds_until(state, item.start_time)
        previous_product_id = state.line_last_product_id.get(item.line_id)
        actual_start, standard_changeover, sampled_changeover = _calculate_actual_start(
            item,
            state,
            data,
            solver_config,
            distributions,
            rng,
        )
        if previous_product_id is not None and previous_product_id != item.product_id:
            changeover_overrun = max(0, sampled_changeover - standard_changeover)
            if changeover_overrun:
                _increment_counter(totals["delay_causes"], EVENT_SETUP_DELAY)
                _increment_counter(totals["event_counts"], EVENT_SETUP_DELAY)
                _append_event_log(
                    totals,
                    event_time=actual_start,
                    event=EVENT_SETUP_DELAY,
                    item=item,
                    data=data,
                    standard_changeover_minutes=standard_changeover,
                    sampled_changeover_minutes=sampled_changeover,
                    delay_minutes=changeover_overrun,
                    delivery_delay_minutes=0,
                    loss_amount=0,
                )
            line_change_delay = sample_line_change_delay_minutes(distributions, rng)
            if line_change_delay:
                actual_start += timedelta(minutes=line_change_delay)
                _increment_counter(totals["delay_causes"], EVENT_LINE_CHANGE_DELAY)
                _increment_counter(totals["event_counts"], EVENT_LINE_CHANGE_DELAY)
                _append_event_log(
                    totals,
                    event_time=actual_start,
                    event=EVENT_LINE_CHANGE_DELAY,
                    item=item,
                    data=data,
                    delay_minutes=line_change_delay,
                    delivery_delay_minutes=0,
                    loss_amount=0,
                )
        material_shortage_delay, material_shortage_quantity = _consume_materials_or_delay(
            item,
            actual_start,
            state,
            data,
            simulation_config.material_shortage_delay_minutes,
        )
        if material_shortage_delay:
            material_shortage_penalty = _material_shortage_penalty_amount(
                material_shortage_quantity,
                solver_config,
            )
            actual_start += timedelta(minutes=material_shortage_delay)
            totals["material_shortage_count"] += 1
            totals["material_shortage_quantity"] += material_shortage_quantity
            totals["material_shortage_penalty_amount"] += material_shortage_penalty
            _increment_counter(totals["delay_causes"], EVENT_MATERIAL_SHORTAGE)
            _increment_counter(totals["event_counts"], EVENT_MATERIAL_SHORTAGE)
            _append_event_log(
                totals,
                event_time=actual_start,
                event=EVENT_MATERIAL_SHORTAGE,
                item=item,
                data=data,
                delay_minutes=material_shortage_delay,
                delivery_delay_minutes=0,
                loss_amount=material_shortage_penalty,
                risk_cost=material_shortage_penalty,
                material_shortage_quantity=material_shortage_quantity,
            )

        setup_delay = sample_setup_delay_minutes(
            distributions,
            rng,
            product_id=item.product_id,
            line_id=item.line_id,
        )
        if setup_delay:
            _increment_counter(totals["delay_causes"], EVENT_SETUP_DELAY)
            _increment_counter(totals["event_counts"], EVENT_SETUP_DELAY)
            _append_event_log(
                totals,
                event_time=actual_start,
                event=EVENT_SETUP_DELAY,
                item=item,
                data=data,
                delay_minutes=setup_delay,
                delivery_delay_minutes=0,
                loss_amount=0,
            )
            actual_start += timedelta(minutes=setup_delay)

        planned_duration = max(1, round((item.end_time - item.start_time).total_seconds() / 60))
        duration = sample_duration_minutes(
            planned_duration,
            distributions,
            rng,
            product_id=item.product_id,
            line_id=item.line_id,
        )
        machine_breakdown_delay = sample_machine_breakdown_minutes(
            distributions,
            rng,
            line_id=item.line_id,
        )
        if machine_breakdown_delay:
            _increment_counter(totals["delay_causes"], EVENT_MACHINE_BREAKDOWN)
            _increment_counter(totals["event_counts"], EVENT_MACHINE_BREAKDOWN)
            _append_event_log(
                totals,
                event_time=actual_start + timedelta(minutes=max(1, duration // 2)),
                event=EVENT_MACHINE_BREAKDOWN,
                item=item,
                data=data,
                delay_minutes=machine_breakdown_delay,
                delivery_delay_minutes=0,
                loss_amount=0,
            )
        operational_delay = sample_operational_delay_minutes(
            distributions,
            rng,
            product_id=item.product_id,
            line_id=item.line_id,
        )
        if operational_delay:
            cause = sample_cause(
                distributions,
                rng,
                product_id=item.product_id,
                line_id=item.line_id,
            )
            event = _event_for_delay_cause(cause)
            _increment_counter(totals["delay_causes"], cause)
            _increment_counter(totals["event_counts"], event)
            _append_event_log(
                totals,
                event_time=actual_start
                + timedelta(minutes=duration + machine_breakdown_delay),
                event=event,
                item=item,
                data=data,
                cause=cause,
                delay_minutes=operational_delay,
                delivery_delay_minutes=0,
                loss_amount=0,
            )
        actual_end = actual_start + timedelta(
            minutes=duration + machine_breakdown_delay + operational_delay
        )
        _finalize_item(item, actual_end, state, data, totals, distributions, rng)

    return ScenarioResult(
        plan_variant_code=plan.plan_variant_code,
        total_tardiness_minutes=totals["total_tardiness_minutes"],
        delayed_order_count=totals["delayed_order_count"],
        late_penalty_amount=totals["late_penalty_amount"],
        changeover_cost=totals["changeover_cost"],
        material_shortage_quantity=totals["material_shortage_quantity"],
        material_shortage_penalty_amount=totals["material_shortage_penalty_amount"],
        total_risk_cost=(
            totals["late_penalty_amount"]
            + totals["changeover_cost"]
            + totals["material_shortage_penalty_amount"]
        ),
        material_shortage_count=totals["material_shortage_count"],
        yield_rate_sum=totals["yield_rate_sum"],
        yield_sample_count=totals["yield_sample_count"],
        defect_quantity=totals["defect_quantity"],
        delay_causes=totals["delay_causes"],
        event_counts=totals["event_counts"],
        event_log=sorted(totals["event_log"], key=lambda event: event["event_time"]),
    )


def initialize_simulation_state(
    data: NormalizedPlanningData,
    distributions: SamplingDistributions,
    rng: Random,
) -> SimulationState:
    """
    Parameters:
        - data: Normalized planning data containing line and material state.

    Methodology:
        - Initialize every active line at planning_start and every material from current
          available quantity.
        - Convert inbound material quantities into sampled future events.

    Output:
        - Mutable SimulationState used by one scenario run.
    """
    inbound_events = []
    for material_id, material in data.materials.items():
        inbound_quantity = get_inbound_material_quantity_scaled(material)
        if inbound_quantity <= 0:
            continue
        sampled_delay = sample_material_inbound_delay_minutes(distributions, rng)
        if sampled_delay is not None:
            inbound_time = data.planning_start + timedelta(minutes=sampled_delay)
            inbound_events.append((inbound_time, material_id, inbound_quantity))

    return SimulationState(
        line_available_at={line_id: data.planning_start for line_id in data.lines},
        line_last_product_id={line_id: None for line_id in data.lines},
        inventory_by_material_id={
            material_id: get_initial_available_material_quantity_scaled(material)
            for material_id, material in data.materials.items()
        },
        inbound_events=sorted(inbound_events, key=lambda item: item[0]),
    )


def _initial_unscheduled_metrics(plan: PlanResult, data: NormalizedPlanningData) -> dict:
    late_penalty = sum(
        data.orders[order_id].order.late_penalty_amount
        for order_id in plan.unscheduled_orders
        if order_id in data.orders
    )
    event_log = []
    for order_id in sorted(plan.unscheduled_orders):
        normalized_order = data.orders.get(order_id)
        if normalized_order is None:
            continue
        event_log.append(
            _build_event_log_row(
                event_time=normalized_order.order.due_date,
                event=EVENT_UNSCHEDULED,
                order_id=order_id,
                product_id=normalized_order.order.product_id,
                line_id=None,
                due_date=normalized_order.order.due_date,
                delivery_delay_minutes=None,
                loss_amount=normalized_order.order.late_penalty_amount,
                risk_cost=normalized_order.order.late_penalty_amount,
            )
        )
    return {
        "total_tardiness_minutes": 0,
        "delayed_order_count": len(plan.unscheduled_orders),
        "late_penalty_amount": late_penalty,
        "changeover_cost": 0,
        "material_shortage_quantity": 0,
        "material_shortage_penalty_amount": 0,
        "material_shortage_count": 0,
        "yield_rate_sum": 0.0,
        "yield_sample_count": 0,
        "defect_quantity": Decimal("0"),
        "delay_causes": {"UNSCHEDULED": len(plan.unscheduled_orders)}
        if plan.unscheduled_orders
        else {},
        "event_counts": {"UNSCHEDULED": len(plan.unscheduled_orders)}
        if plan.unscheduled_orders
        else {},
        "event_log": event_log,
    }


def _calculate_actual_start(
    item: ScheduleItem,
    state: SimulationState,
    data: NormalizedPlanningData,
    solver_config: SolverConfig,
    distributions: SamplingDistributions,
    rng: Random,
) -> tuple[datetime, int, int]:
    line_available = state.line_available_at[item.line_id]
    previous_product_id = state.line_last_product_id.get(item.line_id)
    if previous_product_id is None or previous_product_id == item.product_id:
        return (max(item.start_time, line_available), 0, 0)

    standard_changeover = get_changeover_minutes(
        previous_product_id,
        item.product_id,
        item.line_id,
        data.changeover_rules,
        solver_config.default_changeover_minutes,
    )
    sampled_changeover = sample_changeover_minutes(
        standard_changeover,
        distributions,
        rng,
        from_product_id=previous_product_id,
        to_product_id=item.product_id,
        line_id=item.line_id,
    )
    return (
        max(item.start_time, line_available + timedelta(minutes=sampled_changeover)),
        standard_changeover,
        sampled_changeover,
    )


def _consume_materials_or_delay(
    item: ScheduleItem,
    actual_start: datetime,
    state: SimulationState,
    data: NormalizedPlanningData,
    fallback_delay_minutes: int,
) -> tuple[int, int]:
    _process_inbounds_until(state, actual_start)
    requirements = _required_materials_by_id(item, data)
    if not requirements:
        return 0, 0
    shortage_quantity = _material_shortage_quantity_units(
        requirements,
        state.inventory_by_material_id,
    )
    if not shortage_quantity:
        for material_id, quantity in requirements.items():
            state.inventory_by_material_id[material_id] -= quantity
        return 0, 0

    future_index = state.next_inbound_index
    candidate_inventory = dict(state.inventory_by_material_id)
    while future_index < len(state.inbound_events):
        inbound_time, material_id, quantity = state.inbound_events[future_index]
        candidate_inventory[material_id] = candidate_inventory.get(material_id, 0) + quantity
        future_index += 1
        if all(
            candidate_inventory.get(required_material_id, 0) >= required_quantity
            for required_material_id, required_quantity in requirements.items()
        ):
            state.next_inbound_index = future_index
            state.inventory_by_material_id = candidate_inventory
            for required_material_id, required_quantity in requirements.items():
                state.inventory_by_material_id[required_material_id] -= required_quantity
            return (
                max(1, round((inbound_time - actual_start).total_seconds() / 60)),
                shortage_quantity,
            )

    return fallback_delay_minutes, shortage_quantity


def _material_shortage_quantity_units(
    requirements: dict[str, int],
    inventory_by_material_id: dict[str, int],
) -> int:
    shortage_scaled = sum(
        max(0, required_quantity - inventory_by_material_id.get(material_id, 0))
        for material_id, required_quantity in requirements.items()
    )
    return _scaled_quantity_to_units(shortage_scaled)


def _material_shortage_penalty_amount(
    material_shortage_quantity: int,
    solver_config: SolverConfig,
) -> int:
    return material_shortage_quantity * solver_config.amount_optimization.material_shortage_weight


def _required_materials_by_id(
    item: ScheduleItem,
    data: NormalizedPlanningData,
) -> dict[str, int]:
    normalized_order = data.orders.get(item.order_id)
    if normalized_order is None:
        return {}
    product = data.products[normalized_order.order.product_id]
    requirements = {}
    for bom_item in data.bom_items_by_product_id.get(item.product_id, []):
        quantity = calculate_required_material_quantity_scaled(
            normalized_order.order,
            product,
            bom_item,
        )
        requirements[bom_item.material_id] = requirements.get(bom_item.material_id, 0) + quantity
    return requirements


def _process_inbounds_until(state: SimulationState, event_time: datetime) -> None:
    while state.next_inbound_index < len(state.inbound_events):
        inbound_time, material_id, quantity = state.inbound_events[state.next_inbound_index]
        if inbound_time > event_time:
            break
        state.inventory_by_material_id[material_id] = (
            state.inventory_by_material_id.get(material_id, 0) + quantity
        )
        state.next_inbound_index += 1


def _scaled_quantity_to_units(quantity_scaled: int) -> int:
    return (quantity_scaled + MATERIAL_QUANTITY_SCALE - 1) // MATERIAL_QUANTITY_SCALE


def _finalize_item(
    item: ScheduleItem,
    actual_end: datetime,
    state: SimulationState,
    data: NormalizedPlanningData,
    totals: dict,
    distributions: SamplingDistributions,
    rng: Random,
) -> None:
    normalized_order = data.orders[item.order_id]
    tardiness = max(
        0,
        round((actual_end - normalized_order.order.due_date).total_seconds() / 60),
    )
    if tardiness:
        totals["delayed_order_count"] += 1
        totals["late_penalty_amount"] += normalized_order.order.late_penalty_amount
        _increment_counter(totals["event_counts"], EVENT_LATE_COMPLETION)
        _append_event_log(
            totals,
            event_time=actual_end,
            event=EVENT_LATE_COMPLETION,
            item=item,
            data=data,
            actual_end_time=actual_end,
            delivery_delay_minutes=tardiness,
            loss_amount=normalized_order.order.late_penalty_amount,
            risk_cost=normalized_order.order.late_penalty_amount,
        )
    else:
        _append_event_log(
            totals,
            event_time=actual_end,
            event=EVENT_ORDER_COMPLETED,
            item=item,
            data=data,
            actual_end_time=actual_end,
            delivery_delay_minutes=0,
            loss_amount=0,
            risk_cost=0,
        )
    totals["total_tardiness_minutes"] += tardiness
    totals["yield_rate_sum"] += sample_yield_rate(
        distributions,
        rng,
        product_id=item.product_id,
        line_id=item.line_id,
    )
    totals["yield_sample_count"] += 1
    defect_quantity = Decimal(item.planned_production_quantity) * Decimal(
        str(
            sample_defect_rate(
                distributions,
                rng,
                product_id=item.product_id,
                line_id=item.line_id,
            )
        )
    )
    totals["defect_quantity"] += defect_quantity
    if defect_quantity > 0:
        _increment_counter(totals["event_counts"], EVENT_QUALITY_DEFECT)
        _append_event_log(
            totals,
            event_time=actual_end,
            event=EVENT_QUALITY_DEFECT,
            item=item,
            data=data,
            defect_quantity=float(defect_quantity),
            delivery_delay_minutes=0,
            loss_amount=0,
        )
    _update_order_event_result(
        totals,
        item.order_id,
        tardiness,
        normalized_order.order.late_penalty_amount if tardiness else 0,
    )
    previous_product_id = state.line_last_product_id.get(item.line_id)
    if previous_product_id and previous_product_id != item.product_id:
        totals["changeover_cost"] += get_changeover_cost(
            previous_product_id,
            item.product_id,
            item.line_id,
            data.changeover_rules,
            0,
        )
    state.line_available_at[item.line_id] = actual_end
    state.line_last_product_id[item.line_id] = item.product_id


def _increment_counter(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _event_for_delay_cause(cause: str) -> str:
    return EVENT_BY_DELAY_CAUSE.get(cause, EVENT_LINE_CHANGE_DELAY)


def _update_order_event_result(
    totals: dict,
    order_id: str,
    delivery_delay_minutes: int,
    loss_amount: int,
) -> None:
    for event in totals["event_log"]:
        if event.get("order_id") != order_id:
            continue
        event["delivery_delay_minutes"] = delivery_delay_minutes
        event["delivery_delay_days"] = _minutes_to_days(delivery_delay_minutes)
        event["loss_amount"] = loss_amount
        event["risk_cost"] = loss_amount


def _append_event_log(
    totals: dict,
    *,
    event_time: datetime,
    event: str,
    item: ScheduleItem,
    data: NormalizedPlanningData,
    delivery_delay_minutes: int | None,
    loss_amount: int,
    risk_cost: int | None = None,
    **extra,
) -> None:
    normalized_order = data.orders[item.order_id]
    totals["event_log"].append(
        _build_event_log_row(
            event_time=event_time,
            event=event,
            order_id=item.order_id,
            product_id=item.product_id,
            line_id=item.line_id,
            due_date=normalized_order.order.due_date,
            delivery_delay_minutes=delivery_delay_minutes,
            loss_amount=loss_amount,
            risk_cost=loss_amount if risk_cost is None else risk_cost,
            **extra,
        )
    )


def _build_event_log_row(
    *,
    event_time: datetime,
    event: str,
    order_id: str | None,
    product_id: str | None,
    line_id: str | None,
    due_date: datetime | None,
    delivery_delay_minutes: int | None,
    loss_amount: int,
    risk_cost: int,
    **extra,
) -> dict:
    return {
        "event_time": event_time,
        "event": event,
        "order_id": order_id,
        "product_id": product_id,
        "line_id": line_id,
        "due_date": due_date,
        "delivery_delay_minutes": delivery_delay_minutes,
        "delivery_delay_days": _minutes_to_days(delivery_delay_minutes),
        "loss_amount": loss_amount,
        "risk_cost": risk_cost,
        **extra,
    }


def _minutes_to_days(minutes: int | None) -> float | None:
    if minutes is None:
        return None
    return round(minutes / (24 * 60), 4)
