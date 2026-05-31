from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from random import Random

from app.features.production_planning.config import SimulationConfig, SolverConfig
from app.features.production_planning.preprocessing import (
    calculate_required_material_quantity_scaled,
    get_changeover_cost,
    get_changeover_minutes,
    get_inbound_material_quantity_scaled,
    get_inbound_material_time,
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
    sample_operational_delay_minutes,
    sample_yield_rate,
)


@dataclass(frozen=True)
class ScenarioResult:
    plan_variant_code: str
    total_tardiness_minutes: int
    delayed_order_count: int
    late_penalty_amount: int
    changeover_cost: int
    total_risk_cost: int
    material_shortage_count: int
    yield_rate_sum: float
    yield_sample_count: int
    defect_quantity: float
    delay_causes: dict[str, int] = field(default_factory=dict)


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
    state = initialize_simulation_state(data)
    totals = _initial_unscheduled_metrics(plan, data)

    for item in sorted(plan.schedule_items, key=lambda value: (value.start_time, value.line_id)):
        _process_inbounds_until(state, item.start_time)
        actual_start = _calculate_actual_start(item, state, data, solver_config, distributions, rng)
        material_shortage = _consume_materials_or_delay(item, actual_start, state, data)
        if material_shortage:
            actual_start += timedelta(minutes=simulation_config.material_shortage_delay_minutes)
            totals["material_shortage_count"] += 1
            totals["delay_causes"]["MATERIAL_SHORTAGE"] = (
                totals["delay_causes"].get("MATERIAL_SHORTAGE", 0) + 1
            )

        planned_duration = max(1, round((item.end_time - item.start_time).total_seconds() / 60))
        duration = sample_duration_minutes(
            planned_duration,
            distributions,
            rng,
            product_id=item.product_id,
            line_id=item.line_id,
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
            totals["delay_causes"][cause] = totals["delay_causes"].get(cause, 0) + 1
        actual_end = actual_start + timedelta(minutes=duration + operational_delay)
        _finalize_item(item, actual_end, state, data, totals, distributions, rng)

    return ScenarioResult(
        plan_variant_code=plan.plan_variant_code,
        total_tardiness_minutes=totals["total_tardiness_minutes"],
        delayed_order_count=totals["delayed_order_count"],
        late_penalty_amount=totals["late_penalty_amount"],
        changeover_cost=totals["changeover_cost"],
        total_risk_cost=totals["late_penalty_amount"] + totals["changeover_cost"],
        material_shortage_count=totals["material_shortage_count"],
        yield_rate_sum=totals["yield_rate_sum"],
        yield_sample_count=totals["yield_sample_count"],
        defect_quantity=totals["defect_quantity"],
        delay_causes=totals["delay_causes"],
    )


def initialize_simulation_state(data: NormalizedPlanningData) -> SimulationState:
    """
    Parameters:
        - data: Normalized planning data containing line and material state.

    Methodology:
        - Initialize every active line at planning_start and every material from current
          available quantity.
        - Convert expected inbound material quantities into ordered future events.

    Output:
        - Mutable SimulationState used by one scenario run.
    """
    inbound_events = []
    for material_id, material in data.materials.items():
        inbound_time = get_inbound_material_time(material)
        inbound_quantity = get_inbound_material_quantity_scaled(material)
        if inbound_time is not None and inbound_quantity > 0:
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
    return {
        "total_tardiness_minutes": 0,
        "delayed_order_count": len(plan.unscheduled_orders),
        "late_penalty_amount": late_penalty,
        "changeover_cost": 0,
        "material_shortage_count": 0,
        "yield_rate_sum": 0.0,
        "yield_sample_count": 0,
        "defect_quantity": Decimal("0"),
        "delay_causes": {"UNSCHEDULED": len(plan.unscheduled_orders)}
        if plan.unscheduled_orders
        else {},
    }


def _calculate_actual_start(
    item: ScheduleItem,
    state: SimulationState,
    data: NormalizedPlanningData,
    solver_config: SolverConfig,
    distributions: SamplingDistributions,
    rng: Random,
) -> datetime:
    line_available = state.line_available_at[item.line_id]
    previous_product_id = state.line_last_product_id.get(item.line_id)
    if previous_product_id is None or previous_product_id == item.product_id:
        return max(item.start_time, line_available)

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
    return max(item.start_time, line_available + timedelta(minutes=sampled_changeover))


def _consume_materials_or_delay(
    item: ScheduleItem,
    actual_start: datetime,
    state: SimulationState,
    data: NormalizedPlanningData,
) -> bool:
    _process_inbounds_until(state, actual_start)
    requirements = _required_materials_by_id(item, data)
    if not requirements:
        return False
    has_inventory = all(
        state.inventory_by_material_id.get(material_id, 0) >= quantity
        for material_id, quantity in requirements.items()
    )
    if has_inventory:
        for material_id, quantity in requirements.items():
            state.inventory_by_material_id[material_id] -= quantity
        return False

    future_index = state.next_inbound_index
    while future_index < len(state.inbound_events):
        _, material_id, quantity = state.inbound_events[future_index]
        state.inventory_by_material_id[material_id] = (
            state.inventory_by_material_id.get(material_id, 0) + quantity
        )
        future_index += 1
        if all(
            state.inventory_by_material_id.get(required_material_id, 0) >= required_quantity
            for required_material_id, required_quantity in requirements.items()
        ):
            state.next_inbound_index = future_index
            for required_material_id, required_quantity in requirements.items():
                state.inventory_by_material_id[required_material_id] -= required_quantity
            return True

    return True


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
    totals["total_tardiness_minutes"] += tardiness
    totals["yield_rate_sum"] += sample_yield_rate(
        distributions,
        rng,
        product_id=item.product_id,
        line_id=item.line_id,
    )
    totals["yield_sample_count"] += 1
    totals["defect_quantity"] += Decimal(item.planned_production_quantity) * Decimal(
        str(
            sample_defect_rate(
                distributions,
                rng,
                product_id=item.product_id,
                line_id=item.line_id,
            )
        )
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
