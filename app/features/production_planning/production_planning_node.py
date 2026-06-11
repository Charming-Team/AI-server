from datetime import datetime

from app.core.config import get_settings
from app.features.production_planning.config import SimulationConfig, SolverConfig
from app.features.production_planning.exceptions import PlanningValidationError
from app.features.production_planning.json_formatter import build_dashboard_response
from app.features.production_planning.langgraph_workflow import run_production_planning_graph
from app.features.production_planning.preprocessing import normalize_request
from app.features.production_planning.repositories.planning_data_repository import (
    PlanningDataRepository,
    PlanningInputBundle,
)
from app.features.production_planning.repositories.simulation_data_repository import (
    SimulationDataRepository,
)
from app.features.production_planning.schemas import (
    ExistingScheduleInput,
    LockedPlanInput,
    OrderInput,
    PlanningOrderPatchInput,
    ProductionPlanningAdjustmentRequest,
    ProductionPlanningRequest,
    ProductionPlanningResult,
)


def generate_production_plans(
    request: ProductionPlanningRequest,
) -> ProductionPlanningResult:
    """
    Parameters:
        - request: Production planning request containing orders, resources, constraints, and
          solver configuration.

    Methodology:
        - Delegate orchestration to the LangGraph production planning workflow.
        - The graph connects validation, preprocessing, model-building, solving, extraction,
          comparison, and finalization nodes.

    Output:
        - ProductionPlanningResult with due-date and amount optimal plans plus comparison summary.
    """
    return run_production_planning_graph(request)


def generate_production_plans_from_db(
    planning_start: datetime,
    planning_end: datetime,
    solver_config: SolverConfig | None = None,
) -> ProductionPlanningResult:
    """
    Parameters:
        - planning_start: Inclusive start of the planning window (timezone-aware datetime).
        - planning_end: Exclusive end of the planning window (timezone-aware datetime).
        - solver_config: Optional solver configuration; defaults to SolverConfig() when omitted.

    Methodology:
        - Load all planning input data from ai_planning schema views via PlanningDataRepository.
        - Build a ProductionPlanningRequest from the loaded bundle and the supplied time window.
        - Delegate to generate_production_plans() to run the two-plan optimization.

    Output:
        - ProductionPlanningResult with due-date and amount optimal plans plus comparison summary.
    """
    repository = PlanningDataRepository()
    effective_config = solver_config or SolverConfig()
    bundle = repository.load_planning_input_bundle(
        planning_start,
        planning_end,
        effective_config.excluded_order_statuses,
        use_existing_plans_as_orders=True,
        excluded_rescheduling_plan_statuses=(
            effective_config.excluded_rescheduling_plan_statuses
        ),
    )

    request = ProductionPlanningRequest(
        planning_start=planning_start,
        planning_end=planning_end,
        orders=bundle.orders,
        products=bundle.products,
        production_lines=bundle.production_lines,
        product_line_capabilities=bundle.product_line_capabilities,
        existing_schedules=bundle.existing_schedules,
        changeover_rules=bundle.changeover_rules,
        materials=bundle.material_inventories,
        bom_items=bundle.bom_items,
        operators=bundle.operators,
        solver_config=effective_config,
    )

    return generate_production_plans(request)


def generate_adjusted_production_plans(
    request: ProductionPlanningAdjustmentRequest,
) -> dict:
    """
    Parameters:
        - request: Thin adjustment request containing planning window, edit_orders, and
          add_orders.

    Methodology:
        - Load master and current planning data from DB views.
        - Merge DB rescheduling targets with front-end state: edit_orders become locked,
          add_orders become movable new orders, and untouched DB orders remain movable.
        - Run the existing LangGraph workflow and serialize only production_plans-compatible
          adjusted plan candidates.

    Output:
        - Dictionary shaped as {"adjusted_plan_candidates": [...]}.
    """
    result = generate_adjusted_production_planning_result(request)
    return result.to_adjusted_plan_response()


def generate_adjusted_production_plan_dashboard_response(
    request: ProductionPlanningAdjustmentRequest,
    *,
    include_full_event_timeline: bool = False,
) -> dict:
    """
    Parameters:
        - request: Thin adjustment request containing planning window, edit_orders, and
          add_orders.
        - include_full_event_timeline: Whether to include representative full DES timelines.

    Methodology:
        - Generate adjusted CP-SAT and simulation alternatives through the existing workflow.
        - Load the DB current plan as the baseline comparison source.
        - Delegate final dashboard JSON assembly and delta calculations to the formatter.

    Output:
        - Dashboard JSON comparing DB current-plan baseline and adjusted candidates.
    """
    result, planning_request = _generate_adjusted_production_planning_result_with_request(
        request
    )
    baseline = SimulationDataRepository().load_baseline_simulation_snapshot(
        request.planning_start,
        request.planning_end,
    )
    settings = get_settings()
    dashboard_response = build_dashboard_response(
        result,
        baseline_plans=baseline.current_plan_rows,
        normalized_data=normalize_request(planning_request),
        products=planning_request.products,
        production_lines=planning_request.production_lines,
        product_line_capabilities=planning_request.product_line_capabilities,
        planning_start=request.planning_start,
        planning_end=request.planning_end,
        include_full_event_timeline=include_full_event_timeline,
        enable_llm_evaluation=True,
        settings=settings,
    )
    dashboard_response["warnings"].extend(baseline.warnings)
    return dashboard_response


def generate_adjusted_production_plan_api_response(
    request: ProductionPlanningAdjustmentRequest,
) -> dict:
    """
    Parameters:
        - request: Thin adjustment request containing planning window, edit_orders, and
          add_orders.

    Methodology:
        - Execute the LangGraph planning workflow once.
        - Return both front-end dashboard simulation data and DB-save planning data from the
          same graph result so the two payloads stay consistent.

    Output:
        - Dictionary containing planning_response and simulation_response.
    """
    result = generate_adjusted_production_planning_result(request)
    if result.dashboard_response is None:
        raise PlanningValidationError("Planning dashboard response was not generated.")
    return {
        "planning_response": result.to_adjusted_plan_response(),
        "simulation_response": result.dashboard_response,
    }


def generate_adjusted_production_planning_result(
    request: ProductionPlanningAdjustmentRequest,
) -> ProductionPlanningResult:
    """
    Parameters:
        - request: Thin adjustment request containing front-end edit/add order state.

    Methodology:
        - Build the full internal ProductionPlanningRequest from DB views and request deltas.
        - Delegate all optimization, simulation, and response construction to LangGraph.

    Output:
        - Full ProductionPlanningResult for diagnostics or internal callers.
    """
    result, _ = _generate_adjusted_production_planning_result_with_request(request)
    return result


def _generate_adjusted_production_planning_result_with_request(
    request: ProductionPlanningAdjustmentRequest,
) -> tuple[ProductionPlanningResult, ProductionPlanningRequest]:
    """
    Parameters:
        - request: Thin adjustment request containing front-end edit/add order state.

    Methodology:
        - Load DB planning data once and keep the expanded internal request available for
          dashboard formatting metadata.
        - Delegate optimization and simulation to the existing LangGraph workflow.

    Output:
        - Tuple of ProductionPlanningResult and the full internal planning request.
    """
    solver_config = _default_adjustment_solver_config()
    simulation_config = _default_adjustment_simulation_config()
    is_full_db_replan = _is_full_db_replan_request(request)
    bundle = PlanningDataRepository().load_planning_input_bundle(
        request.planning_start,
        request.planning_end,
        solver_config.excluded_order_statuses,
        use_existing_plans_as_orders=True,
        excluded_rescheduling_plan_statuses=(
            solver_config.excluded_rescheduling_plan_statuses
            if is_full_db_replan
            else None
        ),
    )
    planning_request = build_adjusted_planning_request_from_bundle(
        request,
        bundle,
        solver_config,
        simulation_config,
    )
    return generate_production_plans(planning_request), planning_request


def build_adjusted_planning_request_from_bundle(
    request: ProductionPlanningAdjustmentRequest,
    bundle: PlanningInputBundle,
    solver_config: SolverConfig | None = None,
    simulation_config: SimulationConfig | None = None,
    include_unpatched_db_orders: bool | None = None,
) -> ProductionPlanningRequest:
    """
    Parameters:
        - request: Thin adjustment request containing edit_orders and add_orders.
        - bundle: DB-loaded planning input bundle.
        - solver_config: Optional internal solver config override for tests.
        - simulation_config: Optional internal simulation config override for tests.
        - include_unpatched_db_orders: Whether DB orders absent from the thin request are
          movable planning targets.

    Methodology:
        - Treat an empty edit/add request as full DB replanning.
        - Treat a non-empty edit/add request as local adjustment and only move requested rows.
        - Override matching DB orders with edit_orders and mark them locked.
        - Append add_orders as movable new work.
        - Keep non-requested DB schedules as fixed blockers during local adjustment.
        - Keep all master data from DB views.

    Output:
        - Full ProductionPlanningRequest accepted by the existing LangGraph workflow.
    """
    effective_solver_config = solver_config or _default_adjustment_solver_config()
    effective_simulation_config = simulation_config or _default_adjustment_simulation_config()
    if include_unpatched_db_orders is None:
        include_unpatched_db_orders = _is_full_db_replan_request(request)
    extra_movable_db_order_ids = (
        set()
        if include_unpatched_db_orders
        else _resolve_conflicting_movable_schedule_order_ids(
            bundle.orders,
            bundle.existing_schedules,
            request.edit_orders,
        )
    )
    orders = merge_adjustment_orders(
        bundle.orders,
        request.edit_orders,
        request.add_orders,
        include_unpatched_db_orders=include_unpatched_db_orders,
        extra_movable_db_order_ids=extra_movable_db_order_ids,
    )
    existing_schedules = (
        bundle.existing_schedules
        if include_unpatched_db_orders
        else _filter_requested_schedules_from_blockers(
            bundle.existing_schedules,
            bundle.orders,
            request.edit_orders,
            request.add_orders,
            extra_movable_db_order_ids,
        )
    )
    return ProductionPlanningRequest(
        planning_start=request.planning_start,
        planning_end=request.planning_end,
        orders=orders,
        products=bundle.products,
        production_lines=bundle.production_lines,
        product_line_capabilities=bundle.product_line_capabilities,
        existing_schedules=existing_schedules,
        changeover_rules=bundle.changeover_rules,
        materials=bundle.material_inventories,
        bom_items=bundle.bom_items,
        operators=bundle.operators,
        solver_config=effective_solver_config,
        simulation_config=effective_simulation_config,
    )


def merge_adjustment_orders(
    db_orders: list[OrderInput],
    edit_orders: list[PlanningOrderPatchInput],
    add_orders: list[PlanningOrderPatchInput],
    *,
    include_unpatched_db_orders: bool = True,
    extra_movable_db_order_ids: set[str] | None = None,
) -> list[OrderInput]:
    """
    Parameters:
        - db_orders: DB view orders currently eligible for rescheduling.
        - edit_orders: Front-end locked order state.
        - add_orders: Front-end newly added movable orders.
        - include_unpatched_db_orders: Whether unmentioned DB orders stay movable.
        - extra_movable_db_order_ids: Additional DB orders that must move because they
          conflict with locked front-end edits.

    Methodology:
        - Resolve edit order IDs against DB orders.
        - Replace matching DB orders with locked request values.
        - Replace matching DB orders with add_orders too, but keep them movable.
        - For local adjustment, start from only conflict neighbors so unrelated DB plans
          remain fixed blockers.
        - Reject duplicates inside the same request section to keep the merge deterministic.

    Output:
        - Ordered list of internal OrderInput values.
    """
    db_order_by_id = {order.order_id: order for order in db_orders}
    identity_lookup = _build_order_identity_lookup(db_order_by_id)
    extra_movable_db_order_ids = extra_movable_db_order_ids or set()
    merged_orders = (
        dict(db_order_by_id)
        if include_unpatched_db_orders
        else {
            order_id: db_order_by_id[order_id]
            for order_id in extra_movable_db_order_ids
            if order_id in db_order_by_id
        }
    )
    resolved_edit_ids: set[str] = set()

    for patch in edit_orders:
        if patch.locked_plan is None:
            raise PlanningValidationError(f"edit_order {patch.order_id} requires locked_plan.")
        resolved_order_id = _resolve_existing_order_id(patch.order_id, identity_lookup)
        if resolved_order_id in resolved_edit_ids:
            raise PlanningValidationError(f"Duplicate edit_order {patch.order_id}.")
        resolved_edit_ids.add(resolved_order_id)
        merged_orders[resolved_order_id] = _patch_to_order_input(
            patch,
            order_id=resolved_order_id,
            is_locked=True,
        )

    resolved_add_ids: set[str] = set()
    for patch in add_orders:
        if patch.locked_plan is not None:
            raise PlanningValidationError(
                f"add_order {patch.order_id} must not include locked_plan."
            )
        order_id = _resolve_add_order_id(patch.order_id, identity_lookup)
        if order_id in resolved_add_ids:
            raise PlanningValidationError(f"Duplicate add_order {patch.order_id}.")
        if order_id in resolved_edit_ids:
            raise PlanningValidationError(
                f"add_order {patch.order_id} duplicates an edit_order."
            )
        resolved_add_ids.add(order_id)
        merged_orders[order_id] = _patch_to_order_input(
            patch,
            order_id=order_id,
            is_locked=False,
        )

    return [merged_orders[order_id] for order_id in sorted(merged_orders)]


def _is_full_db_replan_request(request: ProductionPlanningAdjustmentRequest) -> bool:
    return not request.edit_orders and not request.add_orders


def _filter_requested_schedules_from_blockers(
    existing_schedules: list[ExistingScheduleInput],
    db_orders: list[OrderInput],
    edit_orders: list[PlanningOrderPatchInput],
    add_orders: list[PlanningOrderPatchInput],
    extra_movable_db_order_ids: set[str] | None = None,
) -> list[ExistingScheduleInput]:
    requested_order_ids = _resolve_requested_adjustment_order_ids(
        db_orders,
        edit_orders,
        add_orders,
    )
    requested_order_ids.update(extra_movable_db_order_ids or set())
    if not requested_order_ids:
        return existing_schedules
    return [
        schedule
        for schedule in existing_schedules
        if requested_order_ids.isdisjoint(_schedule_identity_keys(schedule))
    ]


def _resolve_requested_adjustment_order_ids(
    db_orders: list[OrderInput],
    edit_orders: list[PlanningOrderPatchInput],
    add_orders: list[PlanningOrderPatchInput],
) -> set[str]:
    db_order_by_id = {order.order_id: order for order in db_orders}
    identity_lookup = _build_order_identity_lookup(db_order_by_id)
    requested_order_ids: set[str] = set()

    for patch in edit_orders:
        requested_order_ids.add(_resolve_existing_order_id(patch.order_id, identity_lookup))
    for patch in add_orders:
        requested_order_ids.add(_resolve_add_order_id(patch.order_id, identity_lookup))

    return requested_order_ids


def _resolve_conflicting_movable_schedule_order_ids(
    db_orders: list[OrderInput],
    existing_schedules: list[ExistingScheduleInput],
    edit_orders: list[PlanningOrderPatchInput],
) -> set[str]:
    db_order_by_id = {order.order_id: order for order in db_orders}
    identity_lookup = _build_order_identity_lookup(db_order_by_id)
    locked_intervals: list[LockedPlanPatchInput] = [
        edit_order.locked_plan
        for edit_order in edit_orders
        if edit_order.locked_plan is not None
    ]
    if not locked_intervals:
        return set()

    movable_order_ids: set[str] = set()
    for schedule in existing_schedules:
        if not _schedule_overlaps_any_locked_interval(schedule, locked_intervals):
            continue
        for key in _schedule_identity_keys(schedule):
            resolved_order_id = identity_lookup.get(key)
            if resolved_order_id is not None:
                movable_order_ids.add(resolved_order_id)
                break
    return movable_order_ids


def _schedule_overlaps_any_locked_interval(
    schedule: ExistingScheduleInput,
    locked_intervals: list[LockedPlanPatchInput],
) -> bool:
    for locked_plan in locked_intervals:
        if str(schedule.line_id) != str(locked_plan.line_id):
            continue
        if schedule.end_time > locked_plan.planned_start_at and schedule.start_time < (
            locked_plan.planned_end_at
        ):
            return True
    return False


def _schedule_identity_keys(schedule: ExistingScheduleInput) -> set[str]:
    keys = set(_order_identity_keys(schedule.schedule_id))
    if schedule.order_id is not None:
        keys.update(_order_identity_keys(schedule.order_id))
    return keys


def _patch_to_order_input(
    patch: PlanningOrderPatchInput,
    *,
    order_id: str,
    is_locked: bool,
) -> OrderInput:
    return OrderInput(
        order_id=order_id,
        order_no=patch.order_no,
        product_id=str(patch.product_id),
        order_quantity=patch.order_quantity,
        due_date=patch.due_date,
        contract_amount=_money_numeric_to_int(patch.contract_amount),
        late_penalty_amount=_money_numeric_to_int(patch.late_penalty_amount),
        order_status=patch.order_status,
        is_locked=is_locked,
        locked_plan=_to_internal_locked_plan(patch) if is_locked else None,
    )


def _build_order_identity_lookup(db_order_by_id: dict[str, OrderInput]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for order_id in db_order_by_id:
        for key in _order_identity_keys(order_id):
            existing = lookup.get(key)
            if existing is not None and existing != order_id:
                raise PlanningValidationError(f"Ambiguous order identity {key}.")
            lookup[key] = order_id
    return lookup


def _order_identity_keys(order_id: str) -> set[str]:
    keys = {str(order_id)}
    if str(order_id).startswith("PLAN-"):
        keys.add(str(order_id).removeprefix("PLAN-"))
    elif str(order_id).isdigit():
        keys.add(f"PLAN-{order_id}")
    return keys


def _resolve_existing_order_id(order_id: str, identity_lookup: dict[str, str]) -> str:
    resolved_order_id = identity_lookup.get(str(order_id))
    if resolved_order_id is None:
        raise PlanningValidationError(f"edit_order {order_id} does not exist in DB planning data.")
    return resolved_order_id


def _resolve_add_order_id(order_id: str, identity_lookup: dict[str, str]) -> str:
    return identity_lookup.get(str(order_id), str(order_id))


def _to_internal_locked_plan(patch: PlanningOrderPatchInput) -> LockedPlanInput | None:
    if patch.locked_plan is None:
        return None
    return LockedPlanInput(
        line_id=str(patch.locked_plan.line_id),
        planned_start_at=patch.locked_plan.planned_start_at,
        planned_end_at=patch.locked_plan.planned_end_at,
    )


def _money_numeric_to_int(value) -> int:
    if value != value.to_integral_value():
        raise PlanningValidationError(
            "contract_amount and late_penalty_amount must be whole-number KRW "
            "before CP-SAT conversion."
        )
    return int(value)


def _default_adjustment_solver_config() -> SolverConfig:
    return SolverConfig(
        time_limit_seconds=60,
        num_search_workers=8,
        allow_unscheduled_orders=True,
        due_date_policy="SOFT",
        use_existing_schedule_locks=True,
        use_material_constraints=True,
    )


def _default_adjustment_simulation_config() -> SimulationConfig:
    return SimulationConfig(
        enabled=True,
        num_iterations=1000,
        random_seed=20260531,
    )
