from __future__ import annotations

from datetime import datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from math import ceil

from app.features.production_planning.config import AmountOptimizationConfig, SolverConfig
from app.features.production_planning.exceptions import PlanningValidationError
from app.features.production_planning.schemas import (
    AmountReferenceData,
    BomItemInput,
    ChangeoverRuleInput,
    ExistingScheduleBlock,
    NormalizedLine,
    NormalizedOrder,
    NormalizedPlanningData,
    OrderInput,
    ProcessingCandidate,
    ProductInput,
    ProductionPlanningRequest,
    ProductLineCapabilityInput,
)

YIELD_SCALE = 10_000
MATERIAL_QUANTITY_SCALE = 10_000
CHANGEOVER_HOUR_TOLERANCE = Decimal("0.0001")


def normalize_request(request: ProductionPlanningRequest) -> NormalizedPlanningData:
    """
    Parameters:
        - request: Validated production planning request.

    Methodology:
        - Convert datetimes to integer minute offsets and build lookup dictionaries used by
          CP-SAT model construction.
        - Prepare only active lines for new jobs while retaining locked schedules as blockers.

    Output:
        - NormalizedPlanningData containing deterministic maps and candidate assignments.
    """
    horizon_minutes = calculate_horizon_minutes(request.planning_start, request.planning_end)
    warnings: list[str] = []
    products = {product.product_id: product for product in request.products}
    active_lines = [line for line in request.production_lines if line.is_active]
    lines = {
        line.line_id: NormalizedLine(
            line=line,
            available_from_offset=max(
                0,
                to_minute_offset(
                    line.available_from or request.planning_start,
                    request.planning_start,
                ),
            ),
            available_to_offset=min(
                horizon_minutes,
                to_minute_offset(line.available_to or request.planning_end, request.planning_start),
            ),
        )
        for line in active_lines
    }
    blocks_by_line_id = build_existing_schedule_blocks(request, lines, horizon_minutes, warnings)
    unavailable_no_changeover_lines = find_unavailable_no_changeover_lines(
        lines,
        blocks_by_line_id,
        warnings,
    )
    locked_no_changeover_product_by_line = find_locked_no_changeover_products(
        lines,
        blocks_by_line_id,
        unavailable_no_changeover_lines,
    )
    capabilities = {
        (capability.product_id, capability.line_id): capability
        for capability in expand_interchangeable_line_capabilities(
            request.product_line_capabilities,
            set(lines),
            request.solver_config.interchangeable_line_groups,
        )
    }
    plannable_orders = filter_plannable_orders(request.orders, request.solver_config, warnings)
    orders = {
        order.order_id: NormalizedOrder(
            order=order,
            due_date_offset=to_minute_offset(order.due_date, request.planning_start),
            priority=order.priority if order.priority is not None else 1,
            normalized_amount=normalize_order_amount(
                order.order_amount,
                request.solver_config.amount_optimization.amount_scale_unit,
            ),
        )
        for order in sorted(plannable_orders, key=lambda item: item.order_id)
    }

    candidates_by_order_id: dict[str, list[ProcessingCandidate]] = {}
    candidates_by_line_id: dict[str, list[ProcessingCandidate]] = {line_id: [] for line_id in lines}
    for normalized_order in orders.values():
        order = normalized_order.order
        for line_id in get_capable_lines(order.product_id, list(lines.values()), capabilities):
            capability = capabilities[(order.product_id, line_id)]
            if line_id in unavailable_no_changeover_lines:
                continue
            locked_product_id = locked_no_changeover_product_by_line.get(line_id)
            if locked_product_id is not None and locked_product_id != order.product_id:
                continue
            duration = calculate_candidate_duration_minutes(
                order,
                line_id,
                capability,
                products[order.product_id],
                request.planning_start,
            )
            planned_quantity = calculate_planned_production_quantity(
                order,
                products[order.product_id],
            )
            candidate = ProcessingCandidate(
                order_id=order.order_id,
                product_id=order.product_id,
                line_id=line_id,
                duration_minutes=duration,
                priority_rank=capability.priority_rank
                or request.solver_config.default_line_priority_rank,
                planned_production_quantity=planned_quantity,
            )
            candidates_by_order_id.setdefault(order.order_id, []).append(candidate)
            candidates_by_line_id.setdefault(line_id, []).append(candidate)

    validate_locked_order_candidates(plannable_orders, candidates_by_order_id)

    bom_items_by_product_id: dict[str, list[BomItemInput]] = {}
    for bom_item in request.bom_items:
        bom_items_by_product_id.setdefault(bom_item.product_id, []).append(bom_item)

    return NormalizedPlanningData(
        planning_start=request.planning_start,
        planning_end=request.planning_end,
        horizon_minutes=horizon_minutes,
        orders=orders,
        products=products,
        lines=lines,
        capabilities=capabilities,
        candidates_by_order_id=candidates_by_order_id,
        candidates_by_line_id=candidates_by_line_id,
        existing_schedule_blocks_by_line_id=blocks_by_line_id,
        materials={material.material_id: material for material in request.materials},
        bom_items_by_product_id=bom_items_by_product_id,
        changeover_rules={
            (rule.from_product_id, rule.to_product_id, rule.line_id): rule
            for rule in request.changeover_rules
        },
        warnings=warnings,
    )


def validate_locked_order_candidates(
    orders: list[OrderInput],
    candidates_by_order_id: dict[str, list[ProcessingCandidate]],
) -> None:
    for order in orders:
        if not order.is_locked or order.locked_plan is None:
            continue
        if not any(
            candidate.line_id == order.locked_plan.line_id
            for candidate in candidates_by_order_id.get(order.order_id, [])
        ):
            raise PlanningValidationError(
                f"Locked order {order.order_id} has no candidate on locked line "
                f"{order.locked_plan.line_id}."
            )


def normalize_status(status: str | None) -> str | None:
    if status is None:
        return None
    return status.strip().upper()


def filter_plannable_orders(
    orders: list[OrderInput],
    config: SolverConfig,
    warnings: list[str],
) -> list[OrderInput]:
    excluded_statuses = {normalize_status(status) for status in config.excluded_order_statuses}
    plannable = [
        order
        for order in orders
        if normalize_status(order.status) not in excluded_statuses
    ]
    excluded_count = len(orders) - len(plannable)
    if excluded_count:
        warnings.append(f"{excluded_count} orders excluded by order status.")
    if not plannable:
        raise PlanningValidationError("No plannable orders remain after status filtering.")
    return plannable


def build_existing_schedule_blocks(
    request: ProductionPlanningRequest,
    lines: dict[str, NormalizedLine],
    horizon_minutes: int,
    warnings: list[str],
) -> dict[str, list[ExistingScheduleBlock]]:
    blocks_by_line_id: dict[str, list[ExistingScheduleBlock]] = {line_id: [] for line_id in lines}
    locked_statuses = {
        normalize_status(status) for status in request.solver_config.locked_plan_statuses
    }
    for schedule in request.existing_schedules:
        if schedule.line_id not in blocks_by_line_id or not schedule.is_locked:
            continue
        normalized_status = normalize_status(schedule.plan_status)
        if normalized_status is not None and normalized_status not in locked_statuses:
            continue
        if (
            schedule.end_time <= request.planning_start
            or schedule.start_time >= request.planning_end
        ):
            continue
        start_offset = max(0, to_minute_offset(schedule.start_time, request.planning_start))
        end_offset = min(
            horizon_minutes,
            to_minute_offset_ceil(schedule.end_time, request.planning_start),
        )
        if start_offset >= end_offset:
            continue
        blocks_by_line_id[schedule.line_id].append(
            ExistingScheduleBlock(
                schedule=schedule,
                start_offset=start_offset,
                end_offset=end_offset,
                duration_minutes=end_offset - start_offset,
            )
        )
    ignored_count = len(request.existing_schedules) - sum(
        len(items) for items in blocks_by_line_id.values()
    )
    if ignored_count:
        warnings.append(f"{ignored_count} existing schedules ignored by lock status or horizon.")
    return blocks_by_line_id


def find_unavailable_no_changeover_lines(
    lines: dict[str, NormalizedLine],
    blocks_by_line_id: dict[str, list[ExistingScheduleBlock]],
    warnings: list[str],
) -> set[str]:
    unavailable_line_ids = set()
    for line_id, line in lines.items():
        if line.line.supports_changeover:
            continue
        product_ids = {
            block.schedule.product_id
            for block in blocks_by_line_id.get(line_id, [])
            if block.schedule.product_id
        }
        if len(product_ids) > 1:
            unavailable_line_ids.add(line_id)
            warnings.append(
                f"Line {line_id} disabled for new assignments because locked schedules "
                "contain multiple products and supports_changeover is false."
            )
    return unavailable_line_ids


def find_locked_no_changeover_products(
    lines: dict[str, NormalizedLine],
    blocks_by_line_id: dict[str, list[ExistingScheduleBlock]],
    unavailable_line_ids: set[str],
) -> dict[str, str]:
    locked_products = {}
    for line_id, line in lines.items():
        if line.line.supports_changeover or line_id in unavailable_line_ids:
            continue
        product_ids = {
            block.schedule.product_id
            for block in blocks_by_line_id.get(line_id, [])
            if block.schedule.product_id
        }
        if len(product_ids) == 1:
            locked_products[line_id] = next(iter(product_ids))
    return locked_products


def to_minute_offset(dt: datetime, planning_start: datetime) -> int:
    """
    Parameters:
        - dt: Datetime to convert.
        - planning_start: Planning window start used as zero offset.

    Methodology:
        - Convert elapsed seconds to whole minutes using floor division.

    Output:
        - Integer minute offset from planning_start.
    """
    return int((dt - planning_start).total_seconds() // 60)


def to_minute_offset_ceil(dt: datetime, planning_start: datetime) -> int:
    """
    Parameters:
        - dt: Datetime to convert.
        - planning_start: Planning window start used as zero offset.

    Methodology:
        - Convert elapsed seconds to whole minutes using ceiling division.
        - This keeps existing schedule blockers conservative when DB timestamps include
          seconds but CP-SAT operates in whole-minute buckets.

    Output:
        - Integer minute offset from planning_start.
    """
    return ceil((dt - planning_start).total_seconds() / 60)


def from_minute_offset(offset: int, planning_start: datetime) -> datetime:
    """
    Parameters:
        - offset: Integer minute offset.
        - planning_start: Planning window start used as zero offset.

    Methodology:
        - Add the integer offset as minutes to the planning start.

    Output:
        - Datetime represented by the offset.
    """
    return planning_start + timedelta(minutes=offset)


def calculate_horizon_minutes(planning_start: datetime, planning_end: datetime) -> int:
    """
    Parameters:
        - planning_start: Inclusive planning window start.
        - planning_end: Exclusive planning window end.

    Methodology:
        - Convert the planning interval length to integer minutes.

    Output:
        - Positive integer planning horizon in minutes.
    """
    return to_minute_offset(planning_end, planning_start)


def calculate_candidate_duration_minutes(
    order: OrderInput,
    line_id: str,
    capability: ProductLineCapabilityInput,
    product: ProductInput,
    planning_start: datetime,
) -> int:
    """
    Parameters:
        - order: Order requiring production.
        - line_id: Candidate production line ID.
        - capability: Product-line capability data.
        - product: Product master data with optional default process time.
        - planning_start: Planning window start used for locked offset conversion.

    Methodology:
        - Use the locked plan's fixed interval length when the order is fixed to this line.
        - Otherwise calculate the standard process duration from quantity, yield, and setup.

    Output:
        - Positive integer candidate duration in minutes.
    """
    if order.is_locked and order.locked_plan is not None and order.locked_plan.line_id == line_id:
        start = to_minute_offset(order.locked_plan.planned_start_at, planning_start)
        end = to_minute_offset_ceil(order.locked_plan.planned_end_at, planning_start)
        duration = end - start
        if duration <= 0:
            raise PlanningValidationError(f"Locked order {order.order_id} duration is invalid.")
        return duration
    return calculate_processing_duration_minutes(order, line_id, capability, product)


def get_capable_lines(
    product_id: str,
    active_lines: list[NormalizedLine],
    capabilities: dict[tuple[str, str], ProductLineCapabilityInput],
) -> list[str]:
    """
    Parameters:
        - product_id: Product that must be produced.
        - active_lines: Active line records normalized for the planning horizon.
        - capabilities: Lookup keyed by product and line.

    Methodology:
        - Select active lines with a matching product-line capability.
        - Sort line IDs for deterministic model construction and solution extraction.

    Output:
        - Sorted list of capable active line IDs.
    """
    active_line_ids = {line.line.line_id for line in active_lines}
    return sorted(
        line_id
        for candidate_product_id, line_id in capabilities
        if candidate_product_id == product_id and line_id in active_line_ids
    )


def expand_interchangeable_line_capabilities(
    capabilities: list[ProductLineCapabilityInput],
    active_line_ids: set[str],
    interchangeable_line_groups: list[list[str]],
) -> list[ProductLineCapabilityInput]:
    """
    Parameters:
        - capabilities: Product-line capability rows from source data.
        - active_line_ids: Active line IDs eligible for new production.
        - interchangeable_line_groups: Line ID groups where lines can substitute each other.

    Methodology:
        - Keep exact active capabilities as-is.
        - For a capability in an interchangeable group, create equivalent active peer-line
          capabilities unless an exact product-line row already exists.
        - Preserve processing time, setup, yield, and priority rank from the source capability.

    Output:
        - Product-line capabilities expanded with interchangeable active line candidates.
    """
    capability_by_key = {
        (capability.product_id, capability.line_id): capability for capability in capabilities
    }
    expanded_by_key: dict[tuple[str, str], ProductLineCapabilityInput] = {}
    peer_line_ids_by_line_id = build_interchangeable_line_lookup(interchangeable_line_groups)

    for capability in capabilities:
        if capability.line_id in active_line_ids:
            expanded_by_key[(capability.product_id, capability.line_id)] = capability
        for peer_line_id in peer_line_ids_by_line_id.get(capability.line_id, set()):
            if peer_line_id not in active_line_ids:
                continue
            peer_key = (capability.product_id, peer_line_id)
            exact_peer_capability = capability_by_key.get(peer_key)
            expanded_by_key[peer_key] = exact_peer_capability or capability.model_copy(
                update={"line_id": peer_line_id}
            )

    return [expanded_by_key[key] for key in sorted(expanded_by_key)]


def build_interchangeable_line_lookup(
    interchangeable_line_groups: list[list[str]],
) -> dict[str, set[str]]:
    """
    Parameters:
        - interchangeable_line_groups: Groups of line IDs that can replace each other.

    Methodology:
        - Convert group definitions into a line-to-peer lookup.
        - A line is considered interchangeable with every line in its group, including itself,
          which simplifies exact and peer candidate expansion.

    Output:
        - Mapping from line ID to the set of interchangeable line IDs.
    """
    lookup: dict[str, set[str]] = {}
    for group in interchangeable_line_groups:
        normalized_group = {line_id for line_id in group if line_id}
        for line_id in normalized_group:
            lookup.setdefault(line_id, set()).update(normalized_group)
    return lookup


def calculate_processing_duration_minutes(
    order: OrderInput,
    line_id: str,
    capability: ProductLineCapabilityInput,
    product: ProductInput,
) -> int:
    """
    Parameters:
        - order: Order requiring production.
        - line_id: Candidate production line ID.
        - capability: Product-line capability data.
        - product: Product master data with optional default process time.

    Methodology:
        - Use capability process time first, then product default.
        - If yield rate is supplied, inflate quantity with ceil(quantity / yield_rate).
        - Add fixed setup time once per order-line assignment.

    Output:
        - Integer processing duration in minutes.
    """
    process_time = capability.process_time_per_unit_minutes or product.default_process_time_minutes
    if process_time is None or process_time <= 0:
        raise PlanningValidationError(
            f"Process time is missing for {order.product_id} on {line_id}."
        )
    quantity = calculate_planned_production_quantity(order, product)
    if capability.yield_rate_scaled is not None:
        quantity = ceil(quantity * YIELD_SCALE / capability.yield_rate_scaled)
    return ceil(quantity * process_time) + (capability.fixed_setup_minutes or 0)


def get_order_quantity(order: OrderInput) -> int:
    if order.order_quantity is not None:
        return order.order_quantity
    if order.quantity is not None:
        return order.quantity
    raise PlanningValidationError(f"Order {order.order_id} has no quantity.")


def calculate_planned_production_quantity(order: OrderInput, product: ProductInput) -> int:
    minimum_quantity = product.min_production_quantity or 0
    return max(get_order_quantity(order), minimum_quantity)


def decimal_to_scaled_ceiling(value: Decimal, scale: int = MATERIAL_QUANTITY_SCALE) -> int:
    return int((value * scale).to_integral_value(rounding=ROUND_CEILING))


def decimal_to_scaled_floor(value: Decimal, scale: int = MATERIAL_QUANTITY_SCALE) -> int:
    return int((value * scale).to_integral_value(rounding=ROUND_FLOOR))


def calculate_required_material_quantity_scaled(
    order: OrderInput,
    product: ProductInput,
    bom_item: BomItemInput,
) -> int:
    planned_quantity = Decimal(calculate_planned_production_quantity(order, product))
    loss_multiplier = Decimal("1") + bom_item.loss_rate
    required = bom_item.required_quantity_per_unit * planned_quantity * loss_multiplier
    return decimal_to_scaled_ceiling(required)


def get_initial_available_material_quantity_scaled(material) -> int:
    available = max(Decimal(str(material.available_quantity or 0)), Decimal("0"))
    safety_stock = max(Decimal(str(material.safety_stock_quantity or 0)), Decimal("0"))
    return decimal_to_scaled_floor(max(available - safety_stock, Decimal("0")))


def get_inbound_material_quantity_scaled(material) -> int:
    inbound_quantity = material.expected_inbound_quantity
    if inbound_quantity is None:
        inbound_quantity = material.confirmed_inbound_quantity
    if inbound_quantity is None:
        return 0
    return decimal_to_scaled_floor(inbound_quantity)


def get_inbound_material_time(material) -> datetime | None:
    return material.expected_inbound_at or material.confirmed_inbound_time


def normalize_order_amount(order_amount: int, amount_scale_unit: int) -> int:
    """
    Parameters:
        - order_amount: Source order amount in integer money unit.
        - amount_scale_unit: Divisor used to keep CP-SAT coefficients compact.

    Methodology:
        - Scale down with ceil so any positive amount remains visible to the optimizer.

    Output:
        - Non-negative normalized integer order amount.
    """
    if order_amount <= 0:
        return 0
    return ceil(order_amount / amount_scale_unit)


def prepare_amount_reference_data(
    orders: list[OrderInput],
    products: list[ProductInput],
    config: SolverConfig,
) -> AmountReferenceData:
    """
    Parameters:
        - orders: Input orders used by amount optimization.
        - products: Product master records reserved for future margin enrichment.
        - config: Solver configuration containing amount normalization policy.

    Methodology:
        - Build explicit amount, normalized amount, priority, and high-amount lookups.
        - Use safe defaults for optional customer priority and margin data and record warnings
          instead of silently inventing business data.

    Output:
        - AmountReferenceData used by amount objective construction.
    """
    amount_config = config.amount_optimization
    high_amount_order_ids = identify_high_amount_orders(orders, amount_config)
    warnings = [
        "customer priority data is unavailable; defaulting customer priority to 1.",
        "margin amount data is unavailable; defaulting margin amount to order amount.",
    ]
    product_ids = {product.product_id for product in products}
    if any(order.product_id not in product_ids for order in orders):
        warnings.append("some orders reference products unavailable to amount reference data.")

    return AmountReferenceData(
        order_amount_by_order_id={order.order_id: order.order_amount for order in orders},
        normalized_amount_by_order_id={
            order.order_id: normalize_order_amount(
                order.order_amount,
                amount_config.amount_scale_unit,
            )
            for order in orders
        },
        priority_by_order_id={
            order.order_id: order.priority if order.priority is not None else 1 for order in orders
        },
        high_amount_order_ids=high_amount_order_ids,
        customer_priority_by_order_id={order.order_id: 1 for order in orders},
        margin_amount_by_order_id={order.order_id: order.order_amount for order in orders},
        warnings=warnings,
    )


def identify_high_amount_orders(
    orders: list[OrderInput],
    amount_config: AmountOptimizationConfig,
) -> set[str]:
    """
    Parameters:
        - orders: Orders to classify.
        - amount_config: Amount optimization threshold policy.

    Methodology:
        - Apply deterministic threshold policy with order_id tie-breaking for top-percent mode.

    Output:
        - Set of order IDs considered high amount.
    """
    if not orders:
        return set()
    policy = amount_config.high_amount_threshold_policy
    if policy == "TOP_20_PERCENT":
        limit = max(1, ceil(len(orders) * 0.2))
        ordered = sorted(orders, key=lambda order: (-order.order_amount, order.order_id))
        return {order.order_id for order in ordered[:limit]}
    if policy == "ABOVE_AVERAGE":
        average_amount = sum(order.order_amount for order in orders) / len(orders)
        return {order.order_id for order in orders if order.order_amount >= average_amount}
    if amount_config.custom_high_amount_threshold is None:
        raise PlanningValidationError(
            "custom_high_amount_threshold is required for CUSTOM_THRESHOLD policy."
        )
    return {
        order.order_id
        for order in orders
        if order.order_amount >= amount_config.custom_high_amount_threshold
    }


def get_changeover_minutes(
    from_product_id: str,
    to_product_id: str,
    line_id: str,
    changeover_rules: dict[tuple[str, str, str | None], ChangeoverRuleInput],
    default_changeover_minutes: int,
) -> int:
    """
    Parameters:
        - from_product_id: Product produced first.
        - to_product_id: Product produced second.
        - line_id: Line where the transition occurs.
        - changeover_rules: Rule lookup keyed by product pair and optional line.
        - default_changeover_minutes: Fallback setup time.

    Methodology:
        - Return zero for same-product continuation.
        - Prefer exact line-specific rule, then global product-pair rule, then default.

    Output:
        - Integer changeover minutes.
    """
    if from_product_id == to_product_id:
        return 0
    line_rule = changeover_rules.get((from_product_id, to_product_id, line_id))
    if line_rule is not None:
        return calculate_changeover_minutes(line_rule)
    global_rule = changeover_rules.get((from_product_id, to_product_id, None))
    if global_rule is not None:
        return calculate_changeover_minutes(global_rule)
    return default_changeover_minutes


def calculate_changeover_minutes(rule: ChangeoverRuleInput) -> int:
    if rule.total_changeover_time_hr is not None:
        return int(
            (rule.total_changeover_time_hr * Decimal("60")).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
    if rule.cleaning_time_hr is not None and rule.stabilization_time_hr is not None:
        total_hours = rule.cleaning_time_hr + rule.stabilization_time_hr
        return int((total_hours * Decimal("60")).to_integral_value(rounding=ROUND_CEILING))
    if rule.changeover_minutes is not None:
        return rule.changeover_minutes
    raise PlanningValidationError("Changeover rule requires total hours or minutes.")


def get_changeover_cost(
    from_product_id: str,
    to_product_id: str,
    line_id: str,
    changeover_rules: dict[tuple[str, str, str | None], ChangeoverRuleInput],
    default_changeover_cost: int,
) -> int:
    """
    Parameters:
        - from_product_id: Product produced first.
        - to_product_id: Product produced second.
        - line_id: Line where the transition occurs.
        - changeover_rules: Rule lookup keyed by product pair and optional line.
        - default_changeover_cost: Fallback setup cost.

    Methodology:
        - Return zero for same-product continuation.
        - Prefer exact line-specific rule, then global product-pair rule, then default.

    Output:
        - Integer changeover cost.
    """
    if from_product_id == to_product_id:
        return 0
    line_rule = changeover_rules.get((from_product_id, to_product_id, line_id))
    if line_rule is not None:
        return line_rule.changeover_cost or 0
    global_rule = changeover_rules.get((from_product_id, to_product_id, None))
    if global_rule is not None:
        return global_rule.changeover_cost or 0
    return default_changeover_cost
