from app.features.production_planning.config import AmountOptimizationConfig
from app.features.production_planning.exceptions import PlanningValidationError
from app.features.production_planning.preprocessing import (
    CHANGEOVER_HOUR_TOLERANCE,
    calculate_changeover_minutes,
    expand_interchangeable_line_capabilities,
    filter_plannable_orders,
    get_order_quantity,
)
from app.features.production_planning.schemas import (
    BomItemInput,
    ChangeoverRuleInput,
    ExistingScheduleInput,
    MaterialInput,
    ProductInput,
    ProductionLineInput,
    ProductionPlanningRequest,
    ProductLineCapabilityInput,
)


def validate_request(request: ProductionPlanningRequest) -> ProductionPlanningRequest:
    """
    Parameters:
        - request: Production planning request received from upstream structured data.

    Methodology:
        - Validate global planning window, required collections, referential integrity,
          material inputs, existing schedules, and amount optimization settings.
        - Keep validation side-effect free so the same request can be reused by both
          objective functions.

    Output:
        - The original request when all validation rules pass.
    """
    if request.planning_start >= request.planning_end:
        raise PlanningValidationError("planning_start must be earlier than planning_end.")
    if not request.orders:
        raise PlanningValidationError("orders must not be empty.")
    if not request.production_lines:
        raise PlanningValidationError("production_lines must not be empty.")

    product_by_id = {product.product_id: product for product in request.products}
    line_by_id = {line.line_id: line for line in request.production_lines}
    active_line_ids = {line.line_id for line in request.production_lines if line.is_active}
    plannable_orders = filter_plannable_orders(request.orders, request.solver_config, [])

    for order in request.orders:
        if order.product_id not in product_by_id:
            raise PlanningValidationError(
                f"Order {order.order_id} references unknown product {order.product_id}."
            )
        if (
            order.quantity is not None
            and order.order_quantity is not None
            and order.quantity != order.order_quantity
        ):
            raise PlanningValidationError(
                f"Order {order.order_id} quantity and order_quantity must match."
            )
        if get_order_quantity(order) <= 0:
            raise PlanningValidationError(f"Order {order.order_id} quantity must be positive.")
        if order.order_amount < 0:
            raise PlanningValidationError(
                f"Order {order.order_id} order_amount must be non-negative."
            )
        if order.late_penalty_amount < 0:
            raise PlanningValidationError(
                f"Order {order.order_id} late_penalty_amount must be non-negative."
            )

    validate_product_master_data(request.products)
    validate_line_master_data(request.production_lines)
    validate_product_line_capability(
        request.product_line_capabilities,
        request.products,
        request.production_lines,
        active_line_ids,
        plannable_orders,
        request.solver_config.interchangeable_line_groups,
    )
    validate_interchangeable_line_groups(
        request.solver_config.interchangeable_line_groups,
        line_by_id,
    )
    validate_existing_schedules(request.existing_schedules, line_by_id)
    validate_material_constraints(request.materials, request.bom_items, product_by_id)
    validate_changeover_rules(request.changeover_rules, line_by_id, product_by_id)
    validate_amount_optimization_config(request.solver_config.amount_optimization)
    return request


def validate_product_master_data(products: list[ProductInput]) -> None:
    for product in products:
        if product.average_yield_rate is not None and product.average_yield_rate <= 0:
            raise PlanningValidationError("average_yield_rate must be positive when provided.")
        if product.min_production_quantity is not None and product.min_production_quantity <= 0:
            raise PlanningValidationError("min_production_quantity must be positive when provided.")


def validate_line_master_data(lines: list[ProductionLineInput]) -> None:
    for line in lines:
        if line.max_capacity_per_day is not None and line.max_capacity_per_day <= 0:
            raise PlanningValidationError("max_capacity_per_day must be positive when provided.")


def validate_product_line_capability(
    capabilities: list[ProductLineCapabilityInput],
    products: list[ProductInput],
    production_lines: list[ProductionLineInput],
    active_line_ids: set[str],
    orders: list,
    interchangeable_line_groups: list[list[str]],
) -> None:
    """
    Parameters:
        - capabilities: Product-line production capability rows.
        - products: Known product master data.
        - production_lines: Known production line master data.
        - active_line_ids: Line IDs currently eligible for new production.
        - orders: Orders that need at least one active capable line.
        - interchangeable_line_groups: Line ID groups that can substitute each other.

    Methodology:
        - Ensure capability rows reference valid products and lines.
        - Expand capabilities through interchangeable line groups before checking active
          candidate coverage.
        - Ensure every ordered product has at least one active capable line and a calculable
          process time using either capability time or product default time.

    Output:
        - None. Raises PlanningValidationError when capability data is unusable.
    """
    product_by_id = {product.product_id: product for product in products}
    line_by_id = {line.line_id: line for line in production_lines}
    line_ids = set(line_by_id)
    capability_by_product: dict[str, list[ProductLineCapabilityInput]] = {}

    for capability in capabilities:
        if capability.product_id not in product_by_id:
            raise PlanningValidationError(
                f"Capability references unknown product {capability.product_id}."
            )
        if capability.line_id not in line_ids:
            raise PlanningValidationError(
                f"Capability references unknown line {capability.line_id}."
            )
        if capability.process_time_per_unit_minutes is not None:
            if capability.process_time_per_unit_minutes <= 0:
                raise PlanningValidationError("process_time_per_unit_minutes must be positive.")
        if capability.fixed_setup_minutes is not None and capability.fixed_setup_minutes < 0:
            raise PlanningValidationError("fixed_setup_minutes must be non-negative.")
        if capability.yield_rate_scaled is not None and capability.yield_rate_scaled <= 0:
            raise PlanningValidationError("yield_rate_scaled must be positive.")
        if capability.capacity_per_day is not None and capability.capacity_per_day <= 0:
            raise PlanningValidationError("capacity_per_day must be positive when provided.")
        if capability.priority_rank is not None and capability.priority_rank <= 0:
            raise PlanningValidationError("priority_rank must be positive when provided.")
        validate_capacity_units(
            product_by_id[capability.product_id],
            line_by_id[capability.line_id],
        )

    expanded_capabilities = expand_interchangeable_line_capabilities(
        capabilities,
        active_line_ids,
        interchangeable_line_groups,
    )
    for capability in expanded_capabilities:
        capability_by_product.setdefault(capability.product_id, []).append(capability)

    for order in orders:
        product = product_by_id[order.product_id]
        active_capabilities = [
            capability
            for capability in capability_by_product.get(order.product_id, [])
            if capability.line_id in active_line_ids
        ]
        if not active_capabilities:
            raise PlanningValidationError(
                f"Order {order.order_id} product {order.product_id} has no active capable line."
            )
        for capability in active_capabilities:
            process_time = (
                capability.process_time_per_unit_minutes
                if capability.process_time_per_unit_minutes is not None
                else product.default_process_time_minutes
            )
            if process_time is None or process_time <= 0:
                raise PlanningValidationError(
                    f"Process time is missing for product {order.product_id} "
                    f"on line {capability.line_id}."
                )


def validate_capacity_units(product: ProductInput, line: ProductionLineInput) -> None:
    if product.unit is None or line.capacity_unit is None:
        return
    if product.unit.strip().upper() != line.capacity_unit.strip().upper():
        raise PlanningValidationError(
            f"Product {product.product_id} unit is incompatible with line {line.line_id} capacity."
        )


def validate_interchangeable_line_groups(
    interchangeable_line_groups: list[list[str]],
    line_by_id: dict[str, ProductionLineInput],
) -> None:
    """
    Parameters:
        - interchangeable_line_groups: Configured groups of substitutable line IDs.
        - line_by_id: Valid production line lookup.

    Methodology:
        - Require at least two unique line IDs in a configured group.
        - Unknown line IDs are tolerated because default groups may describe plant-wide
          substitutability while a request can include only a subset of lines.

    Output:
        - None. Raises PlanningValidationError when line group configuration is invalid.
    """
    for group in interchangeable_line_groups:
        unique_group = {line_id for line_id in group if line_id}
        if len(unique_group) < 2:
            raise PlanningValidationError(
                "Each interchangeable line group must contain at least two line IDs."
            )


def validate_existing_schedules(
    existing_schedules: list[ExistingScheduleInput],
    line_by_id: dict[str, ProductionLineInput],
) -> None:
    """
    Parameters:
        - existing_schedules: Previously committed line schedules.
        - line_by_id: Valid production line lookup.

    Methodology:
        - Validate time ranges and line references so fixed interval constraints can be
          modeled without ambiguity.

    Output:
        - None. Raises PlanningValidationError for invalid lock data.
    """
    for schedule in existing_schedules:
        if schedule.line_id not in line_by_id:
            raise PlanningValidationError(
                f"Existing schedule {schedule.schedule_id} references unknown line."
            )
        if schedule.start_time >= schedule.end_time:
            raise PlanningValidationError(
                f"Existing schedule {schedule.schedule_id} start_time must be before end_time."
            )


def validate_material_constraints(
    materials: list[MaterialInput],
    bom_items: list[BomItemInput],
    product_by_id: dict[str, ProductInput],
) -> None:
    """
    Parameters:
        - materials: Available material inventory and confirmed inbound quantities.
        - bom_items: Product-material consumption rules.
        - product_by_id: Valid product lookup.

    Methodology:
        - Check non-negative material quantities and BOM referential integrity.
        - Require positive per-unit BOM consumption to keep linear material constraints valid.

    Output:
        - None. Raises PlanningValidationError when material inputs are inconsistent.
    """
    material_ids = {material.material_id for material in materials}
    material_by_id = {material.material_id: material for material in materials}
    for material in materials:
        if material.available_quantity < 0:
            raise PlanningValidationError(
                f"Material {material.material_id} available_quantity must be non-negative."
            )
        if material.safety_stock_quantity < 0:
            raise PlanningValidationError(
                f"Material {material.material_id} safety_stock_quantity must be non-negative."
            )
        if (
            material.expected_inbound_quantity is not None
            and material.expected_inbound_quantity < 0
        ):
            raise PlanningValidationError(
                f"Material {material.material_id} expected inbound quantity must be non-negative."
            )
        if (
            material.confirmed_inbound_quantity is not None
            and material.confirmed_inbound_quantity < 0
        ):
            raise PlanningValidationError(
                f"Material {material.material_id} inbound quantity must be non-negative."
            )

    for bom_item in bom_items:
        if bom_item.product_id not in product_by_id:
            raise PlanningValidationError(
                f"BOM references unknown product {bom_item.product_id}."
            )
        if bom_item.material_id not in material_ids:
            raise PlanningValidationError(
                f"BOM references unknown material {bom_item.material_id}."
            )
        if bom_item.required_quantity_per_unit <= 0:
            raise PlanningValidationError("BOM required_quantity_per_unit must be positive.")
        if bom_item.loss_rate < 0:
            raise PlanningValidationError("BOM loss_rate must be non-negative.")
        material = material_by_id[bom_item.material_id]
        if bom_item.unit is not None and material.material_unit is not None:
            if bom_item.unit.strip().upper() != material.material_unit.strip().upper():
                raise PlanningValidationError("BOM unit is incompatible with material unit.")


def validate_changeover_rules(
    changeover_rules: list[ChangeoverRuleInput],
    line_by_id: dict[str, ProductionLineInput],
    product_by_id: dict[str, ProductInput],
) -> None:
    """
    Parameters:
        - changeover_rules: Product transition setup rules.
        - line_by_id: Valid line lookup.
        - product_by_id: Valid product lookup.

    Methodology:
        - Validate product and optional line references.
        - Enforce non-negative time and cost values because CP-SAT uses integer penalties.

    Output:
        - None. Raises PlanningValidationError for invalid transition rules.
    """
    for rule in changeover_rules:
        if rule.from_product_id not in product_by_id or rule.to_product_id not in product_by_id:
            raise PlanningValidationError("Changeover rule references unknown product.")
        if rule.line_id is not None and rule.line_id not in line_by_id:
            raise PlanningValidationError("Changeover rule references unknown line.")
        if rule.cleaning_time_hr is not None and rule.cleaning_time_hr < 0:
            raise PlanningValidationError("cleaning_time_hr must be non-negative.")
        if rule.stabilization_time_hr is not None and rule.stabilization_time_hr < 0:
            raise PlanningValidationError("stabilization_time_hr must be non-negative.")
        if rule.total_changeover_time_hr is not None and rule.total_changeover_time_hr < 0:
            raise PlanningValidationError("total_changeover_time_hr must be non-negative.")
        if rule.changeover_minutes is not None and rule.changeover_minutes < 0:
            raise PlanningValidationError("changeover_minutes must be non-negative.")
        if rule.changeover_cost is not None and rule.changeover_cost < 0:
            raise PlanningValidationError("changeover_cost must be non-negative.")
        if (
            rule.total_changeover_time_hr is not None
            and rule.cleaning_time_hr is not None
            and rule.stabilization_time_hr is not None
        ):
            detail_total = rule.cleaning_time_hr + rule.stabilization_time_hr
            if abs(rule.total_changeover_time_hr - detail_total) > CHANGEOVER_HOUR_TOLERANCE:
                raise PlanningValidationError("total_changeover_time_hr must match detail hours.")
        calculate_changeover_minutes(rule)


def validate_amount_optimization_config(
    amount_config: AmountOptimizationConfig,
) -> None:
    """
    Parameters:
        - amount_config: Configurable weights and threshold policy for amount optimization.

    Methodology:
        - Validate threshold policy requirements that Pydantic field types cannot express.
        - Keep amount policy validation separate from objective construction.

    Output:
        - None. Raises PlanningValidationError when amount policy is incomplete.
    """
    if amount_config.amount_scale_unit <= 0:
        raise PlanningValidationError("amount_scale_unit must be positive.")
    if (
        amount_config.high_amount_threshold_policy == "CUSTOM_THRESHOLD"
        and amount_config.custom_high_amount_threshold is None
    ):
        raise PlanningValidationError(
            "custom_high_amount_threshold is required for CUSTOM_THRESHOLD policy."
        )
