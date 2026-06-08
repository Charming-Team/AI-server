from __future__ import annotations

from typing import Any

from ortools.sat.python import cp_model

from app.features.production_planning.config import AmountOptimizationConfig, SolverConfig
from app.features.production_planning.cpsat_model_builder import CpsatModelBundle
from app.features.production_planning.preprocessing import (
    MATERIAL_QUANTITY_SCALE,
    calculate_required_material_quantity_scaled,
    get_inbound_material_quantity_scaled,
    get_inbound_material_time,
    get_initial_available_material_quantity_scaled,
    normalize_order_amount,
    to_minute_offset,
)
from app.features.production_planning.schemas import (
    AmountObjectiveTerms,
    AmountReferenceData,
    NormalizedPlanningData,
)


def apply_due_date_min_delay_count_objective(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> None:
    """
    Parameters:
        - model: CP-SAT model with hard constraints already built.
        - bundle: Variable bundle produced by the base model builder.
        - data: Normalized planning data.
        - config: Solver configuration.

    Methodology:
        - Minimize delayed order count as the primary business goal.
        - Lexicographic hierarchy is unscheduled, delayed count, tardiness, max tardiness,
          changeover minutes, line priority, and makespan.

    Output:
        - None. Model objective updated in place.
    """
    total_tardiness = build_total_tardiness_var(model, bundle, data)
    max_tardiness = build_max_tardiness_var(model, bundle, data)
    delayed_count = build_delayed_order_count_var(model, bundle, data)
    makespan = build_makespan_var(model, bundle, data)
    total_changeover_minutes = build_total_changeover_minutes_expr(bundle)
    unscheduled_penalty = build_unscheduled_penalty_expr(bundle, config)
    total_line_priority_penalty = build_total_line_priority_penalty_expr(bundle, data)
    completion_priority = build_due_date_completion_priority_expr(bundle, data)
    total_material_shortage = build_total_material_shortage_expr(model, bundle, data)
    weights = config.due_date_optimization

    model.Minimize(
        weights.unscheduled_order_weight * unscheduled_penalty
        + weights.delayed_order_count_weight * delayed_count
        + weights.total_tardiness_weight * total_tardiness
        + weights.max_tardiness_weight * max_tardiness
        + weights.material_shortage_weight * total_material_shortage
        + weights.due_date_completion_priority_weight * completion_priority
        + weights.total_changeover_minutes_weight * total_changeover_minutes
        + weights.line_priority_weight * total_line_priority_penalty
        + weights.makespan_weight * makespan
    )


def apply_due_date_min_total_tardiness_objective(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> None:
    """
    Parameters:
        - model: CP-SAT model with hard constraints already built.
        - bundle: Variable bundle produced by the base model builder.
        - data: Normalized planning data.
        - config: Solver configuration.

    Methodology:
        - Minimize total tardiness minutes as the primary goal.
        - Delayed order count remains a secondary tie-breaker after total tardiness.

    Output:
        - None. Model objective updated in place.
    """
    total_tardiness = build_total_tardiness_var(model, bundle, data)
    max_tardiness = build_max_tardiness_var(model, bundle, data)
    delayed_count = build_delayed_order_count_var(model, bundle, data)
    makespan = build_makespan_var(model, bundle, data)
    total_changeover_minutes = build_total_changeover_minutes_expr(bundle)
    unscheduled_penalty = build_unscheduled_penalty_expr(bundle, config)
    total_line_priority_penalty = build_total_line_priority_penalty_expr(bundle, data)

    model.Minimize(
        100_000_000_000_000 * unscheduled_penalty
        + 100_000_000 * total_tardiness
        + 1_000_000 * delayed_count
        + 1_000 * max_tardiness
        + 10 * total_changeover_minutes
        + total_line_priority_penalty
        + makespan
    )


def apply_due_date_balanced_objective(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> None:
    """
    Parameters:
        - model: CP-SAT model with hard constraints already built.
        - bundle: Variable bundle produced by the base model builder.
        - data: Normalized planning data.
        - config: Solver configuration.

    Methodology:
        - Balance delayed count, total tardiness, and max tardiness into a composite delay-risk
          term. Changeover minimization is secondary.

    Output:
        - None. Model objective updated in place.
    """
    total_tardiness = build_total_tardiness_var(model, bundle, data)
    max_tardiness = build_max_tardiness_var(model, bundle, data)
    delayed_count = build_delayed_order_count_var(model, bundle, data)
    makespan = build_makespan_var(model, bundle, data)
    total_changeover_minutes = build_total_changeover_minutes_expr(bundle)
    unscheduled_penalty = build_unscheduled_penalty_expr(bundle, config)
    total_line_priority_penalty = build_total_line_priority_penalty_expr(bundle, data)
    weighted_delay_risk = (
        10_000_000 * delayed_count
        + 10_000 * total_tardiness
        + 5_000 * max_tardiness
    )

    model.Minimize(
        100_000_000_000_000 * unscheduled_penalty
        + weighted_delay_risk
        + 100 * total_changeover_minutes
        + total_line_priority_penalty
        + makespan
    )


def apply_cost_min_total_cost_objective(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> None:
    """
    Parameters:
        - model: CP-SAT model with hard constraints already built.
        - bundle: Variable bundle produced by the base model builder.
        - data: Normalized planning data.
        - config: Solver configuration.

    Methodology:
        - Minimize total monetary cost from late penalties and changeover costs.
        - Contract amount is not used as a reward.

    Output:
        - None. Model objective updated in place.
    """
    total_tardiness = build_total_tardiness_var(model, bundle, data)
    delayed_count = build_delayed_order_count_var(model, bundle, data)
    makespan = build_makespan_var(model, bundle, data)
    unscheduled_penalty = build_unscheduled_penalty_expr(bundle, config)
    total_line_priority_penalty = build_total_line_priority_penalty_expr(bundle, data)
    total_late_penalty = build_total_late_penalty_amount_expr(model, bundle, data)
    total_changeover_cost = build_total_changeover_cost_expr(bundle)
    total_monetary_cost = total_late_penalty + total_changeover_cost

    model.Minimize(
        100_000_000_000_000 * unscheduled_penalty
        + 1_000 * total_monetary_cost
        + 100_000 * delayed_count
        + 100 * total_tardiness
        + total_line_priority_penalty
        + makespan
    )


def apply_cost_min_changeover_cost_objective(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> None:
    """
    Parameters:
        - model: CP-SAT model with hard constraints already built.
        - bundle: Variable bundle produced by the base model builder.
        - data: Normalized planning data.
        - config: Solver configuration.

    Methodology:
        - Prioritize changeover cost minimization above late penalty cost.
        - This encourages line clustering to reduce setup transitions.

    Output:
        - None. Model objective updated in place.
    """
    total_tardiness = build_total_tardiness_var(model, bundle, data)
    delayed_count = build_delayed_order_count_var(model, bundle, data)
    makespan = build_makespan_var(model, bundle, data)
    unscheduled_penalty = build_unscheduled_penalty_expr(bundle, config)
    total_line_priority_penalty = build_total_line_priority_penalty_expr(bundle, data)
    total_late_penalty = build_total_late_penalty_amount_expr(model, bundle, data)
    total_changeover_cost = build_total_changeover_cost_expr(bundle)

    model.Minimize(
        100_000_000_000_000 * unscheduled_penalty
        + 10_000 * total_changeover_cost
        + 1_000 * total_late_penalty
        + 100_000 * delayed_count
        + 100 * total_tardiness
        + total_line_priority_penalty
        + makespan
    )


def apply_cost_balanced_objective(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> None:
    """
    Parameters:
        - model: CP-SAT model with hard constraints already built.
        - bundle: Variable bundle produced by the base model builder.
        - data: Normalized planning data.
        - config: Solver configuration.

    Methodology:
        - Balance late penalty, changeover cost, delayed count, and total tardiness into one
          composite cost-risk term.

    Output:
        - None. Model objective updated in place.
    """
    total_tardiness = build_total_tardiness_var(model, bundle, data)
    delayed_count = build_delayed_order_count_var(model, bundle, data)
    makespan = build_makespan_var(model, bundle, data)
    unscheduled_penalty = build_unscheduled_penalty_expr(bundle, config)
    total_line_priority_penalty = build_total_line_priority_penalty_expr(bundle, data)
    total_late_penalty = build_total_late_penalty_amount_expr(model, bundle, data)
    total_changeover_cost = build_total_changeover_cost_expr(bundle)
    balanced_cost_risk = (
        1_000 * total_late_penalty
        + 1_000 * total_changeover_cost
        + 100_000 * delayed_count
        + 100 * total_tardiness
    )

    model.Minimize(
        100_000_000_000_000 * unscheduled_penalty
        + balanced_cost_risk
        + total_line_priority_penalty
        + makespan
    )


def apply_objective_by_variant(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    config: SolverConfig,
    variant_code: str,
) -> None:
    """
    Parameters:
        - model: CP-SAT model with hard constraints already built.
        - bundle: Variable bundle from build_base_cpsat_model.
        - data: Normalized planning data.
        - config: Solver configuration.
        - variant_code: One of the supported production planning variant codes.

    Methodology:
        - Dispatch to the matching objective function based on variant_code.

    Output:
        - None. Delegates to the chosen objective function.
    """
    dispatch = {
        "DUE_DATE_OPTIMAL": apply_due_date_objective,
        "AMOUNT_OPTIMAL": apply_amount_objective,
    }
    dispatch[variant_code](model, bundle, data, config)


def build_total_tardiness_var(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> cp_model.IntVar:
    """
    Parameters:
        - model: CP-SAT model receiving the aggregate variable.
        - bundle: Variable bundle containing per-order tardiness.
        - data: Normalized planning data used for variable bounds.

    Methodology:
        - Sum all per-order tardiness variables into a bounded aggregate metric.

    Output:
        - IntVar representing total tardiness minutes.
    """
    if "total_tardiness" not in bundle.metric_vars:
        total = model.NewIntVar(0, data.horizon_minutes * len(data.orders), "total_tardiness")
        model.Add(total == sum(bundle.tardiness_vars.values()))
        bundle.metric_vars["total_tardiness"] = total
    return bundle.metric_vars["total_tardiness"]


def build_max_tardiness_var(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> cp_model.IntVar:
    """
    Parameters:
        - model: CP-SAT model receiving the max variable.
        - bundle: Variable bundle containing per-order tardiness.
        - data: Normalized planning data used for bounds.

    Methodology:
        - Use AddMaxEquality so maximum tardiness remains a true CP-SAT metric variable.

    Output:
        - IntVar representing maximum tardiness minutes.
    """
    if "max_tardiness" not in bundle.metric_vars:
        max_tardiness = model.NewIntVar(0, data.horizon_minutes, "max_tardiness")
        model.AddMaxEquality(max_tardiness, list(bundle.tardiness_vars.values()))
        bundle.metric_vars["max_tardiness"] = max_tardiness
    return bundle.metric_vars["max_tardiness"]


def build_delayed_order_count_var(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> cp_model.IntVar:
    """
    Parameters:
        - model: CP-SAT model receiving delayed indicator variables.
        - bundle: Variable bundle containing per-order tardiness.
        - data: Normalized planning data used for count bounds.

    Methodology:
        - Reify each tardiness variable into a delayed boolean and sum the indicators.

    Output:
        - IntVar representing delayed scheduled order count.
    """
    delayed_vars = _get_or_build_delayed_vars(model, bundle)
    if "delayed_order_count" not in bundle.metric_vars:
        delayed_count = model.NewIntVar(0, len(data.orders), "delayed_order_count")
        model.Add(delayed_count == sum(delayed_vars.values()))
        bundle.metric_vars["delayed_order_count"] = delayed_count
    return bundle.metric_vars["delayed_order_count"]


def build_makespan_var(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> cp_model.IntVar:
    """
    Parameters:
        - model: CP-SAT model receiving the makespan variable.
        - bundle: Variable bundle containing per-order completion variables.
        - data: Normalized planning data used for bounds.

    Methodology:
        - Apply AddMaxEquality over order completion times; unscheduled orders have completion 0.

    Output:
        - IntVar representing makespan minutes from planning start.
    """
    if "makespan" not in bundle.metric_vars:
        makespan = model.NewIntVar(0, data.horizon_minutes, "makespan")
        completion_vars = [order_vars["completion"] for order_vars in bundle.order_vars.values()]
        model.AddMaxEquality(makespan, completion_vars)
        bundle.metric_vars["makespan"] = makespan
    return bundle.metric_vars["makespan"]


def build_total_changeover_minutes_expr(bundle: CpsatModelBundle) -> Any:
    """
    Parameters:
        - bundle: Variable bundle containing sequence booleans and changeover minutes.

    Methodology:
        - Sum selected pairwise transition minutes using the corresponding before variables.

    Output:
        - Linear CP-SAT expression for total changeover minutes.
    """
    return sum(
        item["minutes"] * item["before_var"] for item in bundle.changeover_vars["sequence"]
    )


def build_due_date_completion_priority_expr(
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> Any:
    """
    Parameters:
        - bundle: Variable bundle containing per-order completion variables.
        - data: Normalized planning data containing due-date offsets.

    Methodology:
        - Sort orders by due date and give earlier due orders larger completion weights.
        - This breaks ties among otherwise on-time schedules so earlier due orders are planned
          earlier when hard constraints allow it.

    Output:
        - Linear CP-SAT expression for due-date-aware completion priority.
    """
    ordered_order_ids = sorted(
        data.orders,
        key=lambda order_id: (data.orders[order_id].due_date_offset, order_id),
    )
    order_count = len(ordered_order_ids)
    return sum(
        (order_count - index) * bundle.order_vars[order_id]["completion"]
        for index, order_id in enumerate(ordered_order_ids)
    )


def build_unscheduled_penalty_expr(
    bundle: CpsatModelBundle,
    config: SolverConfig,
) -> Any:
    """
    Parameters:
        - bundle: Variable bundle containing unscheduled booleans.
        - config: Solver configuration retained for interface consistency.

    Methodology:
        - Sum every unscheduled indicator as the penalty basis.
        - The actual weight is loaded from DueDateOptimizationConfig.

    Output:
        - Linear CP-SAT expression for unscheduled penalty.
    """
    return sum(bundle.unscheduled_vars.values())


def load_amount_optimization_config(config: SolverConfig) -> AmountOptimizationConfig:
    """
    Parameters:
        - config: Solver configuration supplied by the planning request.

    Methodology:
        - Return the nested amount optimization configuration so objective construction does
          not depend on hard-coded weights.

    Output:
        - AmountOptimizationConfig for amount objective terms.
    """
    return config.amount_optimization


def build_amount_objective_terms(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    amount_config: AmountOptimizationConfig,
) -> AmountObjectiveTerms:
    """
    Parameters:
        - model: CP-SAT model receiving aggregate metric variables.
        - bundle: Variable bundle from common model construction.
        - data: Normalized planning data.
        - amount_config: Configurable amount objective weights and scale policy.

    Methodology:
        - Build terms using the same names as the AMOUNT_OPTIMAL business objective.
        - Contract amount terms use amount_scale_unit so CP-SAT coefficients stay compact.
        - Cleaning cost is represented by selected transition minutes because the current DB
          input exposes cleaning/setup time, not a separate cleaning money amount.

    Output:
        - AmountObjectiveTerms containing linear expressions and aggregate variables.
    """
    return AmountObjectiveTerms(
        unscheduled_contract_amount=build_unscheduled_contract_amount_expr(
            bundle,
            data,
            amount_config,
        ),
        late_penalty_amount=build_total_late_penalty_amount_expr(model, bundle, data),
        total_cleaning_cost=build_total_cleaning_cost_expr(model, bundle, data),
        line_change_cost=build_line_change_cost_expr(model, bundle, data),
        different_product_sequence_count=build_different_product_sequence_count_expr(
            model,
            bundle,
            data,
        ),
        total_tardiness_minutes=build_total_tardiness_var(model, bundle, data),
        makespan_minutes=build_makespan_var(model, bundle, data),
    )


def build_unscheduled_contract_amount_expr(
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    amount_config: AmountOptimizationConfig,
) -> Any:
    """
    Parameters:
        - bundle: Variable bundle containing unscheduled order booleans.
        - data: Normalized planning data containing order contract amounts.
        - amount_config: Amount optimization config containing amount_scale_unit.

    Methodology:
        - Penalize the normalized contract amount of every unscheduled order.
        - Minimizing this term maximizes protected contract amount when not all orders fit.

    Output:
        - Linear CP-SAT expression for unscheduled contract amount units.
    """
    return sum(
        _normalized_contract_amount(data, amount_config, order_id) * unscheduled_var
        for order_id, unscheduled_var in bundle.unscheduled_vars.items()
    )


def build_total_cleaning_cost_expr(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> Any:
    """
    Parameters:
        - model: CP-SAT model receiving adjacent transition variables.
        - bundle: Variable bundle containing selected pairwise transition minutes.
        - data: Normalized planning data containing order and product IDs.

    Methodology:
        - Use adjacent transition minutes as the current cleaning/setup cost surrogate.
        - The DB view provides cleaning/setup hours and total changeover time, not a separate
          cleaning money column, so the objective weight converts minutes into business cost.

    Output:
        - Linear CP-SAT expression for total selected cleaning/setup minutes.
    """
    return sum(
        item["minutes"] * item["adjacent_var"]
        for item in _get_or_build_adjacent_transition_vars(model, bundle, data)
    )


def build_line_change_cost_expr(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> Any:
    """
    Parameters:
        - model: CP-SAT model receiving adjacent transition variables.
        - bundle: Variable bundle containing selected pairwise transition costs.
        - data: Normalized planning data containing order and product IDs.

    Methodology:
        - Sum adjacent changeover_cost values from product transition rules.
        - This represents the monetary cost of changing products on a line.

    Output:
        - Linear CP-SAT expression for total line change cost.
    """
    return sum(
        item["cost"] * item["adjacent_var"]
        for item in _get_or_build_adjacent_transition_vars(model, bundle, data)
    )


def build_different_product_sequence_count_expr(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> Any:
    """
    Parameters:
        - model: CP-SAT model receiving adjacent transition variables.
        - bundle: Variable bundle containing pairwise sequence booleans.
        - data: Normalized planning data containing product_id by order.

    Methodology:
        - Count adjacent order-to-order transitions where product_id changes on the same line.
        - Same-product consecutive production has zero count, encouraging product clustering.

    Output:
        - Linear CP-SAT expression for the number of selected different-product transitions.
    """
    return sum(
        item["adjacent_var"]
        for item in _get_or_build_adjacent_transition_vars(model, bundle, data)
        if _sequence_changes_product(item, data)
    )


def build_total_scheduled_amount_expr(
    bundle: CpsatModelBundle,
    amount_reference_data: AmountReferenceData,
) -> Any:
    """
    Parameters:
        - bundle: Variable bundle containing scheduled booleans.
        - amount_reference_data: Normalized amount lookup by order ID.

    Methodology:
        - Sum normalized amount for orders selected into the schedule.

    Output:
        - Linear CP-SAT expression for scheduled amount reward.
    """
    return sum(
        amount_reference_data.normalized_amount_by_order_id[order_id]
        * bundle.order_vars[order_id]["scheduled"]
        for order_id in bundle.order_vars
    )


def build_amount_weighted_tardiness_expr(
    bundle: CpsatModelBundle,
    amount_reference_data: AmountReferenceData,
) -> Any:
    """
    Parameters:
        - bundle: Variable bundle containing tardiness variables.
        - amount_reference_data: Normalized amount and priority lookups.

    Methodology:
        - Weight each tardiness minute by normalized amount and priority.

    Output:
        - Linear CP-SAT expression for amount-weighted tardiness.
    """
    return sum(
        amount_reference_data.normalized_amount_by_order_id[order_id]
        * amount_reference_data.priority_by_order_id[order_id]
        * bundle.tardiness_vars[order_id]
        for order_id in bundle.tardiness_vars
    )


def build_unscheduled_amount_penalty_expr(
    bundle: CpsatModelBundle,
    amount_reference_data: AmountReferenceData,
) -> Any:
    """
    Parameters:
        - bundle: Variable bundle containing unscheduled booleans.
        - amount_reference_data: Normalized amount lookup by order ID.

    Methodology:
        - Penalize unscheduled high-value work in proportion to normalized order amount.

    Output:
        - Linear CP-SAT expression for unscheduled amount penalty.
    """
    return sum(
        amount_reference_data.normalized_amount_by_order_id[order_id] * unscheduled_var
        for order_id, unscheduled_var in bundle.unscheduled_vars.items()
    )


def build_delayed_high_amount_penalty_expr(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    amount_reference_data: AmountReferenceData,
) -> Any:
    """
    Parameters:
        - model: CP-SAT model containing delayed indicator constraints.
        - bundle: Variable bundle containing tardiness variables.
        - amount_reference_data: High-amount order classification and normalized amount lookup.

    Methodology:
        - Reuse delayed booleans and apply extra penalty to high-amount orders delayed past due.

    Output:
        - Linear CP-SAT expression for high-amount delay penalty.
    """
    delayed_vars = _get_or_build_delayed_vars(model, bundle)
    return sum(
        amount_reference_data.normalized_amount_by_order_id[order_id] * delayed_vars[order_id]
        for order_id in amount_reference_data.high_amount_order_ids
    )


def build_total_changeover_cost_expr(bundle: CpsatModelBundle) -> Any:
    """
    Parameters:
        - bundle: Variable bundle containing sequence booleans and transition costs.

    Methodology:
        - Sum selected pairwise transition costs using the corresponding before variables.

    Output:
        - Linear CP-SAT expression for total changeover cost.
    """
    return sum(item["cost"] * item["before_var"] for item in bundle.changeover_vars["sequence"])


def build_total_material_shortage_expr(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> Any:
    """
    Parameters:
        - model: CP-SAT model receiving material shortage variables.
        - bundle: Variable bundle containing scheduled decision variables.
        - data: Normalized planning data with materials and BOM consumption rules.

    Methodology:
        - Recompute material requirements from BOM by product_id for scheduled orders.
        - Represent total shortage and pre-inbound shortage as non-negative slack variables.
        - This keeps material feasibility soft while still penalizing plans that consume
          unavailable stock or start before expected inbound can support production.

    Output:
        - Linear expression summing material shortage quantities in material units.
    """
    shortage_vars = []
    for material_id, material in data.materials.items():
        required_terms = []
        required_bound = 0
        for order_id, normalized_order in data.orders.items():
            product = data.products[normalized_order.order.product_id]
            quantity_required = _calculate_order_material_requirement_units(
                normalized_order.order,
                product,
                data,
                material_id,
            )
            if not quantity_required:
                continue
            required_terms.append(quantity_required * bundle.order_vars[order_id]["scheduled"])
            required_bound += quantity_required
        if not required_terms:
            continue

        available = (
            _scaled_quantity_to_units(get_initial_available_material_quantity_scaled(material))
            + _scaled_quantity_to_units(get_inbound_material_quantity_scaled(material))
        )
        shortage = model.NewIntVar(
            0,
            required_bound,
            f"soft_material_shortage_{material_id}",
        )
        model.Add(shortage >= sum(required_terms) - available)
        shortage_vars.append(shortage)

        before_inbound_terms = _build_before_inbound_material_terms(
            model,
            bundle,
            data,
            material_id,
        )
        if before_inbound_terms:
            initial_available = _scaled_quantity_to_units(
                get_initial_available_material_quantity_scaled(material)
            )
            before_inbound_shortage = model.NewIntVar(
                0,
                required_bound,
                f"soft_material_before_inbound_shortage_{material_id}",
            )
            model.Add(before_inbound_shortage >= sum(before_inbound_terms) - initial_available)
            shortage_vars.append(before_inbound_shortage)
    return sum(shortage_vars)


def _build_before_inbound_material_terms(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    material_id: str,
) -> list[Any]:
    material = data.materials[material_id]
    inbound_time = get_inbound_material_time(material)
    if inbound_time is None:
        return []
    inbound_offset = to_minute_offset(inbound_time, data.planning_start)
    if inbound_offset <= 0 or inbound_offset > data.horizon_minutes:
        return []

    terms = []
    for order_id, normalized_order in data.orders.items():
        product = data.products[normalized_order.order.product_id]
        quantity_required = _calculate_order_material_requirement_units(
            normalized_order.order,
            product,
            data,
            material_id,
        )
        if not quantity_required:
            continue
        for line_id, assigned in bundle.order_vars[order_id]["assigned"].items():
            starts_before = model.NewBoolVar(
                f"soft_material_{material_id}_{order_id}_{line_id}_before_inbound"
            )
            model.Add(starts_before <= assigned)
            start = bundle.order_vars[order_id]["start"][line_id]
            model.Add(start <= inbound_offset - 1).OnlyEnforceIf(starts_before)
            model.Add(start >= inbound_offset).OnlyEnforceIf(
                [assigned, starts_before.Not()]
            )
            terms.append(quantity_required * starts_before)
    return terms


def _calculate_order_material_requirement_units(order, product, data, material_id: str) -> int:
    required_scaled = sum(
        calculate_required_material_quantity_scaled(order, product, bom_item)
        for bom_item in data.bom_items_by_product_id.get(order.product_id, [])
        if bom_item.material_id == material_id
    )
    return _scaled_quantity_to_units(required_scaled)


def _scaled_quantity_to_units(quantity_scaled: int) -> int:
    return (quantity_scaled + MATERIAL_QUANTITY_SCALE - 1) // MATERIAL_QUANTITY_SCALE


def _normalized_contract_amount(
    data: NormalizedPlanningData,
    amount_config: AmountOptimizationConfig,
    order_id: str,
) -> int:
    return normalize_order_amount(
        data.orders[order_id].order.order_amount,
        amount_config.amount_scale_unit,
    )


def _sequence_changes_product(item: dict[str, Any], data: NormalizedPlanningData) -> bool:
    from_order = data.orders.get(item["from_order_id"])
    to_order = data.orders.get(item["to_order_id"])
    if from_order is None or to_order is None:
        return False
    return from_order.order.product_id != to_order.order.product_id


def _get_or_build_adjacent_transition_vars(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> list[dict[str, Any]]:
    if "adjacent_transitions" in bundle.metric_vars:
        return bundle.metric_vars["adjacent_transitions"]

    sequence_by_key = {
        (item["line_id"], item["from_order_id"], item["to_order_id"]): item
        for item in bundle.changeover_vars["sequence"]
    }
    adjacent_transitions = []
    for line_id, candidates in data.candidates_by_line_id.items():
        order_ids = sorted({candidate.order_id for candidate in candidates})
        if len(order_ids) < 2:
            continue
        adjacent_transitions.extend(
            _build_line_adjacent_transition_vars(
                model,
                bundle,
                line_id,
                order_ids,
                sequence_by_key,
            )
        )

    bundle.metric_vars["adjacent_transitions"] = adjacent_transitions
    return adjacent_transitions


def _build_line_adjacent_transition_vars(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    line_id: str,
    order_ids: list[str],
    sequence_by_key: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    line_used = model.NewBoolVar(f"line_used_for_adjacency_{line_id}")
    assigned_vars = [
        bundle.order_vars[order_id]["assigned"][line_id]
        for order_id in order_ids
        if line_id in bundle.order_vars[order_id]["assigned"]
    ]
    model.Add(line_used <= sum(assigned_vars))
    for assigned in assigned_vars:
        model.Add(line_used >= assigned)

    adjacent_transitions = []
    successors_by_order_id = {order_id: [] for order_id in order_ids}
    predecessors_by_order_id = {order_id: [] for order_id in order_ids}
    for from_order_id in order_ids:
        for to_order_id in order_ids:
            if from_order_id == to_order_id:
                continue
            sequence = sequence_by_key.get((line_id, from_order_id, to_order_id))
            if sequence is None:
                continue
            adjacent = model.NewBoolVar(
                f"adjacent_{from_order_id}_{to_order_id}_{line_id}"
            )
            model.Add(adjacent <= sequence["before_var"])
            _forbid_intermediate_orders_between_adjacent_pair(
                model,
                adjacent,
                line_id,
                from_order_id,
                to_order_id,
                order_ids,
                sequence_by_key,
            )
            transition = {**sequence, "adjacent_var": adjacent}
            adjacent_transitions.append(transition)
            successors_by_order_id[from_order_id].append(adjacent)
            predecessors_by_order_id[to_order_id].append(adjacent)

    first_vars = []
    last_vars = []
    for order_id in order_ids:
        assigned = bundle.order_vars[order_id]["assigned"][line_id]
        first = model.NewBoolVar(f"first_on_line_{order_id}_{line_id}")
        last = model.NewBoolVar(f"last_on_line_{order_id}_{line_id}")
        model.Add(sum(predecessors_by_order_id[order_id]) + first == assigned)
        model.Add(sum(successors_by_order_id[order_id]) + last == assigned)
        first_vars.append(first)
        last_vars.append(last)

    model.Add(sum(first_vars) == line_used)
    model.Add(sum(last_vars) == line_used)
    return adjacent_transitions


def _forbid_intermediate_orders_between_adjacent_pair(
    model: cp_model.CpModel,
    adjacent: cp_model.IntVar,
    line_id: str,
    from_order_id: str,
    to_order_id: str,
    order_ids: list[str],
    sequence_by_key: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    for intermediate_order_id in order_ids:
        if intermediate_order_id in {from_order_id, to_order_id}:
            continue
        before_intermediate = sequence_by_key.get(
            (line_id, from_order_id, intermediate_order_id)
        )
        intermediate_before_to = sequence_by_key.get(
            (line_id, intermediate_order_id, to_order_id)
        )
        if before_intermediate is None or intermediate_before_to is None:
            continue
        model.Add(
            adjacent
            + before_intermediate["before_var"]
            + intermediate_before_to["before_var"]
            <= 2
        )


def build_total_late_penalty_amount_expr(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> Any:
    """
    Parameters:
        - model: CP-SAT model used to build delayed indicator variables when needed.
        - bundle: Variable bundle with tardiness variables.
        - data: Normalized planning data with late_penalty_amount per order.

    Methodology:
        - Multiply each order's late_penalty_amount by its binary delayed indicator.
        - delayed_var is 1 when tardiness is greater than zero.

    Output:
        - Linear CP-SAT expression for total late penalty cost.
    """
    delayed_vars = _get_or_build_delayed_vars(model, bundle)
    return sum(
        data.orders[order_id].order.late_penalty_amount * delayed_vars[order_id]
        for order_id in bundle.tardiness_vars
    )


def build_total_line_priority_penalty_expr(
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> Any:
    """
    Parameters:
        - bundle: Variable bundle containing order-line assignment booleans.
        - data: Normalized planning data containing candidate line priority ranks.

    Methodology:
        - Penalize selected product-line candidates by priority rank.
        - Lower priority_rank means a preferred line, so minimizing this expression selects
          preferred lines only after higher-order business terms are satisfied.

    Output:
        - Linear CP-SAT expression for selected line priority penalty.
    """
    return sum(
        candidate.priority_rank
        * bundle.order_vars[candidate.order_id]["assigned"][candidate.line_id]
        for candidates in data.candidates_by_order_id.values()
        for candidate in candidates
    )


def _get_or_build_delayed_vars(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
) -> dict[str, cp_model.IntVar]:
    if "delayed_vars" in bundle.metric_vars:
        return bundle.metric_vars["delayed_vars"]

    delayed_vars = {}
    for order_id, tardiness in bundle.tardiness_vars.items():
        delayed = model.NewBoolVar(f"delayed_{order_id}")
        model.Add(tardiness >= 1).OnlyEnforceIf(delayed)
        model.Add(tardiness == 0).OnlyEnforceIf(delayed.Not())
        delayed_vars[order_id] = delayed
    bundle.metric_vars["delayed_vars"] = delayed_vars
    return delayed_vars


def apply_due_date_objective(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> None:
    """
    Parameters:
        - model: CP-SAT model with hard constraints already built.
        - bundle: Variable bundle from build_base_cpsat_model.
        - data: Normalized planning data.
        - config: Solver configuration.

    Methodology:
        - Preserve the old entry point by delegating to the min-delay-count due-date variant.

    Output:
        - None. Model objective updated in place.
    """
    apply_due_date_min_delay_count_objective(model, bundle, data, config)


def apply_amount_objective(
    model: cp_model.CpModel,
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> None:
    """
    Parameters:
        - model: CP-SAT model with hard constraints already built.
        - bundle: Variable bundle from build_base_cpsat_model.
        - data: Normalized planning data.
        - config: Solver configuration.

    Methodology:
        - Maximize protected contract amount by minimizing unscheduled contract amount.
        - Minimize late penalties, cleaning/setup time, line change cost, and product changes.
        - Keep due dates soft through total tardiness penalties.

    Output:
        - None. Model objective updated in place.
    """
    amount_config = load_amount_optimization_config(config)
    terms = build_amount_objective_terms(
        model,
        bundle,
        data,
        amount_config,
    )

    model.Minimize(
        amount_config.unscheduled_contract_amount_weight
        * terms.unscheduled_contract_amount
        + amount_config.late_penalty_amount_weight * terms.late_penalty_amount
        + amount_config.total_cleaning_cost_weight * terms.total_cleaning_cost
        + amount_config.line_change_cost_weight * terms.line_change_cost
        + amount_config.different_product_sequence_count_weight
        * terms.different_product_sequence_count
        + amount_config.total_tardiness_minutes_weight * terms.total_tardiness_minutes
        + amount_config.makespan_minutes_weight * terms.makespan_minutes
    )
