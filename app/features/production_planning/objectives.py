from __future__ import annotations

from typing import Any

from ortools.sat.python import cp_model

from app.features.production_planning.config import AmountOptimizationConfig, SolverConfig
from app.features.production_planning.cpsat_model_builder import CpsatModelBundle
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

    model.Minimize(
        100_000_000_000_000 * unscheduled_penalty
        + 100_000_000_000 * delayed_count
        + 10_000 * total_tardiness
        + 1_000 * max_tardiness
        + 10 * total_changeover_minutes
        + total_line_priority_penalty
        + makespan
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
        "DUE_DATE_MIN_DELAY_COUNT": apply_due_date_min_delay_count_objective,
        "DUE_DATE_MIN_TOTAL_TARDINESS": apply_due_date_min_total_tardiness_objective,
        "DUE_DATE_BALANCED": apply_due_date_balanced_objective,
        "COST_MIN_TOTAL_COST": apply_cost_min_total_cost_objective,
        "COST_MIN_CHANGEOVER_COST": apply_cost_min_changeover_cost_objective,
        "COST_BALANCED": apply_cost_balanced_objective,
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
    amount_reference_data: AmountReferenceData,
    amount_config: AmountOptimizationConfig,
) -> AmountObjectiveTerms:
    """
    Parameters:
        - model: CP-SAT model receiving aggregate metric variables.
        - bundle: Variable bundle from common model construction.
        - data: Normalized planning data.
        - amount_reference_data: Prepared amount and high-value order lookups.
        - amount_config: Configurable amount objective weights and scale policy.

    Methodology:
        - Build amount objective expressions from prepared reference data only.
        - Keep scheduled reward, amount-weighted tardiness, unscheduled amount, high-amount
          delay, changeover cost, makespan, and due-date safety as named terms.

    Output:
        - AmountObjectiveTerms containing linear expressions and aggregate variables.
    """
    return AmountObjectiveTerms(
        total_scheduled_amount=build_total_scheduled_amount_expr(bundle, amount_reference_data),
        amount_weighted_tardiness=build_amount_weighted_tardiness_expr(
            bundle,
            amount_reference_data,
        ),
        unscheduled_amount_penalty=build_unscheduled_amount_penalty_expr(
            bundle,
            amount_reference_data,
        ),
        high_amount_delay_penalty=build_delayed_high_amount_penalty_expr(
            model,
            bundle,
            amount_reference_data,
        ),
        total_changeover_cost=build_total_changeover_cost_expr(bundle),
        total_line_priority_penalty=build_total_line_priority_penalty_expr(bundle, data),
        makespan=build_makespan_var(model, bundle, data),
        total_tardiness=build_total_tardiness_var(model, bundle, data),
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
        - Preserve the old entry point by delegating to the cost-balanced variant.

    Output:
        - None. Model objective updated in place.
    """
    apply_cost_balanced_objective(model, bundle, data, config)
