from __future__ import annotations

from app.features.production_planning.schemas import (
    FinalPlanRecommendation,
    PlanResult,
    PlanSimulationResult,
    SimulationComparisonSummary,
    SimulationRankingItem,
)

DUE_DATE_CODES = {
    "DUE_DATE_OPTIMAL",
}
AMOUNT_CODES = {
    "AMOUNT_OPTIMAL",
}


def compare_simulation_results(
    simulation_results: list[PlanSimulationResult],
) -> SimulationComparisonSummary:
    """
    Parameters:
        - simulation_results: Simulation metrics for all CP-SAT candidates.

    Methodology:
        - Split results by due-date and amount families.
        - Rank due-date plans by simulated delivery reliability.
        - Rank amount plans by simulated total risk cost and tail penalty.

    Output:
        - SimulationComparisonSummary containing both family recommendations and rankings.
    """
    due_date_ranked = sorted(
        _filter_results(simulation_results, DUE_DATE_CODES),
        key=_due_date_rank_key,
    )
    cost_ranked = sorted(
        _filter_results(simulation_results, AMOUNT_CODES),
        key=_cost_rank_key,
    )
    return SimulationComparisonSummary(
        due_date_recommended_plan_variant_code=due_date_ranked[0].plan_variant_code,
        cost_recommended_plan_variant_code=cost_ranked[0].plan_variant_code,
        due_date_rankings=[
            _ranking_item(result, index + 1, "Due-date simulation ranking.")
            for index, result in enumerate(due_date_ranked)
        ],
        cost_rankings=[
            _ranking_item(result, index + 1, "Cost simulation ranking.")
            for index, result in enumerate(cost_ranked)
        ],
    )


def recommend_final_plans(
    plan_results: list[PlanResult],
    simulation_results: list[PlanSimulationResult],
    comparison_summary: SimulationComparisonSummary,
) -> tuple[FinalPlanRecommendation, FinalPlanRecommendation]:
    """
    Parameters:
        - plan_results: CP-SAT plan candidates with schedule items.
        - simulation_results: Simulation metrics for each candidate.
        - comparison_summary: Family-specific simulation rankings.

    Methodology:
        - Join the selected simulation result back to its source schedule.
        - Return exactly one due-date recommendation and one amount recommendation.

    Output:
        - Tuple of due-date and amount FinalPlanRecommendation objects.
    """
    plan_by_code = {plan.plan_variant_code: plan for plan in plan_results}
    simulation_by_code = {result.plan_variant_code: result for result in simulation_results}
    due_code = comparison_summary.due_date_recommended_plan_variant_code
    cost_code = comparison_summary.cost_recommended_plan_variant_code
    return (
        _recommendation(
            plan_by_code[due_code],
            simulation_by_code[due_code],
            "Lowest simulated delivery risk among due-date candidates.",
        ),
        _recommendation(
            plan_by_code[cost_code],
            simulation_by_code[cost_code],
            "Lowest simulated risk among amount candidates.",
        ),
    )


def _filter_results(
    simulation_results: list[PlanSimulationResult],
    allowed_codes: set[str],
) -> list[PlanSimulationResult]:
    filtered = [
        result
        for result in simulation_results
        if result.plan_variant_code in allowed_codes
    ]
    if not filtered:
        raise ValueError("Simulation recommendations require at least one candidate per family.")
    return filtered


def _due_date_rank_key(result: PlanSimulationResult) -> tuple[float, ...]:
    return (
        result.delay_probability,
        result.expected_delayed_order_count,
        result.p95_tardiness_minutes,
        result.expected_tardiness_minutes,
        result.material_shortage_probability,
        result.expected_late_penalty_amount,
        result.expected_total_risk_cost,
    )


def _cost_rank_key(result: PlanSimulationResult) -> tuple[float, ...]:
    return (
        result.expected_total_risk_cost,
        result.p95_late_penalty_amount,
        result.expected_late_penalty_amount,
        result.expected_changeover_cost,
        result.delay_probability,
        result.material_shortage_probability,
        result.p95_tardiness_minutes,
    )


def _ranking_item(
    result: PlanSimulationResult,
    rank: int,
    note: str,
) -> SimulationRankingItem:
    return SimulationRankingItem(
        rank=rank,
        plan_variant_code=result.plan_variant_code,
        plan_family=result.plan_family,
        delay_probability=result.delay_probability,
        expected_delayed_order_count=result.expected_delayed_order_count,
        expected_total_risk_cost=result.expected_total_risk_cost,
        material_shortage_probability=result.material_shortage_probability,
        recommendation_note=note,
    )


def _recommendation(
    plan: PlanResult,
    simulation_result: PlanSimulationResult,
    reason: str,
) -> FinalPlanRecommendation:
    return FinalPlanRecommendation(
        plan_variant_code=plan.plan_variant_code,
        plan_family=plan.plan_family,
        recommendation_reason=reason,
        simulation_metrics=simulation_result,
        schedule_items=plan.schedule_items,
        risk_summary={
            "delay_probability": simulation_result.delay_probability,
            "expected_late_penalty_amount": (
                simulation_result.expected_late_penalty_amount
            ),
            "expected_changeover_cost": simulation_result.expected_changeover_cost,
            "expected_material_shortage_penalty_amount": (
                simulation_result.expected_material_shortage_penalty_amount
            ),
            "expected_total_risk_cost": simulation_result.expected_total_risk_cost,
            "material_shortage_probability": simulation_result.material_shortage_probability,
            "top_delay_causes": simulation_result.top_delay_causes,
        },
    )
