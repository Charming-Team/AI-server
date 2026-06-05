from app.features.production_planning.schemas import (
    PlanComparisonSummary,
    PlanRankingItem,
    PlanResult,
)

SOLVED_STATUSES = {"OPTIMAL", "FEASIBLE"}


def compare_plans(plan_results: list[PlanResult]) -> PlanComparisonSummary:
    """
    Parameters:
        - plan_results: All plan variant results.

    Methodology:
        - Exclude infeasible or unresolved plans from solved ranking.
        - Rank solved plans by unscheduled count, estimated total cost, delayed order count,
          and total tardiness.
        - Prefer DUE_DATE_OPTIMAL unless AMOUNT_OPTIMAL is materially better with similar
          delivery reliability.

    Output:
        - PlanComparisonSummary with recommendation and ranking items.
    """
    solved_plans = [plan for plan in plan_results if plan.status in SOLVED_STATUSES]
    if not solved_plans:
        rankings = [
            PlanRankingItem(
                rank=index + 1,
                plan_variant_code=plan.plan_variant_code,
                plan_family=plan.plan_family,
                plan_variant_name=plan.plan_variant_name,
                status=plan.status,
                delayed_order_count=0,
                total_tardiness_minutes=0,
                total_late_penalty_amount=0,
                total_changeover_cost=0,
                estimated_total_cost=0,
                unscheduled_count=len(plan.unscheduled_orders),
                recommendation_note="No feasible solution found.",
            )
            for index, plan in enumerate(plan_results)
        ]
        return PlanComparisonSummary(
            recommended_plan_variant_code=plan_results[0].plan_variant_code,
            recommendation_reason="All plans are infeasible. No valid recommendation possible.",
            plan_rankings=rankings,
        )

    ranked = sorted(
        solved_plans,
        key=lambda plan: (
            plan.metrics.unscheduled_count,
            plan.metrics.estimated_total_cost,
            plan.metrics.delayed_order_count,
            plan.metrics.total_tardiness_minutes,
        ),
    )
    recommended_code, reason = _select_recommendation(plan_results)
    return PlanComparisonSummary(
        recommended_plan_variant_code=recommended_code,
        recommendation_reason=reason,
        plan_rankings=_build_rankings(ranked, plan_results),
    )


def _select_recommendation(plan_results: list[PlanResult]) -> tuple[str, str]:
    by_code = {plan.plan_variant_code: plan for plan in plan_results}
    due_plan = by_code.get("DUE_DATE_OPTIMAL")
    amount_plan = by_code.get("AMOUNT_OPTIMAL")

    if due_plan and due_plan.status in SOLVED_STATUSES:
        if amount_plan and amount_plan.status in SOLVED_STATUSES:
            due_delayed = due_plan.metrics.delayed_order_count
            amount_delayed = amount_plan.metrics.delayed_order_count
            due_cost = due_plan.metrics.estimated_total_cost
            amount_cost = amount_plan.metrics.estimated_total_cost
            if due_delayed <= amount_delayed + 1 and (
                amount_cost == 0 or due_cost <= amount_cost * 1.15
            ):
                return (
                    "DUE_DATE_OPTIMAL",
                    "Due-date plan has similar delay count and cost within 15%.",
                )
            return (
                "AMOUNT_OPTIMAL",
                "Amount plan has lower estimated cost or significantly fewer delays.",
            )
        return (
            "DUE_DATE_OPTIMAL",
            "DUE_DATE_OPTIMAL is the default recommendation when amount plan is unavailable.",
        )

    solved_plans = [plan for plan in plan_results if plan.status in SOLVED_STATUSES]
    if not solved_plans:
        return plan_results[0].plan_variant_code, "No feasible plans available."
    best = min(
        solved_plans,
        key=lambda plan: (
            plan.metrics.unscheduled_count,
            plan.metrics.estimated_total_cost,
            plan.metrics.delayed_order_count,
        ),
    )
    return best.plan_variant_code, f"{best.plan_variant_code} is the best available solved plan."


def _build_rankings(
    ranked_solved: list[PlanResult],
    all_plans: list[PlanResult],
) -> list[PlanRankingItem]:
    solved_codes = {plan.plan_variant_code for plan in ranked_solved}
    items: list[PlanRankingItem] = []
    rank = 1
    for plan in ranked_solved:
        items.append(_ranking_item(plan, rank, _rank_note(rank)))
        rank += 1
    for plan in all_plans:
        if plan.plan_variant_code not in solved_codes:
            items.append(_ranking_item(plan, rank, "Infeasible — excluded from ranking."))
            rank += 1
    return items


def _ranking_item(
    plan: PlanResult,
    rank: int,
    recommendation_note: str,
) -> PlanRankingItem:
    return PlanRankingItem(
        rank=rank,
        plan_variant_code=plan.plan_variant_code,
        plan_family=plan.plan_family,
        plan_variant_name=plan.plan_variant_name,
        status=plan.status,
        delayed_order_count=plan.metrics.delayed_order_count,
        total_tardiness_minutes=plan.metrics.total_tardiness_minutes,
        total_late_penalty_amount=plan.metrics.total_late_penalty_amount,
        total_changeover_cost=plan.metrics.total_changeover_cost,
        estimated_total_cost=plan.metrics.estimated_total_cost,
        unscheduled_count=plan.metrics.unscheduled_count,
        recommendation_note=recommendation_note,
    )


def _rank_note(rank: int) -> str:
    if rank == 1:
        return "Best overall: lowest unscheduled count and estimated cost."
    if rank == 2:
        return "Runner-up by cost and delay metrics."
    return f"Rank {rank} by combined cost and delay score."
