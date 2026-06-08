from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from secrets import choice
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.config import Settings, get_settings
from app.features.production_planning.config import SolverConfig
from app.features.production_planning.cpsat_model_builder import (
    CpsatModelBundle,
    build_base_cpsat_model,
)
from app.features.production_planning.exceptions import PlanningDataAccessError
from app.features.production_planning.json_formatter import build_dashboard_response
from app.features.production_planning.objectives import apply_objective_by_variant
from app.features.production_planning.plan_comparator import compare_plans
from app.features.production_planning.preprocessing import normalize_request, to_minute_offset
from app.features.production_planning.recommendation import (
    compare_simulation_results,
    recommend_final_plans,
)
from app.features.production_planning.repositories.simulation_data_repository import (
    SimulationDataRepository,
    SimulationInputBundle,
)
from app.features.production_planning.schemas import (
    PLAN_VARIANTS,
    AdjustedPlanCandidate,
    AdjustedPlanRow,
    FinalPlanRecommendation,
    NormalizedPlanningData,
    PlanComparisonSummary,
    PlanResult,
    PlanSimulationResult,
    ProductionPlanningRequest,
    ProductionPlanningResult,
    SimulationComparisonSummary,
)
from app.features.production_planning.simulation import (
    build_sampling_distributions,
    simulate_plan_candidates,
)
from app.features.production_planning.simulation.sampling import SamplingDistributions
from app.features.production_planning.solution_extractor import extract_plan_result
from app.features.production_planning.solver import solve_model
from app.features.production_planning.validators import validate_request

logger = logging.getLogger(__name__)

OPERATOR_ID_POOL = (6, 7, 8, 9, 10, 11, 13, 14, 15)


class ProductionPlanningGraphState(TypedDict, total=False):
    request: ProductionPlanningRequest
    validated_request: ProductionPlanningRequest
    normalized_data: NormalizedPlanningData
    plan_results: list[PlanResult]
    simulation_input: SimulationInputBundle | None
    sampling_distributions: SamplingDistributions
    simulation_results: list[PlanSimulationResult]
    comparison_summary: PlanComparisonSummary
    simulation_comparison_summary: SimulationComparisonSummary
    recommended_due_date_plan: FinalPlanRecommendation
    recommended_cost_plan: FinalPlanRecommendation
    result: ProductionPlanningResult
    dashboard_response: dict[str, Any]


def build_production_planning_graph():
    """
    Parameters:
        - None.

    Methodology:
        - Build a LangGraph StateGraph that validates, normalizes, generates due-date and
          amount optimal plans, compares them, and finalizes a single response.

    Output:
        - A compiled LangGraph workflow that accepts ProductionPlanningGraphState.
    """
    graph = StateGraph(ProductionPlanningGraphState)
    graph.add_node("validate_request", validate_request_node)
    graph.add_node("normalize_request", normalize_request_node)
    graph.add_node("generate_plan_variants", generate_plan_variants_node)
    graph.add_node("load_simulation_input_from_db", load_simulation_input_from_db_node)
    graph.add_node("build_sampling_distributions", build_sampling_distributions_node)
    graph.add_node("simulate_plan_candidates", simulate_plan_candidates_node)
    graph.add_node("compare_plan_variants", compare_plan_variants_node)
    graph.add_node("compare_simulation_results", compare_simulation_results_node)
    graph.add_node("recommend_final_plans", recommend_final_plans_node)
    graph.add_node("finalize_result", finalize_result_node)
    graph.add_node("format_dashboard_response", format_dashboard_response_node)

    graph.add_edge(START, "validate_request")
    graph.add_edge("validate_request", "normalize_request")
    graph.add_edge("normalize_request", "generate_plan_variants")
    graph.add_edge("generate_plan_variants", "load_simulation_input_from_db")
    graph.add_edge("load_simulation_input_from_db", "build_sampling_distributions")
    graph.add_edge("build_sampling_distributions", "simulate_plan_candidates")
    graph.add_edge("simulate_plan_candidates", "compare_plan_variants")
    graph.add_edge("compare_plan_variants", "compare_simulation_results")
    graph.add_edge("compare_simulation_results", "recommend_final_plans")
    graph.add_edge("recommend_final_plans", "finalize_result")
    graph.add_edge("finalize_result", "format_dashboard_response")
    graph.add_edge("format_dashboard_response", END)
    return graph.compile()


def run_production_planning_graph(
    request: ProductionPlanningRequest,
) -> ProductionPlanningResult:
    """
    Parameters:
        - request: Production planning request to execute through the LangGraph workflow.

    Methodology:
        - Invoke the compiled graph with the request as initial state.
        - Read the final ProductionPlanningResult from the terminal graph state.

    Output:
        - ProductionPlanningResult generated by connected LangGraph nodes.
    """
    configure_langsmith_tracing_from_settings(get_settings())
    state = build_production_planning_graph().invoke({"request": request})
    return state["result"]


def configure_langsmith_tracing_from_settings(settings: Settings) -> None:
    """
    Parameters:
        - settings: Application settings loaded from environment or .env.

    Methodology:
        - Copy optional LangSmith settings into os.environ before graph invocation.
        - Keep existing shell-provided environment values when they are already present.
        - Do not log or expose the LangSmith API key.

    Output:
        - None. Updates process environment for LangGraph/LangSmith auto tracing.
    """
    if not settings.langsmith_tracing:
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint


def validate_request_node(
    state: ProductionPlanningGraphState,
) -> dict[str, ProductionPlanningRequest]:
    """
    Parameters:
        - state: LangGraph state containing the original planning request.

    Methodology:
        - Run domain validation before any model construction.
        - Store the validated request for downstream nodes.

    Output:
        - Partial graph state with validated_request.
    """
    request = state["request"]
    logger.info("production_planning.validation.started")
    validated = validate_request(request)
    logger.info("production_planning.validation.completed", extra={"orders": len(validated.orders)})
    return {"validated_request": validated}


def normalize_request_node(
    state: ProductionPlanningGraphState,
) -> dict[str, NormalizedPlanningData]:
    """
    Parameters:
        - state: LangGraph state containing a validated request.

    Methodology:
        - Convert datetimes and business inputs into integer CP-SAT-ready data.
        - Share normalized data with all objective variants.

    Output:
        - Partial graph state with normalized_data.
    """
    validated = state["validated_request"]
    logger.info("production_planning.preprocessing.started")
    normalized = normalize_request(validated)
    logger.info(
        "production_planning.preprocessing.completed",
        extra={"orders": len(normalized.orders), "lines": len(normalized.lines)},
    )
    return {"normalized_data": normalized}


def generate_plan_variants_node(
    state: ProductionPlanningGraphState,
) -> dict[str, list[PlanResult]]:
    """
    Parameters:
        - state: LangGraph state with validated_request and normalized_data.

    Methodology:
        - For each plan variant, build a fresh CP-SAT model, apply the variant objective,
          solve, and extract results.

    Output:
        - Partial graph state with plan_results.
    """
    validated = state["validated_request"]
    normalized = state["normalized_data"]
    plan_results = []

    for variant_code in PLAN_VARIANTS:
        if variant_code == "DUE_DATE_OPTIMAL":
            plan_results.append(_generate_due_date_plan(normalized, validated))
            continue
        if variant_code == "AMOUNT_OPTIMAL":
            plan_results.append(_generate_amount_plan(normalized, validated))
            continue

        variant_config = _solver_config_for_variant(validated.solver_config, variant_code)
        bundle = _build_objective_bundle(normalized, variant_config, variant_code)
        apply_objective_by_variant(
            bundle.model,
            bundle,
            normalized,
            variant_config,
            variant_code,
        )
        plan_results.append(_solve_extract_plan(bundle, normalized, variant_config, variant_code))

    return {"plan_results": plan_results}


def compare_plan_variants_node(
    state: ProductionPlanningGraphState,
) -> dict[str, PlanComparisonSummary]:
    """
    Parameters:
        - state: LangGraph state containing all solved plan variants.

    Methodology:
        - Compare all plan variants with the ranking and recommendation policy.

    Output:
        - Partial graph state with comparison_summary.
    """
    return {"comparison_summary": compare_plans(state["plan_results"])}


def load_simulation_input_from_db_node(
    state: ProductionPlanningGraphState,
) -> dict[str, SimulationInputBundle | None]:
    """
    Parameters:
        - state: LangGraph state after CP-SAT plan candidate generation.

    Methodology:
        - Skip DB loading when simulation is disabled.
        - Load all ai_planning sampling views through SimulationDataRepository.
        - On sampling DB access failure, log a warning and allow downstream fallback
          distributions to keep the workflow usable.

    Output:
        - Partial graph state with simulation_input.
    """
    validated = state["validated_request"]
    if not validated.simulation_config.enabled:
        return {"simulation_input": None}

    logger.info("production_planning.simulation.db_load.started")
    try:
        bundle = SimulationDataRepository().load_simulation_input_bundle(
            validated.planning_start,
            validated.planning_end,
        )
        logger.info(
            "production_planning.simulation.db_load.completed",
            extra={
                "production_results": len(bundle.production_results),
                "production_result_causes": len(bundle.production_result_causes),
                "changeover_sequences": len(bundle.changeover_sequences),
            },
        )
        return {"simulation_input": bundle}
    except PlanningDataAccessError as exc:
        logger.warning(
            "production_planning.simulation.db_load.failed",
            extra={"error": str(exc)},
        )
        return {"simulation_input": None}


def build_sampling_distributions_node(
    state: ProductionPlanningGraphState,
) -> dict[str, SamplingDistributions]:
    """
    Parameters:
        - state: LangGraph state containing plan candidates and normalized planning data.

    Methodology:
        - Build simulation sampling distributions before DES execution.
        - Use deterministic fallback distributions when historical sampling views are not loaded
          through this in-memory request path.

    Output:
        - Partial graph state with sampling_distributions.
    """
    logger.info("production_planning.simulation.sampling.started")
    distributions = build_sampling_distributions(
        state["plan_results"],
        state["normalized_data"],
        state["validated_request"].simulation_config,
        state.get("simulation_input"),
    )
    logger.info(
        "production_planning.simulation.sampling.completed",
        extra={"warnings": len(distributions.warnings)},
    )
    return {"sampling_distributions": distributions}


def simulate_plan_candidates_node(
    state: ProductionPlanningGraphState,
) -> dict[str, list[PlanSimulationResult]]:
    """
    Parameters:
        - state: LangGraph state containing plan candidates, normalized data, and sampling data.

    Methodology:
        - Run Monte Carlo + DES for every CP-SAT candidate plan.
        - Preserve one simulation result per plan variant.

    Output:
        - Partial graph state with simulation_results.
    """
    validated = state["validated_request"]
    logger.info(
        "production_planning.simulation.started",
        extra={
            "plans": len(state["plan_results"]),
            "iterations": validated.simulation_config.num_iterations,
        },
    )
    results = simulate_plan_candidates(
        state["plan_results"],
        state["normalized_data"],
        validated.solver_config,
        validated.simulation_config,
        state["sampling_distributions"],
    )
    logger.info(
        "production_planning.simulation.completed",
        extra={"simulation_results": len(results)},
    )
    return {"simulation_results": results}


def compare_simulation_results_node(
    state: ProductionPlanningGraphState,
) -> dict[str, SimulationComparisonSummary]:
    """
    Parameters:
        - state: LangGraph state containing simulation results for all plan variants.

    Methodology:
        - Rank simulated due-date and amount families separately.
        - Produce the family-specific recommendation summary required by downstream response
          assembly.

    Output:
        - Partial graph state with simulation_comparison_summary.
    """
    return {
        "simulation_comparison_summary": compare_simulation_results(
            state["simulation_results"]
        )
    }


def recommend_final_plans_node(
    state: ProductionPlanningGraphState,
) -> dict[str, FinalPlanRecommendation]:
    """
    Parameters:
        - state: LangGraph state containing plan candidates and simulation comparison summary.

    Methodology:
        - Select exactly one simulated due-date recommendation and one simulated amount
          recommendation.

    Output:
        - Partial graph state with recommended_due_date_plan and recommended_cost_plan.
    """
    due_date_plan, cost_plan = recommend_final_plans(
        state["plan_results"],
        state["simulation_results"],
        state["simulation_comparison_summary"],
    )
    return {
        "recommended_due_date_plan": due_date_plan,
        "recommended_cost_plan": cost_plan,
    }


def finalize_result_node(
    state: ProductionPlanningGraphState,
) -> dict[str, ProductionPlanningResult]:
    """
    Parameters:
        - state: LangGraph state containing plan results and comparison summary.

    Methodology:
        - Assemble the final response model and collect warnings and solver metadata by variant.

    Output:
        - Partial graph state with result.
    """
    plan_results = state["plan_results"]
    comparison_summary = state["comparison_summary"]
    warnings = []
    for plan in plan_results:
        warnings.extend(plan.warnings)
    warnings.extend(state["sampling_distributions"].warnings)

    return {
        "result": ProductionPlanningResult(
            status="COMPLETED",
            adjusted_plan_candidates=_build_adjusted_plan_candidates(
                plan_results,
                state["simulation_results"],
            ),
            plan_results=plan_results,
            recommended_plan_variant_code=comparison_summary.recommended_plan_variant_code,
            comparison_summary=comparison_summary,
            simulation_results=state["simulation_results"],
            simulation_comparison_summary=state["simulation_comparison_summary"],
            recommended_due_date_plan=state["recommended_due_date_plan"],
            recommended_cost_plan=state["recommended_cost_plan"],
            warnings=warnings,
            solver_metadata={
                plan.plan_variant_code: plan.solver_metadata for plan in plan_results
            },
        )
    }


def format_dashboard_response_node(
    state: ProductionPlanningGraphState,
) -> dict[str, Any]:
    """
    Parameters:
        - state: LangGraph state containing the finalized internal planning result.

    Methodology:
        - Build the dashboard-facing JSON from the same graph state used for planning.
        - Use the DB current-plan snapshot as the sole dashboard baseline source.

    Output:
        - Partial graph state with dashboard_response and an updated ProductionPlanningResult.
    """
    validated = state["validated_request"]
    simulation_input = state.get("simulation_input")
    settings = get_settings()
    baseline_inputs, baseline_warnings = _load_dashboard_baseline_inputs(
        validated,
        simulation_input,
    )
    dashboard_response = build_dashboard_response(
        state["result"],
        baseline_plans=baseline_inputs["baseline_plans"],
        products=validated.products,
        production_lines=validated.production_lines,
        product_line_capabilities=validated.product_line_capabilities,
        planning_start=validated.planning_start,
        planning_end=validated.planning_end,
        enable_llm_evaluation=True,
        settings=settings,
    )
    dashboard_response["warnings"].extend(baseline_warnings)
    result = state["result"].model_copy(
        update={"dashboard_response": dashboard_response}
    )
    return {
        "dashboard_response": dashboard_response,
        "result": result,
    }


def _load_dashboard_baseline_inputs(
    validated: ProductionPlanningRequest,
    simulation_input: SimulationInputBundle | None,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """
    Parameters:
        - validated: Validated planning request from the graph state.
        - simulation_input: Sampling/history bundle loaded earlier in the graph.

    Methodology:
        - Use the DB current-plan snapshot as the sole baseline source.
        - Fall back to in-request existing schedules when the graph is used without DB access.

    Output:
        - Tuple of formatter baseline input rows and non-fatal warning messages.
    """
    fallback = {
        "baseline_plans": [
            schedule.model_dump(mode="python")
            for schedule in validated.existing_schedules
        ],
    }
    if simulation_input is None:
        return fallback, []

    try:
        snapshot = SimulationDataRepository().load_baseline_simulation_snapshot(
            validated.planning_start,
            validated.planning_end,
        )
    except PlanningDataAccessError as exc:
        return fallback, [f"Dashboard baseline DB snapshot fallback used: {exc}"]

    return {"baseline_plans": snapshot.current_plan_rows}, snapshot.warnings


def _build_adjusted_plan_candidates(
    plan_results: list[PlanResult],
    simulation_results: list[PlanSimulationResult],
) -> list[AdjustedPlanCandidate]:
    """
    Parameters:
        - plan_results: Solved CP-SAT plan variants.
        - simulation_results: Simulation metrics containing order-level duration estimates.

    Methodology:
        - Convert internal ScheduleItem rows into production_plans-compatible response rows.
        - Assign plan_sequence independently inside each line by planned start order.
        - Use simulation estimated_duration_hr when available and fall back to planned duration.

    Output:
        - Adjusted plan candidates ready to serialize as API/DB-save payloads.
    """
    updated_at = datetime.now(UTC)
    estimates_by_variant = _duration_estimates_by_variant(simulation_results)
    candidates = []
    for plan in plan_results:
        candidates.append(
            AdjustedPlanCandidate(
                plan_variant_code=plan.plan_variant_code,
                plan_variant_name=plan.plan_variant_name,
                status=plan.status,
                plans=_build_adjusted_plan_rows(
                    plan,
                    estimates_by_variant.get(plan.plan_variant_code, {}),
                    updated_at,
                ),
            )
        )
    return candidates


def _duration_estimates_by_variant(
    simulation_results: list[PlanSimulationResult],
) -> dict[str, dict[str, float]]:
    estimates = {}
    for result in simulation_results:
        estimates[result.plan_variant_code] = {
            row["order_id"]: float(row["estimated_duration_hr"])
            for row in result.order_duration_estimates
            if "order_id" in row and "estimated_duration_hr" in row
        }
    return estimates


def _build_adjusted_plan_rows(
    plan: PlanResult,
    duration_estimates_by_order_id: dict[str, float],
    updated_at: datetime,
) -> list[AdjustedPlanRow]:
    line_sequence_by_order_id = _line_sequence_by_order_id(plan)
    rows = []
    for item in sorted(plan.schedule_items, key=lambda value: (value.line_id, value.start_time)):
        rows.append(
            AdjustedPlanRow(
                order_id=item.order_id,
                product_id=item.product_id,
                line_id=item.line_id,
                operator_id=choice(OPERATOR_ID_POOL),
                planned_start_at=item.start_time,
                planned_end_at=item.end_time,
                estimated_duration_hr=duration_estimates_by_order_id.get(
                    item.order_id,
                    _planned_duration_hr(item),
                ),
                planned_quantity=item.planned_production_quantity,
                plan_sequence=line_sequence_by_order_id[item.order_id],
                plan_status="SCHEDULED",
                updated_at=updated_at,
            )
        )
    return rows


def _line_sequence_by_order_id(plan: PlanResult) -> dict[str, int]:
    sequence_by_order_id = {}
    items_by_line: dict[str, list] = {}
    for item in plan.schedule_items:
        items_by_line.setdefault(item.line_id, []).append(item)
    for line_items in items_by_line.values():
        for index, item in enumerate(
            sorted(
                line_items,
                key=lambda value: (value.start_time, value.end_time, value.order_id),
            ),
            start=1,
        ):
            sequence_by_order_id[item.order_id] = index
    return sequence_by_order_id


def _planned_duration_hr(item) -> float:
    duration_minutes = max(1, round((item.end_time - item.start_time).total_seconds() / 60))
    return round(duration_minutes / 60, 4)


def _build_objective_bundle(
    normalized: NormalizedPlanningData,
    solver_config: SolverConfig,
    variant_code: str,
) -> CpsatModelBundle:
    logger.info("production_planning.model_build.started", extra={"variant_code": variant_code})
    bundle = build_base_cpsat_model(normalized, solver_config)
    logger.info(
        "production_planning.model_build.completed",
        extra={
            "variant_code": variant_code,
            "orders": len(normalized.orders),
            "lines": len(normalized.lines),
            "optional_intervals": bundle.metric_vars["optional_interval_count"],
            "changeover_constraints": bundle.metric_vars["changeover_constraint_count"],
        },
    )
    return bundle


def _solve_extract_plan(
    bundle: CpsatModelBundle,
    normalized: NormalizedPlanningData,
    solver_config: SolverConfig,
    variant_code: str,
) -> PlanResult:
    logger.info("production_planning.solver.started", extra={"variant_code": variant_code})
    raw = solve_model(bundle, variant_code, solver_config)
    logger.info(
        "production_planning.solver.completed",
        extra={
            "variant_code": variant_code,
            "status": raw.status,
            "objective_value": raw.objective_value,
            "wall_time_seconds": raw.wall_time_seconds,
        },
    )
    plan = extract_plan_result(raw, bundle, variant_code, normalized, solver_config)
    logger.info(
        "production_planning.extraction.completed",
        extra={
            "variant_code": variant_code,
            "scheduled": plan.metrics.scheduled_count,
            "unscheduled": plan.metrics.unscheduled_count,
            "delayed": plan.metrics.delayed_order_count,
        },
    )
    return plan


def _generate_due_date_plan(
    normalized: NormalizedPlanningData,
    validated: ProductionPlanningRequest,
) -> PlanResult:
    """
    Parameters:
        - normalized: Shared normalized planning data.
        - validated: Validated request containing the base solver configuration.

    Methodology:
        - Solve a SOFT due-date model first to find the maximum schedulable order count under
          physical hard constraints.
        - Try a HARD due-date model constrained to keep at least that scheduled count.
        - Use HARD only when it does not reduce assignment; otherwise keep the SOFT result.

    Output:
        - Due-date optimal PlanResult that maximizes assignment before enforcing hard due dates.
    """
    variant_code = "DUE_DATE_OPTIMAL"
    soft_config = validated.solver_config.model_copy(
        update={"due_date_policy": "SOFT", "use_material_constraints": False}
    )
    soft_bundle = _build_objective_bundle(normalized, soft_config, variant_code)
    apply_objective_by_variant(
        soft_bundle.model,
        soft_bundle,
        normalized,
        soft_config,
        variant_code,
    )
    soft_plan = _solve_extract_plan(soft_bundle, normalized, soft_config, variant_code)
    if soft_plan.status not in {"OPTIMAL", "FEASIBLE"}:
        return soft_plan

    hard_config = validated.solver_config.model_copy(
        update={"due_date_policy": "HARD", "use_material_constraints": False}
    )
    hard_bundle = _build_objective_bundle(normalized, hard_config, variant_code)
    _add_minimum_scheduled_count_constraint(
        hard_bundle,
        soft_plan.metrics.scheduled_count,
    )
    apply_objective_by_variant(
        hard_bundle.model,
        hard_bundle,
        normalized,
        hard_config,
        variant_code,
    )
    hard_plan = _solve_extract_plan(hard_bundle, normalized, hard_config, variant_code)
    if (
        hard_plan.status in {"OPTIMAL", "FEASIBLE"}
        and hard_plan.metrics.scheduled_count >= soft_plan.metrics.scheduled_count
    ):
        hard_plan.warnings.append(
            "DUE_DATE_OPTIMAL used HARD due-date constraints without reducing scheduled count."
        )
        return hard_plan

    soft_plan.warnings.append(
        "DUE_DATE_OPTIMAL relaxed due-date constraints because HARD due dates would reduce "
        "the maximum schedulable order count."
    )
    return soft_plan


def _generate_amount_plan(
    normalized: NormalizedPlanningData,
    validated: ProductionPlanningRequest,
) -> PlanResult:
    """
    Parameters:
        - normalized: Shared normalized planning data.
        - validated: Validated request containing the base solver configuration.

    Methodology:
        - Solve AMOUNT_OPTIMAL once with SOFT due dates to discover a good product-clustering
          candidate.
        - Re-solve with HARD due dates, using the SOFT solution as a hint and preserving the
          same scheduled count when possible.
        - If HARD cannot keep the SOFT scheduled count, keep HARD due dates and let the
          objective choose the highest-value feasible subset instead of returning a delayed plan.

    Output:
        - Amount optimal PlanResult whose scheduled orders satisfy HARD due dates.
    """
    variant_code = "AMOUNT_OPTIMAL"
    soft_config = validated.solver_config.model_copy(
        update={
            "allow_unscheduled_orders": True,
            "due_date_policy": "SOFT",
            "use_material_constraints": False,
        }
    )
    soft_bundle = _build_objective_bundle(normalized, soft_config, variant_code)
    apply_objective_by_variant(
        soft_bundle.model,
        soft_bundle,
        normalized,
        soft_config,
        variant_code,
    )
    soft_plan = _solve_extract_plan(soft_bundle, normalized, soft_config, variant_code)
    if soft_plan.status not in {"OPTIMAL", "FEASIBLE"}:
        return soft_plan

    hard_config = validated.solver_config.model_copy(
        update={
            "allow_unscheduled_orders": True,
            "due_date_policy": "HARD",
            "use_material_constraints": False,
        }
    )
    hard_bundle = _build_objective_bundle(normalized, hard_config, variant_code)
    _add_plan_solution_hint(hard_bundle, normalized, soft_plan)
    _add_minimum_scheduled_count_constraint(
        hard_bundle,
        soft_plan.metrics.scheduled_count,
    )
    apply_objective_by_variant(
        hard_bundle.model,
        hard_bundle,
        normalized,
        hard_config,
        variant_code,
    )
    hard_plan = _solve_extract_plan(hard_bundle, normalized, hard_config, variant_code)
    if (
        hard_plan.status in {"OPTIMAL", "FEASIBLE"}
        and hard_plan.metrics.scheduled_count >= soft_plan.metrics.scheduled_count
    ):
        hard_plan.warnings.append(
            "AMOUNT_OPTIMAL enforced HARD due dates without reducing scheduled count."
        )
        return hard_plan

    fallback_bundle = _build_objective_bundle(normalized, hard_config, variant_code)
    _add_plan_solution_hint(fallback_bundle, normalized, soft_plan)
    apply_objective_by_variant(
        fallback_bundle.model,
        fallback_bundle,
        normalized,
        hard_config,
        variant_code,
    )
    fallback_plan = _solve_extract_plan(
        fallback_bundle,
        normalized,
        hard_config,
        variant_code,
    )
    fallback_plan.warnings.append(
        "AMOUNT_OPTIMAL enforced HARD due dates and reduced scheduled orders because "
        "the SOFT scheduled count was infeasible without delay."
    )
    return fallback_plan


def _add_minimum_scheduled_count_constraint(
    bundle: CpsatModelBundle,
    minimum_scheduled_count: int,
) -> None:
    """
    Parameters:
        - bundle: CP-SAT model bundle containing scheduled decision variables.
        - minimum_scheduled_count: Required lower bound for scheduled orders.

    Methodology:
        - Preserve the maximum assignment discovered by the SOFT due-date probe while testing
          whether due dates can be enforced as hard constraints.

    Output:
        - None. Adds one aggregate linear constraint to the model.
    """
    bundle.model.Add(
        sum(order_vars["scheduled"] for order_vars in bundle.order_vars.values())
        >= minimum_scheduled_count
    )


def _add_plan_solution_hint(
    bundle: CpsatModelBundle,
    normalized: NormalizedPlanningData,
    plan: PlanResult,
) -> None:
    """
    Parameters:
        - bundle: Fresh CP-SAT bundle receiving solution hints.
        - normalized: Normalized planning data with planning_start.
        - plan: Previously solved plan whose assignments guide the next solve.

    Methodology:
        - Hint scheduled, unscheduled, assignment, start, end, and completion variables.
        - Hints guide CP-SAT toward product clustering from the SOFT solve without making those
          assignments hard constraints.

    Output:
        - None. Adds CP-SAT solution hints in place.
    """
    item_by_order_id = {item.order_id: item for item in plan.schedule_items}
    for order_id, order_vars in bundle.order_vars.items():
        item = item_by_order_id.get(order_id)
        if item is None:
            bundle.model.AddHint(order_vars["scheduled"], 0)
            if order_id in bundle.unscheduled_vars:
                bundle.model.AddHint(bundle.unscheduled_vars[order_id], 1)
            for assigned in order_vars["assigned"].values():
                bundle.model.AddHint(assigned, 0)
            continue

        start_offset = to_minute_offset(item.start_time, normalized.planning_start)
        end_offset = to_minute_offset(item.end_time, normalized.planning_start)
        bundle.model.AddHint(order_vars["scheduled"], 1)
        bundle.model.AddHint(order_vars["completion"], end_offset)
        if order_id in bundle.unscheduled_vars:
            bundle.model.AddHint(bundle.unscheduled_vars[order_id], 0)
        for line_id, assigned in order_vars["assigned"].items():
            is_selected_line = line_id == item.line_id
            bundle.model.AddHint(assigned, int(is_selected_line))
            if is_selected_line:
                bundle.model.AddHint(order_vars["start"][line_id], start_offset)
                bundle.model.AddHint(order_vars["end"][line_id], end_offset)


def _solver_config_for_variant(
    solver_config: SolverConfig,
    variant_code: str,
) -> SolverConfig:
    """
    Parameters:
        - solver_config: Request-level solver configuration.
        - variant_code: Production planning variant currently being built.

    Methodology:
        - Keep amount/cost optimization on SOFT due-date policy so late-penalty tradeoffs can
          be optimized by the objective.

    Output:
        - SolverConfig used for this specific CP-SAT model build and solve.
    """
    if variant_code == "AMOUNT_OPTIMAL":
        return solver_config.model_copy(
            update={"due_date_policy": "SOFT", "use_material_constraints": False}
        )
    return solver_config
