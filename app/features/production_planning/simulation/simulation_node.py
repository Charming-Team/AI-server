from __future__ import annotations

from collections import Counter
from random import Random

from app.features.production_planning.config import SimulationConfig, SolverConfig
from app.features.production_planning.repositories.simulation_data_repository import (
    SimulationInputBundle,
)
from app.features.production_planning.schemas import (
    NormalizedPlanningData,
    PlanResult,
    PlanSimulationResult,
)
from app.features.production_planning.simulation.des import (
    ScenarioResult,
    run_discrete_event_simulation,
)
from app.features.production_planning.simulation.sampling import (
    SamplingDistributions,
    build_empirical_sampling_distributions,
)
from app.features.production_planning.simulation.sampling import (
    build_sampling_distributions as build_fallback_sampling_distributions,
)


def build_sampling_distributions(
    plan_results: list[PlanResult],
    data: NormalizedPlanningData,
    config: SimulationConfig,
    simulation_input: SimulationInputBundle | None = None,
) -> SamplingDistributions:
    """
    Parameters:
        - plan_results: Six CP-SAT candidate plans.
        - data: Normalized planning data shared by planning and simulation.
        - config: Simulation configuration.
        - simulation_input: Optional historical rows from ai_planning sampling views.

    Methodology:
        - Use empirical historical distributions when production result history is available.
        - Use configured fallback distributions as the safe default when DB sampling data is
          unavailable or sparse.

    Output:
        - SamplingDistributions for Monte Carlo simulation.
    """
    if simulation_input is not None and simulation_input.production_results:
        return build_empirical_sampling_distributions(simulation_input, data, config)
    fallback = build_fallback_sampling_distributions(plan_results, data, config)
    if simulation_input is not None:
        fallback.warnings.extend(simulation_input.warnings)
        fallback.warnings.append(
            "Production result sampling history is empty; using fallback distributions."
        )
    return fallback


def simulate_plan_candidates(
    plan_results: list[PlanResult],
    data: NormalizedPlanningData,
    solver_config: SolverConfig,
    simulation_config: SimulationConfig,
    distributions: SamplingDistributions,
) -> list[PlanSimulationResult]:
    """
    Parameters:
        - plan_results: Six CP-SAT candidate plans.
        - data: Normalized planning data used by DES.
        - solver_config: Solver configuration containing changeover defaults.
        - simulation_config: Monte Carlo run configuration.
        - distributions: Sampling distributions for stochastic values.

    Methodology:
        - Run deterministic-seeded Monte Carlo iterations for each candidate.
        - Aggregate scenario-level DES metrics into PlanSimulationResult values.

    Output:
        - One PlanSimulationResult for each input plan candidate.
    """
    if not simulation_config.enabled:
        return [_disabled_result(plan) for plan in plan_results]

    results = []
    for plan_index, plan in enumerate(plan_results):
        scenarios = []
        for iteration in range(simulation_config.num_iterations):
            rng = Random(
                simulation_config.random_seed
                + plan_index * 1_000_003
                + iteration
            )
            scenarios.append(
                run_discrete_event_simulation(
                    plan,
                    data,
                    solver_config,
                    simulation_config,
                    distributions,
                    rng,
                )
            )
        results.append(_aggregate_plan_simulation(plan, scenarios, simulation_config))
    return results


def _aggregate_plan_simulation(
    plan: PlanResult,
    scenarios: list[ScenarioResult],
    config: SimulationConfig,
) -> PlanSimulationResult:
    total_orders = max(1, plan.metrics.scheduled_count + plan.metrics.unscheduled_count)
    tardiness_values = [scenario.total_tardiness_minutes for scenario in scenarios]
    late_penalty_values = [scenario.late_penalty_amount for scenario in scenarios]
    delayed_counts = [scenario.delayed_order_count for scenario in scenarios]
    shortage_counts = [scenario.material_shortage_count for scenario in scenarios]
    delay_causes = Counter()
    for scenario in scenarios:
        delay_causes.update(scenario.delay_causes)

    expected_delayed = _mean(delayed_counts)
    expected_shortages = _mean(shortage_counts)
    expected_late_penalty = _mean(late_penalty_values)
    expected_changeover_cost = _mean([scenario.changeover_cost for scenario in scenarios])
    expected_risk_cost = _mean([scenario.total_risk_cost for scenario in scenarios])
    yield_sample_count = sum(scenario.yield_sample_count for scenario in scenarios)
    yield_rate = None
    if yield_sample_count:
        yield_rate = sum(scenario.yield_rate_sum for scenario in scenarios) / yield_sample_count

    return PlanSimulationResult(
        plan_variant_code=plan.plan_variant_code,
        plan_family=plan.plan_family,
        iterations=config.num_iterations,
        delay_probability=expected_delayed / total_orders,
        expected_delayed_order_count=expected_delayed,
        expected_tardiness_minutes=_mean(tardiness_values),
        p50_tardiness_minutes=_percentile(tardiness_values, 0.50),
        p90_tardiness_minutes=_percentile(tardiness_values, 0.90),
        p95_tardiness_minutes=_percentile(tardiness_values, 0.95),
        expected_late_penalty_amount=expected_late_penalty,
        p95_late_penalty_amount=_percentile(late_penalty_values, 0.95),
        expected_changeover_cost=expected_changeover_cost,
        expected_total_risk_cost=expected_risk_cost,
        material_shortage_probability=sum(1 for value in shortage_counts if value > 0)
        / len(shortage_counts),
        expected_material_shortage_count=expected_shortages,
        expected_yield_rate=yield_rate,
        expected_defect_quantity=_mean([scenario.defect_quantity for scenario in scenarios]),
        top_delay_causes=[
            {"cause": cause, "count": count}
            for cause, count in delay_causes.most_common(5)
        ],
        scenario_summary={
            "scheduled_count": plan.metrics.scheduled_count,
            "unscheduled_count": plan.metrics.unscheduled_count,
            "material_shortage_scenarios": sum(1 for value in shortage_counts if value > 0),
        },
    )


def _disabled_result(plan: PlanResult) -> PlanSimulationResult:
    return PlanSimulationResult(
        plan_variant_code=plan.plan_variant_code,
        plan_family=plan.plan_family,
        iterations=0,
        delay_probability=0.0,
        expected_delayed_order_count=float(plan.metrics.delayed_order_count),
        expected_tardiness_minutes=float(plan.metrics.total_tardiness_minutes),
        p50_tardiness_minutes=float(plan.metrics.total_tardiness_minutes),
        p90_tardiness_minutes=float(plan.metrics.total_tardiness_minutes),
        p95_tardiness_minutes=float(plan.metrics.total_tardiness_minutes),
        expected_late_penalty_amount=float(plan.metrics.total_late_penalty_amount),
        p95_late_penalty_amount=float(plan.metrics.total_late_penalty_amount),
        expected_changeover_cost=float(plan.metrics.total_changeover_cost),
        expected_total_risk_cost=float(plan.metrics.estimated_total_cost),
        material_shortage_probability=0.0,
        expected_material_shortage_count=0.0,
        expected_yield_rate=None,
        expected_defect_quantity=None,
        top_delay_causes=[],
        scenario_summary={"simulation_enabled": False},
    )


def _mean(values) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def _percentile(values, percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]
