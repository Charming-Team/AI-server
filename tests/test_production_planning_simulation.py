from datetime import UTC, datetime, timedelta

from app.features.production_planning.config import SimulationConfig, SolverConfig
from app.features.production_planning.production_planning_node import generate_production_plans
from app.features.production_planning.schemas import (
    OrderInput,
    ProductInput,
    ProductionLineInput,
    ProductionPlanningRequest,
    ProductLineCapabilityInput,
)


def _request() -> ProductionPlanningRequest:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    return ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=4),
        orders=[
            OrderInput(
                order_id="O-1",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=2),
                order_amount=100_000,
                late_penalty_amount=100,
            ),
            OrderInput(
                order_id="O-2",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=3),
                order_amount=80_000,
                late_penalty_amount=100,
            ),
        ],
        products=[
            ProductInput(
                product_id="P-1",
                product_name="Product",
                default_process_time_minutes=30,
            )
        ],
        production_lines=[
            ProductionLineInput(line_id="L-1", line_name="Line 1", is_active=True),
        ],
        product_line_capabilities=[
            ProductLineCapabilityInput(product_id="P-1", line_id="L-1"),
        ],
        solver_config=SolverConfig(time_limit_seconds=5, num_search_workers=1),
        simulation_config=SimulationConfig(num_iterations=5, random_seed=7),
    )


def test_simulation_node_returns_one_result_per_plan_candidate() -> None:
    result = generate_production_plans(_request())

    assert len(result.simulation_results) == 6
    assert result.simulation_comparison_summary is not None
    assert result.recommended_due_date_plan is not None
    assert result.recommended_cost_plan is not None
    assert result.recommended_due_date_plan.plan_family == "DUE_DATE_OPTIMAL"
    assert result.recommended_cost_plan.plan_family == "COST_OPTIMAL"
