from datetime import UTC, datetime, timedelta

from app.features.production_planning.config import SolverConfig
from app.features.production_planning.production_planning_node import generate_production_plans
from app.features.production_planning.schemas import (
    OrderInput,
    ProductInput,
    ProductionLineInput,
    ProductionPlanningRequest,
    ProductLineCapabilityInput,
)


def test_cost_objective_can_prioritize_penalty_reduction_over_due_date_order() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(minutes=300),
        orders=[
            OrderInput(
                order_id="O-LOW",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(minutes=100),
                order_amount=1_000,
                late_penalty_amount=1,
            ),
            OrderInput(
                order_id="O-HIGH",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(minutes=150),
                order_amount=100_000,
                late_penalty_amount=100_000,
            ),
            OrderInput(
                order_id="O-TAIL",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(minutes=300),
                order_amount=1_000,
                late_penalty_amount=1,
            ),
        ],
        products=[
            ProductInput(
                product_id="P-1",
                product_name="Product",
                default_process_time_minutes=100,
            )
        ],
        production_lines=[
            ProductionLineInput(line_id="L-1", line_name="Line 1", is_active=True),
        ],
        product_line_capabilities=[
            ProductLineCapabilityInput(product_id="P-1", line_id="L-1"),
        ],
        solver_config=SolverConfig(time_limit_seconds=5, num_search_workers=1),
    )

    result = generate_production_plans(request)
    due_plan = next(
        plan
        for plan in result.plan_results
        if plan.plan_variant_code == "DUE_DATE_MIN_DELAY_COUNT"
    )
    cost_plan = next(
        plan for plan in result.plan_results if plan.plan_variant_code == "COST_MIN_TOTAL_COST"
    )

    assert due_plan.status in {"OPTIMAL", "FEASIBLE"}
    assert cost_plan.status in {"OPTIMAL", "FEASIBLE"}
    due_starts = {item.order_id: item.start_time for item in due_plan.schedule_items}
    cost_starts = {item.order_id: item.start_time for item in cost_plan.schedule_items}
    assert due_starts["O-LOW"] < due_starts["O-HIGH"]
    assert cost_starts["O-HIGH"] < cost_starts["O-LOW"]
    assert cost_plan.metrics.estimated_total_cost < due_plan.metrics.estimated_total_cost
