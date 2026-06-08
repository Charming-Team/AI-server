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


def test_amount_objective_enforces_hard_due_dates_instead_of_delaying_orders() -> None:
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
                order_amount=1_000_000,
                late_penalty_amount=1,
            ),
            OrderInput(
                order_id="O-HIGH",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(minutes=150),
                order_amount=1_000,
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
        if plan.plan_variant_code == "DUE_DATE_OPTIMAL"
    )
    cost_plan = next(
        plan for plan in result.plan_results if plan.plan_variant_code == "AMOUNT_OPTIMAL"
    )

    assert due_plan.status in {"OPTIMAL", "FEASIBLE"}
    assert cost_plan.status in {"OPTIMAL", "FEASIBLE"}
    assert due_plan.metrics.unscheduled_count == 0
    assert due_plan.metrics.delayed_order_count > 0
    assert cost_plan.metrics.delayed_order_count == 0
    assert cost_plan.metrics.total_tardiness_minutes == 0
    assert cost_plan.metrics.unscheduled_count == 1
    assert any(
        "AMOUNT_OPTIMAL enforced HARD due dates" in warning
        for warning in cost_plan.warnings
    )


def test_amount_objective_clusters_same_product_to_reduce_sequence_changes() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(minutes=60),
        orders=[
            OrderInput(
                order_id="O-A1",
                product_id="P-A",
                quantity=1,
                due_date=start + timedelta(minutes=60),
                order_amount=100_000,
                late_penalty_amount=0,
            ),
            OrderInput(
                order_id="O-B",
                product_id="P-B",
                quantity=1,
                due_date=start + timedelta(minutes=60),
                order_amount=100_000,
                late_penalty_amount=0,
            ),
            OrderInput(
                order_id="O-A2",
                product_id="P-A",
                quantity=1,
                due_date=start + timedelta(minutes=60),
                order_amount=100_000,
                late_penalty_amount=0,
            ),
        ],
        products=[
            ProductInput(
                product_id="P-A",
                product_name="Product A",
                default_process_time_minutes=10,
            ),
            ProductInput(
                product_id="P-B",
                product_name="Product B",
                default_process_time_minutes=10,
            ),
        ],
        production_lines=[
            ProductionLineInput(line_id="L-1", line_name="Line 1", is_active=True),
        ],
        product_line_capabilities=[
            ProductLineCapabilityInput(product_id="P-A", line_id="L-1"),
            ProductLineCapabilityInput(product_id="P-B", line_id="L-1"),
        ],
        solver_config=SolverConfig(time_limit_seconds=5, num_search_workers=1),
    )

    result = generate_production_plans(request)
    amount_plan = next(
        plan for plan in result.plan_results if plan.plan_variant_code == "AMOUNT_OPTIMAL"
    )
    products_by_start = [
        item.product_id
        for item in sorted(amount_plan.schedule_items, key=lambda item: item.start_time)
    ]
    product_change_count = sum(
        left != right
        for left, right in zip(products_by_start, products_by_start[1:], strict=False)
    )

    assert amount_plan.status in {"OPTIMAL", "FEASIBLE"}
    assert amount_plan.metrics.delayed_order_count == 0
    assert product_change_count == 1
