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


def test_due_date_objective_schedules_earlier_due_order_earlier_when_feasible() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=4),
        orders=[
            OrderInput(
                order_id="O-EARLY",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(minutes=60),
                order_amount=100,
            ),
            OrderInput(
                order_id="O-MID",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(minutes=120),
                order_amount=100,
            ),
            OrderInput(
                order_id="O-LATE",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(minutes=180),
                order_amount=100,
            ),
        ],
        products=[
            ProductInput(
                product_id="P-1",
                product_name="Product",
                default_process_time_minutes=60,
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
    plan = next(
        plan
        for plan in result.plan_results
        if plan.plan_variant_code == "DUE_DATE_MIN_DELAY_COUNT"
    )

    assert plan.status in {"OPTIMAL", "FEASIBLE"}
    starts = {item.order_id: item.start_time for item in plan.schedule_items}
    assert starts["O-EARLY"] <= starts["O-MID"] <= starts["O-LATE"]
    line_items = sorted(plan.schedule_items, key=lambda item: item.start_time)
    for previous, current in zip(line_items, line_items[1:], strict=False):
        assert previous.end_time <= current.start_time
