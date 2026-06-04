from datetime import UTC, datetime, timedelta

from app.features.production_planning.config import SolverConfig
from app.features.production_planning.production_planning_node import generate_production_plans
from app.features.production_planning.schemas import (
    ExistingScheduleInput,
    OrderInput,
    ProductInput,
    ProductionLineInput,
    ProductionPlanningRequest,
    ProductLineCapabilityInput,
)


def test_solution_extractor_respects_locked_existing_schedule() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=3),
        orders=[
            OrderInput(
                order_id="O-1",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=3),
                order_amount=100,
            )
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
        existing_schedules=[
            ExistingScheduleInput(
                schedule_id="S-LOCK",
                line_id="L-1",
                product_id="P-1",
                start_time=start,
                end_time=start + timedelta(minutes=60),
            )
        ],
        solver_config=SolverConfig(time_limit_seconds=5, num_search_workers=1),
    )

    result = generate_production_plans(request)
    plan = next(
        plan
        for plan in result.plan_results
        if plan.plan_variant_code == "DUE_DATE_MIN_DELAY_COUNT"
    )
    item = plan.schedule_items[0]

    assert item.start_time >= start + timedelta(minutes=60)
    assert plan.metrics.makespan_minutes >= 120
