from datetime import UTC, datetime, timedelta

from app.features.production_planning.config import SolverConfig
from app.features.production_planning.preprocessing import normalize_request
from app.features.production_planning.production_planning_node import generate_production_plans
from app.features.production_planning.schemas import (
    BomItemInput,
    MaterialInput,
    OrderInput,
    ProductInput,
    ProductionLineInput,
    ProductionPlanningRequest,
    ProductLineCapabilityInput,
)
from app.features.production_planning.solution_validator import validate_solution
from app.features.production_planning.validators import validate_request


def _request() -> ProductionPlanningRequest:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    return ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=3),
        orders=[
            OrderInput(
                order_id="O-1",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=1),
                order_amount=100,
            ),
            OrderInput(
                order_id="O-2",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=2),
                order_amount=100,
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
        materials=[
            MaterialInput(material_id="M-1", material_name="Material", available_quantity=2),
        ],
        bom_items=[
            BomItemInput(product_id="P-1", material_id="M-1", required_quantity_per_unit=1),
        ],
        solver_config=SolverConfig(time_limit_seconds=5, num_search_workers=1),
    )


def _normalized_request(request: ProductionPlanningRequest):
    return normalize_request(validate_request(request))


def _plan_index(result, variant_code: str) -> int:
    return next(
        index
        for index, plan in enumerate(result.plan_results)
        if plan.plan_variant_code == variant_code
    )


def test_solution_validator_accepts_generated_solution() -> None:
    request = _request()
    result = generate_production_plans(request)
    data = _normalized_request(request)

    report = validate_solution(result, data, request.solver_config)

    assert report.is_valid


def test_solution_validator_detects_line_overlap_after_schedule_corruption() -> None:
    request = _request()
    result = generate_production_plans(request)
    data = _normalized_request(request)
    plan_index = _plan_index(result, "DUE_DATE_MIN_DELAY_COUNT")
    plan = result.plan_results[plan_index]
    first_item = plan.schedule_items[0]
    second_item = plan.schedule_items[1]
    corrupted_first_item = first_item.model_copy(update={"end_time": second_item.end_time})
    corrupted_plan = plan.model_copy(
        update={"schedule_items": [corrupted_first_item, second_item]}
    )
    plan_results = list(result.plan_results)
    plan_results[plan_index] = corrupted_plan
    corrupted_result = result.model_copy(update={"plan_results": plan_results})

    report = validate_solution(corrupted_result, data, request.solver_config)

    assert not report.is_valid
    assert report.constraint_violations


def test_solution_validator_detects_metric_mismatch() -> None:
    request = _request()
    result = generate_production_plans(request)
    data = _normalized_request(request)
    plan_index = _plan_index(result, "DUE_DATE_MIN_DELAY_COUNT")
    plan = result.plan_results[plan_index]
    corrupted_metrics = plan.metrics.model_copy(update={"total_tardiness_minutes": 999})
    corrupted_plan = plan.model_copy(update={"metrics": corrupted_metrics})
    plan_results = list(result.plan_results)
    plan_results[plan_index] = corrupted_plan
    corrupted_result = result.model_copy(update={"plan_results": plan_results})

    report = validate_solution(corrupted_result, data, request.solver_config)

    assert report.metric_mismatches


def test_solution_validator_detects_hard_due_date_business_rule_violation() -> None:
    request = _request()
    result = generate_production_plans(request)
    data = _normalized_request(request)
    plan_index = _plan_index(result, "DUE_DATE_MIN_DELAY_COUNT")
    plan = result.plan_results[plan_index]
    item = plan.schedule_items[0]
    corrupted_item = item.model_copy(update={"end_time": request.planning_end})
    corrupted_plan = plan.model_copy(update={"schedule_items": [corrupted_item]})
    plan_results = list(result.plan_results)
    plan_results[plan_index] = corrupted_plan
    corrupted_result = result.model_copy(update={"plan_results": plan_results})
    config = request.solver_config.model_copy(update={"due_date_policy": "HARD"})

    report = validate_solution(corrupted_result, data, config)

    assert report.business_rule_violations
