from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.features.production_planning.config import DueDateOptimizationConfig, SolverConfig
from app.features.production_planning.exceptions import PlanningValidationError
from app.features.production_planning.plan_comparator import compare_plans
from app.features.production_planning.production_planning_node import generate_production_plans
from app.features.production_planning.schemas import (
    PLAN_VARIANTS,
    BomItemInput,
    ExistingScheduleInput,
    MaterialInput,
    OrderInput,
    PlanMetrics,
    PlanResult,
    ProductInput,
    ProductionLineInput,
    ProductionPlanningRequest,
    ProductLineCapabilityInput,
)


def _material_request(allow_unscheduled: bool) -> ProductionPlanningRequest:
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
                order_amount=100,
                late_penalty_amount=10,
            ),
            OrderInput(
                order_id="O-2",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=3),
                order_amount=100,
                late_penalty_amount=20,
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
        materials=[
            MaterialInput(material_id="M-1", material_name="Material", available_quantity=6),
        ],
        bom_items=[
            BomItemInput(product_id="P-1", material_id="M-1", required_quantity_per_unit=6),
        ],
        solver_config=SolverConfig(
            time_limit_seconds=5,
            num_search_workers=1,
            allow_unscheduled_orders=allow_unscheduled,
        ),
    )


def _plan_by_code(result, variant_code: str) -> PlanResult:
    return next(plan for plan in result.plan_results if plan.plan_variant_code == variant_code)


def test_generate_production_plans_returns_two_variants_and_metadata() -> None:
    result = generate_production_plans(_material_request(allow_unscheduled=True))

    assert len(result.plan_results) == 2
    assert {plan.plan_variant_code for plan in result.plan_results} == set(PLAN_VARIANTS)
    assert sum(plan.plan_family == "DUE_DATE_OPTIMAL" for plan in result.plan_results) == 1
    assert sum(plan.plan_family == "AMOUNT_OPTIMAL" for plan in result.plan_results) == 1
    assert result.recommended_plan_variant_code in PLAN_VARIANTS
    assert len(result.comparison_summary.plan_rankings) == 2
    assert set(result.solver_metadata) == set(PLAN_VARIANTS)


def test_material_shortage_is_soft_and_tracked_without_penalty_cost() -> None:
    result = generate_production_plans(_material_request(allow_unscheduled=True))
    due_plan = _plan_by_code(result, "DUE_DATE_OPTIMAL")
    amount_plan = _plan_by_code(result, "AMOUNT_OPTIMAL")

    assert due_plan.status in {"OPTIMAL", "FEASIBLE"}
    assert due_plan.metrics.scheduled_count == 2
    assert due_plan.metrics.unscheduled_count == 0
    assert due_plan.metrics.total_material_shortage_quantity > 0
    assert due_plan.metrics.total_material_shortage_penalty_amount == 0
    assert amount_plan.status in {"OPTIMAL", "FEASIBLE"}
    assert amount_plan.metrics.scheduled_count == 2
    assert amount_plan.metrics.unscheduled_count == 0
    assert amount_plan.metrics.total_material_shortage_quantity > 0
    assert amount_plan.metrics.total_material_shortage_penalty_amount == 0


def test_soft_material_shortage_does_not_make_mandatory_orders_infeasible() -> None:
    result = generate_production_plans(_material_request(allow_unscheduled=False))
    plan = _plan_by_code(result, "DUE_DATE_OPTIMAL")

    assert plan.status in {"OPTIMAL", "FEASIBLE"}
    assert plan.metrics.scheduled_count == 2
    assert plan.metrics.unscheduled_count == 0
    assert plan.metrics.total_material_shortage_quantity > 0


def test_plan_comparison_recommendation_and_rankings() -> None:
    due_plan = _plan_result(
        "DUE_DATE_OPTIMAL",
        "DUE_DATE_OPTIMAL",
        delayed_order_count=0,
        estimated_total_cost=55,
    )
    cost_plan = _plan_result(
        "AMOUNT_OPTIMAL",
        "AMOUNT_OPTIMAL",
        delayed_order_count=3,
        estimated_total_cost=50,
    )

    summary = compare_plans([due_plan, cost_plan])

    assert summary.recommended_plan_variant_code == "DUE_DATE_OPTIMAL"
    assert len(summary.plan_rankings) == 2


def test_due_date_plan_relaxes_hard_due_date_when_mandatory_assignment_needs_delay() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=3),
        orders=[
            OrderInput(
                order_id="O-LATE",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(minutes=30),
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
        solver_config=SolverConfig(
            time_limit_seconds=5,
            num_search_workers=1,
            allow_unscheduled_orders=False,
            due_date_policy="HARD",
        ),
    )

    result = generate_production_plans(request)
    plan = _plan_by_code(result, "DUE_DATE_OPTIMAL")

    assert plan.status in {"OPTIMAL", "FEASIBLE"}
    assert plan.metrics.scheduled_count == 1
    assert plan.metrics.delayed_order_count == 1
    assert any("relaxed due-date constraints" in warning for warning in plan.warnings)


def test_due_date_variant_keeps_maximum_assignment_before_hard_due_date_policy() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=3),
        orders=[
            OrderInput(
                order_id="O-LATE",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(minutes=30),
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
        solver_config=SolverConfig(
            time_limit_seconds=5,
            num_search_workers=1,
            allow_unscheduled_orders=True,
            due_date_policy="SOFT",
        ),
    )

    result = generate_production_plans(request)
    due_plan = _plan_by_code(result, "DUE_DATE_OPTIMAL")

    assert due_plan.metrics.scheduled_count == 1
    assert due_plan.metrics.unscheduled_count == 0
    assert due_plan.metrics.delayed_order_count == 1
    assert any("relaxed due-date constraints" in warning for warning in due_plan.warnings)
    amount_plan = _plan_by_code(result, "AMOUNT_OPTIMAL")
    assert amount_plan.metrics.scheduled_count == 0
    assert amount_plan.metrics.unscheduled_count == 1
    assert amount_plan.metrics.delayed_order_count == 0


def test_line_priority_rank_prefers_lower_rank_line_when_business_terms_are_tied() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=3),
        orders=[
            OrderInput(
                order_id="O-1",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=2),
                order_amount=100,
            )
        ],
        products=[
            ProductInput(
                product_id="P-1",
                product_name="Product",
                default_process_time_minutes=30,
            )
        ],
        production_lines=[
            ProductionLineInput(line_id="1", line_name="Line 1", is_active=True),
            ProductionLineInput(line_id="2", line_name="Line 2", is_active=True),
        ],
        product_line_capabilities=[
            ProductLineCapabilityInput(product_id="P-1", line_id="1", priority_rank=3),
            ProductLineCapabilityInput(product_id="P-1", line_id="2", priority_rank=1),
        ],
        solver_config=SolverConfig(
            time_limit_seconds=5,
            num_search_workers=1,
            due_date_optimization=DueDateOptimizationConfig(line_priority_weight=100),
        ),
    )

    result = generate_production_plans(request)
    plan = _plan_by_code(result, "DUE_DATE_OPTIMAL")

    assert plan.schedule_items[0].line_id == "2"


def test_capability_is_not_expanded_to_peer_line_without_explicit_group() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=3),
        orders=[
            OrderInput(
                order_id="O-1",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=2),
                order_amount=100,
            )
        ],
        products=[
            ProductInput(
                product_id="P-1",
                product_name="Product",
                default_process_time_minutes=30,
            )
        ],
        production_lines=[
            ProductionLineInput(line_id="1", line_name="Line 1", is_active=False),
            ProductionLineInput(line_id="2", line_name="Line 2", is_active=True),
        ],
        product_line_capabilities=[
            ProductLineCapabilityInput(product_id="P-1", line_id="1", priority_rank=2),
        ],
        solver_config=SolverConfig(time_limit_seconds=5, num_search_workers=1),
    )

    with pytest.raises(PlanningValidationError):
        generate_production_plans(request)


def test_interchangeable_line_group_allows_peer_line_without_exact_capability() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=3),
        orders=[
            OrderInput(
                order_id="O-1",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=2),
                order_amount=100,
            )
        ],
        products=[
            ProductInput(
                product_id="P-1",
                product_name="Product",
                default_process_time_minutes=30,
            )
        ],
        production_lines=[
            ProductionLineInput(line_id="1", line_name="Line 1", is_active=False),
            ProductionLineInput(line_id="2", line_name="Line 2", is_active=True),
        ],
        product_line_capabilities=[
            ProductLineCapabilityInput(product_id="P-1", line_id="1", priority_rank=2),
        ],
        solver_config=SolverConfig(
            time_limit_seconds=5,
            num_search_workers=1,
            interchangeable_line_groups=[["1", "2"]],
        ),
    )

    result = generate_production_plans(request)
    plan = _plan_by_code(result, "DUE_DATE_OPTIMAL")

    assert plan.status in {"OPTIMAL", "FEASIBLE"}
    assert plan.schedule_items[0].line_id == "2"


def test_completed_and_in_progress_orders_are_excluded_from_new_planning() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=3),
        orders=[
            OrderInput(
                order_id="O-DONE",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=1),
                order_amount=100,
                status="COMPLETED",
            ),
            OrderInput(
                order_id="O-OPEN",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=2),
                order_amount=100,
                status="SCHEDULED",
            ),
            OrderInput(
                order_id="O-RUNNING",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=3),
                order_amount=100,
                status="IN_PROGRESS",
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
    )

    result = generate_production_plans(request)
    plan = _plan_by_code(result, "DUE_DATE_OPTIMAL")

    assert [item.order_id for item in plan.schedule_items] == ["O-OPEN"]
    assert "O-DONE" not in plan.unscheduled_orders
    assert "O-RUNNING" not in plan.unscheduled_orders


def test_line_daily_capacity_limits_scheduled_quantity() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=4),
        orders=[
            OrderInput(
                order_id="O-1",
                product_id="P-1",
                quantity=6,
                due_date=start + timedelta(hours=2),
                order_amount=100,
            ),
            OrderInput(
                order_id="O-2",
                product_id="P-1",
                quantity=6,
                due_date=start + timedelta(hours=3),
                order_amount=100,
            ),
        ],
        products=[
            ProductInput(
                product_id="P-1",
                product_name="Product",
                unit="KG",
                default_process_time_minutes=10,
            )
        ],
        production_lines=[
            ProductionLineInput(
                line_id="L-1",
                line_name="Line 1",
                is_active=True,
                max_capacity_per_day=10,
                capacity_unit="KG",
            ),
        ],
        product_line_capabilities=[
            ProductLineCapabilityInput(product_id="P-1", line_id="L-1"),
        ],
        solver_config=SolverConfig(time_limit_seconds=5, num_search_workers=1),
    )

    result = generate_production_plans(request)
    plan = _plan_by_code(result, "DUE_DATE_OPTIMAL")

    assert plan.metrics.scheduled_count == 1
    assert plan.metrics.unscheduled_count == 1


def test_oversized_single_order_uses_duration_instead_of_completion_bucket_capacity() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=8),
        orders=[
            OrderInput(
                order_id="O-LARGE",
                product_id="P-1",
                quantity=12,
                due_date=start + timedelta(hours=6),
                order_amount=100,
            )
        ],
        products=[
            ProductInput(
                product_id="P-1",
                product_name="Product",
                unit="KG",
                default_process_time_minutes=10,
            )
        ],
        production_lines=[
            ProductionLineInput(
                line_id="L-1",
                line_name="Line 1",
                is_active=True,
                max_capacity_per_day=10,
                capacity_unit="KG",
            ),
        ],
        product_line_capabilities=[
            ProductLineCapabilityInput(product_id="P-1", line_id="L-1"),
        ],
        solver_config=SolverConfig(time_limit_seconds=5, num_search_workers=1),
    )

    result = generate_production_plans(request)
    plan = _plan_by_code(result, "DUE_DATE_OPTIMAL")

    assert plan.metrics.scheduled_count == 1
    assert plan.metrics.unscheduled_count == 0
    assert any("completion-bucket capacity" in warning for warning in plan.warnings)


def test_minimum_production_quantity_is_reported_separately_from_order_quantity() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=3),
        orders=[
            OrderInput(
                order_id="O-1",
                product_id="P-1",
                order_quantity=80,
                due_date=start + timedelta(hours=2),
                order_amount=100,
            )
        ],
        products=[
            ProductInput(
                product_id="P-1",
                product_name="Product",
                min_production_quantity=100,
                default_process_time_minutes=1,
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
    item = _plan_by_code(result, "DUE_DATE_OPTIMAL").schedule_items[0]

    assert item.order_quantity == 80
    assert item.planned_production_quantity == 100
    assert item.quantity == 100


def test_no_changeover_line_with_conflicting_locked_products_is_disabled() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=4),
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
                product_name="Product 1",
                default_process_time_minutes=30,
            ),
            ProductInput(product_id="P-2", product_name="Product 2"),
            ProductInput(product_id="P-3", product_name="Product 3"),
        ],
        production_lines=[
            ProductionLineInput(
                line_id="L-1",
                line_name="Line 1",
                is_active=True,
                supports_changeover=False,
            ),
        ],
        product_line_capabilities=[
            ProductLineCapabilityInput(product_id="P-1", line_id="L-1"),
        ],
        existing_schedules=[
            ExistingScheduleInput(
                schedule_id="S-1",
                line_id="L-1",
                product_id="P-2",
                start_time=start,
                end_time=start + timedelta(minutes=30),
                plan_status="SCHEDULED",
            ),
            ExistingScheduleInput(
                schedule_id="S-2",
                line_id="L-1",
                product_id="P-3",
                start_time=start + timedelta(minutes=30),
                end_time=start + timedelta(hours=1),
                plan_status="IN_PROGRESS",
            ),
        ],
        solver_config=SolverConfig(time_limit_seconds=5, num_search_workers=1),
    )

    result = generate_production_plans(request)
    plan = _plan_by_code(result, "DUE_DATE_OPTIMAL")

    assert plan.metrics.scheduled_count == 0
    assert plan.metrics.unscheduled_count == 1
    assert any("supports_changeover is false" in warning for warning in plan.warnings)


def test_safety_stock_and_loss_rate_limit_material_usage() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=3),
        orders=[
            OrderInput(
                order_id="O-1",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=2),
                order_amount=100,
            )
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
            MaterialInput(
                material_id="M-1",
                material_name="Material",
                available_quantity=Decimal("1.1"),
                safety_stock_quantity=Decimal("0.1"),
            )
        ],
        bom_items=[
            BomItemInput(
                product_id="P-1",
                material_id="M-1",
                required_quantity_per_unit=Decimal("1.0"),
                loss_rate=Decimal("0.2"),
            )
        ],
        solver_config=SolverConfig(time_limit_seconds=5, num_search_workers=1),
    )

    result = generate_production_plans(request)
    plan = _plan_by_code(result, "DUE_DATE_OPTIMAL")

    assert plan.metrics.scheduled_count == 1
    assert plan.metrics.unscheduled_count == 0
    assert plan.metrics.total_material_shortage_quantity == 1
    assert plan.metrics.total_material_shortage_penalty_amount == 0


def test_expected_inbound_at_delays_material_usage_until_arrival() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    inbound_at = start + timedelta(hours=1)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=3),
        orders=[
            OrderInput(
                order_id="O-1",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=2),
                order_amount=100,
            )
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
            MaterialInput(
                material_id="M-1",
                material_name="Material",
                available_quantity=0,
                expected_inbound_quantity=1,
                expected_inbound_at=inbound_at,
            )
        ],
        bom_items=[
            BomItemInput(product_id="P-1", material_id="M-1", required_quantity_per_unit=1)
        ],
        solver_config=SolverConfig(time_limit_seconds=5, num_search_workers=1),
    )

    result = generate_production_plans(request)
    item = _plan_by_code(result, "DUE_DATE_OPTIMAL").schedule_items[0]

    assert item.start_time >= inbound_at


def test_zero_inventory_material_without_inbound_does_not_block_family() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=3),
        orders=[
            OrderInput(
                order_id="O-1",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=2),
                order_amount=100,
            )
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
            MaterialInput(
                material_id="M-1",
                material_name="Material",
                available_quantity=0,
            )
        ],
        bom_items=[
            BomItemInput(product_id="P-1", material_id="M-1", required_quantity_per_unit=1)
        ],
        solver_config=SolverConfig(time_limit_seconds=5, num_search_workers=1),
    )

    result = generate_production_plans(request)
    plan = _plan_by_code(result, "DUE_DATE_OPTIMAL")

    assert plan.metrics.scheduled_count == 1
    assert plan.metrics.unscheduled_count == 0


def _plan_result(
    variant_code: str,
    family: str,
    delayed_order_count: int,
    estimated_total_cost: int,
) -> PlanResult:
    return PlanResult(
        plan_family=family,
        plan_variant_code=variant_code,
        plan_variant_name=variant_code,
        status="OPTIMAL",
        schedule_items=[],
        unscheduled_orders=[],
        metrics=PlanMetrics(
            scheduled_count=3,
            unscheduled_count=0,
            delayed_order_count=delayed_order_count,
            on_time_order_count=3 - delayed_order_count,
            total_tardiness_minutes=delayed_order_count * 10,
            max_tardiness_minutes=10 if delayed_order_count else 0,
            total_scheduled_amount=100,
            on_time_amount=100,
            delayed_amount=0,
            unscheduled_amount=0,
            makespan_minutes=100,
            total_changeover_minutes=0,
            total_late_penalty_amount=estimated_total_cost,
            total_changeover_cost=0,
            estimated_total_cost=estimated_total_cost,
        ),
    )
