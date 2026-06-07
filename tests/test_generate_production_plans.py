from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.features.production_planning.config import DueDateOptimizationConfig, SolverConfig
from app.features.production_planning.exceptions import (
    PlanningDataAccessError,
    PlanningInfeasibleError,
    PlanningValidationError,
    SolverExecutionError,
)
from app.features.production_planning.plan_comparator import compare_plans
from app.features.production_planning.production_planning_node import (
    build_adjusted_planning_request_from_bundle,
    generate_adjusted_production_plan_dashboard_response,
    generate_production_plans,
)
from app.features.production_planning.repositories.planning_data_repository import (
    PlanningDataRepository,
    PlanningInputBundle,
)
from app.features.production_planning.repositories.simulation_data_repository import (
    BaselineSimulationSnapshot,
    SimulationDataRepository,
)
from app.features.production_planning.schemas import (
    PLAN_VARIANTS,
    BomItemInput,
    ExistingScheduleInput,
    LockedPlanInput,
    LockedPlanPatchInput,
    MaterialInput,
    OrderInput,
    PlanMetrics,
    PlanningOrderPatchInput,
    PlanResult,
    ProductInput,
    ProductionLineInput,
    ProductionPlanningAdjustmentRequest,
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


def _planning_bundle_for_adjustment() -> PlanningInputBundle:
    request = _material_request(allow_unscheduled=True)
    request.orders[0].order_id = "PLAN-1"
    request.orders[0].product_id = "1"
    request.orders[1].order_id = "PLAN-2"
    request.orders[1].product_id = "1"
    request.products[0].product_id = "1"
    request.production_lines[0].line_id = "1"
    request.product_line_capabilities[0].product_id = "1"
    request.product_line_capabilities[0].line_id = "1"
    request.bom_items[0].product_id = "1"
    return PlanningInputBundle(
        orders=request.orders,
        products=request.products,
        production_lines=request.production_lines,
        product_line_capabilities=request.product_line_capabilities,
        existing_schedules=[],
        changeover_rules=[],
        material_inventories=request.materials,
        bom_items=request.bom_items,
        line_statuses=[],
        machine_statuses=[],
    )


def _plan_by_code(result, variant_code: str) -> PlanResult:
    return next(plan for plan in result.plan_results if plan.plan_variant_code == variant_code)


def test_generate_production_plans_returns_two_variants_and_metadata() -> None:
    result = generate_production_plans(_material_request(allow_unscheduled=True))

    assert len(result.plan_results) == 2
    assert len(result.adjusted_plan_candidates) == 2
    assert {plan.plan_variant_code for plan in result.plan_results} == set(PLAN_VARIANTS)
    assert {
        candidate.plan_variant_code for candidate in result.adjusted_plan_candidates
    } == set(PLAN_VARIANTS)
    assert sum(plan.plan_family == "DUE_DATE_OPTIMAL" for plan in result.plan_results) == 1
    assert sum(plan.plan_family == "AMOUNT_OPTIMAL" for plan in result.plan_results) == 1
    assert result.recommended_plan_variant_code in PLAN_VARIANTS
    assert len(result.comparison_summary.plan_rankings) == 2
    assert set(result.solver_metadata) == set(PLAN_VARIANTS)


def test_adjusted_plan_candidates_use_production_plan_response_columns() -> None:
    result = generate_production_plans(_material_request(allow_unscheduled=True))
    candidate = next(
        item
        for item in result.adjusted_plan_candidates
        if item.plan_variant_code == "DUE_DATE_OPTIMAL"
    )

    assert candidate.plans
    first_row = candidate.plans[0]
    row = first_row.model_dump()

    assert "plan_id" not in row
    assert row["plan_status"] == "SCHEDULED"
    assert row["operator_id"] in {6, 7, 8, 9, 10, 11, 13, 14, 15}
    assert row["plan_sequence"] >= 1
    assert row["estimated_duration_hr"] > 0
    assert row["updated_at"] is not None

    response = result.to_adjusted_plan_response()
    response_row = response["adjusted_plan_candidates"][0]["plans"][0]

    assert set(response) == {"adjusted_plan_candidates"}
    assert "plan_id" not in response_row
    assert response_row["plan_status"] == "SCHEDULED"
    assert response_row["operator_id"] in {6, 7, 8, 9, 10, 11, 13, 14, 15}


def test_planning_exceptions_return_compact_error_response() -> None:
    assert PlanningValidationError("계획 대상 주문이 없습니다.").to_response_error() == {
        "status": "400 BAD_REQUEST",
        "message": "계획 대상 주문이 없습니다.",
    }
    infeasible_error = PlanningInfeasibleError(
        "현재 제약 조건에서는 생산 계획을 생성할 수 없습니다."
    )
    assert infeasible_error.to_response_error() == {
        "status": "409 CONFLICT",
        "message": "현재 제약 조건에서는 생산 계획을 생성할 수 없습니다.",
    }
    solver_error = SolverExecutionError("제한 시간 내에 생산 계획을 찾지 못했습니다.")
    assert solver_error.to_response_error() == {
        "status": "408 REQUEST_TIMEOUT",
        "message": "제한 시간 내에 생산 계획을 찾지 못했습니다.",
    }
    assert PlanningDataAccessError("DB view 조회에 실패했습니다.").to_response_error() == {
        "status": "503 SERVICE_UNAVAILABLE",
        "message": "DB view 조회에 실패했습니다.",
    }


def test_adjustment_request_merges_edit_and_add_orders() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    adjustment = ProductionPlanningAdjustmentRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=8),
        edit_orders=[
            PlanningOrderPatchInput(
                order_id=1,
                product_id=1,
                order_quantity=1,
                due_date=start + timedelta(hours=3),
                contract_amount=Decimal("200.00"),
                late_penalty_amount=Decimal("30.00"),
                order_status="SCHEDULED",
                locked_plan=LockedPlanPatchInput(
                    line_id=1,
                    planned_start_at=start + timedelta(hours=1),
                    planned_end_at=start + timedelta(hours=2),
                ),
            )
        ],
        add_orders=[
            PlanningOrderPatchInput(
                order_id=900000001,
                product_id=1,
                order_quantity=1,
                due_date=start + timedelta(hours=4),
                contract_amount=Decimal("150.00"),
                late_penalty_amount=Decimal("10.00"),
                order_status="SCHEDULED",
            )
        ],
    )

    request = build_adjusted_planning_request_from_bundle(
        adjustment,
        _planning_bundle_for_adjustment(),
        SolverConfig(time_limit_seconds=5, num_search_workers=1),
        simulation_config=None,
    )
    order_by_id = {order.order_id: order for order in request.orders}

    assert order_by_id["PLAN-1"].is_locked is True
    assert order_by_id["PLAN-1"].locked_plan is not None
    assert order_by_id["PLAN-1"].order_amount == 200
    assert order_by_id["900000001"].is_locked is False
    assert order_by_id["PLAN-2"].is_locked is False


def test_adjustment_request_requires_db_numeric_field_types() -> None:
    start = "2026-05-22 09:00:00.000 +0900"
    valid_payload = {
        "planning_start": start,
        "planning_end": "2026-05-22 17:00:00.000 +0900",
        "edit_orders": [
            {
                "order_id": 1,
                "product_id": 1,
                "order_quantity": 1,
                "due_date": "2026-05-22 12:00:00.000 +0900",
                "contract_amount": Decimal("200.00"),
                "late_penalty_amount": Decimal("30.00"),
                "order_status": "SCHEDULED",
                "locked_plan": {
                    "line_id": 1,
                    "planned_start_at": "2026-05-22 10:00:00.000 +0900",
                    "planned_end_at": "2026-05-22 11:00:00.000 +0900",
                },
            }
        ],
        "add_orders": [],
    }

    parsed = ProductionPlanningAdjustmentRequest.model_validate(valid_payload)

    assert parsed.edit_orders[0].order_id == 1
    assert parsed.edit_orders[0].product_id == 1
    assert parsed.edit_orders[0].contract_amount == Decimal("200.00")
    assert parsed.edit_orders[0].late_penalty_amount == Decimal("30.00")
    assert parsed.edit_orders[0].locked_plan.line_id == 1

    invalid_payload = {
        **valid_payload,
        "edit_orders": [
            {
                **valid_payload["edit_orders"][0],
                "order_id": "1",
            }
        ],
    }
    with pytest.raises(ValueError):
        ProductionPlanningAdjustmentRequest.model_validate(invalid_payload)


def test_adjustment_empty_edit_and_add_orders_runs_full_db_replan() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    adjustment = ProductionPlanningAdjustmentRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=4),
    )

    request = build_adjusted_planning_request_from_bundle(
        adjustment,
        _planning_bundle_for_adjustment(),
        SolverConfig(time_limit_seconds=5, num_search_workers=1),
        simulation_config=None,
    )

    assert {order.order_id for order in request.orders} == {"PLAN-1", "PLAN-2"}
    assert all(not order.is_locked for order in request.orders)


def test_adjustment_add_order_overrides_db_order_when_ids_overlap() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    adjustment = ProductionPlanningAdjustmentRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=4),
        add_orders=[
            PlanningOrderPatchInput(
                order_id=1,
                product_id=1,
                order_quantity=1,
                due_date=start + timedelta(hours=3),
                contract_amount=Decimal("999.00"),
                late_penalty_amount=Decimal("77.00"),
            )
        ],
    )

    request = build_adjusted_planning_request_from_bundle(
        adjustment,
        _planning_bundle_for_adjustment(),
        SolverConfig(time_limit_seconds=5, num_search_workers=1),
        simulation_config=None,
    )
    order_by_id = {order.order_id: order for order in request.orders}

    assert order_by_id["PLAN-1"].is_locked is False
    assert order_by_id["PLAN-1"].order_amount == 999
    assert order_by_id["PLAN-1"].late_penalty_amount == 77


def test_adjusted_dashboard_response_runs_db_langgraph_formatter_pipeline(
    monkeypatch,
) -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    adjustment = ProductionPlanningAdjustmentRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=4),
    )

    def fake_load_planning_bundle(self, *args, **kwargs):
        return _planning_bundle_for_adjustment()

    def fake_load_baseline_snapshot(self, planning_start, planning_end):
        return BaselineSimulationSnapshot(
            current_plan_rows=[
                {
                    "schedule_id": 1,
                    "order_id": 1,
                    "product_id": 1,
                    "line_id": 1,
                    "start_time": planning_start,
                    "end_time": planning_start + timedelta(hours=1),
                    "plan_status": "SCHEDULED",
                }
            ],
            simulation_result_rows=[
                {
                    "created_at": planning_start - timedelta(days=1),
                    "delay_probability": 0.20,
                    "expected_delayed_order_count": 2.0,
                    "p95_tardiness_minutes": 1000.0,
                    "expected_total_risk_cost": 1_000_000.0,
                    "material_shortage_probability": 0.50,
                }
            ],
            simulation_detail_rows=[],
        )

    monkeypatch.setattr(
        PlanningDataRepository,
        "load_planning_input_bundle",
        fake_load_planning_bundle,
    )
    monkeypatch.setattr(
        SimulationDataRepository,
        "load_baseline_simulation_snapshot",
        fake_load_baseline_snapshot,
    )

    response = generate_adjusted_production_plan_dashboard_response(adjustment)

    assert response["data_sources"] == {
        "baseline": "DB_CURRENT_PLAN_AND_SIMULATION",
        "alternative": "CP_SAT_AND_SIMULATION",
    }
    assert response["planning_window"] == {
        "planning_start": start.isoformat(),
        "planning_end": (start + timedelta(hours=4)).isoformat(),
    }
    assert response["baseline"]["plans"][0]["plan_id"] == 1
    assert len(response["alternatives"]) == 2
    assert {
        "plans",
        "simulation_metrics",
        "simulation_comparison_table",
        "application_conditions",
        "selected_plan_change_schedule",
        "important_events",
        "ai_evaluation",
    }.issubset(response["alternatives"][0])
    assert "llm_evidence" not in response["alternatives"][0]
    assert "llm_prompts" not in response["alternatives"][0]


def test_adjustment_duplicate_add_orders_fail_validation() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    adjustment = ProductionPlanningAdjustmentRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=4),
        add_orders=[
            PlanningOrderPatchInput(
                order_id=900000001,
                product_id=1,
                order_quantity=1,
                due_date=start + timedelta(hours=3),
                contract_amount=Decimal("100.00"),
            ),
            PlanningOrderPatchInput(
                order_id=900000001,
                product_id=1,
                order_quantity=1,
                due_date=start + timedelta(hours=3),
                contract_amount=Decimal("100.00"),
            ),
        ],
    )

    with pytest.raises(PlanningValidationError):
        build_adjusted_planning_request_from_bundle(
            adjustment,
            _planning_bundle_for_adjustment(),
            SolverConfig(time_limit_seconds=5, num_search_workers=1),
            simulation_config=None,
        )


def test_locked_order_keeps_line_start_and_end_in_cpsat_result() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    request = _material_request(allow_unscheduled=True)
    request.planning_end = start + timedelta(hours=6)
    request.orders[0].due_date = start + timedelta(minutes=30)
    request.orders[0].is_locked = True
    request.orders[0].locked_plan = LockedPlanInput(
        line_id="L-1",
        planned_start_at=start + timedelta(hours=1, seconds=31),
        planned_end_at=start + timedelta(hours=2, minutes=30, seconds=31),
    )
    request.materials = [
        MaterialInput(material_id="M-1", material_name="Material", available_quantity=20),
    ]
    request.solver_config = SolverConfig(
        time_limit_seconds=5,
        num_search_workers=1,
        allow_unscheduled_orders=True,
    )

    result = generate_production_plans(request)

    for plan in result.plan_results:
        item = next(schedule for schedule in plan.schedule_items if schedule.order_id == "O-1")
        assert item.line_id == "L-1"
        assert item.start_time == start + timedelta(hours=1, seconds=31)
        assert item.end_time == start + timedelta(hours=2, minutes=30, seconds=31)


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
