from datetime import UTC, datetime, timedelta

import pytest

from app.features.production_planning.config import AmountOptimizationConfig, SolverConfig
from app.features.production_planning.exceptions import PlanningValidationError
from app.features.production_planning.schemas import (
    BomItemInput,
    ExistingScheduleInput,
    MaterialInput,
    OrderInput,
    ProductInput,
    ProductionLineInput,
    ProductionPlanningRequest,
    ProductLineCapabilityInput,
)
from app.features.production_planning.validators import validate_request


def _base_request() -> ProductionPlanningRequest:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    return ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=8),
        orders=[
            OrderInput(
                order_id="O-1",
                product_id="P-1",
                customer_id=None,
                quantity=10,
                due_date=start + timedelta(hours=4),
                order_amount=100_000,
            )
        ],
        products=[
            ProductInput(
                product_id="P-1",
                product_name="Product 1",
                grade="A",
                default_process_time_minutes=5,
            )
        ],
        production_lines=[
            ProductionLineInput(line_id="L-1", line_name="Line 1", is_active=True)
        ],
        product_line_capabilities=[
            ProductLineCapabilityInput(product_id="P-1", line_id="L-1")
        ],
    )


def test_missing_product_capability_should_fail() -> None:
    request = _base_request()
    request.product_line_capabilities = []

    with pytest.raises(PlanningValidationError):
        validate_request(request)


def test_negative_quantity_should_fail() -> None:
    request = _base_request()
    request.orders[0].quantity = -1

    with pytest.raises(PlanningValidationError):
        validate_request(request)


def test_negative_late_penalty_amount_should_fail() -> None:
    request = _base_request()
    request.orders[0].late_penalty_amount = -1

    with pytest.raises(PlanningValidationError):
        validate_request(request)


def test_unknown_product_id_should_fail() -> None:
    request = _base_request()
    request.orders[0].product_id = "UNKNOWN"

    with pytest.raises(PlanningValidationError):
        validate_request(request)


def test_inactive_line_should_not_be_used() -> None:
    request = _base_request()
    request.production_lines[0].is_active = False

    with pytest.raises(PlanningValidationError):
        validate_request(request)


def test_existing_schedule_with_invalid_time_should_fail() -> None:
    request = _base_request()
    request.existing_schedules = [
        ExistingScheduleInput(
            schedule_id="S-1",
            line_id="L-1",
            product_id="P-1",
            start_time=request.planning_start + timedelta(hours=2),
            end_time=request.planning_start + timedelta(hours=1),
        )
    ]

    with pytest.raises(PlanningValidationError):
        validate_request(request)


def test_invalid_amount_optimization_config_should_fail() -> None:
    request = _base_request()
    request.solver_config = SolverConfig(
        amount_optimization=AmountOptimizationConfig(
            high_amount_threshold_policy="CUSTOM_THRESHOLD",
        )
    )

    with pytest.raises(PlanningValidationError):
        validate_request(request)


def test_product_unit_must_match_line_capacity_unit() -> None:
    request = _base_request()
    request.products[0].unit = "EA"
    request.production_lines[0].capacity_unit = "KG"
    request.production_lines[0].max_capacity_per_day = 100

    with pytest.raises(PlanningValidationError):
        validate_request(request)


def test_bom_unit_must_match_material_unit_when_material_unit_exists() -> None:
    request = _base_request()
    request.materials = [
        MaterialInput(
            material_id="M-1",
            material_name="Material",
            material_unit="KG",
            available_quantity=100,
        )
    ]
    request.bom_items = [
        BomItemInput(
            product_id="P-1",
            material_id="M-1",
            required_quantity_per_unit=1,
            unit="L",
        )
    ]

    with pytest.raises(PlanningValidationError):
        validate_request(request)
