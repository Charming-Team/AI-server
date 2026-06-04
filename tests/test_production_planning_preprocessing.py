from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.features.production_planning.config import AmountOptimizationConfig
from app.features.production_planning.exceptions import PlanningValidationError
from app.features.production_planning.preprocessing import (
    calculate_changeover_minutes,
    calculate_horizon_minutes,
    calculate_planned_production_quantity,
    calculate_processing_duration_minutes,
    expand_interchangeable_line_capabilities,
    from_minute_offset,
    get_capable_lines,
    get_changeover_cost,
    get_changeover_minutes,
    identify_high_amount_orders,
    normalize_order_amount,
    to_minute_offset,
)
from app.features.production_planning.repositories.planning_data_repository import (
    _apply_loss_rate,
    _to_process_time_minutes,
)
from app.features.production_planning.schemas import (
    ChangeoverRuleInput,
    NormalizedLine,
    OrderInput,
    ProductInput,
    ProductionLineInput,
    ProductLineCapabilityInput,
)


def test_datetime_minute_offset_conversion() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    target = start + timedelta(minutes=75)

    assert to_minute_offset(target, start) == 75
    assert from_minute_offset(75, start) == target


def test_horizon_calculation() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)

    assert calculate_horizon_minutes(start, start + timedelta(hours=6)) == 360


def test_capable_line_lookup_is_sorted_and_active_only() -> None:
    active_lines = [
        NormalizedLine(
            ProductionLineInput(line_id="L-2", line_name="Line 2", is_active=True),
            0,
            60,
        ),
        NormalizedLine(
            ProductionLineInput(line_id="L-1", line_name="Line 1", is_active=True),
            0,
            60,
        ),
    ]
    capabilities = {
        ("P-1", "L-2"): ProductLineCapabilityInput(product_id="P-1", line_id="L-2"),
        ("P-1", "L-1"): ProductLineCapabilityInput(product_id="P-1", line_id="L-1"),
        ("P-1", "L-3"): ProductLineCapabilityInput(product_id="P-1", line_id="L-3"),
    }

    assert get_capable_lines("P-1", active_lines, capabilities) == ["L-1", "L-2"]


def test_interchangeable_line_group_expands_capability_to_active_peer_line() -> None:
    capabilities = [
        ProductLineCapabilityInput(
            product_id="P-1",
            line_id="1",
            process_time_per_unit_minutes=10,
            priority_rank=2,
        )
    ]

    expanded = expand_interchangeable_line_capabilities(
        capabilities,
        active_line_ids={"2"},
        interchangeable_line_groups=[["1", "2"]],
    )

    assert [(capability.product_id, capability.line_id) for capability in expanded] == [
        ("P-1", "2")
    ]
    assert expanded[0].priority_rank == 2


def test_duration_calculation_with_yield_and_setup() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    order = OrderInput(
        order_id="O-1",
        product_id="P-1",
        quantity=10,
        due_date=start,
        order_amount=1,
    )
    product = ProductInput(product_id="P-1", product_name="Product 1")
    capability = ProductLineCapabilityInput(
        product_id="P-1",
        line_id="L-1",
        process_time_per_unit_minutes=2,
        fixed_setup_minutes=5,
        yield_rate_scaled=5_000,
    )

    assert calculate_processing_duration_minutes(order, "L-1", capability, product) == 45


def test_repository_process_time_converts_hours_per_ton_to_minutes_per_kg() -> None:
    assert _to_process_time_minutes(Decimal("1.5")) == 0.09


def test_repository_bom_loss_rate_preserves_fractional_decimal() -> None:
    assert _apply_loss_rate(Decimal("0.6"), Decimal("0.05")) == Decimal("0.630")


def test_standard_yield_rate_and_changeover_hours_are_converted() -> None:
    capability = ProductLineCapabilityInput(
        product_id="P-1",
        line_id="L-1",
        standard_yield_rate=Decimal("0.9480"),
    )
    rule = ChangeoverRuleInput(
        from_product_id="P-1",
        to_product_id="P-2",
        line_id="L-1",
        total_changeover_time_hr=Decimal("1.5"),
    )

    assert capability.yield_rate_scaled == 9480
    assert calculate_changeover_minutes(rule) == 90


def test_planned_production_quantity_uses_minimum_production_quantity() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    order = OrderInput(
        order_id="O-1",
        product_id="P-1",
        quantity=80,
        due_date=start,
        order_amount=1,
    )
    product = ProductInput(
        product_id="P-1",
        product_name="Product 1",
        min_production_quantity=100,
    )

    assert calculate_planned_production_quantity(order, product) == 100


def test_changeover_rule_priority() -> None:
    rules = {
        ("P-1", "P-2", None): ChangeoverRuleInput(
            from_product_id="P-1",
            to_product_id="P-2",
            changeover_minutes=30,
            changeover_cost=300,
        ),
        ("P-1", "P-2", "L-1"): ChangeoverRuleInput(
            from_product_id="P-1",
            to_product_id="P-2",
            line_id="L-1",
            changeover_minutes=10,
            changeover_cost=100,
        ),
    }

    assert get_changeover_minutes("P-1", "P-2", "L-1", rules, 99) == 10
    assert get_changeover_cost("P-1", "P-2", "L-1", rules, 99) == 100
    assert get_changeover_minutes("P-1", "P-2", "L-2", rules, 99) == 30
    assert get_changeover_minutes("P-2", "P-1", "L-1", rules, 99) == 99


def test_amount_normalization_and_high_amount_policy() -> None:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    orders = [
        OrderInput(order_id="O-1", product_id="P", quantity=1, due_date=start, order_amount=100),
        OrderInput(order_id="O-2", product_id="P", quantity=1, due_date=start, order_amount=300),
        OrderInput(order_id="O-3", product_id="P", quantity=1, due_date=start, order_amount=200),
    ]

    assert normalize_order_amount(1_001, 1_000) == 2
    assert identify_high_amount_orders(orders, AmountOptimizationConfig()) == {"O-2"}
    assert identify_high_amount_orders(
        orders,
        AmountOptimizationConfig(high_amount_threshold_policy="ABOVE_AVERAGE"),
    ) == {"O-2", "O-3"}
    with pytest.raises(PlanningValidationError):
        identify_high_amount_orders(
            orders,
            AmountOptimizationConfig(high_amount_threshold_policy="CUSTOM_THRESHOLD"),
        )
