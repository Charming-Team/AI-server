from datetime import UTC, datetime, timedelta
from decimal import Decimal
from random import Random

from app.features.production_planning.config import SimulationConfig, SolverConfig
from app.features.production_planning.preprocessing import normalize_request
from app.features.production_planning.production_planning_node import generate_production_plans
from app.features.production_planning.repositories.simulation_data_repository import (
    SimulationDataRepository,
    SimulationInputBundle,
)
from app.features.production_planning.schemas import (
    BomItemInput,
    MaterialInput,
    OrderInput,
    ProductInput,
    ProductionLineInput,
    ProductionPlanningRequest,
    ProductLineCapabilityInput,
)
from app.features.production_planning.simulation.des import initialize_simulation_state
from app.features.production_planning.simulation.sampling import (
    build_empirical_sampling_distributions,
)
from app.features.production_planning.simulation.sampling import (
    build_sampling_distributions as build_fallback_sampling_distributions,
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

    assert len(result.simulation_results) == 2
    assert result.simulation_results[0].sampling_summary
    assert "delay_probability" in result.simulation_results[0].sampling_summary
    assert result.simulation_comparison_summary is not None
    assert result.recommended_due_date_plan is not None
    assert result.recommended_cost_plan is not None
    assert result.recommended_due_date_plan.plan_family == "DUE_DATE_OPTIMAL"
    assert result.recommended_cost_plan.plan_family == "AMOUNT_OPTIMAL"


def test_simulation_result_reports_event_response_values() -> None:
    request = _request()
    request.simulation_config = SimulationConfig(
        num_iterations=3,
        random_seed=7,
        delay_probability=1.0,
        min_delay_minutes=10,
        max_delay_minutes=10,
        setup_delay_probability=1.0,
        min_setup_delay_minutes=5,
        max_setup_delay_minutes=5,
        machine_breakdown_probability=1.0,
        min_machine_repair_minutes=15,
        max_machine_repair_minutes=15,
    )

    result = generate_production_plans(request)

    first_result = result.simulation_results[0]
    assert first_result.sampling_summary["delay_probability"] == 1.0
    assert first_result.event_timeline
    assert {
        "event_time",
        "event",
        "order_amount",
        "delivery_delay_days",
        "loss_amount",
    }.issubset(first_result.event_timeline[0])
    assert any(
        event["event"] == "LINE_CHANGE_DELAY"
        and event["occurrence_count"] > 0
        and "avg_risk_cost_when_event_occurs" in event
        for event in first_result.event_summary
    )
    assert {event["event"] for event in first_result.event_summary} >= {
        "MACHINE_BREAKDOWN",
        "SETUP_DELAY",
    }


def test_empirical_sampling_distributions_use_production_history() -> None:
    request = _request()
    raw_data = SimulationInputBundle(
        production_results=[
            {
                "product_id": "P-1",
                "line_id": "L-1",
                "product_category": None,
                "actual_duration_hr": 1.0,
                "actual_setup_time_hr": 0.1,
                "actual_delay_hr": 0.5,
                "is_delayed": True,
                "yield_rate": 0.95,
                "actual_quantity": 100,
                "defect_quantity": 2,
            }
        ],
        production_result_causes=[
            {"product_id": "P-1", "line_id": "L-1", "cause_type": "MACHINE_DELAY"}
        ],
        changeover_sequences=[],
        orders_history=[],
        production_plans_history=[],
        ai_prediction_results=[],
        ai_prediction_causes=[],
        simulation_results_history=[],
        simulation_details_history=[],
    )

    distributions = build_empirical_sampling_distributions(
        raw_data,
        normalize_request(request),
        SimulationConfig(min_samples=1),
    )

    assert ("P-1", "L-1") in distributions.duration_params_by_context
    assert distributions.cause_weights_by_context[("P-1", "L-1")] == {
        "MACHINE_ABNORMAL": 1.0
    }


def test_simulation_samples_material_inbound_time_instead_of_using_expected_date() -> None:
    request = _request()
    expected_inbound_at = request.planning_start + timedelta(days=3)
    request.materials = [
        MaterialInput(
            material_id="M-1",
            material_name="Material",
            available_quantity=Decimal("0"),
            expected_inbound_quantity=Decimal("10"),
            expected_inbound_at=expected_inbound_at,
        )
    ]
    request.bom_items = [
        BomItemInput(
            product_id="P-1",
            material_id="M-1",
            required_quantity_per_unit=Decimal("1"),
        )
    ]
    data = normalize_request(request)
    distributions = build_fallback_sampling_distributions(
        [],
        data,
        SimulationConfig(
            material_inbound_probability=1.0,
            min_material_inbound_delay_minutes=0,
            max_material_inbound_delay_minutes=0,
        ),
    )

    state = initialize_simulation_state(data, distributions, Random(7))

    assert state.inbound_events[0][0] == request.planning_start
    assert state.inbound_events[0][0] != expected_inbound_at


def test_simulation_tracks_material_shortage_without_penalty_cost() -> None:
    request = _request()
    request.simulation_config = SimulationConfig(
        num_iterations=1,
        random_seed=7,
        delay_probability=0.0,
        setup_delay_probability=0.0,
        machine_breakdown_probability=0.0,
        line_change_delay_probability=0.0,
        material_shortage_delay_minutes=10,
    )
    request.materials = [
        MaterialInput(
            material_id="M-1",
            material_name="Material",
            available_quantity=Decimal("0"),
        )
    ]
    request.bom_items = [
        BomItemInput(
            product_id="P-1",
            material_id="M-1",
            required_quantity_per_unit=Decimal("1"),
        )
    ]

    result = generate_production_plans(request)
    simulation_result = result.simulation_results[0]

    assert simulation_result.expected_material_shortage_count > 0
    assert simulation_result.expected_material_shortage_penalty_amount == 0
    assert simulation_result.expected_total_risk_cost == (
        simulation_result.expected_late_penalty_amount
        + simulation_result.expected_changeover_cost
    )


def test_order_estimated_duration_excludes_material_shortage_delay() -> None:
    request = _request()
    request.simulation_config = SimulationConfig(
        num_iterations=1,
        random_seed=7,
        duration_variation_ratio=0.0,
        delay_probability=0.0,
        setup_delay_probability=0.0,
        machine_breakdown_probability=0.0,
        line_change_delay_probability=0.0,
        material_shortage_delay_minutes=600,
    )
    request.materials = [
        MaterialInput(
            material_id="M-1",
            material_name="Material",
            available_quantity=Decimal("0"),
        )
    ]
    request.bom_items = [
        BomItemInput(
            product_id="P-1",
            material_id="M-1",
            required_quantity_per_unit=Decimal("1"),
        )
    ]

    result = generate_production_plans(request)
    simulation_result = result.simulation_results[0]

    assert simulation_result.expected_material_shortage_count > 0
    assert simulation_result.order_duration_estimates
    assert {
        row["estimated_duration_minutes"]
        for row in simulation_result.order_duration_estimates
    } == {30.0}
    assert {
        row["estimated_duration_hr"]
        for row in simulation_result.order_duration_estimates
    } == {0.5}
    assert {
        row["order_amount"]
        for row in simulation_result.order_duration_estimates
    } == {100_000, 80_000}


def test_baseline_simulation_selection_prefers_window_overlapping_details(monkeypatch) -> None:
    repository = SimulationDataRepository()
    start = datetime(2026, 5, 1, tzinfo=UTC)
    end = datetime(2026, 6, 8, tzinfo=UTC)
    calls = []

    def fake_execute_auxiliary_query(conn, query, view, params=None):
        calls.append({"query": str(query), "params": params, "view": view})
        return [{"simulation_id": 11}]

    monkeypatch.setattr(
        repository,
        "_execute_auxiliary_query",
        fake_execute_auxiliary_query,
    )

    warnings = []
    rows = repository._get_latest_baseline_simulation_result_rows(
        object(),
        start,
        end,
        warnings,
    )

    assert rows == [{"simulation_id": 11}]
    assert warnings == []
    assert len(calls) == 1
    assert "v_schedule_simulation_details_history_for_sampling" in calls[0]["query"]
    assert calls[0]["params"] == {"planning_start": start, "planning_end": end}


def test_baseline_simulation_selection_falls_back_with_warning(monkeypatch) -> None:
    repository = SimulationDataRepository()
    start = datetime(2026, 5, 1, tzinfo=UTC)
    end = datetime(2026, 6, 8, tzinfo=UTC)
    calls = []

    def fake_execute_auxiliary_query(conn, query, view, params=None):
        calls.append({"query": str(query), "params": params, "view": view})
        if len(calls) == 1:
            return []
        return [{"simulation_id": 99}]

    monkeypatch.setattr(
        repository,
        "_execute_auxiliary_query",
        fake_execute_auxiliary_query,
    )

    warnings = []
    rows = repository._get_latest_baseline_simulation_result_rows(
        object(),
        start,
        end,
        warnings,
    )

    assert rows == [{"simulation_id": 99}]
    assert len(calls) == 2
    assert warnings == [
        "No baseline simulation result overlaps the planning window; falling back to latest result."
    ]
