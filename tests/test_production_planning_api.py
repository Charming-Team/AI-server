from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.routes import planning as planning_route
from app.features.production_planning.exceptions import PlanningValidationError
from app.features.production_planning.schemas import ProductionPlanningAdjustmentRequest


def _test_client() -> TestClient:
    app = FastAPI()
    app.include_router(planning_route.router, prefix="/api/v1")
    return TestClient(app)


def _fake_success_payload() -> dict:
    return {
        "planning_response": {
            "adjusted_plan_candidates": [
                {
                    "plan_variant_code": "DUE_DATE_OPTIMAL",
                    "plan_variant_name": "Due-Date Optimal",
                    "status": "FEASIBLE",
                    "plans": [],
                }
            ]
        },
        "simulation_response": {
            "generated_at": "2026-06-08T00:00:00+00:00",
            "planning_window": {
                "planning_start": "2026-05-01T09:00:00+09:00",
                "planning_end": "2026-06-09T08:59:00+09:00",
            },
            "data_sources": {
                "baseline": "DB_CURRENT_PLAN",
                "alternative": "CP_SAT_AND_SIMULATION",
            },
            "baseline": {
                "source": "DB_CURRENT_PLAN",
                "plans": [],
                "simulation_metrics": {
                    "calculation_status": "OK",
                    "missing_fields": [],
                },
                "current_state_summary": {
                    "calculation_status": "OK",
                },
                "provenance": {
                    "source": "ai_planning.v_existing_schedules_for_planning",
                    "plan_count": 0,
                },
            },
            "alternatives": [
                {
                    "plan_variant_code": "DUE_DATE_OPTIMAL",
                    "plan_variant_name": "Due-Date Optimal",
                    "status": "FEASIBLE",
                    "plans": [],
                    "simulation_metrics": {
                        "calculation_status": "OK",
                        "missing_fields": [],
                    },
                    "computed_deltas": {},
                    "simulation_comparison_table": [],
                    "application_conditions": {
                        "available_lines": [],
                        "target_products": [],
                        "applicable_period": {},
                        "unchanged_overlapping_orders": [],
                    },
                    "selected_plan_change_schedule": [],
                    "important_events": [],
                    "ai_evaluation": {
                        "status": "FAILED",
                        "current_state_summary": {
                            "risk_analysis_text": "정보 없음",
                        },
                        "risk_interpretation": {
                            "text": "정보 없음",
                        },
                        "ai_recommendation": {
                            "summary_text": "정보 없음",
                            "reasons": ["정보 없음"],
                        },
                        "recommendation_level": "INFO_ONLY",
                        "recommendation_grade_label": "보통",
                        "recommendation_grade_basis": ["정보 없음"],
                    },
                }
            ],
            "warnings": [],
        },
    }


def test_planning_api_returns_planning_and_simulation_payloads(monkeypatch) -> None:
    captured = {}

    def fake_generate(request: ProductionPlanningAdjustmentRequest) -> dict:
        captured["request"] = request
        return _fake_success_payload()

    monkeypatch.setattr(
        planning_route,
        "generate_adjusted_production_plan_api_response",
        fake_generate,
    )

    client = _test_client()
    response = client.post(
        "/api/v1/planning",
        json={
            "planning_start": "2026-05-01 09:00:00.000 +0900",
            "planning_end": "2026-06-09 08:59:00.000 +0900",
            "edit_orders": [],
            "add_orders": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["planning_response"]["adjusted_plan_candidates"][0][
        "plan_variant_code"
    ] == "DUE_DATE_OPTIMAL"
    assert payload["simulation_response"]["baseline"]["source"] == "DB_CURRENT_PLAN"
    assert payload["simulation_response"]["data_sources"] == {
        "baseline": "DB_CURRENT_PLAN",
        "alternative": "CP_SAT_AND_SIMULATION",
    }
    assert payload["simulation_response"]["alternatives"][0][
        "plan_variant_code"
    ] == "DUE_DATE_OPTIMAL"
    assert isinstance(captured["request"], ProductionPlanningAdjustmentRequest)


def test_planning_api_returns_compact_planning_error(monkeypatch) -> None:
    def fake_generate(request: ProductionPlanningAdjustmentRequest) -> dict:
        raise PlanningValidationError("계획 요청 값이 올바르지 않습니다.")

    monkeypatch.setattr(
        planning_route,
        "generate_adjusted_production_plan_api_response",
        fake_generate,
    )

    client = _test_client()
    response = client.post(
        "/api/v1/planning",
        json={
            "planning_start": "2026-05-01 09:00:00.000 +0900",
            "planning_end": "2026-06-09 08:59:00.000 +0900",
            "edit_orders": [],
            "add_orders": [],
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "status": "400 BAD_REQUEST",
        "message": "계획 요청 값이 올바르지 않습니다.",
    }


def test_planning_health_route() -> None:
    client = _test_client()

    response = client.get("/api/v1/planning/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "feature": "production-planning",
    }


def test_planning_router_is_registered_in_api_router() -> None:
    router_source = Path("app/api/router.py").read_text(encoding="utf-8")

    assert "planning_router" in router_source
    assert "api_router.include_router(planning_router)" in router_source


def test_planning_openapi_uses_executable_request_examples() -> None:
    client = _test_client()

    schema = client.get("/openapi.json").json()
    request_body = schema["paths"]["/api/v1/planning"]["post"]["requestBody"]
    examples = request_body["content"]["application/json"]["examples"]
    example = examples["edit_and_add_orders"]["value"]

    assert example["planning_start"] == "2026-05-01 09:00:00.000 +0900"
    assert example["edit_orders"][0]["order_id"] == 399
    assert example["edit_orders"][0]["product_id"] == 10
    assert example["edit_orders"][0]["locked_plan"]["line_id"] == 6
    assert example["edit_orders"][0]["contract_amount"] == "30752426.00"
    assert example["add_orders"][0]["order_id"] == 900000001
