import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.features.production_planning.exceptions import PlanningValidationError
from app.features.production_planning.json_formatter import (
    build_dashboard_response,
    build_evaluation_draft_prompt,
    build_json_normalization_prompt,
    generate_ai_evaluation_from_evidence,
    validate_evaluation_draft,
    validate_final_ai_evaluation,
)
from app.features.production_planning.schemas import (
    AdjustedPlanCandidate,
    AdjustedPlanRow,
    PlanComparisonSummary,
    PlanMetrics,
    PlanRankingItem,
    PlanResult,
    PlanSimulationResult,
    ProductionPlanningResult,
    ScheduleItem,
)


def _simulation_result() -> PlanSimulationResult:
    return PlanSimulationResult(
        plan_variant_code="DUE_DATE_OPTIMAL",
        plan_family="DUE_DATE_OPTIMAL",
        iterations=100,
        delay_probability=0.10,
        expected_delayed_order_count=2.0,
        expected_tardiness_minutes=250.0,
        p50_tardiness_minutes=100.0,
        p90_tardiness_minutes=350.0,
        p95_tardiness_minutes=400.0,
        expected_late_penalty_amount=500_000.0,
        p95_late_penalty_amount=700_000.0,
        expected_changeover_cost=100_000.0,
        expected_material_shortage_penalty_amount=0.0,
        expected_total_risk_cost=600_000.0,
        material_shortage_probability=0.30,
        expected_material_shortage_count=1.0,
        expected_material_shortage_quantity=10.0,
        expected_yield_rate=0.97,
        expected_defect_quantity=3.0,
        top_delay_causes=[{"cause": "MACHINE_ABNORMAL", "count": 3}],
        order_duration_estimates=[],
        sampling_summary={},
        event_summary=[
            {
                "event": "ORDER_COMPLETED",
                "occurrence_count": 20,
                "scenario_count": 10,
                "scenario_probability": 1.0,
                "expected_count_per_iteration": 2.0,
                "avg_tardiness_minutes_when_event_occurs": 0.0,
                "avg_late_penalty_amount_when_event_occurs": 0.0,
                "avg_material_shortage_penalty_when_event_occurs": 0.0,
                "avg_risk_cost_when_event_occurs": 0.0,
            },
            {
                "event": "MACHINE_BREAKDOWN",
                "occurrence_count": 5,
                "scenario_count": 4,
                "scenario_probability": 0.4,
                "expected_count_per_iteration": 0.5,
                "avg_tardiness_minutes_when_event_occurs": 200.0,
                "avg_late_penalty_amount_when_event_occurs": 500_000.0,
                "avg_material_shortage_penalty_when_event_occurs": 0.0,
                "avg_risk_cost_when_event_occurs": 600_000.0,
            },
        ],
        event_timeline=[
            {
                "event_time": datetime(2026, 5, 1, 10, tzinfo=UTC),
                "event": "MACHINE_BREAKDOWN",
                "order_id": "PLAN-1",
                "line_id": "1",
                "delivery_delay_minutes": 60,
                "loss_amount": 1000,
            }
        ],
        scenario_summary={},
    )


def _planning_result() -> ProductionPlanningResult:
    start = datetime(2026, 5, 1, 9, tzinfo=UTC)
    schedule_item = ScheduleItem(
        order_id="PLAN-1",
        product_id="1",
        line_id="1",
        start_time=start,
        end_time=start + timedelta(hours=1),
        quantity=100,
        order_quantity=100,
        planned_production_quantity=100,
        order_amount=1_000_000,
        tardiness_minutes=0,
        status="ON_TIME",
    )
    metrics = PlanMetrics(
        scheduled_count=1,
        unscheduled_count=0,
        delayed_order_count=0,
        on_time_order_count=1,
        total_tardiness_minutes=0,
        max_tardiness_minutes=0,
        total_scheduled_amount=1_000_000,
        on_time_amount=1_000_000,
        delayed_amount=0,
        unscheduled_amount=0,
        makespan_minutes=60,
        total_changeover_minutes=0,
    )
    plan = PlanResult(
        plan_family="DUE_DATE_OPTIMAL",
        plan_variant_code="DUE_DATE_OPTIMAL",
        plan_variant_name="Due-Date Optimal",
        status="FEASIBLE",
        schedule_items=[schedule_item],
        unscheduled_orders=[],
        metrics=metrics,
    )
    candidate = AdjustedPlanCandidate(
        plan_variant_code="DUE_DATE_OPTIMAL",
        plan_variant_name="Due-Date Optimal",
        status="FEASIBLE",
        plans=[
            AdjustedPlanRow(
                order_id="PLAN-1",
                product_id="1",
                line_id="1",
                operator_id=6,
                planned_start_at=start,
                planned_end_at=start + timedelta(hours=1),
                estimated_duration_hr=1.0,
                planned_quantity=100,
                plan_sequence=1,
                plan_status="SCHEDULED",
                updated_at=start,
            )
        ],
    )
    return ProductionPlanningResult(
        adjusted_plan_candidates=[candidate],
        plan_results=[plan],
        recommended_plan_variant_code="DUE_DATE_OPTIMAL",
        comparison_summary=PlanComparisonSummary(
            recommended_plan_variant_code="DUE_DATE_OPTIMAL",
            recommendation_reason="test",
            plan_rankings=[
                PlanRankingItem(
                    rank=1,
                    plan_variant_code="DUE_DATE_OPTIMAL",
                    plan_family="DUE_DATE_OPTIMAL",
                    plan_variant_name="Due-Date Optimal",
                    status="FEASIBLE",
                    delayed_order_count=0,
                    total_tardiness_minutes=0,
                    total_late_penalty_amount=0,
                    total_changeover_cost=0,
                    estimated_total_cost=0,
                    unscheduled_count=0,
                    recommendation_note="test",
                )
            ],
        ),
        simulation_results=[_simulation_result()],
        solver_metadata={},
    )


def test_dashboard_response_computes_deltas_and_filters_important_events() -> None:
    response = build_dashboard_response(
        _planning_result(),
        include_full_event_timeline=True,
        include_internal_debug_payloads=True,
        baseline_plans=[
            {
                "schedule_id": 10,
                "order_id": 1,
                "product_id": 1,
                "line_id": 1,
                "start_time": datetime(2026, 5, 1, 9, tzinfo=UTC),
                "end_time": datetime(2026, 5, 1, 10, tzinfo=UTC),
                "plan_status": "SCHEDULED",
            }
        ],
        baseline_simulation_rows=[
            {
                "created_at": datetime(2026, 4, 30, tzinfo=UTC),
                "delay_probability": 0.20,
                "expected_delayed_order_count": 5.0,
                "p95_tardiness_minutes": 1000.0,
                "expected_total_risk_cost": 1_000_000.0,
                "material_shortage_probability": 0.50,
            }
        ],
    )

    alternative = response["alternatives"][0]

    assert response["baseline"]["plans"][0]["plan_id"] == 10
    assert alternative["computed_deltas"]["delay_probability_reduction_percent"] == 50.0
    assert alternative["computed_deltas"]["expected_delayed_order_reduction"] == 3.0
    assert alternative["computed_deltas"]["p95_tardiness_reduction_minutes"] == 600.0
    assert alternative["computed_deltas"]["risk_cost_saving_amount"] == "400000.00"
    assert alternative["computed_deltas"]["risk_cost_saving_percent"] == 40.0
    assert {event["event"] for event in alternative["important_events"]} == {
        "MACHINE_BREAKDOWN"
    }
    assert alternative["full_event_timeline_included"] is True
    assert alternative["event_timeline"][0]["event_time"] == "2026-05-01T10:00:00+00:00"
    assert "evaluation_draft" in alternative["llm_prompts"]
    assert "json_normalization_template" in alternative["llm_prompts"]
    json.dumps(response, ensure_ascii=False)


def test_dashboard_response_builds_baseline_comparison_from_prediction_and_detail() -> None:
    start = datetime(2026, 5, 1, 9, tzinfo=UTC)
    response = build_dashboard_response(
        _planning_result(),
        baseline_plans=[
            {
                "schedule_id": 10,
                "order_id": 1,
                "product_id": 1,
                "line_id": 1,
                "start_time": start,
                "end_time": start + timedelta(hours=1),
                "plan_status": "SCHEDULED",
            }
        ],
        baseline_simulation_rows=[
            {
                "simulation_id": 10,
                "simulation_type": "DUE_DATE_OPTIMIZATION",
                "simulation_name": "납기일 최적화 기준",
                "before_avg_line_utilization_rate": Decimal("0.72"),
                "before_bottleneck_line_id": 1,
                "cost_change_amount": Decimal("620000.00"),
                "created_at": start - timedelta(days=1),
            }
        ],
        baseline_simulation_detail_rows=[
            {
                "simulation_id": 10,
                "order_id": 1,
                "plan_id": 10,
                "before_line_id": 1,
                "before_sequence": 3,
                "before_start_at": start,
                "before_end_at": start + timedelta(days=2),
                "expected_completion_date": start.date(),
                "before_quantity": 100,
                "after_line_id": 2,
                "after_sequence": 1,
                "after_start_at": start - timedelta(hours=1),
                "after_end_at": start + timedelta(hours=1),
                "after_quantity": 100,
                "after_is_delayed": False,
                "change_reason": "과거 대응안",
            }
        ],
        baseline_prediction_rows=[
            {
                "prediction_id": 99,
                "order_id": 1,
                "plan_id": 10,
                "product_id": 1,
                "delay_probability": Decimal("0.8"),
                "predicted_delay_days": Decimal("2.0"),
                "predicted_at": start - timedelta(hours=1),
            }
        ],
        products=[
            {
                "product_id": 1,
                "product_code": "P-1",
                "product_name": "Product",
                "unit": "KG",
            }
        ],
        production_lines=[
            {"line_id": 1, "line_code": "L-1", "line_name": "Line 1"},
        ],
        product_line_capabilities=[
            {"product_id": 1, "line_id": 1},
        ],
    )

    baseline = response["baseline"]
    alternative = response["alternatives"][0]

    assert baseline["current_state_summary"]["expected_delay_days"] == 2.0
    assert baseline["current_state_summary"]["delay_risk_order_count"] == 1
    assert baseline["current_state_summary"]["delivery_fulfillment_rate_percent"] == 0.0
    assert baseline["current_state_summary"]["avg_line_utilization_percent"] == 72.0
    assert baseline["provenance"]["historical_action_cost_change_amount"] == "620000.00"
    assert baseline["historical_applied_result"]["details"][0]["after_line_id"] == 2
    assert alternative["computed_deltas"]["expected_delay_days_reduction"] == 1.83
    assert "expected_delay_days" in {
        row["metric_code"] for row in alternative["simulation_comparison_table"]
    }
    assert "material_shortage_quantity" not in {
        row["metric_code"] for row in alternative["simulation_comparison_table"]
    }
    assert alternative["selected_plan_change_schedule"][0]["order_id"] == 1
    assert alternative["selected_plan_change_schedule"][0]["sequence_change"] == "3 -> 1"
    assert alternative["application_conditions"]["target_products"][0]["product_id"] == 1


def test_dashboard_response_populates_ai_evaluation_with_configured_llm() -> None:
    class FakeLlmClient:
        def __init__(self) -> None:
            self.prompts = []
            self.answers = [
                {
                    "risk_analysis_text": "위험이 낮습니다.",
                    "risk_interpretation_text": "안정적입니다.",
                    "recommendation_summary_text": "추천합니다.",
                    "recommendation_reasons": ["근거가 있습니다."],
                    "recommendation_cautions": ["추가 확인이 필요합니다."],
                    "final_action": "계획을 검토하세요.",
                    "recommendation_level": "RECOMMEND",
                },
                {
                    "current_state_summary": {
                        "risk_analysis_text": "위험이 낮습니다."
                    },
                    "risk_interpretation": {"text": "안정적입니다."},
                    "ai_recommendation": {
                        "summary_text": "추천합니다.",
                        "reasons": ["근거가 있습니다."],
                        "cautions": ["추가 확인이 필요합니다."],
                        "final_action": "계획을 검토하세요.",
                    },
                    "recommendation_level": "RECOMMEND",
                },
            ]

        async def generate_completion(self, prompt):
            self.prompts.append(prompt)
            return SimpleNamespace(answer=json.dumps(self.answers.pop(0), ensure_ascii=False))

    fake_client = FakeLlmClient()
    response = build_dashboard_response(
        _planning_result(),
        enable_llm_evaluation=True,
        settings=Settings(
            llm_enabled=True,
            llm_provider="openai_compatible",
            llm_base_url="http://llm.test/v1",
            llm_model="test-model",
        ),
        llm_client=fake_client,
    )

    evaluation = response["alternatives"][0]["ai_evaluation"]

    assert evaluation["status"] == "COMPLETED"
    assert evaluation["recommendation_level"] == "RECOMMEND"
    assert evaluation["ai_recommendation"]["summary_text"] == "추천합니다."
    assert len(fake_client.prompts) == 2
    assert "Output JSON Schema" in fake_client.prompts[0].user_prompt
    assert "Final Output JSON Schema" in fake_client.prompts[1].user_prompt


def test_dashboard_response_runs_json_normalization_for_each_variant() -> None:
    class FakeLlmClient:
        def __init__(self) -> None:
            self.prompts = []

        async def generate_completion(self, prompt):
            self.prompts.append(prompt)
            if "Final Output JSON Schema" in prompt.user_prompt:
                payload = {
                    "current_state_summary": {"risk_analysis_text": "정보 없음"},
                    "risk_interpretation": {"text": "정보 없음"},
                    "ai_recommendation": {
                        "summary_text": "정보 없음",
                        "reasons": ["정보 없음"],
                        "cautions": ["정보 없음"],
                        "final_action": "정보 없음",
                    },
                    "recommendation_level": "INFO_ONLY",
                    "recommendation_grade_label": "보통",
                    "recommendation_grade_basis": ["정보 없음"],
                }
            else:
                payload = {
                    "risk_analysis_text": "정보 없음",
                    "risk_interpretation_text": "정보 없음",
                    "recommendation_summary_text": "정보 없음",
                    "recommendation_reasons": ["정보 없음"],
                    "recommendation_cautions": ["정보 없음"],
                    "final_action": "정보 없음",
                    "recommendation_level": "INFO_ONLY",
                    "recommendation_grade_label": "보통",
                    "recommendation_grade_basis": ["정보 없음"],
                }
            return SimpleNamespace(answer=json.dumps(payload, ensure_ascii=False))

    result = _planning_result()
    amount_candidate = result.adjusted_plan_candidates[0].model_copy(
        update={
            "plan_variant_code": "AMOUNT_OPTIMAL",
            "plan_variant_name": "Amount Optimal",
        }
    )
    amount_plan = result.plan_results[0].model_copy(
        update={
            "plan_family": "AMOUNT_OPTIMAL",
            "plan_variant_code": "AMOUNT_OPTIMAL",
            "plan_variant_name": "Amount Optimal",
        }
    )
    amount_simulation = result.simulation_results[0].model_copy(
        update={
            "plan_family": "AMOUNT_OPTIMAL",
            "plan_variant_code": "AMOUNT_OPTIMAL",
        }
    )
    result = result.model_copy(
        update={
            "adjusted_plan_candidates": [
                result.adjusted_plan_candidates[0],
                amount_candidate,
            ],
            "plan_results": [result.plan_results[0], amount_plan],
            "simulation_results": [result.simulation_results[0], amount_simulation],
        }
    )

    fake_client = FakeLlmClient()
    response = build_dashboard_response(
        result,
        enable_llm_evaluation=True,
        settings=Settings(
            llm_enabled=True,
            llm_provider="openai_compatible",
            llm_base_url="http://llm.test/v1",
            llm_model="test-model",
        ),
        llm_client=fake_client,
    )

    amount = response["alternatives"][1]

    assert len(fake_client.prompts) == 4
    assert "Final Output JSON Schema" in fake_client.prompts[3].user_prompt
    assert amount["plan_variant_code"] == "AMOUNT_OPTIMAL"
    assert amount["ai_evaluation"]["status"] == "COMPLETED"


def test_dashboard_response_keeps_fallback_when_llm_is_disabled() -> None:
    response = build_dashboard_response(
        _planning_result(),
        enable_llm_evaluation=True,
        settings=Settings(llm_enabled=False),
    )

    evaluation = response["alternatives"][0]["ai_evaluation"]

    assert evaluation["status"] == "FAILED"
    assert "비활성화" in evaluation["ai_recommendation"]["summary_text"]


def test_two_prompt_builders_insert_evidence_and_draft_json() -> None:
    evidence = {
        "computed_deltas": {"risk_cost_saving_percent": 40.0},
        "important_events": [{"event": "MACHINE_BREAKDOWN"}],
    }
    draft = {
        "risk_analysis_text": "위험 비용 절감률은 40.0입니다.",
        "risk_interpretation_text": "MACHINE_BREAKDOWN 영향이 있습니다.",
        "recommendation_summary_text": "추천합니다.",
        "recommendation_reasons": ["근거가 있습니다."],
        "recommendation_cautions": [],
        "final_action": "검토하세요.",
        "recommendation_level": "RECOMMEND",
    }

    first_prompt = build_evaluation_draft_prompt(evidence)
    second_prompt = build_json_normalization_prompt(evidence, draft)

    assert "Output JSON Schema" in first_prompt.user_prompt
    assert '"risk_cost_saving_percent": 40.0' in first_prompt.user_prompt
    assert "Final Output JSON Schema" in second_prompt.user_prompt
    assert '"recommendation_level": "RECOMMEND"' in second_prompt.user_prompt
    assert "비판하는 글이 아니라 추천 이유" in first_prompt.user_prompt
    assert "비판이 아니라 실행 전 확인사항" in second_prompt.user_prompt
    assert "AMOUNT_OPTIMAL" in first_prompt.user_prompt
    assert "가격/비용 관점의 정량 수치" in first_prompt.user_prompt
    assert "가격/비용 정량 수치" in second_prompt.user_prompt


def test_dashboard_response_hides_missing_value_tokens_from_llm_evidence() -> None:
    response = build_dashboard_response(
        _planning_result(),
        include_internal_debug_payloads=True,
        baseline_simulation_rows=[{"delay_probability": 0.20}],
    )

    evidence = response["alternatives"][0]["llm_evidence"]
    prompt = response["alternatives"][0]["llm_prompts"]["evaluation_draft"]
    evidence_text = json.dumps(evidence, ensure_ascii=False)
    prompt_text = json.dumps(prompt, ensure_ascii=False)

    assert "MISSING_VALUE" not in evidence_text
    assert "missing_fields" not in evidence_text
    assert "calculation_status" not in evidence_text
    assert "MISSING_VALUE" not in prompt_text


def test_ai_evaluation_returns_failed_evaluation_on_grounding_error() -> None:
    class FakeLlmClient:
        async def generate_completion(self, prompt):
            payload = {
                "risk_analysis_text": "근거에 없는 41.0 값을 생성했습니다.",
                "risk_interpretation_text": "정보 없음",
                "recommendation_summary_text": "정보 없음",
                "recommendation_reasons": ["정보 없음"],
                "recommendation_cautions": ["정보 없음"],
                "final_action": "정보 없음",
                "recommendation_level": "INFO_ONLY",
            }
            return SimpleNamespace(answer=json.dumps(payload, ensure_ascii=False))

    evaluation = asyncio.run(
        generate_ai_evaluation_from_evidence(
            {
                "baseline": {
                    "metrics": {
                        "calculation_status": "MISSING_VALUE",
                        "missing_fields": ["total_risk_cost"],
                        "total_risk_cost": None,
                    }
                },
                "alternative": {
                    "plan_variant_code": "AMOUNT_OPTIMAL",
                    "metrics": {"total_risk_cost": "4286442.79"},
                },
                "computed_deltas": {"risk_cost_saving_percent": None},
                "important_events": [],
            },
            settings=Settings(
                llm_enabled=True,
                llm_provider="openai_compatible",
                llm_base_url="http://llm.test/v1",
                llm_model="test-model",
            ),
            llm_client=FakeLlmClient(),
        )
    )

    assert evaluation.status == "FAILED"
    assert "LLM output contains numbers" in evaluation.ai_recommendation.summary_text


def test_llm_draft_validation_rejects_numbers_not_in_evidence() -> None:
    evidence = {
        "computed_deltas": {"risk_cost_saving_percent": 40.0},
        "important_events": [{"event": "MACHINE_BREAKDOWN"}],
    }
    payload = {
        "risk_analysis_text": "위험 비용 절감률은 41.0입니다.",
        "risk_interpretation_text": "MACHINE_BREAKDOWN 영향이 있습니다.",
        "recommendation_summary_text": "추천합니다.",
        "recommendation_reasons": ["근거가 있습니다."],
        "recommendation_cautions": [],
        "final_action": "검토하세요.",
        "recommendation_level": "RECOMMEND",
    }

    with pytest.raises(PlanningValidationError):
        validate_evaluation_draft(payload, evidence)


def test_llm_draft_validation_allows_comma_formatted_evidence_numbers() -> None:
    evidence = {
        "computed_deltas": {
            "risk_cost_saving_amount": "11034336.57",
            "cost_per_order": 146881.44,
        },
        "important_events": [],
    }
    payload = {
        "risk_analysis_text": "위험 비용은 11,034,336.57원, 건당 비용은 146,881.44원입니다.",
        "risk_interpretation_text": "근거 숫자만 사용했습니다.",
        "recommendation_summary_text": "추천합니다.",
        "recommendation_reasons": ["근거가 있습니다."],
        "recommendation_cautions": [],
        "final_action": "검토하세요.",
        "recommendation_level": "RECOMMEND",
    }

    draft = validate_evaluation_draft(payload, evidence)

    assert draft.recommendation_level == "RECOMMEND"


def test_final_ai_evaluation_validation_rejects_unknown_event_tokens() -> None:
    evidence = {
        "computed_deltas": {"risk_cost_saving_percent": 40.0},
        "important_events": [{"event": "MACHINE_BREAKDOWN"}],
    }
    draft = {
        "risk_analysis_text": "위험 비용 절감률은 40.0입니다.",
        "risk_interpretation_text": "MACHINE_BREAKDOWN 영향이 있습니다.",
        "recommendation_summary_text": "추천합니다.",
        "recommendation_reasons": ["근거가 있습니다."],
        "recommendation_cautions": [],
        "final_action": "검토하세요.",
        "recommendation_level": "RECOMMEND",
    }
    payload = {
        "current_state_summary": {
            "risk_analysis_text": "FIRE_EVENT 영향이 있습니다."
        },
        "risk_interpretation": {"text": "MACHINE_BREAKDOWN 영향이 있습니다."},
        "ai_recommendation": {
            "summary_text": "추천합니다.",
            "reasons": ["위험 비용 절감률은 40.0입니다."],
            "cautions": [],
            "final_action": "검토하세요.",
        },
        "recommendation_level": "RECOMMEND",
    }

    with pytest.raises(PlanningValidationError):
        validate_final_ai_evaluation(payload, evidence, draft)
