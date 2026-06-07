from __future__ import annotations

# ruff: noqa: E501
import asyncio
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.config import Settings, get_settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.grounded_prompt_builder import GroundedPrompt
from app.features.chat.llm_client import LlmClient, LlmCompletion, validate_llm_settings
from app.features.production_planning.exceptions import PlanningValidationError
from app.features.production_planning.schemas import (
    PLAN_VARIANT_NAMES,
    AdjustedPlanCandidate,
    PlanSimulationResult,
    ProductionPlanningResult,
)

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - LangGraph normally installs langsmith.

    def traceable(*args, **kwargs):
        def decorator(func):
            return func

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return decorator


logger = logging.getLogger(__name__)

INFO_UNAVAILABLE = "정보 없음"
LLM_DISABLED_REASON = "LLM 평가 생성 기능이 비활성화되어 있습니다."

RecommendationLevel = Literal[
    "STRONG_RECOMMEND",
    "RECOMMEND",
    "CAUTION",
    "NOT_RECOMMENDED",
    "INFO_ONLY",
]
RecommendationGradeLabel = Literal["매우 낮음", "낮음", "보통", "높음", "매우 높음"]

RECOMMENDATION_LEVELS: set[str] = {
    "STRONG_RECOMMEND",
    "RECOMMEND",
    "CAUTION",
    "NOT_RECOMMENDED",
    "INFO_ONLY",
}

IMPORTANT_EVENT_TYPES: set[str] = {
    "MATERIAL_SHORTAGE",
    "MACHINE_BREAKDOWN",
    "SETUP_DELAY",
    "LINE_CHANGE_DELAY",
    "QUALITY_DEFECT",
    "LATE_COMPLETION",
}

_NUMERIC_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9_.-])-?\d+(?:[,.]\d+)*")
_ID_TOKEN_PATTERN = re.compile(
    r"\b(?:PLAN|ORDER|LINE|PRODUCT|MATERIAL|MACHINE|O|P|L|M)-[A-Za-z0-9]+\b"
)
_UPPERCASE_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
_MONEY_QUANT = Decimal("0.01")
_PERCENT_QUANT = Decimal("0.01")

EVALUATION_DRAFT_SYSTEM_PROMPT = """You are a manufacturing production planning analyst.

Your role is to generate concise Korean dashboard evaluation text using only the provided evidence JSON.

Rules:
1. The baseline plan is retrieved from the database.
2. The alternative plan is generated from CP-SAT planning and simulation results.
3. Do not assume any value that is not included in the input JSON.
4. Do not perform new calculations. Use only values already present in metrics and computed_deltas.
5. Do not invent order IDs, line IDs, product IDs, event causes, costs, dates, or durations.
6. Explain why the alternative is recommended using delivery delay reduction, cost reduction, and schedule completion evidence.
7. For AMOUNT_OPTIMAL, emphasize quantitative money/cost evidence first when those values are present in evidence.
8. Write in Korean.
9. Return valid JSON only.
10. Do not include Markdown.
11. If a required value is missing, write "정보 없음".
12. Keep every sentence suitable for a production planning dashboard.
13. Do not criticize the generated plan. Write recommendation rationale and neutral execution checks.
14. Do not mention internal calculation status values or missing-field key names.
15. Choose recommendation_grade_label from exactly one of: 매우 낮음, 낮음, 보통, 높음, 매우 높음."""

EVALUATION_DRAFT_USER_PROMPT_TEMPLATE = """다음 Evidence JSON은 DB에서 조회한 기존 생산계획 평가값과 신규 대응안 시뮬레이션 결과를 비교한 데이터입니다.

이 데이터를 기반으로 대시보드 평가 문구 초안을 생성하세요.

주의사항:
- 이 문구는 대응안을 비판하는 글이 아니라 추천 이유를 설명하는 글입니다.
- 숫자는 반드시 input JSON의 metrics 또는 computed_deltas 값을 그대로 사용하세요.
- 변화량, 감소율, 절감률은 computed_deltas 값만 사용하세요.
- alternative.plan_variant_code가 "AMOUNT_OPTIMAL"이면 가격/비용 관점의 정량 수치를 가장 먼저 부각하세요.
- AMOUNT_OPTIMAL의 recommendation_summary_text와 recommendation_reasons에는 가능한 경우 total_risk_cost, expected_late_penalty_amount, risk_cost_saving_amount, risk_cost_saving_percent, contract/value 보호 관련 수치를 우선 사용하세요.
- AMOUNT_OPTIMAL에서도 납기/자재/일정 안정성은 보조 근거로만 설명하고, 가격 측면의 판단 축이 흐려지지 않게 하세요.
- recommendation_grade_label은 아래 우선순위로 판단하세요: 1순위 지연 일수 감소, 2순위 비용 감소, 3순위 계획 종료 시각 단축.
- 비용 감소나 계획 종료 시각 단축 값이 없으면 사용하지 말고, 있는 근거만으로 추천도를 산정하세요.
- 새로운 주문 ID, 라인 ID, 제품 ID, 비용, 기간, 이벤트 원인을 만들지 마세요.
- 값이 없거나 계산 불가능하면 "정보 없음"이라고만 쓰고, 데이터 부족을 추천 반대 사유로 쓰지 마세요.
- recommendation_reasons에는 대응안을 선택할 수 있는 근거만 작성하세요.
- recommendation_cautions에는 비판이 아니라 실행 전 확인사항만 작성하세요.
- 내부 계산 상태값이나 누락 필드명은 출력하지 마세요.
- 출력은 JSON만 반환하세요.

Output JSON Schema:
{
  "risk_analysis_text": "string",
  "risk_interpretation_text": "string",
  "recommendation_summary_text": "string",
  "recommendation_reasons": ["string"],
  "recommendation_cautions": ["string"],
  "final_action": "string",
  "recommendation_level": "STRONG_RECOMMEND | RECOMMEND | CAUTION | NOT_RECOMMENDED | INFO_ONLY",
  "recommendation_grade_label": "매우 낮음 | 낮음 | 보통 | 높음 | 매우 높음",
  "recommendation_grade_basis": ["string"]
}

Input Evidence JSON:
{{EVIDENCE_JSON}}"""

JSON_NORMALIZATION_SYSTEM_PROMPT = """You are a strict JSON formatter for a manufacturing planning dashboard.

Your task is to convert the draft evaluation JSON into the final dashboard JSON schema.

Rules:
1. Use only the provided evidence JSON and draft evaluation JSON.
2. Do not add new facts, numbers, causes, IDs, dates, costs, or durations.
3. Do not perform calculations.
4. Preserve numeric statements only if they already appear in evidence metrics, computed_deltas, or draft text.
5. If a required field is missing or unsupported by evidence, use "정보 없음".
6. Return valid JSON only.
7. Do not include Markdown.
8. Follow the output schema exactly.
9. Preserve the draft's recommendation rationale tone. Do not turn it into criticism.
10. For AMOUNT_OPTIMAL, preserve money/cost-focused quantitative wording from the draft when present.
11. Do not mention internal calculation status values or missing-field key names.
12. Preserve recommendation_grade_label from the draft or use 보통 when it is missing."""

JSON_NORMALIZATION_USER_PROMPT_TEMPLATE = """아래 Draft Evaluation JSON을 최종 프론트엔드 응답의 ai_evaluation schema로 변환하세요.

주의사항:
- schema key를 변경하지 마세요.
- 값이 없으면 "정보 없음"을 사용하세요.
- Evidence JSON에 없는 새로운 숫자나 원인을 추가하지 마세요.
- 추천 문구는 대응안의 장점과 선택 근거 중심으로 유지하세요.
- AMOUNT_OPTIMAL의 최종 JSON에서는 가격/비용 정량 수치가 recommendation summary와 reasons에서 먼저 보이게 정리하세요.
- cautions는 비판이 아니라 실행 전 확인사항으로만 작성하세요.
- 내부 계산 상태값이나 누락 필드명은 출력하지 마세요.
- recommendation_grade_label은 매우 낮음, 낮음, 보통, 높음, 매우 높음 중 하나만 사용하세요.
- recommendation_grade_basis에는 추천도 산정 기준을 지연 일수 감소, 비용 감소, 계획 종료 시각 단축 우선순위에 맞춰 짧게 작성하세요.
- 출력은 JSON만 반환하세요.

Final Output JSON Schema:
{
  "current_state_summary": {
    "risk_analysis_text": "string"
  },
  "risk_interpretation": {
    "text": "string"
  },
  "ai_recommendation": {
    "summary_text": "string",
    "reasons": ["string"],
    "cautions": ["string"],
    "final_action": "string"
  },
  "recommendation_level": "STRONG_RECOMMEND | RECOMMEND | CAUTION | NOT_RECOMMENDED | INFO_ONLY",
  "recommendation_grade_label": "매우 낮음 | 낮음 | 보통 | 높음 | 매우 높음",
  "recommendation_grade_basis": ["string"]
}

Evidence JSON:
{{EVIDENCE_JSON}}

Draft Evaluation JSON:
{{DRAFT_EVALUATION_JSON}}"""


class LlmPrompt(BaseModel):
    system_prompt: str
    user_prompt: str


class DashboardEvaluationDraft(BaseModel):
    risk_analysis_text: str
    risk_interpretation_text: str
    recommendation_summary_text: str
    recommendation_reasons: list[str] = Field(default_factory=list)
    recommendation_cautions: list[str] = Field(default_factory=list)
    final_action: str
    recommendation_level: RecommendationLevel
    recommendation_grade_label: RecommendationGradeLabel = "보통"
    recommendation_grade_basis: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def default_grade_basis(self) -> DashboardEvaluationDraft:
        if not self.recommendation_grade_basis:
            self.recommendation_grade_basis = [INFO_UNAVAILABLE]
        return self


class DashboardCurrentStateSummary(BaseModel):
    risk_analysis_text: str


class DashboardRiskInterpretation(BaseModel):
    text: str


class DashboardAiRecommendation(BaseModel):
    summary_text: str
    reasons: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    final_action: str


class DashboardAiEvaluation(BaseModel):
    status: Literal["COMPLETED", "FAILED"] = "COMPLETED"
    current_state_summary: DashboardCurrentStateSummary
    risk_interpretation: DashboardRiskInterpretation
    ai_recommendation: DashboardAiRecommendation
    recommendation_level: RecommendationLevel
    recommendation_grade_label: RecommendationGradeLabel = "보통"
    recommendation_grade_basis: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def default_grade_basis(self) -> DashboardAiEvaluation:
        if not self.recommendation_grade_basis:
            self.recommendation_grade_basis = [INFO_UNAVAILABLE]
        return self


def build_dashboard_response(
    result: ProductionPlanningResult,
    *,
    baseline_plans: list[dict[str, Any]] | None = None,
    baseline_simulation_rows: list[dict[str, Any]] | None = None,
    baseline_simulation_detail_rows: list[dict[str, Any]] | None = None,
    baseline_prediction_rows: list[dict[str, Any]] | None = None,
    products: list[Any] | None = None,
    production_lines: list[Any] | None = None,
    product_line_capabilities: list[Any] | None = None,
    planning_start: datetime | None = None,
    planning_end: datetime | None = None,
    generated_at: datetime | None = None,
    include_full_event_timeline: bool = False,
    include_internal_debug_payloads: bool = False,
    ai_evaluations_by_variant: dict[str, DashboardAiEvaluation | dict[str, Any]] | None = None,
    enable_llm_evaluation: bool = False,
    settings: Settings | None = None,
    llm_client: LlmClient | None = None,
) -> dict[str, Any]:
    """
    Parameters:
        - result: Internal production planning result with adjusted plans and simulations.
        - baseline_plans: DB current production plan rows for the same planning window.
        - baseline_simulation_rows: DB current simulation result rows for baseline metrics.
        - baseline_simulation_detail_rows: DB simulation detail rows matching baseline history.
        - baseline_prediction_rows: DB prediction rows used to quantify current delay risk.
        - products: Product master rows used for unit-aware fulfillment-rate calculations.
        - production_lines: Production line master rows used for application conditions.
        - product_line_capabilities: Product-line capability rows used for target products.
        - planning_start: Explicit comparison window start from the request.
        - planning_end: Explicit comparison window end from the request.
        - generated_at: Optional response timestamp.
        - include_full_event_timeline: Whether to include the representative full event timeline.
        - include_internal_debug_payloads: Whether to include LLM evidence/prompts and raw
          simulation event summaries in the final response.
        - ai_evaluations_by_variant: Optional validated or raw LLM evaluation by variant.
        - enable_llm_evaluation: Whether to call the configured LLM for ai_evaluation text.
        - settings: Optional application settings; defaults to .env-backed Settings.
        - llm_client: Optional LLM client, usually supplied by tests.

    Methodology:
        - Normalize DB baseline rows and CP-SAT simulation metrics into one response shape.
        - Compute percentage and amount deltas before any LLM text generation.
        - Include only important simulation events by default to keep dashboard payloads compact.
        - When LLM evaluation is enabled, run the two-prompt flow per plan variant and replace
          fallback ai_evaluation values with grounded validated LLM output.

    Output:
        - Front-end dashboard JSON comparing baseline and adjusted plan alternatives.
    """
    response_generated_at = generated_at or datetime.now(UTC)
    baseline_context = build_baseline_simulation_context(
        baseline_simulation_rows or [],
        baseline_detail_rows=baseline_simulation_detail_rows or [],
        baseline_prediction_rows=baseline_prediction_rows or [],
        product_rows=products or [],
    )
    baseline_metrics = baseline_context["metrics"]
    plan_results_by_code = {
        plan.plan_variant_code: plan for plan in result.plan_results
    }
    simulations_by_code = {
        simulation.plan_variant_code: simulation for simulation in result.simulation_results
    }
    completion_rank_by_code = _completion_rank_by_variant(result.adjusted_plan_candidates)
    risk_cost_rank_by_code = _risk_cost_rank_by_variant(result.simulation_results)
    ai_evaluations = ai_evaluations_by_variant or {}

    alternatives = []
    for candidate in result.adjusted_plan_candidates:
        simulation = simulations_by_code.get(candidate.plan_variant_code)
        plan_result = plan_results_by_code.get(candidate.plan_variant_code)
        alternative_metrics = build_alternative_simulation_metrics(simulation)
        alternative_current_state = build_alternative_current_state_summary(
            simulation,
            candidate,
            plan_result,
            product_rows=products or [],
            completion_rank=completion_rank_by_code.get(candidate.plan_variant_code),
            risk_cost_rank=risk_cost_rank_by_code.get(candidate.plan_variant_code),
        )
        alternative_metrics.update(alternative_current_state)
        comparison = build_comparison_metrics(baseline_metrics, alternative_metrics)
        simulation_comparison_table = build_simulation_comparison_table(
            baseline_context["current_state_summary"],
            alternative_current_state,
        )
        selected_plan_change_schedule = build_selected_plan_change_schedule(
            candidate,
            baseline_context["detail_rows"],
            baseline_context["matched_predictions"],
            plan_result,
        )
        application_conditions = build_application_conditions(
            candidate,
            baseline_plans or [],
            selected_plan_change_schedule,
            products=products or [],
            production_lines=production_lines or [],
            product_line_capabilities=product_line_capabilities or [],
        )
        important_events = _json_safe(select_important_events(simulation))
        event_summary = _json_safe(simulation.event_summary if simulation else [])
        event_timeline = _json_safe(
            simulation.event_timeline
            if include_full_event_timeline and simulation
            else []
        )
        evidence = build_evidence_json(
            baseline_metrics=baseline_metrics,
            alternative_metrics=alternative_metrics,
            comparison_metrics=comparison,
            important_events=important_events,
            plan_variant_code=candidate.plan_variant_code,
        )
        alternatives.append(
            {
                "plan_variant_code": candidate.plan_variant_code,
                "plan_variant_name": candidate.plan_variant_name,
                "status": candidate.status,
                "plans": _serialize_adjusted_candidate_plans(candidate),
                "simulation_metrics": alternative_metrics,
                "comparison_metrics": comparison["metrics"],
                "computed_deltas": comparison["computed_deltas"],
                "simulation_comparison_table": simulation_comparison_table,
                "application_conditions": application_conditions,
                "selected_plan_change_schedule": selected_plan_change_schedule,
                "important_events": important_events,
                "event_summary": event_summary,
                "full_event_timeline_included": include_full_event_timeline,
                "event_timeline": event_timeline,
                "llm_evidence": evidence,
                "llm_prompts": build_two_step_llm_prompts(evidence),
                "ai_evaluation": _serialize_ai_evaluation(
                    ai_evaluations.get(candidate.plan_variant_code),
                    evidence,
                ),
            }
        )

    response = {
        "generated_at": response_generated_at.isoformat(),
        "planning_window": {
            "planning_start": _iso_or_none(planning_start),
            "planning_end": _iso_or_none(planning_end),
        },
        "data_sources": {
            "baseline": "DB_CURRENT_PLAN_AND_SIMULATION",
            "alternative": "CP_SAT_AND_SIMULATION",
        },
        "baseline": {
            "source": "DB_CURRENT_PLAN_AND_SIMULATION",
            "plans": normalize_baseline_plan_rows(baseline_plans or []),
            "simulation_metrics": baseline_metrics,
            "current_state_summary": baseline_context["current_state_summary"],
            "provenance": baseline_context["provenance"],
            "historical_applied_result": baseline_context["historical_applied_result"],
        },
        "alternatives": alternatives,
        "warnings": result.warnings,
    }
    response = _json_safe(response)
    if not enable_llm_evaluation:
        return strip_internal_dashboard_fields(
            response,
            include_internal_debug_payloads=include_internal_debug_payloads,
            include_full_event_timeline=include_full_event_timeline,
        )
    evaluated_response = populate_dashboard_ai_evaluations(
        response,
        settings=settings,
        llm_client=llm_client,
    )
    return strip_internal_dashboard_fields(
        _json_safe(evaluated_response),
        include_internal_debug_payloads=include_internal_debug_payloads,
        include_full_event_timeline=include_full_event_timeline,
    )


def build_baseline_simulation_context(
    baseline_simulation_rows: list[dict[str, Any]],
    *,
    baseline_detail_rows: list[dict[str, Any]] | None = None,
    baseline_prediction_rows: list[dict[str, Any]] | None = None,
    product_rows: list[Any] | None = None,
) -> dict[str, Any]:
    """
    Parameters:
        - baseline_simulation_rows: DB rows from schedule simulation result history.
        - baseline_detail_rows: DB rows from schedule simulation detail history.
        - baseline_prediction_rows: DB rows from AI prediction history.
        - product_rows: Product master rows used to validate quantity units.

    Methodology:
        - Pick the latest baseline simulation row by applied/created time and simulation_id.
        - Use only detail rows with that simulation_id.
        - Match each detail row to the latest prediction by order_id + plan_id, then order_id.
        - Derive current-state metrics only from DB-supported values.

    Output:
        - Baseline context containing metrics, current-state summary, details, provenance, and
          matched predictions.
    """
    row = _select_latest_simulation_row(baseline_simulation_rows)
    simulation_id = _first_present(row, ["simulation_id"])
    detail_rows = _filter_detail_rows_by_simulation_id(
        baseline_detail_rows or [],
        simulation_id,
    )
    matched_predictions = _match_baseline_predictions(
        detail_rows,
        baseline_prediction_rows or [],
    )
    current_state_summary = build_baseline_current_state_summary(
        row,
        detail_rows,
        matched_predictions,
        product_rows or [],
    )
    metrics = build_baseline_simulation_metrics(
        baseline_simulation_rows,
        current_state_summary=current_state_summary,
    )
    return {
        "source_row": row,
        "detail_rows": detail_rows,
        "matched_predictions": matched_predictions,
        "metrics": metrics,
        "current_state_summary": current_state_summary,
        "provenance": build_baseline_provenance(row),
        "historical_applied_result": build_historical_applied_result(row, detail_rows),
    }


def strip_internal_dashboard_fields(
    dashboard_response: dict[str, Any],
    *,
    include_internal_debug_payloads: bool,
    include_full_event_timeline: bool,
) -> dict[str, Any]:
    """
    Parameters:
        - dashboard_response: Dashboard response after optional LLM evaluation.
        - include_internal_debug_payloads: Whether to preserve internal LLM/debug payloads.
        - include_full_event_timeline: Whether representative event timelines were requested.

    Methodology:
        - Keep the response front-end focused by default.
        - Remove prompts/evidence/raw summaries that are useful for debugging but noisy in UI.

    Output:
        - The same response object with internal-only fields removed unless explicitly requested.
    """
    if include_internal_debug_payloads:
        return dashboard_response
    for alternative in dashboard_response.get("alternatives", []):
        alternative.pop("llm_evidence", None)
        alternative.pop("llm_prompts", None)
        alternative.pop("event_summary", None)
        alternative.pop("comparison_metrics", None)
        if not include_full_event_timeline:
            alternative.pop("event_timeline", None)
            alternative.pop("full_event_timeline_included", None)
    return dashboard_response


def build_baseline_simulation_metrics(
    baseline_simulation_rows: list[dict[str, Any]],
    *,
    current_state_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Parameters:
        - baseline_simulation_rows: DB rows from current schedule simulation history.

    Methodology:
        - Pick the latest available row using common timestamp columns.
        - Map known DB/result column aliases into dashboard metric names.
        - Mark missing fields explicitly instead of fabricating baseline values.

    Output:
        - Baseline metrics dictionary with values and missing-value metadata.
    """
    row = _select_latest_simulation_row(baseline_simulation_rows)
    metrics = {
        "delay_probability_percent": _coerce_percent(
            _first_present(
                row,
                [
                    "delay_probability_percent",
                    "delay_probability",
                    "delay_rate",
                    "expected_delay_probability",
                ],
            )
        ),
        "expected_delayed_orders": _coerce_decimal_or_none(
            _first_present(
                row,
                [
                    "expected_delayed_order_count",
                    "expected_delayed_orders",
                    "delayed_order_count",
                    "avg_delayed_order_count",
                ],
            )
        ),
        "p95_tardiness_minutes": _coerce_decimal_or_none(
            _first_present(
                row,
                [
                    "p95_tardiness_minutes",
                    "p95_delay_minutes",
                    "p95_late_minutes",
                ],
            )
        ),
        "total_risk_cost": _money_or_none(
            _first_present(
                row,
                [
                    "expected_total_risk_cost",
                    "total_risk_cost",
                    "risk_cost",
                    "expected_risk_cost",
                ],
            )
        ),
        "material_shortage_probability_percent": _coerce_percent(
            _first_present(
                row,
                [
                    "material_shortage_probability_percent",
                    "material_shortage_probability",
                    "material_shortage_rate",
                ],
            )
        ),
    }
    if current_state_summary:
        metrics.update(
            {
                "expected_delay_days": current_state_summary.get("expected_delay_days"),
                "delivery_fulfillment_rate_percent": current_state_summary.get(
                    "delivery_fulfillment_rate_percent"
                ),
                "delivery_miss_rate_percent": current_state_summary.get(
                    "delivery_miss_rate_percent"
                ),
                "delay_risk_order_count": current_state_summary.get(
                    "delay_risk_order_count"
                ),
                "avg_line_utilization_percent": current_state_summary.get(
                    "avg_line_utilization_percent"
                ),
                "bottleneck_line_id": current_state_summary.get("bottleneck_line_id"),
                "baseline_source_simulation_id": current_state_summary.get(
                    "baseline_source_simulation_id"
                ),
                "unit_basis": current_state_summary.get("unit_basis"),
            }
        )
    missing = [
        key
        for key, value in metrics.items()
        if value is None
    ]
    metrics["calculation_status"] = "OK" if not missing else "MISSING_VALUE"
    metrics["missing_fields"] = missing
    return metrics


def build_baseline_current_state_summary(
    baseline_row: dict[str, Any],
    detail_rows: list[dict[str, Any]],
    matched_predictions: dict[str, dict[str, Any]],
    product_rows: list[Any],
) -> dict[str, Any]:
    """
    Parameters:
        - baseline_row: Latest DB simulation result row.
        - detail_rows: DB simulation details matching baseline_row.simulation_id.
        - matched_predictions: Prediction rows matched to detail rows.
        - product_rows: Product master rows used to verify quantity unit consistency.

    Methodology:
        - Sum predicted_delay_days for current-state expected delay.
        - Calculate delivery fulfillment from risk quantities when product units are compatible.
        - Fall back to order-count basis when product units are missing or mixed.

    Output:
        - Current-state baseline summary for dashboard cards and comparison tables.
    """
    risk_detail_keys = {
        key
        for key, detail in _detail_rows_by_key(detail_rows).items()
        if _baseline_detail_has_delay_risk(detail, matched_predictions.get(key))
    }
    expected_delay_days = _sum_prediction_delay_days(matched_predictions.values())
    unit_basis = _delivery_rate_unit_basis(
        _prediction_product_ids(matched_predictions.values()),
        product_rows,
    )
    total, delayed = _baseline_delivery_denominator(
        detail_rows,
        risk_detail_keys,
        unit_basis,
    )
    fulfillment_rate = _fulfillment_rate(total, delayed)
    utilization = _coerce_percent(
        _first_present(baseline_row, ["before_avg_line_utilization_rate"])
    )
    return {
        "expected_delay_days": _decimal_to_float(expected_delay_days),
        "delivery_fulfillment_rate_percent": fulfillment_rate,
        "delivery_miss_rate_percent": _miss_rate(fulfillment_rate),
        "delay_risk_order_count": len(risk_detail_keys),
        "avg_line_utilization_percent": utilization,
        "bottleneck_line_id": _id_or_none(
            _first_present(baseline_row, ["before_bottleneck_line_id"])
        ),
        "baseline_source_simulation_id": _id_or_none(
            _first_present(baseline_row, ["simulation_id"])
        ),
        "unit_basis": unit_basis,
        "total_quantity_or_orders": _decimal_to_float(total),
        "delayed_quantity_or_orders": _decimal_to_float(delayed),
        "prediction_match_count": len(matched_predictions),
        "detail_count": len(detail_rows),
        "calculation_status": "OK" if baseline_row else "MISSING_VALUE",
    }


def build_baseline_provenance(row: dict[str, Any]) -> dict[str, Any]:
    """
    Parameters:
        - row: Latest DB simulation result row.

    Methodology:
        - Preserve DB source identifiers and timestamps separately from derived metrics.

    Output:
        - Provenance metadata for audit/debug display.
    """
    return {
        "simulation_id": _id_or_none(_first_present(row, ["simulation_id"])),
        "simulation_group_id": _first_present(row, ["simulation_group_id"]),
        "simulation_name": _first_present(row, ["simulation_name"]),
        "simulation_type": _first_present(row, ["simulation_type"]),
        "created_at": _iso_or_none(_first_present(row, ["created_at"])),
        "applied_at": _iso_or_none(_first_present(row, ["applied_at"])),
        "recommendation_grade": _first_present(row, ["recommendation_grade"]),
        "historical_action_cost_change_amount": _money_or_none(
            _first_present(row, ["cost_change_amount"])
        ),
    }


def build_historical_applied_result(
    row: dict[str, Any],
    detail_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Parameters:
        - row: Latest DB simulation result row.
        - detail_rows: DB simulation detail rows for that simulation.

    Methodology:
        - Keep after_* DB values as historical-applied context only.
        - Do not mix these values into the new CP-SAT alternative comparison.

    Output:
        - Historical after-state metadata and order-level details.
    """
    return {
        "after_total_delay_hr": _decimal_to_float(
            _to_decimal(_first_present(row, ["after_total_delay_hr"]))
        ),
        "delay_reduction_hr": _decimal_to_float(
            _to_decimal(_first_present(row, ["delay_reduction_hr"]))
        ),
        "after_avg_line_utilization_percent": _coerce_percent(
            _first_present(row, ["after_avg_line_utilization_rate"])
        ),
        "after_total_production_quantity": _first_present(
            row, ["after_total_production_quantity"]
        ),
        "after_bottleneck_line_id": _id_or_none(
            _first_present(row, ["after_bottleneck_line_id"])
        ),
        "action_result": _first_present(row, ["action_result"]),
        "details": [
            {
                "order_id": _id_or_none(_first_present(detail, ["order_id"])),
                "plan_id": _id_or_none(_first_present(detail, ["plan_id"])),
                "after_line_id": _id_or_none(_first_present(detail, ["after_line_id"])),
                "after_sequence": _first_present(detail, ["after_sequence"]),
                "after_start_at": _iso_or_none(_first_present(detail, ["after_start_at"])),
                "after_end_at": _iso_or_none(_first_present(detail, ["after_end_at"])),
                "after_quantity": _first_present(detail, ["after_quantity"]),
                "after_is_delayed": _first_present(detail, ["after_is_delayed"]),
                "change_reason": _first_present(detail, ["change_reason"]),
            }
            for detail in detail_rows
        ],
    }


def build_alternative_simulation_metrics(
    simulation: PlanSimulationResult | None,
) -> dict[str, Any]:
    """
    Parameters:
        - simulation: Simulation result generated from one adjusted plan candidate.

    Methodology:
        - Convert internal probability ratios to dashboard percentages.
        - Serialize money values as two-decimal strings to avoid float money drift.

    Output:
        - Alternative simulation metrics dictionary.
    """
    if simulation is None:
        return {
            "delay_probability_percent": None,
            "expected_delayed_orders": None,
            "p95_tardiness_minutes": None,
            "total_risk_cost": None,
            "material_shortage_probability_percent": None,
            "calculation_status": "MISSING_VALUE",
            "missing_fields": [
                "delay_probability_percent",
                "expected_delayed_orders",
                "p95_tardiness_minutes",
                "total_risk_cost",
                "material_shortage_probability_percent",
            ],
        }

    return {
        "delay_probability_percent": _round_decimal(
            Decimal(str(simulation.delay_probability)) * Decimal("100")
        ),
        "expected_delayed_orders": _round_decimal(
            Decimal(str(simulation.expected_delayed_order_count))
        ),
        "p95_tardiness_minutes": _round_decimal(
            Decimal(str(simulation.p95_tardiness_minutes))
        ),
        "total_risk_cost": _money_or_none(simulation.expected_total_risk_cost),
        "material_shortage_probability_percent": _round_decimal(
            Decimal(str(simulation.material_shortage_probability)) * Decimal("100")
        ),
        "calculation_status": "OK",
        "missing_fields": [],
    }


def build_alternative_current_state_summary(
    simulation: PlanSimulationResult | None,
    candidate: AdjustedPlanCandidate,
    plan_result,
    *,
    product_rows: list[Any],
    completion_rank: int | None,
    risk_cost_rank: int | None,
) -> dict[str, Any]:
    """
    Parameters:
        - simulation: Simulation result for the adjusted candidate.
        - candidate: Adjusted plan candidate rows.
        - plan_result: CP-SAT plan result for deterministic order statuses.
        - product_rows: Product master rows used to validate quantity unit consistency.
        - completion_rank: Rank of this candidate by earliest planned completion.
        - risk_cost_rank: Rank of this candidate by lowest simulated risk cost.

    Methodology:
        - Convert expected tardiness minutes to expected delay days.
        - Use simulated expected delayed order count for order-level risk.
        - Use deterministic delayed planned quantities only when product units are compatible.

    Output:
        - Alternative summary aligned to baseline current-state fields.
    """
    unit_basis = _delivery_rate_unit_basis(
        [row.product_id for row in candidate.plans],
        product_rows,
    )
    delayed_order_ids = _delayed_order_ids_from_plan(plan_result)
    if unit_basis == "QUANTITY":
        total = sum(
            (Decimal(str(row.planned_quantity)) for row in candidate.plans),
            Decimal("0"),
        )
        delayed = sum(
            (
                Decimal(str(row.planned_quantity))
                for row in candidate.plans
                if _canonical_id(row.order_id) in delayed_order_ids
            ),
            Decimal("0"),
        )
    else:
        total = Decimal(str(len(candidate.plans)))
        delayed = _to_decimal(
            None if simulation is None else simulation.expected_delayed_order_count
        ) or Decimal("0")

    expected_delay_days = None
    delay_risk_order_count = None
    if simulation is not None:
        expected_delay_days = _decimal_to_float(
            Decimal(str(simulation.expected_tardiness_minutes)) / Decimal("1440")
        )
        delay_risk_order_count = _round_decimal(
            Decimal(str(simulation.expected_delayed_order_count))
        )

    fulfillment_rate = _fulfillment_rate(total, delayed)
    return {
        "expected_delay_days": expected_delay_days,
        "delivery_fulfillment_rate_percent": fulfillment_rate,
        "delivery_miss_rate_percent": _miss_rate(fulfillment_rate),
        "delay_risk_order_count": delay_risk_order_count,
        "avg_line_utilization_percent": None,
        "bottleneck_line_id": None,
        "plan_completion_at": _candidate_completion_at(candidate),
        "plan_completion_rank": completion_rank,
        "risk_cost_rank": risk_cost_rank,
        "unit_basis": unit_basis,
        "total_quantity_or_orders": _decimal_to_float(total),
        "delayed_quantity_or_orders": _decimal_to_float(delayed),
    }


def build_comparison_metrics(
    baseline_metrics: dict[str, Any],
    alternative_metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Parameters:
        - baseline_metrics: Normalized DB current-plan simulation metrics.
        - alternative_metrics: Normalized adjusted-plan simulation metrics.

    Methodology:
        - Compute all dashboard deltas before LLM generation.
        - Return both flat computed deltas and detailed before/after audit entries.

    Output:
        - Dictionary containing computed_deltas and metric-level calculation details.
    """
    risk_cost_saving_amount = _decimal_delta(
        baseline_metrics.get("total_risk_cost"),
        alternative_metrics.get("total_risk_cost"),
    )
    computed_deltas = {
        "expected_delay_days_reduction": _decimal_delta(
            baseline_metrics.get("expected_delay_days"),
            alternative_metrics.get("expected_delay_days"),
        ),
        "expected_delay_days_reduction_percent": _reduction_percent(
            baseline_metrics.get("expected_delay_days"),
            alternative_metrics.get("expected_delay_days"),
        ),
        "delivery_fulfillment_rate_delta_percent_points": _decimal_delta(
            alternative_metrics.get("delivery_fulfillment_rate_percent"),
            baseline_metrics.get("delivery_fulfillment_rate_percent"),
        ),
        "delay_risk_order_reduction": _decimal_delta(
            baseline_metrics.get("delay_risk_order_count"),
            alternative_metrics.get("delay_risk_order_count"),
        ),
        "delay_probability_reduction_percent": _reduction_percent(
            baseline_metrics.get("delay_probability_percent"),
            alternative_metrics.get("delay_probability_percent"),
        ),
        "expected_delayed_order_reduction": _decimal_delta(
            baseline_metrics.get("expected_delayed_orders"),
            alternative_metrics.get("expected_delayed_orders"),
        ),
        "p95_tardiness_reduction_minutes": _decimal_delta(
            baseline_metrics.get("p95_tardiness_minutes"),
            alternative_metrics.get("p95_tardiness_minutes"),
        ),
        "risk_cost_saving_amount": _money_string(risk_cost_saving_amount),
        "risk_cost_saving_percent": _reduction_percent(
            baseline_metrics.get("total_risk_cost"),
            alternative_metrics.get("total_risk_cost"),
        ),
        "material_shortage_probability_reduction_percent": _reduction_percent(
            baseline_metrics.get("material_shortage_probability_percent"),
            alternative_metrics.get("material_shortage_probability_percent"),
        ),
    }
    metrics = {
        "expected_delay_days": _comparison_entry(
            baseline_metrics.get("expected_delay_days"),
            alternative_metrics.get("expected_delay_days"),
            "days",
        ),
        "delivery_fulfillment_rate": _comparison_entry(
            baseline_metrics.get("delivery_fulfillment_rate_percent"),
            alternative_metrics.get("delivery_fulfillment_rate_percent"),
            "percent",
        ),
        "delay_risk_orders": _comparison_entry(
            baseline_metrics.get("delay_risk_order_count"),
            alternative_metrics.get("delay_risk_order_count"),
            "orders",
        ),
        "delay_probability": _comparison_entry(
            baseline_metrics.get("delay_probability_percent"),
            alternative_metrics.get("delay_probability_percent"),
            "percent",
        ),
        "expected_delayed_orders": _comparison_entry(
            baseline_metrics.get("expected_delayed_orders"),
            alternative_metrics.get("expected_delayed_orders"),
            "orders",
        ),
        "p95_tardiness": _comparison_entry(
            baseline_metrics.get("p95_tardiness_minutes"),
            alternative_metrics.get("p95_tardiness_minutes"),
            "minutes",
        ),
        "total_risk_cost": _comparison_entry(
            baseline_metrics.get("total_risk_cost"),
            alternative_metrics.get("total_risk_cost"),
            "KRW",
            money=True,
        ),
        "material_shortage_probability": _comparison_entry(
            baseline_metrics.get("material_shortage_probability_percent"),
            alternative_metrics.get("material_shortage_probability_percent"),
            "percent",
        ),
    }
    return {
        "computed_deltas": computed_deltas,
        "metrics": metrics,
    }


def build_simulation_comparison_table(
    baseline_summary: dict[str, Any],
    alternative_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Parameters:
        - baseline_summary: Current-state baseline metrics derived from DB.
        - alternative_summary: Adjusted-plan metrics derived from CP-SAT and simulation.

    Methodology:
        - Build front-end table rows only for metrics supported by the current data sources.
        - Mark utilization as NOT_COMPARABLE until alternative utilization is available.

    Output:
        - List of metric comparison rows with baseline, alternative, delta, status, and unit.
    """
    return [
        _comparison_table_row(
            "expected_delay_days",
            "예상 지연일",
            baseline_summary.get("expected_delay_days"),
            alternative_summary.get("expected_delay_days"),
            "days",
            improvement_direction="lower",
        ),
        _comparison_table_row(
            "delivery_fulfillment_rate_percent",
            "납기 충족률",
            baseline_summary.get("delivery_fulfillment_rate_percent"),
            alternative_summary.get("delivery_fulfillment_rate_percent"),
            "percent_point",
            improvement_direction="higher",
        ),
        _comparison_table_row(
            "delay_risk_order_count",
            "지연 위험 주문수",
            baseline_summary.get("delay_risk_order_count"),
            alternative_summary.get("delay_risk_order_count"),
            "orders",
            improvement_direction="lower",
        ),
        _comparison_table_row(
            "avg_line_utilization_percent",
            "라인 가동률",
            baseline_summary.get("avg_line_utilization_percent"),
            alternative_summary.get("avg_line_utilization_percent"),
            "percent_point",
            improvement_direction="higher",
        ),
    ]


def build_selected_plan_change_schedule(
    candidate: AdjustedPlanCandidate,
    baseline_detail_rows: list[dict[str, Any]],
    matched_predictions: dict[str, dict[str, Any]],
    plan_result,
) -> list[dict[str, Any]]:
    """
    Parameters:
        - candidate: Adjusted plan candidate.
        - baseline_detail_rows: Baseline simulation details containing before_* schedule.
        - matched_predictions: Baseline predictions keyed by detail identity.
        - plan_result: CP-SAT plan result with deterministic delay status.

    Methodology:
        - Match each adjusted order to baseline detail by normalized order_id.
        - Compare DB before_* schedule values to the new CP-SAT adjusted plan.

    Output:
        - Order-level schedule change rows for dashboard display.
    """
    detail_by_order_id = _detail_rows_by_order_id(baseline_detail_rows)
    delayed_order_ids = _delayed_order_ids_from_plan(plan_result)
    rows = []
    for plan in sorted(candidate.plans, key=lambda row: (row.line_id, row.plan_sequence)):
        order_key = _canonical_id(plan.order_id)
        detail = detail_by_order_id.get(order_key, {})
        if not detail:
            continue
        prediction = matched_predictions.get(_detail_key(detail), {})
        before_line_id = _id_or_none(_first_present(detail, ["before_line_id"]))
        after_line_id = _id_or_none(plan.line_id)
        before_delayed = _baseline_detail_has_delay_risk(detail, prediction)
        after_delayed = order_key in delayed_order_ids
        rows.append(
            {
                "order_id": _id_or_none(plan.order_id),
                "line_change": _line_change_text(before_line_id, after_line_id),
                "before_schedule": _schedule_text(
                    _first_present(detail, ["before_start_at"]),
                    _first_present(detail, ["before_end_at"]),
                ),
                "after_schedule": _schedule_text(
                    plan.planned_start_at,
                    plan.planned_end_at,
                ),
                "sequence_change": _sequence_change_text(
                    _first_present(detail, ["before_sequence"]),
                    plan.plan_sequence,
                ),
                "quantity_change": _quantity_change_text(
                    _first_present(detail, ["before_quantity"]),
                    plan.planned_quantity,
                ),
                "delay_status_change": _delay_status_change(before_delayed, after_delayed),
            }
        )
    return rows


def build_application_conditions(
    candidate: AdjustedPlanCandidate,
    baseline_plan_rows: list[dict[str, Any]],
    change_schedule: list[dict[str, Any]],
    *,
    products: list[Any],
    production_lines: list[Any],
    product_line_capabilities: list[Any],
) -> dict[str, Any]:
    """
    Parameters:
        - candidate: Adjusted plan candidate.
        - baseline_plan_rows: Current DB schedules in the planning window.
        - change_schedule: Order-level change rows already derived for the candidate.
        - products: Product master rows.
        - production_lines: Production line master rows.
        - product_line_capabilities: Product-line capability rows.

    Methodology:
        - Treat changed or newly assigned lines as applicable lines.
        - Resolve products producible on those lines through capability data.
        - Report unchanged baseline schedules overlapping the adjusted candidate period.

    Output:
        - Application-condition block for the dashboard.
    """
    changed_line_ids = _changed_line_ids(change_schedule)
    candidate_line_ids = {_canonical_id(row.line_id) for row in candidate.plans}
    applicable_line_ids = changed_line_ids or candidate_line_ids
    period = _applicable_period(candidate)
    return {
        "available_lines": _available_line_rows(applicable_line_ids, production_lines),
        "target_products": _target_product_rows(
            applicable_line_ids,
            products,
            product_line_capabilities,
        ),
        "applicable_period": period,
        "unchanged_overlapping_orders": _unchanged_overlapping_orders(
            baseline_plan_rows,
            change_schedule,
            period,
        ),
    }


def select_important_events(
    simulation: PlanSimulationResult | None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Parameters:
        - simulation: Simulation result containing aggregate event summary and timeline.
        - limit: Maximum important events returned to the dashboard.

    Methodology:
        - Prefer events with delay, cost, shortage, or configured important event types.
        - Rank summary events by risk cost, tardiness, occurrence, and deterministic name.
        - Keep the full timeline outside the default payload to protect response readability.

    Output:
        - Important event dictionaries suitable for compact dashboard rendering.
    """
    if simulation is None or limit <= 0:
        return []

    selected = []
    ranked_summary = sorted(
        simulation.event_summary,
        key=lambda row: (
            -float(row.get("avg_risk_cost_when_event_occurs") or 0),
            -float(row.get("avg_tardiness_minutes_when_event_occurs") or 0),
            -int(row.get("occurrence_count") or 0),
            str(row.get("event") or ""),
        ),
    )
    for row in ranked_summary:
        if len(selected) >= limit:
            break
        event_name = str(row.get("event") or "")
        if _is_important_summary_event(row, event_name):
            selected.append({"source": "SUMMARY", **row})

    for row in simulation.event_timeline:
        if len(selected) >= limit:
            break
        event_name = str(row.get("event") or "")
        if _is_important_timeline_event(row, event_name):
            selected.append({"source": "TIMELINE", **row})

    return selected[:limit]


def build_evidence_json(
    *,
    baseline_metrics: dict[str, Any],
    alternative_metrics: dict[str, Any],
    comparison_metrics: dict[str, Any],
    important_events: list[dict[str, Any]],
    plan_variant_code: str,
) -> dict[str, Any]:
    """
    Parameters:
        - baseline_metrics: Normalized DB current-plan metrics.
        - alternative_metrics: Normalized adjusted-plan metrics.
        - comparison_metrics: Formatter-computed delta metrics.
        - important_events: Filtered simulation events for explanation.
        - plan_variant_code: Adjusted plan variant being explained.

    Methodology:
        - Build a compact evidence document for the two-step LLM prompt.
        - Expose computed deltas so the LLM never needs to calculate percentages or money.

    Output:
        - Evidence JSON consumed by the evaluation-draft prompt.
    """
    baseline_llm_metrics = _llm_dashboard_metrics(baseline_metrics)
    alternative_llm_metrics = _llm_dashboard_metrics(alternative_metrics)
    computed_delta_metrics = _llm_visible_metrics(
        {
            key: comparison_metrics["computed_deltas"].get(key)
            for key in [
                "expected_delay_days_reduction",
                "expected_delay_days_reduction_percent",
                "delivery_fulfillment_rate_delta_percent_points",
                "delay_risk_order_reduction",
            ]
        }
    )
    return {
        "recommendation_judge_criteria": {
            "priority_order": [
                "지연 일수 감소",
                "비용 감소 또는 낮은 총 위험 비용",
                "계획 종료 시각 단축",
            ],
            "rule": "상위 기준의 근거가 없을 때만 다음 기준을 사용합니다.",
        },
        "baseline": {
            "source": "DB_CURRENT_PLAN_AND_SIMULATION",
            "metrics": baseline_llm_metrics,
        },
        "alternative": {
            "plan_variant_code": plan_variant_code,
            "plan_variant_name": PLAN_VARIANT_NAMES.get(plan_variant_code, plan_variant_code),
            "source": "CP_SAT_AND_SIMULATION",
            "metrics": alternative_llm_metrics,
        },
        "computed_deltas": computed_delta_metrics,
        "important_events": important_events,
    }


def _llm_dashboard_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = [
        "expected_delay_days",
        "delivery_fulfillment_rate_percent",
        "delivery_miss_rate_percent",
        "delay_risk_order_count",
        "avg_line_utilization_percent",
        "total_risk_cost",
        "plan_completion_at",
        "plan_completion_rank",
        "risk_cost_rank",
    ]
    return _llm_visible_metrics({key: metrics.get(key) for key in allowed_keys})


def build_two_step_llm_prompts(evidence_json: dict[str, Any]) -> dict[str, dict[str, str]]:
    """
    Parameters:
        - evidence_json: Formatter-generated evidence document.

    Methodology:
        - Create prompt 1 for Korean evaluation draft generation.
        - Create prompt 2 template for strict final JSON normalization after draft creation.

    Output:
        - Dictionary containing the draft prompt and normalization prompt template.
    """
    draft_prompt = build_evaluation_draft_prompt(evidence_json)
    return {
        "evaluation_draft": draft_prompt.model_dump(),
        "json_normalization_template": {
            "system_prompt": JSON_NORMALIZATION_SYSTEM_PROMPT,
            "user_prompt_template": JSON_NORMALIZATION_USER_PROMPT_TEMPLATE,
        },
    }


def _llm_visible_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    hidden_keys = {"calculation_status", "missing_fields"}
    return {
        key: value
        for key, value in metrics.items()
        if key not in hidden_keys and value is not None
    }


def _sanitize_llm_evidence(value: Any) -> Any:
    hidden_keys = {"calculation_status", "missing_fields"}
    if isinstance(value, dict):
        return {
            key: _sanitize_llm_evidence(item)
            for key, item in value.items()
            if key not in hidden_keys and item is not None
        }
    if isinstance(value, list):
        return [
            sanitized
            for item in value
            if (sanitized := _sanitize_llm_evidence(item)) is not None
        ]
    return value


def build_evaluation_draft_prompt(evidence_json: dict[str, Any]) -> LlmPrompt:
    """
    Parameters:
        - evidence_json: Formatter-generated evidence document.

    Methodology:
        - Insert stable JSON into prompt 1.
        - Keep calculation instructions explicit so the LLM only writes Korean text.

    Output:
        - LlmPrompt for the evaluation draft step.
    """
    return LlmPrompt(
        system_prompt=EVALUATION_DRAFT_SYSTEM_PROMPT,
        user_prompt=EVALUATION_DRAFT_USER_PROMPT_TEMPLATE.replace(
            "{{EVIDENCE_JSON}}",
            _json_dumps(evidence_json),
        ),
    )


def build_json_normalization_prompt(
    evidence_json: dict[str, Any],
    draft_evaluation_json: dict[str, Any],
) -> LlmPrompt:
    """
    Parameters:
        - evidence_json: Formatter-generated evidence document.
        - draft_evaluation_json: Prompt 1 output after schema validation.

    Methodology:
        - Insert evidence and draft JSON into prompt 2.
        - The second prompt only maps existing draft text into the final schema.

    Output:
        - LlmPrompt for the strict JSON normalization step.
    """
    return LlmPrompt(
        system_prompt=JSON_NORMALIZATION_SYSTEM_PROMPT,
        user_prompt=JSON_NORMALIZATION_USER_PROMPT_TEMPLATE.replace(
            "{{EVIDENCE_JSON}}",
            _json_dumps(evidence_json),
        ).replace(
            "{{DRAFT_EVALUATION_JSON}}",
            _json_dumps(draft_evaluation_json),
        ),
    )


def _process_ai_evaluation_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    settings = inputs.get("settings")
    evidence = _sanitize_llm_evidence(inputs.get("evidence_json", {}))
    alternative = evidence.get("alternative", {}) if isinstance(evidence, dict) else {}
    base = {
        "plan_variant_code": alternative.get("plan_variant_code"),
        "model": getattr(settings, "llm_model", None),
        "provider": getattr(settings, "llm_provider", None),
        "trace_payloads": _should_trace_payloads(settings),
    }
    if _should_trace_payloads(settings):
        base["evidence_json"] = _json_safe(evidence)
        return base
    if isinstance(evidence, dict):
        base["evidence_summary"] = {
            "baseline_metric_keys": sorted(
                (evidence.get("baseline", {}).get("metrics", {}) or {}).keys()
            ),
            "alternative_metric_keys": sorted(
                (evidence.get("alternative", {}).get("metrics", {}) or {}).keys()
            ),
            "computed_delta_keys": sorted(
                (evidence.get("computed_deltas", {}) or {}).keys()
            ),
            "important_event_count": len(evidence.get("important_events", []) or []),
        }
    return base


def _process_ai_evaluation_trace_outputs(output: DashboardAiEvaluation) -> dict[str, Any]:
    if _should_trace_payloads(None):
        return _json_safe(output)
    return {
        "status": output.status,
        "recommendation_level": output.recommendation_level,
        "recommendation_grade_label": output.recommendation_grade_label,
        "recommendation_grade_basis_count": len(output.recommendation_grade_basis),
        "reason_count": len(output.ai_recommendation.reasons),
        "caution_count": len(output.ai_recommendation.cautions),
    }


def _evidence_variant_code(evidence_json: dict[str, Any]) -> str | None:
    alternative = evidence_json.get("alternative")
    if isinstance(alternative, dict):
        value = alternative.get("plan_variant_code")
        return None if value is None else str(value)
    return None


def _process_llm_completion_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    settings = inputs.get("settings")
    prompt = inputs.get("prompt")
    payload = {
        "step_name": inputs.get("step_name"),
        "plan_variant_code": inputs.get("plan_variant_code"),
        "model": getattr(settings, "llm_model", None),
        "provider": getattr(settings, "llm_provider", None),
        "trace_payloads": _should_trace_payloads(settings),
    }
    if isinstance(prompt, LlmPrompt):
        payload.update(
            {
                "system_prompt_chars": len(prompt.system_prompt),
                "user_prompt_chars": len(prompt.user_prompt),
            }
        )
        if _should_trace_payloads(settings):
            payload["system_prompt"] = prompt.system_prompt
            payload["user_prompt"] = prompt.user_prompt
    return payload


def _process_llm_completion_trace_outputs(output: LlmCompletion) -> dict[str, Any]:
    payload = {
        "answer_chars": len(output.answer),
        "usage": None if output.usage is None else output.usage.model_dump(mode="json"),
    }
    if _should_trace_payloads(None):
        payload["answer"] = output.answer
    return payload


def _should_trace_payloads(settings: Settings | None) -> bool:
    if settings is not None:
        return settings.langsmith_trace_payloads
    try:
        return get_settings().langsmith_trace_payloads
    except Exception:
        return False


def populate_dashboard_ai_evaluations(
    dashboard_response: dict[str, Any],
    *,
    settings: Settings | None = None,
    llm_client: LlmClient | None = None,
) -> dict[str, Any]:
    """
    Parameters:
        - dashboard_response: Dashboard JSON containing alternatives with llm_evidence.
        - settings: Optional .env-backed application settings.
        - llm_client: Optional LLM client, usually supplied by tests.

    Methodology:
        - Read LLM enablement and connection settings from Settings.
        - For each alternative, call Prompt 1 and Prompt 2 through the configured LLM API.
        - Validate both LLM JSON outputs before replacing the fallback ai_evaluation.
        - Keep one variant's failure isolated so the rest of the dashboard remains usable.

    Output:
        - The same dashboard response object with ai_evaluation populated per alternative.
    """
    effective_settings = settings or get_settings()
    if not effective_settings.llm_enabled:
        _set_all_ai_evaluations_to_fallback(
            dashboard_response,
            LLM_DISABLED_REASON,
        )
        return dashboard_response

    try:
        validate_llm_settings(effective_settings)
    except ChatExternalServiceError as exc:
        _set_all_ai_evaluations_to_fallback(dashboard_response, exc.message)
        return dashboard_response

    client = llm_client or LlmClient(effective_settings)
    for alternative in dashboard_response.get("alternatives", []):
        evidence = alternative.get("llm_evidence")
        if not isinstance(evidence, dict):
            alternative["ai_evaluation"] = build_fallback_ai_evaluation(
                "LLM evidence JSON이 누락되었습니다."
            ).model_dump(mode="json")
            continue

        variant_code = str(alternative.get("plan_variant_code") or INFO_UNAVAILABLE)
        try:
            evaluation = _run_async_llm_evaluation(
                lambda evidence=evidence: generate_ai_evaluation_from_evidence(
                    evidence,
                    settings=effective_settings,
                    llm_client=client,
                )
            )
            alternative["ai_evaluation"] = evaluation.model_dump(mode="json")
        except Exception as exc:
            logger.warning(
                "production_planning.llm_evaluation.failed",
                extra={"variant_code": variant_code, "error": str(exc)},
            )
            alternative["ai_evaluation"] = build_fallback_ai_evaluation(
                str(exc)
            ).model_dump(mode="json")
    return dashboard_response


@traceable(
    name="production_planning.ai_evaluation",
    run_type="chain",
    process_inputs=_process_ai_evaluation_trace_inputs,
    process_outputs=_process_ai_evaluation_trace_outputs,
)
async def generate_ai_evaluation_from_evidence(
    evidence_json: dict[str, Any],
    *,
    settings: Settings | None = None,
    llm_client: LlmClient | None = None,
) -> DashboardAiEvaluation:
    """
    Parameters:
        - evidence_json: Formatter-generated grounding document for one plan variant.
        - settings: Optional .env-backed LLM settings.
        - llm_client: Optional LLM client implementation.

    Methodology:
        - Call Prompt 1 to create a Korean evaluation draft.
        - Validate the draft schema and grounding against the evidence.
        - Call Prompt 2 to normalize the validated draft into the dashboard schema.
        - Validate the final JSON against both evidence and the draft output.

    Output:
        - Validated DashboardAiEvaluation ready to serialize in the response JSON.
    """
    effective_settings = settings or get_settings()
    validate_llm_settings(effective_settings)
    client = llm_client or LlmClient(effective_settings)
    safe_evidence_json = _sanitize_llm_evidence(evidence_json)

    try:
        draft_prompt = build_evaluation_draft_prompt(safe_evidence_json)
        draft_completion = await _generate_traced_llm_completion(
            prompt=draft_prompt,
            step_name="evaluation_draft",
            plan_variant_code=_evidence_variant_code(safe_evidence_json),
            settings=effective_settings,
            llm_client=client,
        )
        draft_payload = _parse_llm_json_object(draft_completion.answer, "evaluation draft")
        draft = validate_evaluation_draft(draft_payload, safe_evidence_json)

        final_prompt = build_json_normalization_prompt(
            safe_evidence_json,
            draft.model_dump(mode="json"),
        )
        final_completion = await _generate_traced_llm_completion(
            prompt=final_prompt,
            step_name="json_normalization",
            plan_variant_code=_evidence_variant_code(safe_evidence_json),
            settings=effective_settings,
            llm_client=client,
        )
        final_payload = _parse_llm_json_object(final_completion.answer, "final evaluation")
        return validate_final_ai_evaluation(
            final_payload,
            safe_evidence_json,
            draft.model_dump(mode="json"),
        )
    except PlanningValidationError as exc:
        logger.warning(
            "production_planning.llm_evaluation.validation_failed",
            extra={"error": str(exc)},
        )
        return build_fallback_ai_evaluation(str(exc))


@traceable(
    name="production_planning.llm_completion",
    run_type="llm",
    process_inputs=_process_llm_completion_trace_inputs,
    process_outputs=_process_llm_completion_trace_outputs,
)
async def _generate_traced_llm_completion(
    *,
    prompt: LlmPrompt,
    step_name: str,
    plan_variant_code: str | None,
    settings: Settings,
    llm_client: LlmClient,
) -> LlmCompletion:
    return await llm_client.generate_completion(
        GroundedPrompt(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
        )
    )


def validate_evaluation_draft(
    payload: dict[str, Any],
    evidence_json: dict[str, Any],
) -> DashboardEvaluationDraft:
    """
    Parameters:
        - payload: Raw Prompt 1 LLM output parsed as JSON.
        - evidence_json: Formatter-generated evidence document.

    Methodology:
        - Validate the draft schema.
        - Reject numeric, ID, or event tokens not grounded in the evidence JSON.

    Output:
        - Validated DashboardEvaluationDraft.
    """
    try:
        draft = DashboardEvaluationDraft.model_validate(payload)
    except ValidationError as exc:
        raise PlanningValidationError(
            f"Evaluation draft JSON schema is invalid: {exc}"
        ) from exc
    validate_llm_output_against_evidence(draft.model_dump(), evidence_json)
    return draft


def validate_final_ai_evaluation(
    payload: dict[str, Any],
    evidence_json: dict[str, Any],
    draft_evaluation_json: dict[str, Any] | None = None,
) -> DashboardAiEvaluation:
    """
    Parameters:
        - payload: Raw Prompt 2 LLM output parsed as JSON.
        - evidence_json: Formatter-generated evidence document.
        - draft_evaluation_json: Optional validated Prompt 1 output.

    Methodology:
        - Validate the final dashboard schema.
        - Ground final text against evidence and the validated draft.

    Output:
        - Validated DashboardAiEvaluation.
    """
    try:
        evaluation = DashboardAiEvaluation.model_validate(payload)
    except ValidationError as exc:
        raise PlanningValidationError(
            f"Final AI evaluation JSON schema is invalid: {exc}"
        ) from exc
    grounding_source = evidence_json
    if draft_evaluation_json is not None:
        grounding_source = {
            "evidence": evidence_json,
            "draft_evaluation": draft_evaluation_json,
        }
    validate_llm_output_against_evidence(evaluation.model_dump(), grounding_source)
    return evaluation


def validate_llm_output_against_evidence(
    payload: dict[str, Any],
    evidence_json: dict[str, Any],
) -> None:
    """
    Parameters:
        - payload: LLM output JSON to inspect.
        - evidence_json: Allowed evidence values and tokens.

    Methodology:
        - Compare numbers, DB-like IDs, and uppercase event/cause tokens in output text
          against the evidence document.
        - Fail closed because dashboard text must not invent production facts.

    Output:
        - None. Raises PlanningValidationError when unsupported values are present.
    """
    allowed_numbers = _collect_numeric_tokens(evidence_json)
    normalized_allowed_numbers = {
        normalized
        for token in allowed_numbers
        if (normalized := _normalize_numeric_token(token)) is not None
    }
    allowed_ids = _collect_id_tokens(evidence_json)
    allowed_uppercase_tokens = (
        _collect_uppercase_tokens(evidence_json)
        | RECOMMENDATION_LEVELS
        | {"COMPLETED", "FAILED", "JSON", "LLM", "DB", "SAT"}
    )

    for text in _iter_text_values(payload):
        unsupported_numbers = [
            token
            for token in _NUMERIC_TOKEN_PATTERN.findall(text)
            if token not in allowed_numbers
            and _normalize_numeric_token(token) not in normalized_allowed_numbers
        ]
        if unsupported_numbers:
            raise PlanningValidationError(
                "LLM output contains numbers not present in evidence: "
                + ", ".join(sorted(set(unsupported_numbers)))
            )

        unsupported_ids = [
            token
            for token in _ID_TOKEN_PATTERN.findall(text)
            if token not in allowed_ids
        ]
        if unsupported_ids:
            raise PlanningValidationError(
                "LLM output contains IDs not present in evidence: "
                + ", ".join(sorted(set(unsupported_ids)))
            )

        unsupported_uppercase = [
            token
            for token in _UPPERCASE_TOKEN_PATTERN.findall(text)
            if token not in allowed_uppercase_tokens
        ]
        if unsupported_uppercase:
            raise PlanningValidationError(
                "LLM output contains event/cause tokens not present in evidence: "
                + ", ".join(sorted(set(unsupported_uppercase)))
            )


def build_fallback_ai_evaluation(
    reason: str = INFO_UNAVAILABLE,
) -> DashboardAiEvaluation:
    """
    Parameters:
        - reason: Human-readable reason used in every fallback field.

    Methodology:
        - Produce a schema-valid evaluation when LLM parsing or grounding fails.
        - Avoid carrying partial untrusted LLM text into the dashboard.

    Output:
        - DashboardAiEvaluation marked as FAILED.
    """
    return DashboardAiEvaluation(
        status="FAILED",
        current_state_summary=DashboardCurrentStateSummary(risk_analysis_text=reason),
        risk_interpretation=DashboardRiskInterpretation(text=reason),
        ai_recommendation=DashboardAiRecommendation(
            summary_text=reason,
            reasons=[reason],
            cautions=[reason],
            final_action=reason,
        ),
        recommendation_level="INFO_ONLY",
        recommendation_grade_label="보통",
        recommendation_grade_basis=[reason],
    )


def _set_all_ai_evaluations_to_fallback(
    dashboard_response: dict[str, Any],
    reason: str,
) -> None:
    for alternative in dashboard_response.get("alternatives", []):
        alternative["ai_evaluation"] = build_fallback_ai_evaluation(reason).model_dump(
            mode="json"
        )


def _run_async_llm_evaluation(async_factory):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(async_factory())

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(async_factory())).result()


def _parse_llm_json_object(answer: str, step_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise PlanningValidationError(
            f"LLM {step_name} 응답 JSON 파싱에 실패했습니다."
        ) from exc
    if not isinstance(payload, dict):
        raise PlanningValidationError(
            f"LLM {step_name} 응답은 JSON object여야 합니다."
        )
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return _decimal_to_float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return value


def normalize_baseline_plan_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Parameters:
        - rows: Raw DB current production plan rows.

    Methodology:
        - Preserve DB IDs separately from display labels.
        - Normalize common schedule column aliases into production_plans-compatible keys.

    Output:
        - Baseline plan rows suitable for dashboard display.
    """
    normalized = []
    for row in rows:
        normalized.append(
            {
                "plan_id": _first_present(row, ["plan_id", "schedule_id"]),
                "order_id": _first_present(row, ["order_id"]),
                "product_id": _first_present(row, ["product_id"]),
                "line_id": _first_present(row, ["line_id"]),
                "operator_id": _first_present(row, ["operator_id"]),
                "planned_start_at": _iso_or_none(
                    _first_present(row, ["planned_start_at", "start_time"])
                ),
                "planned_end_at": _iso_or_none(
                    _first_present(row, ["planned_end_at", "end_time"])
                ),
                "estimated_duration_hr": _first_present(
                    row,
                    ["estimated_duration_hr", "actual_duration_hr"],
                ),
                "planned_quantity": _first_present(
                    row,
                    ["planned_quantity", "order_quantity"],
                ),
                "plan_sequence": _first_present(row, ["plan_sequence"]),
                "plan_status": _first_present(row, ["plan_status", "status"]),
                "updated_at": _iso_or_none(_first_present(row, ["updated_at"])),
            }
        )
    return normalized


def _serialize_adjusted_candidate_plans(
    candidate: AdjustedPlanCandidate,
) -> list[dict[str, Any]]:
    return [row.model_dump(mode="json") for row in candidate.plans]


def _serialize_ai_evaluation(
    evaluation: DashboardAiEvaluation | dict[str, Any] | None,
    evidence_json: dict[str, Any],
) -> dict[str, Any]:
    if evaluation is None:
        return build_fallback_ai_evaluation().model_dump(mode="json")
    if isinstance(evaluation, DashboardAiEvaluation):
        return evaluation.model_dump(mode="json")
    try:
        return validate_final_ai_evaluation(evaluation, evidence_json).model_dump(mode="json")
    except PlanningValidationError as exc:
        return build_fallback_ai_evaluation(str(exc)).model_dump(mode="json")


def _select_latest_simulation_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
        timestamp = _first_present(row, ["applied_at", "created_at"])
        return (
            str(timestamp or ""),
            str(_first_present(row, ["simulation_id"]) or ""),
            str(row),
        )

    return max(rows, key=sort_key)


def _select_latest_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    timestamp_keys = [
        "updated_at",
        "created_at",
        "simulated_at",
        "recorded_at",
        "simulation_created_at",
    ]

    def sort_key(row: dict[str, Any]) -> tuple[str, str]:
        for key in timestamp_keys:
            if row.get(key) is not None:
                return (str(row[key]), str(row))
        return ("", str(row))

    return max(rows, key=sort_key)


def _filter_detail_rows_by_simulation_id(
    detail_rows: list[dict[str, Any]],
    simulation_id: Any,
) -> list[dict[str, Any]]:
    if simulation_id is None:
        return []
    expected = _canonical_id(simulation_id)
    return [
        row
        for row in detail_rows
        if _canonical_id(_first_present(row, ["simulation_id"])) == expected
    ]


def _match_baseline_predictions(
    detail_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    predictions_by_order_plan: dict[tuple[str, str], list[dict[str, Any]]] = {}
    predictions_by_order: dict[str, list[dict[str, Any]]] = {}
    for prediction in prediction_rows:
        order_id = _canonical_id(_first_present(prediction, ["order_id"]))
        plan_id = _canonical_id(_first_present(prediction, ["plan_id"]))
        if order_id is None:
            continue
        predictions_by_order.setdefault(order_id, []).append(prediction)
        if plan_id is not None:
            predictions_by_order_plan.setdefault((order_id, plan_id), []).append(prediction)

    matches = {}
    for detail in detail_rows:
        order_id = _canonical_id(_first_present(detail, ["order_id"]))
        plan_id = _canonical_id(_first_present(detail, ["plan_id"]))
        candidates = []
        if order_id is not None and plan_id is not None:
            candidates = predictions_by_order_plan.get((order_id, plan_id), [])
        if not candidates and order_id is not None:
            candidates = predictions_by_order.get(order_id, [])
        if candidates:
            matches[_detail_key(detail)] = _select_latest_prediction(candidates)
    return matches


def _select_latest_prediction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def sort_key(row: dict[str, Any]) -> tuple[str, str]:
        return (
            str(_first_present(row, ["predicted_at", "created_at"]) or ""),
            str(_first_present(row, ["prediction_id"]) or ""),
        )

    return max(rows, key=sort_key)


def _detail_rows_by_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_detail_key(row): row for row in rows}


def _detail_rows_by_order_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        order_id = _canonical_id(_first_present(row, ["order_id"]))
        if order_id is not None:
            result[order_id] = row
    return result


def _detail_key(row: dict[str, Any]) -> str:
    order_id = _canonical_id(_first_present(row, ["order_id"]))
    plan_id = _canonical_id(_first_present(row, ["plan_id"]))
    return f"{order_id or ''}:{plan_id or ''}"


def _canonical_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text.startswith("PLAN-") and text.removeprefix("PLAN-").isdigit():
        return text.removeprefix("PLAN-")
    return text


def _id_or_none(value: Any) -> int | str | None:
    canonical = _canonical_id(value)
    if canonical is None:
        return None
    if canonical.isdigit():
        return int(canonical)
    return canonical


def _sum_prediction_delay_days(predictions) -> Decimal:
    total = Decimal("0")
    for prediction in predictions:
        delay = _to_decimal(_first_present(prediction, ["predicted_delay_days"]))
        if delay is not None:
            total += delay
    return total


def _prediction_product_ids(predictions) -> list[str]:
    ids = []
    for prediction in predictions:
        product_id = _canonical_id(_first_present(prediction, ["product_id"]))
        if product_id is not None:
            ids.append(product_id)
    return ids


def _delivery_rate_unit_basis(product_ids: list[Any], product_rows: list[Any]) -> str:
    product_unit_by_id = {
        _canonical_id(_first_present(row, ["product_id"])): _first_present(row, ["unit"])
        for row in (_plain_row(item) for item in product_rows)
    }
    unique_product_ids = {_canonical_id(product_id) for product_id in product_ids}
    unique_product_ids.discard(None)
    units = [
        product_unit_by_id.get(product_id)
        for product_id in unique_product_ids
    ]
    if units and all(unit is not None for unit in units) and len({str(unit) for unit in units}) == 1:
        return "QUANTITY"
    return "ORDER_COUNT"


def _baseline_delivery_denominator(
    detail_rows: list[dict[str, Any]],
    risk_detail_keys: set[str],
    unit_basis: str,
) -> tuple[Decimal, Decimal]:
    if unit_basis == "QUANTITY":
        total = sum(
            (
                _to_decimal(_first_present(row, ["before_quantity"])) or Decimal("0")
                for row in detail_rows
            ),
            Decimal("0"),
        )
        delayed = sum(
            (
                _to_decimal(_first_present(row, ["before_quantity"])) or Decimal("0")
                for row in detail_rows
                if _detail_key(row) in risk_detail_keys
            ),
            Decimal("0"),
        )
        return total, delayed
    return Decimal(str(len(detail_rows))), Decimal(str(len(risk_detail_keys)))


def _fulfillment_rate(total: Decimal, delayed: Decimal) -> float | None:
    if total <= 0:
        return None
    fulfilled = max(total - delayed, Decimal("0"))
    return _round_decimal((fulfilled / total) * Decimal("100"))


def _miss_rate(fulfillment_rate: float | None) -> float | None:
    if fulfillment_rate is None:
        return None
    return _round_decimal(Decimal("100") - Decimal(str(fulfillment_rate)))


def _baseline_detail_has_delay_risk(
    detail: dict[str, Any],
    prediction: dict[str, Any] | None,
) -> bool:
    if prediction:
        delay_days = _to_decimal(_first_present(prediction, ["predicted_delay_days"]))
        delay_probability = _to_decimal(_first_present(prediction, ["delay_probability"]))
        return (delay_days is not None and delay_days > 0) or (
            delay_probability is not None and delay_probability > 0
        )
    return _detail_is_delayed_by_expected_completion_date(detail)


def _detail_is_delayed_by_expected_completion_date(row: dict[str, Any]) -> bool:
    before_end = _as_datetime(_first_present(row, ["before_end_at"]))
    expected_completion = _as_date(_first_present(row, ["expected_completion_date"]))
    if before_end is None or expected_completion is None:
        return False
    return before_end.date() > expected_completion


def _delayed_order_ids_from_plan(plan_result) -> set[str]:
    if plan_result is None:
        return set()
    return {
        _canonical_id(item.order_id)
        for item in plan_result.schedule_items
        if item.status == "DELAYED" or item.tardiness_minutes > 0
    }


def _comparison_table_row(
    metric_code: str,
    metric_name: str,
    baseline_value: Any,
    alternative_value: Any,
    unit: str,
    *,
    improvement_direction: Literal["lower", "higher"],
) -> dict[str, Any]:
    baseline = _to_decimal(baseline_value)
    alternative = _to_decimal(alternative_value)
    if baseline is None or alternative is None:
        return {
            "metric_code": metric_code,
            "metric_name": metric_name,
            "baseline_value": baseline_value,
            "alternative_value": alternative_value,
            "delta": None,
            "change_text": "비교 불가",
            "unit": unit,
            "calculation_status": "NOT_COMPARABLE",
        }
    delta = baseline - alternative if improvement_direction == "lower" else alternative - baseline
    return {
        "metric_code": metric_code,
        "metric_name": metric_name,
        "baseline_value": baseline_value,
        "alternative_value": alternative_value,
        "delta": _decimal_to_float(delta),
        "change_text": _change_text(delta, unit),
        "unit": unit,
        "calculation_status": "OK",
    }


def _change_text(delta: Decimal, unit: str) -> str:
    value = _decimal_to_float(abs(delta))
    if delta > 0:
        direction = "개선"
    elif delta < 0:
        direction = "악화"
    else:
        direction = "변화 없음"
    suffix = {
        "days": "일",
        "orders": "건",
        "percent_point": "%p",
    }.get(unit, f" {unit}")
    return f"{value}{suffix} {direction}" if direction != "변화 없음" else direction


def _line_change_text(before_line_id: Any, after_line_id: Any) -> str:
    if before_line_id is None:
        return "-"
    if str(before_line_id) == str(after_line_id):
        return "-"
    return f"{before_line_id} -> {after_line_id}"


def _schedule_text(start: Any, end: Any) -> str | None:
    if start is None or end is None:
        return None
    return f"{_iso_or_none(start)} ~ {_iso_or_none(end)}"


def _sequence_change_text(before: Any, after: Any) -> str | None:
    if before is None or after is None:
        return None
    return "-" if str(before) == str(after) else f"{before} -> {after}"


def _quantity_change_text(before: Any, after: Any) -> str | None:
    if before is None or after is None:
        return None
    return "-" if str(before) == str(after) else f"{before} -> {after}"


def _delay_status_change(before_delayed: bool, after_delayed: bool) -> str:
    if before_delayed and not after_delayed:
        return "IMPROVED"
    if not before_delayed and after_delayed:
        return "WORSENED"
    return "SAME"


def _changed_line_ids(change_schedule: list[dict[str, Any]]) -> set[str]:
    result = set()
    for row in change_schedule:
        line_change = row.get("line_change")
        if isinstance(line_change, str) and "->" in line_change:
            result.add(line_change.split("->", 1)[1].strip())
    return result


def _completion_rank_by_variant(
    candidates: list[AdjustedPlanCandidate],
) -> dict[str, int]:
    ranked = sorted(
        (
            (
                candidate.plan_variant_code,
                max((row.planned_end_at for row in candidate.plans), default=datetime.max.replace(tzinfo=UTC)),
            )
            for candidate in candidates
        ),
        key=lambda item: (item[1], item[0]),
    )
    return {variant_code: index + 1 for index, (variant_code, _) in enumerate(ranked)}


def _risk_cost_rank_by_variant(
    simulation_results: list[PlanSimulationResult],
) -> dict[str, int]:
    ranked = sorted(
        (
            (result.plan_variant_code, result.expected_total_risk_cost)
            for result in simulation_results
        ),
        key=lambda item: (item[1], item[0]),
    )
    return {variant_code: index + 1 for index, (variant_code, _) in enumerate(ranked)}


def _candidate_completion_at(candidate: AdjustedPlanCandidate) -> str | None:
    if not candidate.plans:
        return None
    return _iso_or_none(max(row.planned_end_at for row in candidate.plans))


def _applicable_period(candidate: AdjustedPlanCandidate) -> dict[str, Any]:
    if not candidate.plans:
        return {"start_at": None, "end_at": None}
    start_at = min(row.planned_start_at for row in candidate.plans)
    end_at = max(row.planned_end_at for row in candidate.plans)
    return {"start_at": _iso_or_none(start_at), "end_at": _iso_or_none(end_at)}


def _available_line_rows(line_ids: set[str], production_lines: list[Any]) -> list[dict[str, Any]]:
    line_by_id = {
        _canonical_id(_first_present(row, ["line_id"])): row
        for row in (_plain_row(item) for item in production_lines)
    }
    rows = []
    for line_id in sorted(line_ids):
        line = line_by_id.get(_canonical_id(line_id), {})
        rows.append(
            {
                "line_id": _id_or_none(line_id),
                "line_name": _first_present(line, ["line_name"]),
                "line_code": _first_present(line, ["line_code"]),
            }
        )
    return rows


def _target_product_rows(
    line_ids: set[str],
    products: list[Any],
    product_line_capabilities: list[Any],
) -> list[dict[str, Any]]:
    product_by_id = {
        _canonical_id(_first_present(row, ["product_id"])): row
        for row in (_plain_row(item) for item in products)
    }
    product_lines: dict[str, set[str]] = {}
    for capability in (_plain_row(item) for item in product_line_capabilities):
        line_id = _canonical_id(_first_present(capability, ["line_id"]))
        product_id = _canonical_id(_first_present(capability, ["product_id"]))
        if line_id in line_ids and product_id is not None:
            product_lines.setdefault(product_id, set()).add(line_id)
    rows = []
    for product_id in sorted(product_lines):
        product = product_by_id.get(product_id, {})
        rows.append(
            {
                "product_id": _id_or_none(product_id),
                "product_code": _first_present(product, ["product_code"]),
                "product_name": _first_present(product, ["product_name"]),
                "line_ids": [_id_or_none(line_id) for line_id in sorted(product_lines[product_id])],
            }
        )
    return rows


def _unchanged_overlapping_orders(
    baseline_plan_rows: list[dict[str, Any]],
    change_schedule: list[dict[str, Any]],
    period: dict[str, Any],
) -> list[dict[str, Any]]:
    changed_order_ids = {
        _canonical_id(row.get("order_id"))
        for row in change_schedule
        if row.get("before_schedule") != row.get("after_schedule")
        or row.get("line_change") != "-"
    }
    start_at = _as_datetime(period.get("start_at"))
    end_at = _as_datetime(period.get("end_at"))
    rows = []
    for plan in baseline_plan_rows:
        order_id = _canonical_id(_first_present(plan, ["order_id"]))
        plan_start = _as_datetime(_first_present(plan, ["planned_start_at", "start_time"]))
        plan_end = _as_datetime(_first_present(plan, ["planned_end_at", "end_time"]))
        if (
            order_id is None
            or order_id in changed_order_ids
            or start_at is None
            or end_at is None
            or plan_start is None
            or plan_end is None
            or not (plan_end > start_at and plan_start < end_at)
        ):
            continue
        rows.append(
            {
                "order_id": _id_or_none(order_id),
                "plan_id": _id_or_none(_first_present(plan, ["plan_id", "schedule_id"])),
                "line_id": _id_or_none(_first_present(plan, ["line_id"])),
                "planned_start_at": _iso_or_none(plan_start),
                "planned_end_at": _iso_or_none(plan_end),
            }
        )
    return rows


def _plain_row(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, dict):
        return value
    return {}


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _as_date(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _coerce_percent(value: Any) -> float | None:
    decimal_value = _to_decimal(value)
    if decimal_value is None:
        return None
    if Decimal("-1") <= decimal_value <= Decimal("1"):
        decimal_value *= Decimal("100")
    return _round_decimal(decimal_value)


def _coerce_decimal_or_none(value: Any) -> float | None:
    decimal_value = _to_decimal(value)
    if decimal_value is None:
        return None
    return _round_decimal(decimal_value)


def _money_or_none(value: Any) -> str | None:
    decimal_value = _to_decimal(value)
    if decimal_value is None:
        return None
    return _money_string(decimal_value)


def _money_string(value: Any) -> str | None:
    decimal_value = _to_decimal(value)
    if decimal_value is None:
        return None
    return str(decimal_value.quantize(_MONEY_QUANT, rounding=ROUND_HALF_UP))


def _round_decimal(value: Decimal) -> float:
    return float(value.quantize(_PERCENT_QUANT, rounding=ROUND_HALF_UP))


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _decimal_delta(baseline_value: Any, alternative_value: Any) -> Decimal | None:
    baseline = _to_decimal(baseline_value)
    alternative = _to_decimal(alternative_value)
    if baseline is None or alternative is None:
        return None
    return baseline - alternative


def _reduction_percent(baseline_value: Any, alternative_value: Any) -> float | None:
    baseline = _to_decimal(baseline_value)
    alternative = _to_decimal(alternative_value)
    if baseline is None or alternative is None or baseline == 0:
        return None
    return _round_decimal(((baseline - alternative) / baseline) * Decimal("100"))


def _comparison_entry(
    baseline_value: Any,
    alternative_value: Any,
    unit: str,
    *,
    money: bool = False,
) -> dict[str, Any]:
    delta = _decimal_delta(baseline_value, alternative_value)
    status = "OK" if delta is not None else "MISSING_VALUE"
    serialized_delta: Any = _money_string(delta) if money else _decimal_to_float(delta)
    return {
        "baseline_value": baseline_value,
        "alternative_value": alternative_value,
        "delta": serialized_delta,
        "unit": unit,
        "calculation_status": status,
    }


def _decimal_to_float(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return _round_decimal(value)


def _is_important_summary_event(row: dict[str, Any], event_name: str) -> bool:
    return (
        event_name in IMPORTANT_EVENT_TYPES
        or float(row.get("avg_risk_cost_when_event_occurs") or 0) > 0
        or float(row.get("avg_tardiness_minutes_when_event_occurs") or 0) > 0
        or float(row.get("avg_material_shortage_penalty_when_event_occurs") or 0) > 0
    )


def _is_important_timeline_event(row: dict[str, Any], event_name: str) -> bool:
    return (
        event_name in IMPORTANT_EVENT_TYPES
        or float(row.get("delivery_delay_minutes") or 0) > 0
        or float(row.get("risk_cost") or 0) > 0
        or float(row.get("loss_amount") or 0) > 0
    )


def _json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _iter_text_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_text_values(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_text_values(item)


def _collect_numeric_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for text in _iter_grounding_values(value):
        tokens.update(_NUMERIC_TOKEN_PATTERN.findall(text))
    return tokens


def _normalize_numeric_token(token: str) -> str | None:
    decimal_value = _to_decimal(token.replace(",", ""))
    if decimal_value is None:
        return None
    return format(decimal_value.normalize(), "f")


def _collect_id_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for text in _iter_grounding_values(value):
        tokens.update(_ID_TOKEN_PATTERN.findall(text))
    return tokens


def _collect_uppercase_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    for text in _iter_grounding_values(value):
        tokens.update(_UPPERCASE_TOKEN_PATTERN.findall(text))
    return tokens


def _iter_grounding_values(value: Any):
    if isinstance(value, Decimal):
        text = str(value)
        yield text
        yield _money_string(value) or text
    elif isinstance(value, (int, float)):
        yield str(value)
        decimal_value = _to_decimal(value)
        if decimal_value is not None:
            yield str(decimal_value.normalize())
            yield _money_string(decimal_value) or str(value)
            yield str(int(decimal_value)) if decimal_value == int(decimal_value) else str(value)
    elif isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_grounding_values(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _iter_grounding_values(item)


def _iso_or_none(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value is not None else None
