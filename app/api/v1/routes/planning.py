import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse
from pydantic import ConfigDict, Field

from app.api.v1.dependencies import verify_internal_api_token
from app.features.production_planning.exceptions import (
    PlanningDataAccessError,
    PlanningInfeasibleError,
    PlanningValidationError,
    SolutionExtractionError,
    SolverExecutionError,
)
from app.features.production_planning.production_planning_node import (
    generate_adjusted_production_plan_api_response,
)
from app.features.production_planning.schemas import ProductionPlanningAdjustmentRequest
from app.schemas.base import ApiSchema

router = APIRouter(
    prefix="/production-planning",
    tags=["Production Planning"],
    dependencies=[Depends(verify_internal_api_token)],
)
legacy_router = APIRouter(
    prefix="/planning",
    tags=["Production Planning"],
    dependencies=[Depends(verify_internal_api_token)],
    include_in_schema=False,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models — planning_response
# ---------------------------------------------------------------------------


class AdjustedPlanRowItem(ApiSchema):
    order_id: int = Field(..., description="수주 ID (DB 정수 키)")
    product_id: int = Field(..., description="제품 ID")
    line_id: int = Field(..., description="배정된 생산 라인 ID")
    operator_id: int | None = Field(
        None,
        description=(
            "작업자 ID. ai_planning.v_operators_for_planning에서 조회한 operator id 중 "
            "하나를 랜덤 배정하며, 사용 가능한 작업자가 없으면 null입니다."
        ),
    )
    planned_start_at: str = Field(..., description="계획 시작 시각 (ISO 8601, timezone-aware)")
    planned_end_at: str = Field(..., description="계획 종료 시각 (ISO 8601, timezone-aware)")
    estimated_duration_hr: float = Field(
        ..., description="계획 소요 시간 (시간 단위, 수량/라인 UPH 기반)"
    )
    planned_quantity: int = Field(..., description="계획 생산 수량")
    plan_sequence: int = Field(..., description="라인 내 생산 순서 (1부터 시작)")
    plan_status: str = Field(
        ...,
        description="계획 상태. 예: SCHEDULED, DELAYED, IN_PROGRESS, COMPLETED",
    )


class AdjustedPlanCandidateItem(ApiSchema):
    plan_variant_code: str = Field(
        ...,
        description=(
            "최적화 변형 코드. "
            "DUE_DATE_OPTIMAL: 납기 지연 주문 수 최소화 우선, "
            "AMOUNT_OPTIMAL: 계약 금액 보호·비용 최소화 우선"
        ),
    )
    plan_variant_name: str = Field(..., description="변형 코드의 한국어 표시 이름")
    status: str = Field(
        ...,
        description=(
            "CP-SAT 솔버 결과 상태. OPTIMAL: 최적해 도달, "
            "FEASIBLE: 시간 내 실현 가능한 해, INFEASIBLE: 해 없음"
        ),
    )
    plans: list[AdjustedPlanRowItem] = Field(
        ..., description="해당 변형의 조정된 생산 계획 row 목록 (라인 × 시퀀스 순 정렬)"
    )
    unscheduled_orders: list[str] = Field(
        default_factory=list,
        description=(
            "해당 변형에서 스케줄에 배정되지 못한 내부 order ID 목록. "
            "기존 생산계획 기반 항목은 PLAN-{planId} 형식입니다."
        ),
    )
    unscheduled_plan_ids: list[int] = Field(
        default_factory=list,
        description="unscheduled_orders 중 PLAN-{planId} 형식에서 추출한 생산계획 ID 목록",
    )


class PlanningResponseBody(ApiSchema):
    adjusted_plan_candidates: list[AdjustedPlanCandidateItem] = Field(
        ...,
        description=(
            "CP-SAT가 생성한 조정 계획 후보 목록. "
            "프론트엔드 '반영하기' 플로우에서 DB 저장용으로 사용됩니다. "
            "통상 DUE_DATE_OPTIMAL, AMOUNT_OPTIMAL 2개 변형이 포함됩니다."
        ),
    )


# ---------------------------------------------------------------------------
# Response models — simulation_response / baseline
# ---------------------------------------------------------------------------


class BaselineMetrics(ApiSchema):
    delay_probability_percent: float | None = Field(
        None,
        description=(
            "현재 계획 기준 딜레이 확률 (%). "
            "현재 계획의 지연 주문 수 ÷ 전체 주문 수로 산출됩니다."
        ),
    )
    delay_probability_basis: str | None = Field(
        None,
        description="딜레이 확률 산출 방식. CURRENT_PLAN_DELAYED_ORDER_RATE: 현재 계획 기준 지연율",
    )
    expected_delayed_orders: float | None = Field(
        None, description="현재 계획 기준 지연 예상 주문 수 (order 단위)"
    )
    p95_tardiness_minutes: float | None = Field(
        None,
        description=(
            "현재 계획 기준 최대 지연 시간 (분). "
            "95분위 대신 max_tardiness_minutes를 사용합니다."
        ),
    )
    total_risk_cost: str | None = Field(
        None,
        description="총 위험 비용 (KRW, 소수점 2자리 문자열). 지연 위약금 합계입니다.",
    )
    material_shortage_probability_percent: None = Field(
        None,
        description=(
            "자재 부족 확률 (%). 현재 계획 기준에서는 구조적으로 산출 불가하여 "
            "항상 null입니다."
        ),
    )
    expected_delay_days: float | None = Field(
        None,
        description=(
            "평균 예상 지연 일수. 전체 주문의 total_tardiness_minutes ÷ 1440으로 "
            "산출됩니다."
        ),
    )
    delayed_orders_days: float | None = Field(
        None,
        description="expected_delay_days와 동일한 납기 지연일 alias입니다.",
    )
    delivery_fulfillment_rate_percent: float | None = Field(
        None,
        description=(
            "납기 충족률 (%). "
            "제품 단위가 통일된 경우 수량 기준(QUANTITY), 그렇지 않으면 "
            "주문 수 기준(ORDER)으로 산출됩니다."
        ),
    )
    delivery_miss_rate_percent: float | None = Field(
        None, description="납기 미달율 (%). 100 - delivery_fulfillment_rate_percent"
    )
    delay_risk_order_count: float | None = Field(
        None,
        description=(
            "지연 위험 주문 수. planned_end_at > due_date 인 주문의 고유 order_id 수입니다. "
            "동일 주문이 여러 plan row로 분할된 경우 1건으로 집계됩니다."
        ),
    )
    avg_line_utilization_percent: float | None = Field(
        None,
        description="평균 라인 가동률 (%). 계획 기간 대비 각 라인의 점유 시간 비율 평균입니다.",
    )
    bottleneck_line_id: int | None = Field(
        None, description="병목 라인 ID. 가동률이 가장 높은 라인입니다."
    )
    unit_basis: str | None = Field(
        None,
        description="납기 충족률 계산 기준. ORDER: 주문 수 기준, QUANTITY: 생산 수량 기준",
    )
    total_tardiness_minutes: int | None = Field(
        None, description="전체 주문의 총 지연 시간 합계 (분)"
    )
    total_late_penalty_amount: str | None = Field(
        None, description="총 지연 위약금 합계 (KRW, 소수점 2자리 문자열)"
    )
    baseline_plan_count: int | None = Field(
        None, description="기준 계획 DB row 수 (plan_id 단위)"
    )
    baseline_plan_ids: list[int] | None = Field(
        None, description="기준 계획 plan_id 목록"
    )
    plan_completion_at: str | None = Field(
        None,
        description="현재 계획 전체 완료 예상 시각 (ISO 8601). 가장 늦은 planned_end_at 값입니다.",
    )
    calculation_status: str = Field(
        ...,
        description=(
            "지표 산출 상태. "
            "OK: 필수 필드 모두 정상 산출, "
            "MISSING_VALUE: 일부 필드 누락 (missing_fields 참고)"
        ),
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="calculation_status가 MISSING_VALUE일 때 누락된 필드명 목록",
    )
    baseline_metric_source: str | None = Field(
        None, description="기준 지표 소스 식별자. 항상 DB_CURRENT_PLAN입니다."
    )


class BaselinePlanRow(ApiSchema):
    model_config = ConfigDict(
        alias_generator=ApiSchema.model_config["alias_generator"],
        populate_by_name=True,
        extra="allow",
    )

    plan_id: int | None = Field(None, description="계획 ID (DB 기본키)")
    order_id: int | None = Field(None, description="수주 ID")
    line_id: int | None = Field(None, description="생산 라인 ID")
    product_id: int | None = Field(None, description="제품 ID")
    planned_start_at: str | None = Field(None, description="계획 시작 시각 (ISO 8601)")
    planned_end_at: str | None = Field(None, description="계획 종료 시각 (ISO 8601)")
    planned_quantity: int | None = Field(None, description="계획 생산 수량")
    due_date: str | None = Field(None, description="납기일 (ISO 8601)")
    late_penalty_amount: str | None = Field(
        None, description="지연 위약금 (KRW, 문자열)"
    )
    plan_status: str | None = Field(None, description="계획 상태")


class CurrentStateSummary(ApiSchema):
    expected_delay_days: float | None = Field(
        None, description="평균 예상 지연 일수 (total_tardiness_minutes ÷ 1440)"
    )
    delayed_orders_days: float | None = Field(
        None, description="expected_delay_days와 동일한 납기 지연일 alias"
    )
    total_tardiness_minutes: int | None = Field(
        None, description="총 지연 시간 합계 (분)"
    )
    max_tardiness_minutes: int | None = Field(
        None, description="단일 주문 최대 지연 시간 (분)"
    )
    delivery_fulfillment_rate_percent: float | None = Field(
        None, description="납기 충족률 (%)"
    )
    delivery_miss_rate_percent: float | None = Field(
        None, description="납기 미달율 (%)"
    )
    delay_probability_percent: float | None = Field(
        None, description="딜레이 확률 (%, 현재 계획 기준)"
    )
    delay_probability_basis: str | None = Field(
        None, description="딜레이 확률 산출 방식"
    )
    delay_risk_order_count: int | None = Field(
        None, description="지연 위험 고유 주문 수"
    )
    expected_delayed_orders: int | None = Field(
        None, description="예상 지연 주문 수 (delay_risk_order_count와 동일)"
    )
    avg_line_utilization_percent: float | None = Field(
        None, description="평균 라인 가동률 (%)"
    )
    bottleneck_line_id: int | None = Field(None, description="병목 라인 ID")
    unit_basis: str | None = Field(
        None, description="비율 계산 기준 (ORDER 또는 QUANTITY)"
    )
    total_quantity_or_orders: float | None = Field(
        None,
        description="전체 주문 수 또는 수량 (unit_basis에 따라 다름)",
    )
    delayed_quantity_or_orders: float | None = Field(
        None, description="지연된 주문 수 또는 수량"
    )
    total_late_penalty_amount: str | None = Field(
        None, description="총 지연 위약금 (KRW, 문자열)"
    )
    total_risk_cost: str | None = Field(
        None, description="총 위험 비용 (KRW, 문자열). total_late_penalty_amount와 동일합니다."
    )
    baseline_plan_count: int | None = Field(
        None, description="기준 계획 row 수"
    )
    baseline_plan_ids: list[int] | None = Field(
        None, description="기준 계획 ID 목록"
    )
    plan_completion_at: str | None = Field(
        None, description="계획 완료 예상 시각 (ISO 8601)"
    )
    calculation_status: str = Field(
        ..., description="산출 상태 (OK 또는 MISSING_VALUE)"
    )


class BaselineProvenance(ApiSchema):
    source: str = Field(
        ...,
        description=(
            "기준 데이터 소스 뷰명. 항상 "
            "ai_planning.v_existing_schedules_for_planning입니다."
        ),
    )
    plan_count: int = Field(..., description="조회된 기준 계획 row 수")
    planning_start: str | None = Field(
        None, description="조회 기간 시작 (ISO 8601)"
    )
    planning_end: str | None = Field(
        None, description="조회 기간 종료 (ISO 8601)"
    )
    first_planned_start_at: str | None = Field(
        None, description="기준 계획 중 가장 이른 시작 시각 (ISO 8601)"
    )
    last_planned_end_at: str | None = Field(
        None, description="기준 계획 중 가장 늦은 종료 시각 (ISO 8601)"
    )


class PlanValueAnalysis(ApiSchema):
    contract_total: int = Field(
        ...,
        description="계획에 포함된 주문의 총 수주 금액입니다. 단위는 KRW입니다.",
    )
    expected_penalty_total: int = Field(
        ...,
        description=(
            "지연 확률이 threshold 이상인 주문에 대해 반영한 예상 지연 패널티 합계입니다. "
            "단위는 KRW입니다."
        ),
    )
    material_adj_quantity_total: float = Field(
        ...,
        description=(
            "BOM 소요량, loss_rate, line yield를 반영한 자재 조정 수량 합계입니다. "
            "자재 단가가 없으므로 금액이 아니라 수량 단위입니다."
        ),
    )
    line_change_fee_total: int = Field(
        ...,
        description=(
            "같은 라인에서 이전 제품과 다음 제품이 달라질 때 발생하는 changeover_cost 합계입니다. "
            "from_product_id와 to_product_id가 같으면 0으로 계산됩니다."
        ),
    )
    plan_net_value_monetary: int = Field(
        ...,
        description=(
            "금액 기준 계획 순가치입니다. "
            "contract_total - expected_penalty_total - line_change_fee_total로 계산합니다."
        ),
    )
    delay_threshold_percent: float = Field(
        ...,
        description="예상 지연 패널티를 반영할 order-level 지연 확률 기준값입니다. 단위는 %입니다.",
    )
    delay_flag_order_count: int = Field(
        ...,
        description="지연 확률이 delay_threshold_percent 이상이라 패널티가 반영된 주문 수입니다.",
    )
    note: str = Field(
        ...,
        description=(
            "material_adj_quantity_total이 금액이 아니라 수량 단위임을 설명하는 메모입니다."
        ),
    )


class BaselineBlock(ApiSchema):
    source: str = Field(
        ..., description="기준 데이터 출처 식별자. 항상 DB_CURRENT_PLAN입니다."
    )
    plans: list[dict[str, Any]] = Field(
        ...,
        description=(
            "DB에서 조회한 현재 생산 계획 row 목록. "
            "plan_id, order_id, line_id, planned_start_at, planned_end_at, "
            "due_date 등의 필드를 포함합니다."
        ),
    )
    simulation_metrics: BaselineMetrics = Field(
        ..., description="현재 계획 기반으로 산출된 기준 KPI 지표"
    )
    current_state_summary: CurrentStateSummary = Field(
        ...,
        description=(
            "납기·지연·가동률 등 현재 상태 요약. "
            "simulation_metrics의 원본 산출 데이터입니다."
        ),
    )
    plan_value_analysis: PlanValueAnalysis | None = Field(
        None,
        description=(
            "현재 DB 생산 계획의 계획 순가치 분석입니다. "
            "수주 금액, 예상 지연 패널티, 자재 조정 수량, 라인 전환 비용을 포함합니다."
        ),
    )
    provenance: BaselineProvenance = Field(
        ..., description="기준 데이터 출처 메타데이터"
    )


# ---------------------------------------------------------------------------
# Response models — simulation_response / alternatives
# ---------------------------------------------------------------------------


class AlternativeSimulationMetrics(ApiSchema):
    delay_probability_percent: float | None = Field(
        None,
        description=(
            "Monte Carlo 시뮬레이션 기반 딜레이 확률 (%). "
            "1000회 반복 시뮬레이션 중 지연이 발생한 비율입니다."
        ),
    )
    expected_delayed_orders: float | None = Field(
        None,
        description=(
            "대안 계획 기준 지연 주문 수. planned_end_at > due_date 인 "
            "고유 order_id 수로 산출합니다."
        ),
    )
    p95_tardiness_minutes: float | None = Field(
        None, description="시뮬레이션 95분위 지연 시간 (분). 최악 시나리오 수준의 지표입니다."
    )
    total_risk_cost: str | None = Field(
        None,
        description=(
            "시뮬레이션 기반 평균 총 위험 비용 (KRW, 소수점 2자리 문자열). "
            "지연 위약금 + 자재 부족 패널티 포함입니다."
        ),
    )
    material_shortage_probability_percent: float | None = Field(
        None,
        description="시뮬레이션 기반 자재 부족 발생 확률 (%). BOM 소비량 vs 재고 기반 산출입니다.",
    )
    expected_delay_days: float | None = Field(
        None,
        description=(
            "대안 계획 기준 납기 지연일. baseline과 동일하게 "
            "total_tardiness_minutes ÷ 1440으로 산출합니다."
        ),
    )
    delayed_orders_days: float | None = Field(
        None,
        description="expected_delay_days와 동일한 납기 지연일 alias입니다.",
    )
    total_tardiness_minutes: int | None = Field(
        None, description="대안 계획의 총 지연 시간 합계 (분)"
    )
    max_tardiness_minutes: int | None = Field(
        None, description="대안 계획의 단일 주문 최대 지연 시간 (분)"
    )
    delivery_fulfillment_rate_percent: float | None = Field(
        None, description="납기 충족률 (%)"
    )
    delivery_miss_rate_percent: float | None = Field(
        None, description="납기 미달율 (%)"
    )
    delay_risk_order_count: float | None = Field(
        None,
        description=(
            "대안 계획 기준 지연 위험 주문 수. planned_end_at > due_date 인 "
            "고유 order_id 수입니다."
        ),
    )
    avg_line_utilization_percent: None = Field(
        None,
        description="평균 라인 가동률 (%). 대안 계획에서는 현재 미지원으로 항상 null입니다.",
    )
    bottleneck_line_id: None = Field(
        None, description="병목 라인 ID. 대안 계획에서는 현재 미지원으로 항상 null입니다."
    )
    plan_completion_at: str | None = Field(
        None, description="대안 계획 전체 완료 예상 시각 (ISO 8601)"
    )
    alternative_plan_count: int | None = Field(
        None, description="대안 계획 row 수"
    )
    matched_baseline_plan_count: int | None = Field(
        None,
        description="기준 계획과 매칭된 대안 계획 수 (plan_id 또는 order_id 기준)",
    )
    new_plan_count: int | None = Field(
        None, description="신규 추가된 계획 수 (기준 계획에 없는 add_orders)"
    )
    plan_completion_rank: int | None = Field(
        None,
        description="계획 완료 시각 기준 순위 (1=가장 빠름). 변형 간 비교 순위입니다.",
    )
    risk_cost_rank: int | None = Field(
        None,
        description="총 위험 비용 기준 순위 (1=가장 낮음). 변형 간 비교 순위입니다.",
    )
    unit_basis: str | None = Field(
        None, description="비율 계산 기준 (ORDER 또는 QUANTITY)"
    )
    total_quantity_or_orders: float | None = Field(
        None, description="전체 주문 수 또는 수량"
    )
    delayed_quantity_or_orders: float | None = Field(
        None, description="지연된 주문 수 또는 수량"
    )
    calculation_status: str = Field(
        ...,
        description="지표 산출 상태 (OK 또는 MISSING_VALUE)",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="누락된 필드명 목록",
    )


class ComputedDeltas(ApiSchema):
    delivery_fulfillment_rate_delta_percent_points: float | None = Field(
        None,
        description=(
            "납기 충족률 변화량 (%p). "
            "alternative - baseline. 양수: 개선, 음수: 악화"
        ),
    )
    delay_risk_order_reduction: float | None = Field(
        None,
        description=(
            "지연 위험 주문 수 감소량. "
            "baseline - alternative. 양수: 감소(개선), 음수: 증가(악화)"
        ),
    )
    delay_probability_reduction_percent: float | None = Field(
        None,
        description=(
            "딜레이 확률 감소율 (%). "
            "(baseline - alternative) ÷ baseline × 100. 양수: 개선"
        ),
    )
    delay_probability_delta_percent_points: float | None = Field(
        None,
        description=(
            "딜레이 확률 변화량 (%p). "
            "baseline - alternative. 양수: 개선"
        ),
    )
    expected_delayed_order_reduction: float | None = Field(
        None,
        description=(
            "예상 지연 주문 수 감소량. "
            "baseline - alternative. 양수: 감소(개선)"
        ),
    )
    risk_cost_saving_amount: str | None = Field(
        None,
        description=(
            "위험 비용 절감액 (KRW, 소수점 2자리 문자열). "
            "baseline - alternative. 양수: 절감(개선), 음수: 증가(악화)"
        ),
    )
    risk_cost_saving_percent: float | None = Field(
        None,
        description="위험 비용 절감율 (%). (baseline - alternative) ÷ baseline × 100",
    )
    material_shortage_probability_reduction_percent: float | None = Field(
        None, description="자재 부족 확률 감소율 (%)"
    )
    plan_count_delta: float | None = Field(
        None,
        description=(
            "계획 row 수 변화량. "
            "baseline_plan_count - alternative_plan_count. 양수: 계획 수 감소"
        ),
    )
    matched_baseline_plan_count: int | None = Field(
        None, description="기준 계획과 매칭된 대안 계획 수"
    )
    new_plan_count: int | None = Field(
        None, description="신규 추가된 계획 수"
    )
    plan_completion_delta_hours: float | None = Field(
        None,
        description=(
            "계획 완료 시각 차이 (시간). "
            "baseline - alternative. 양수: 대안이 더 빠름(개선)"
        ),
    )
    expected_delay_days_reduction: float | None = Field(
        None,
        description=(
            "예상 지연 일수 감소량. "
            "baseline - alternative. 양수: 개선, 음수: 악화"
        ),
    )
    delayed_orders_days_reduction: float | None = Field(
        None,
        description=(
            "납기 지연일 감소량 alias. "
            "baseline - alternative. 양수: 개선, 음수: 악화"
        ),
    )


class ComparisonTableRow(ApiSchema):
    metric_code: str = Field(..., description="지표 코드입니다.")
    metric_name: str = Field(..., description="지표 한국어 이름입니다.")
    baseline_value: float | None = Field(None, description="기준 계획 값")
    alternative_value: float | None = Field(None, description="대안 계획 값")
    delta: float | None = Field(None, description="변화량 (방향은 improvement_direction 기준)")
    change_text: str | None = Field(
        None,
        description="화면 표시용 변화 설명입니다. 예: '10.51 percent_point 개선'",
    )
    unit: str = Field(..., description="단위 (percent_point, orders 등)")
    calculation_status: str = Field(
        ...,
        description=(
            "비교 상태. "
            "OK: 비교 가능, "
            "NOT_COMPARABLE: 비교 불가"
        ),
    )


class PlanChangeRow(ApiSchema):
    order_id: int | None = Field(None, description="수주 ID")
    plan_id: int | None = Field(None, description="기존 계획 ID")
    line_change: str | None = Field(
        None, description="라인 변경 텍스트. 예: '라인 2 → 라인 3' 또는 '라인 유지'"
    )
    before_schedule: str | None = Field(
        None, description="기존 계획 일정 요약 텍스트. 예: '06/01 09:00 ~ 06/02 18:00'"
    )
    after_schedule: str | None = Field(
        None, description="변경 후 계획 일정 요약 텍스트"
    )
    sequence_change: str | None = Field(
        None, description="시퀀스 변경 텍스트. 예: '순서 3 → 순서 1'"
    )
    quantity_change: str | None = Field(
        None, description="수량 변경 텍스트. 예: '1,000 → 1,200 EA'"
    )
    delay_status_change: str | None = Field(
        None,
        description=(
            "지연 상태 변화 텍스트. 예: '지연 → 정시', '정시 → 지연', '정시 유지', '지연 유지'"
        ),
    )


class ApplicationConditions(ApiSchema):
    available_lines: list[dict[str, Any]] = Field(
        ...,
        description=(
            "적용 대상 생산 라인 목록. "
            "변경된 라인 또는 대안 계획에 포함된 라인의 마스터 정보입니다. "
            "line_id, line_name 등의 필드를 포함합니다."
        ),
    )
    target_products: list[dict[str, Any]] = Field(
        ...,
        description=(
            "적용 대상 제품 목록. "
            "적용 라인에서 생산 가능한 제품의 마스터 정보입니다. "
            "product_id, product_name 등의 필드를 포함합니다."
        ),
    )
    applicable_period: dict[str, Any] = Field(
        ...,
        description=(
            "적용 기간 정보. "
            "대안 계획의 시작~종료 구간을 나타냅니다. "
            "start, end 필드를 포함합니다."
        ),
    )
    unchanged_overlapping_orders: list[dict[str, Any]] = Field(
        ...,
        description=(
            "적용 기간과 겹치지만 변경되지 않은 기존 계획 목록. "
            "반영 시 영향 받을 수 있는 인접 계획을 확인하는 데 사용됩니다."
        ),
    )


class AiRecommendation(ApiSchema):
    summary_text: str = Field(
        ...,
        description=(
            "AI 추천 요약 텍스트 (한국어). "
            "DUE_DATE_OPTIMAL은 납기 지연 감소를 주 근거로, "
            "AMOUNT_OPTIMAL은 비용 절감을 주 근거로 작성됩니다."
        ),
    )
    reasons: list[str] = Field(
        ...,
        description=(
            "추천 이유 목록 (한국어 완전 문장). "
            "각 항목은 정량 수치를 포함합니다. "
            "예: '딜레이 확률이 12.50%p 낮아져 납기 안정성이 개선됩니다.'"
        ),
    )


class RiskInterpretation(ApiSchema):
    text: str = Field(
        ...,
        description="리스크 해석 텍스트입니다.",
    )


class AiEvaluationBlock(ApiSchema):
    status: str = Field(
        ...,
        description=(
            "AI 평가 상태. "
            "COMPLETED: LLM 평가 정상 완료, "
            "FAILED: LLM 호출 실패 (fallback 텍스트 사용)"
        ),
    )
    current_state_summary: dict[str, Any] = Field(
        ...,
        description=(
            "현재 상태 요약 (AI 평가용). "
            "baseline 지표와 alternative 지표를 비교 요약한 구조입니다."
        ),
    )
    risk_interpretation: RiskInterpretation = Field(
        ..., description="리스크 해석 텍스트 (LLM 생성)"
    )
    ai_recommendation: AiRecommendation = Field(
        ..., description="AI 추천 내용 (LLM 생성)"
    )
    recommendation_level: str = Field(
        ...,
        description=(
            "추천 수준 코드. "
            "STRONG_RECOMMEND, RECOMMEND, NEUTRAL, NOT_RECOMMEND 중 하나입니다."
        ),
    )
    recommendation_grade_label: str = Field(
        ...,
        description=(
            "추천 등급 레이블 (한국어). "
            "매우 낮음, 낮음, 보통, 높음, 매우 높음 중 하나입니다."
        ),
    )
    recommendation_grade_basis: list[str] = Field(
        ...,
        description=(
            "추천 등급 근거 목록 (한국어 완전 문장). "
            "각 항목은 정량 수치와 개선 방향을 포함합니다. "
            "예: '딜레이 확률이 12.50%p 감소해 일정 리스크가 낮아집니다.'"
        ),
    )


class AlternativeBlock(ApiSchema):
    plan_variant_code: str = Field(
        ...,
        description="최적화 변형 코드 (DUE_DATE_OPTIMAL 또는 AMOUNT_OPTIMAL)",
    )
    plan_variant_name: str = Field(..., description="변형 코드의 한국어 표시 이름")
    status: str = Field(
        ...,
        description="CP-SAT 솔버 결과 상태 (OPTIMAL, FEASIBLE, INFEASIBLE)",
    )
    plans: list[dict[str, Any]] = Field(
        ...,
        description=(
            "대안 계획 row 목록. "
            "order_id, line_id, planned_start_at, planned_end_at, planned_quantity, "
            "plan_sequence 등을 포함합니다."
        ),
    )
    simulation_metrics: AlternativeSimulationMetrics = Field(
        ...,
        description="Monte Carlo 시뮬레이션 및 CP-SAT 결과 기반 KPI 지표",
    )
    computed_deltas: ComputedDeltas = Field(
        ...,
        description=(
            "baseline 대비 대안의 지표 변화량. "
            "양수 = 개선, 음수 = 악화 (지표별 방향 정의 참고)"
        ),
    )
    simulation_comparison_table: list[ComparisonTableRow] = Field(
        ...,
        description=(
            "주요 지표별 비교 테이블. "
            "딜레이 확률, 납기 충족률, 지연 위험 주문 수, 라인 가동률 항목을 포함합니다."
        ),
    )
    plan_value_analysis: PlanValueAnalysis | None = Field(
        None,
        description=(
            "대안 계획의 계획 순가치 분석입니다. "
            "order별 Monte Carlo 지연 확률을 기준으로 예상 패널티를 반영합니다."
        ),
    )
    application_conditions: ApplicationConditions = Field(
        ..., description="계획 반영 시 영향 범위 정보"
    )
    selected_plan_change_schedule: list[PlanChangeRow] = Field(
        ...,
        description=(
            "주문별 일정 변경 내역. "
            "기존 계획 대비 라인·시각·시퀀스·수량·지연 상태 변화를 나타냅니다."
        ),
    )
    important_events: list[dict[str, Any]] = Field(
        ...,
        description=(
            "시뮬레이션 주요 이벤트 목록 (최대 10개). "
            "지연·자재 부족·설비 고장 등 위험도가 높은 이벤트를 우선 포함합니다. "
            "source(SUMMARY|TIMELINE), event, occurrence_count 등의 필드를 포함합니다."
        ),
    )
    ai_evaluation: AiEvaluationBlock = Field(
        ...,
        description=(
            "LLM 기반 AI 평가 결과. "
            "enable_llm_evaluation=True 일 때 실제 LLM 텍스트가 채워지며, "
            "False일 때는 fallback 기본값이 사용됩니다."
        ),
    )


# ---------------------------------------------------------------------------
# Response models — top level
# ---------------------------------------------------------------------------


class PlanningWindow(ApiSchema):
    planning_start: str | None = Field(
        None, description="계획 기간 시작 (ISO 8601, timezone-aware)"
    )
    planning_end: str | None = Field(
        None, description="계획 기간 종료 (ISO 8601, timezone-aware)"
    )


class DataSources(ApiSchema):
    baseline: str = Field(
        ...,
        description="기준 데이터 소스 식별자. 항상 DB_CURRENT_PLAN입니다.",
    )
    alternative: str = Field(
        ...,
        description="대안 데이터 소스 식별자. 항상 CP_SAT_AND_SIMULATION입니다.",
    )


class SimulationResponseBody(ApiSchema):
    generated_at: str = Field(
        ...,
        description="응답 생성 시각 (ISO 8601, UTC)",
    )
    planning_window: PlanningWindow = Field(
        ..., description="최적화 요청 기간 정보"
    )
    data_sources: DataSources = Field(
        ..., description="baseline과 alternative 데이터 출처 식별자"
    )
    baseline: BaselineBlock = Field(
        ...,
        description=(
            "현재 DB 생산 계획 기반 기준선. "
            "대안 비교의 기준이 되며 plan row·지표·요약·출처 정보를 포함합니다."
        ),
    )
    alternatives: list[AlternativeBlock] = Field(
        ...,
        description=(
            "CP-SAT로 생성된 대안 계획 평가 결과 목록. "
            "통상 DUE_DATE_OPTIMAL, AMOUNT_OPTIMAL 2개 변형이 포함됩니다."
        ),
    )
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "비치명적 경고 메시지 목록. "
            "DB 조회 실패로 fallback 사용 시, 시뮬레이션 누락 시 등 발생합니다."
        ),
    )


class PlanningGenerateResponse(ApiSchema):
    planning_response: PlanningResponseBody = Field(
        ...,
        description=(
            "프론트엔드 '반영하기' 플로우에서 DB 저장에 사용하는 계획 데이터. "
            "adjusted_plan_candidates 목록을 포함합니다."
        ),
    )
    simulation_response: SimulationResponseBody = Field(
        ...,
        description=(
            "대시보드 표시용 분석 결과. "
            "baseline(현재 계획) 대비 대안 계획의 KPI 비교·AI 평가·이벤트 목록을 포함합니다."
        ),
    )


class PlanningHealthResponse(ApiSchema):
    status: str = Field(..., description="서비스 상태. 정상 시 'ok'를 반환합니다.")
    feature: str = Field(
        ..., description="기능 식별자. 항상 'production-planning'입니다."
    )


class PlanningErrorResponse(ApiSchema):
    status: str = Field(..., description="HTTP 상태 코드 문자열. 예: '400 Bad Request'")
    message: str = Field(..., description="오류 상세 메시지")


planning_error_responses = {
    400: {
        "model": PlanningErrorResponse,
        "description": "요청 데이터 유효성 오류 또는 계획 인자 검증 실패 (PlanningValidationError)",
    },
    408: {
        "model": PlanningErrorResponse,
        "description": "CP-SAT 솔버 실행 시간 초과 또는 실행 오류 (SolverExecutionError)",
    },
    409: {
        "model": PlanningErrorResponse,
        "description": "모든 주문이 스케줄 불가능한 경우 (PlanningInfeasibleError)",
    },
    500: {
        "model": PlanningErrorResponse,
        "description": "솔버 결과 추출 실패 (SolutionExtractionError)",
    },
    503: {
        "model": PlanningErrorResponse,
        "description": "DB 조회 실패 또는 외부 데이터 접근 오류 (PlanningDataAccessError)",
    },
}

planning_request_examples = {
    "edit_and_add_orders": {
        "summary": "Edit one existing plan and add one new order",
        "description": (
            "Locks the edited DB plan schedule and lets the newly added order be "
            "optimized by CP-SAT."
        ),
        "value": {
            "planningStart": "2026-05-01 09:00:00.000 +0900",
            "planningEnd": "2026-06-09 08:59:00.000 +0900",
            "editOrders": [
                {
                    "orderId": 399,
                    "productId": 10,
                    "orderQuantity": 16800,
                    "dueDate": "2026-05-22 08:59:59.000 +0900",
                    "contractAmount": "30752426.00",
                    "latePenaltyAmount": "833160.00",
                    "orderStatus": "DELAYED",
                    "lockedPlan": {
                        "lineId": 6,
                        "plannedStartAt": "2026-06-02 00:57:31.000 +0900",
                        "plannedEndAt": "2026-06-03 02:33:31.000 +0900",
                    },
                }
            ],
            "addOrders": [
                {
                    "orderId": 900000001,
                    "productId": 10,
                    "orderQuantity": 1200,
                    "dueDate": "2026-05-21 09:00:00.000 +0900",
                    "contractAmount": "1500000.00",
                    "latePenaltyAmount": "120000.00",
                    "orderStatus": "SCHEDULED",
                }
            ],
        },
    },
    "add_orders_only": {
        "summary": "Add one movable order",
        "description": "Adds a new order and lets CP-SAT choose the line and time.",
        "value": {
            "planningStart": "2026-05-01 09:00:00.000 +0900",
            "planningEnd": "2026-06-09 08:59:00.000 +0900",
            "editOrders": [],
            "addOrders": [
                {
                    "orderId": 900000002,
                    "productId": 10,
                    "orderQuantity": 1200,
                    "dueDate": "2026-05-21 09:00:00.000 +0900",
                    "contractAmount": "1500000.00",
                    "latePenaltyAmount": "120000.00",
                    "orderStatus": "SCHEDULED",
                }
            ],
        },
    },
}


@router.post(
    "/analyze",
    response_model=PlanningGenerateResponse,
    response_model_by_alias=True,
    responses=planning_error_responses,
    summary="생산 계획 조정 및 대시보드 분석 생성",
    description=(
        "CP-SAT 기반 생산 계획 최적화를 수행하고 대시보드 분석 결과를 반환합니다.\n\n"
        "**처리 흐름:**\n"
        "1. `editOrders`: 기존 계획을 지정한 일정으로 고정(locked)합니다.\n"
        "2. `addOrders`: 신규 주문을 CP-SAT가 최적 라인과 시각에 배치합니다.\n"
        "3. 두 최적화 변형(DUE_DATE_OPTIMAL, AMOUNT_OPTIMAL)을 생성하고 "
        "Monte Carlo 시뮬레이션으로 평가합니다.\n"
        "4. `planning_response`: DB 저장용 계획 데이터 (반영하기 플로우)\n"
        "5. `simulation_response`: 대시보드 표시용 KPI 비교·AI 평가 결과\n\n"
        "**최적화 변형:**\n"
        "- `DUE_DATE_OPTIMAL`: 납기 지연 주문 수 최소화 우선 "
        "(unscheduled > delayed_count > tardiness)\n"
        "- `AMOUNT_OPTIMAL`: 계약 금액 보호·비용 최소화 우선, 납기는 soft constraint로 처리 "
        "(unscheduled > delayed_count > late_penalty > cleaning/sequence > tardiness)"
    ),
)
def generate_planning(
    request: Annotated[
        ProductionPlanningAdjustmentRequest,
        Body(openapi_examples=planning_request_examples),
    ],
) -> PlanningGenerateResponse | JSONResponse:
    """
    Parameters:
        - request: Planning adjustment request containing edit_orders and add_orders.

    Methodology:
        - Run the production planning LangGraph workflow once.
        - Return simulation_response for dashboard display and planning_response for the
          front-end '반영하기' flow.

    Output:
        - Combined planning and simulation response, or a compact planning error response.
    """
    try:
        logger.info(
            "[Planning] generate requested planning_start=%s planning_end=%s "
            "edit_orders=%s add_orders=%s",
            request.planning_start,
            request.planning_end,
            len(request.edit_orders),
            len(request.add_orders),
        )
        response = generate_adjusted_production_plan_api_response(request)
        _log_planning_response_summary(response)
        return PlanningGenerateResponse(**response)
    except (
        PlanningValidationError,
        PlanningInfeasibleError,
        SolverExecutionError,
        SolutionExtractionError,
        PlanningDataAccessError,
    ) as exc:
        logger.warning(
            "[Planning] generate failed error_type=%s status=%s message=%s",
            type(exc).__name__,
            exc.response_status,
            str(exc),
        )
        return JSONResponse(
            status_code=_http_status_code(exc.response_status),
            content=exc.to_response_error(),
        )


legacy_router.add_api_route(
    "",
    generate_planning,
    methods=["POST"],
    response_model=PlanningGenerateResponse,
    response_model_by_alias=True,
    responses=planning_error_responses,
)


@router.get(
    "/health",
    response_model=PlanningHealthResponse,
    summary="생산 계획 API 헬스 체크",
    description="솔버나 DB를 호출하지 않고 라우트 가용성만 확인합니다.",
)
def get_planning_health() -> PlanningHealthResponse:
    """
    Parameters:
        - None.

    Methodology:
        - Report route availability without touching the solver or database.

    Output:
        - Lightweight health payload for the production planning API route.
    """
    return PlanningHealthResponse(
        status="ok",
        feature="production-planning",
    )


legacy_router.add_api_route(
    "/health",
    get_planning_health,
    methods=["GET"],
    response_model=PlanningHealthResponse,
)


def _http_status_code(response_status: str) -> int:
    return int(response_status.split(" ", maxsplit=1)[0])


def _log_planning_response_summary(response: dict[str, Any]) -> None:
    planning_response = response.get("planning_response") or {}
    simulation_response = response.get("simulation_response") or {}
    candidates = planning_response.get("adjusted_plan_candidates") or []
    baseline = simulation_response.get("baseline") or {}
    provenance = baseline.get("provenance") or {}
    summary = [
        {
            "variant": candidate.get("plan_variant_code"),
            "plan_count": len(candidate.get("plans") or []),
            "unscheduled_plan_ids": candidate.get("unscheduled_plan_ids") or [],
            "unscheduled_orders": candidate.get("unscheduled_orders") or [],
        }
        for candidate in candidates
    ]

    logger.info(
        "[Planning] generate completed baseline_plan_count=%s candidate_summary=%s",
        provenance.get("plan_count"),
        summary,
    )
