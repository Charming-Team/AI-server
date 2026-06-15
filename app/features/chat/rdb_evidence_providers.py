from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from app.core.config import Settings
from app.features.chat.access_control import (
    EXECUTIVE_ROLE,
    MANUFACTURING_MANAGER_ROLE,
    OPERATOR_ROLE,
)
from app.features.chat.rdb_evidence_view_catalog import (
    RDB_EVIDENCE_VIEW_DEFINITIONS,
    RdbEvidenceViewDefinition,
)
from app.features.chat.rdb_evidence_view_client import (
    AsyncpgRdbEvidenceViewClient,
    RdbEvidenceViewClient,
)
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatIntent,
    EvidenceItem,
    EvidenceLookupFilters,
)

ROLE_RESPONSE_ORDER = (
    OPERATOR_ROLE,
    EXECUTIVE_ROLE,
    MANUFACTURING_MANAGER_ROLE,
)

COLUMN_LABELS = {
    "abnormal_machine_count": "비정상 설비 수",
    "action_result": "대응 결과",
    "after_end_at": "변경 후 종료",
    "after_is_delayed": "변경 후 지연 여부",
    "after_line_code": "변경 후 라인",
    "after_start_at": "변경 후 시작",
    "after_total_delay_hr": "변경 후 전체 지연",
    "analysis_summary": "분석 요약",
    "available_quantity": "가용 재고",
    "before_end_at": "변경 전 종료",
    "before_line_code": "변경 전 라인",
    "before_start_at": "변경 전 시작",
    "before_total_delay_hr": "변경 전 전체 지연",
    "cause_detail": "원인 상세",
    "change_reason": "변경 사유",
    "created_at": "생성 시각",
    "current_yield_rate": "현재 수율",
    "customer_name": "고객사",
    "delay_probability": "지연 확률",
    "delay_reduction_hr": "지연 감소 시간",
    "due_date": "납기일",
    "estimated_duration_hr": "예상 소요 시간",
    "expected_completion_date": "예상 완료일",
    "expected_inbound_at": "입고 예정 시각",
    "expected_inbound_quantity": "입고 예정량",
    "inventory_status": "재고 상태",
    "line_code": "라인 코드",
    "line_name": "라인명",
    "line_waiting_time_hr": "라인 대기 시간",
    "main_cause_type": "주요 원인",
    "material_code": "자재 코드",
    "material_name": "자재명",
    "material_plan_status": "자재 상태",
    "operation_status": "가동 상태",
    "order_no": "주문 번호",
    "plan_id": "생산계획 ID",
    "plan_sequence": "생산 순서",
    "plan_status": "계획 상태",
    "planned_end_at": "계획 종료",
    "planned_quantity": "계획 수량",
    "planned_start_at": "계획 시작",
    "predicted_at": "예측 시각",
    "predicted_delay_days": "예상 지연 일수",
    "priority_rank": "우선순위",
    "product_code": "제품 코드",
    "product_name": "제품명",
    "progress_rate": "진행률",
    "recommendation_grade": "추천 등급",
    "recommended_action": "추천 조치",
    "recorded_at": "기록 시각",
    "report_title": "보고서 제목",
    "report_type": "보고서 유형",
    "required_quantity": "필요 수량",
    "reserved_quantity": "예약 수량",
    "risk_level": "위험 등급",
    "safety_stock_quantity": "안전 재고",
    "shortage_count": "부족 자재 수",
    "shortage_quantity": "부족 수량",
    "simulation_name": "대응안",
    "simulation_type": "대응 유형",
    "target_end_date": "대상 종료일",
    "target_start_date": "대상 시작일",
    "throughput_rate": "처리량",
    "unit": "단위",
    "updated_at": "수정 시각",
    "utilization_rate": "가동률",
    "waiting_quantity": "대기 수량",
    "waiting_time_hr": "대기 시간",
}

HOUR_COLUMNS = frozenset(
    {
        "actual_delay_hr",
        "actual_duration_hr",
        "actual_setup_time_hr",
        "after_total_delay_hr",
        "before_total_delay_hr",
        "delay_reduction_hr",
        "estimated_duration_hr",
        "line_waiting_time_hr",
        "waiting_time_hr",
    }
)

DAY_COLUMNS = frozenset(
    {
        "predicted_delay_days",
    }
)

RATE_COLUMNS = frozenset(
    column
    for column in (
        "after_avg_line_utilization_rate",
        "average_yield_rate",
        "before_avg_line_utilization_rate",
        "current_yield_rate",
        "delay_probability",
        "progress_rate",
        "standard_yield_rate",
        "throughput_rate",
        "utilization_rate",
        "yield_rate",
    )
)


class CatalogRdbEvidenceProvider:
    def __init__(
        self,
        definition: RdbEvidenceViewDefinition,
        view_client: RdbEvidenceViewClient,
    ) -> None:
        self.definition = definition
        self.view_client = view_client

    @property
    def intent(self) -> ChatIntent:
        return self.definition.intent

    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        filters: EvidenceLookupFilters,
    ) -> list[EvidenceItem]:
        role = request.user.role.strip().upper()
        if role not in self.definition.allowed_roles:
            return []

        rows = await self.view_client.fetch_rows(
            self.definition,
            filters,
            filters.limit,
        )
        return [self._to_evidence_item(row) for row in rows]

    def _to_evidence_item(self, row: Mapping[str, Any]) -> EvidenceItem:
        reference_id = self._to_reference_id(row.get(self.definition.reference_id_column))
        return EvidenceItem(
            type=self.definition.source_type,
            title=self._build_title(row),
            summary=self._build_summary(row),
            url=self._build_url(row),
            source=self.definition.source,
            referenceId=reference_id,
            data=self._build_data(row),
            allowedRoles=self._allowed_role_names(),
        )

    def _build_title(self, row: Mapping[str, Any]) -> str:
        values = [
            self._stringify(row.get(column))
            for column in self.definition.title_columns
            if self._has_value(row.get(column))
        ]
        if values:
            return " ".join(values)
        reference_id = row.get(self.definition.reference_id_column)
        return f"{self.definition.source_type} Evidence {reference_id}"

    def _build_summary(self, row: Mapping[str, Any]) -> str:
        fragments = [
            (
                f"{COLUMN_LABELS.get(column, column)}: "
                f"{self._format_summary_value(column, row.get(column))}"
            )
            for column in self.definition.summary_columns
            if self._has_value(row.get(column))
        ]
        return ", ".join(fragments) if fragments else "조회된 RDB Evidence입니다."

    def _build_url(self, row: Mapping[str, Any]) -> str | None:
        source_type = self.definition.source_type
        if source_type == "MATERIAL" and self._has_value(row.get("material_id")):
            return f"/materials/inventory/{row['material_id']}?mode=read"
        if source_type == "PREDICTION" and self._has_value(row.get("prediction_id")):
            return f"/predictions/{row['prediction_id']}?mode=read"
        if source_type == "PLAN" and self._has_value(row.get("plan_id")):
            return f"/production-plans/{row['plan_id']}?mode=read"
        if source_type == "LINE" and self._has_value(row.get("line_id")):
            return f"/production-lines/{row['line_id']}?mode=read"
        if source_type == "ORDER" and self._has_value(row.get("order_id")):
            return f"/orders/{row['order_id']}?mode=read"
        if source_type == "REPORT" and self._has_value(row.get("report_id")):
            return f"/reports/{row['report_id']}?mode=read"
        return None

    def _build_data(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            self._snake_to_camel(column): self._json_safe_value(row.get(column))
            for column in self.definition.data_columns
            if self._has_value(row.get(column))
        }

    def _allowed_role_names(self) -> list[str]:
        return [
            role
            for role in ROLE_RESPONSE_ORDER
            if role in self.definition.allowed_roles
        ]

    def _to_reference_id(self, value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    def _json_safe_value(self, value: Any) -> Any:
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, datetime | date):
            return value.isoformat()
        if isinstance(value, tuple | set):
            return list(value)
        return value

    def _stringify(self, value: Any) -> str:
        return str(self._json_safe_value(value))

    def _format_summary_value(self, column: str, value: Any) -> str:
        if isinstance(value, bool):
            return "예" if value else "아니오"
        if isinstance(value, datetime):
            return value.strftime("%Y.%m.%d %H:%M")
        if isinstance(value, date):
            return value.strftime("%Y.%m.%d")
        if column in HOUR_COLUMNS:
            return f"{self._format_number(value)}시간"
        if column in DAY_COLUMNS:
            return f"{self._format_number(value)}일"
        if column in RATE_COLUMNS:
            return f"{self._format_rate(value)}%"
        return self._stringify(value)

    def _format_rate(self, value: Any) -> str:
        number = self._to_decimal(value)
        if number is None:
            return self._stringify(value)
        if abs(number) <= Decimal("1"):
            number *= Decimal("100")
        return self._format_decimal(number)

    def _format_number(self, value: Any) -> str:
        number = self._to_decimal(value)
        if number is None:
            return self._stringify(value)
        return self._format_decimal(number)

    def _to_decimal(self, value: Any) -> Decimal | None:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int | float | str):
            try:
                return Decimal(str(value))
            except (InvalidOperation, ValueError):
                return None
        return None

    def _format_decimal(self, value: Decimal) -> str:
        normalized = (
            value.quantize(Decimal("0.1"))
            if value != value.to_integral()
            else value
        )
        return format(normalized.normalize(), "f")

    def _has_value(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    def _snake_to_camel(self, value: str) -> str:
        head, *tail = value.split("_")
        return head + "".join(part.capitalize() for part in tail)


def build_default_rdb_evidence_providers(
    settings: Settings,
    view_client: RdbEvidenceViewClient | None = None,
) -> list[CatalogRdbEvidenceProvider]:
    client = view_client or AsyncpgRdbEvidenceViewClient(settings)
    return [
        CatalogRdbEvidenceProvider(definition, client)
        for definition in RDB_EVIDENCE_VIEW_DEFINITIONS
    ]
