from decimal import Decimal, InvalidOperation
from typing import Any

from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatIntent,
    EvidenceItem,
    EvidenceResult,
)

LINE_COUNT_SUMMARY_TITLE = "공정 라인 전체 현황"
LINE_COMPOSITION_SUMMARY_TITLE = "생산 라인 구성 전체 현황"
RUNNING_LINE_SUMMARY_TITLE = "가동 라인 전체 현황"
URGENT_ORDER_IMPACT_SUMMARY_TITLE = "긴급 주문 전체 생산계획 영향"
MATERIAL_SHORTAGE_IMPACT_SUMMARY_TITLE = "자재 부족 영향 생산계획"
LINE_SUMMARY_TITLES = frozenset(
    {
        LINE_COUNT_SUMMARY_TITLE,
        LINE_COMPOSITION_SUMMARY_TITLE,
        RUNNING_LINE_SUMMARY_TITLE,
    }
)
RUNNING_STATUS = "RUNNING"


class EvidenceAggregationPolicy:
    def apply(
        self,
        request: ChatAnswerRequest,
        result: EvidenceResult,
    ) -> EvidenceResult:
        if result.intent == ChatIntent.MATERIAL_SHORTAGE:
            return self._apply_material_shortage_impact_summary(request, result)

        if result.intent == ChatIntent.URGENT_ORDER_IMPACT:
            return self._apply_urgent_order_impact_summary(request, result)

        if result.intent != ChatIntent.LINE_BOTTLENECK or not result.items:
            return result

        summary_type = self._select_line_summary_type(request.question)
        if summary_type is None or result.items[0].title in LINE_SUMMARY_TITLES:
            return result

        if summary_type == "RUNNING":
            line_summary_item = self._build_running_line_summary_item(result.items)
        elif summary_type == "COMPOSITION":
            line_summary_item = self._build_line_composition_summary_item(result.items)
        else:
            line_summary_item = self._build_line_count_summary_item(result.items)

        if line_summary_item is None:
            return result
        return result.model_copy(update={"items": [line_summary_item, *result.items]})

    def _apply_material_shortage_impact_summary(
        self,
        request: ChatAnswerRequest,
        result: EvidenceResult,
    ) -> EvidenceResult:
        if not result.items:
            return result
        if result.items[0].title == MATERIAL_SHORTAGE_IMPACT_SUMMARY_TITLE:
            return result
        if not self._is_material_shortage_impact_question(request.question):
            return result

        summary_item = self._build_material_shortage_impact_summary_item(result.items)
        if summary_item is None:
            return result
        return result.model_copy(update={"items": [summary_item]})

    def _build_material_shortage_impact_summary_item(
        self,
        items: list[EvidenceItem],
    ) -> EvidenceItem | None:
        plan_summaries = self._collect_material_shortage_plan_summaries(items)
        if not plan_summaries:
            return None

        plan_count = len(plan_summaries)
        material_counts = self._count_values(
            plan_summary.get("materialCode") for plan_summary in plan_summaries
        )
        status_counts = self._count_values(
            plan_summary.get("materialPlanStatus") for plan_summary in plan_summaries
        )
        plan_details = self._material_shortage_plan_details(plan_summaries)
        plan_ids = [
            plan_summary["planId"]
            for plan_summary in plan_summaries
            if plan_summary.get("planId") is not None
        ]

        summary_parts = [f"자재 부족으로 영향받는 생산계획은 총 {plan_count}건"]
        if plan_details:
            summary_parts.append(f"영향 계획: {', '.join(plan_details[:5])}")
        if material_counts:
            summary_parts.append(f"부족 자재: {self._format_status_counts(material_counts)}")
        if status_counts:
            summary_parts.append(f"자재 상태: {self._format_status_counts(status_counts)}")

        first_item = items[0]
        return EvidenceItem(
            type="PLAN",
            title=MATERIAL_SHORTAGE_IMPACT_SUMMARY_TITLE,
            summary=". ".join(summary_parts) + ".",
            url="/production-plans?mode=read",
            source=first_item.source,
            data={
                "affectedPlanCount": plan_count,
                "affectedPlanIds": plan_ids,
                "materialCounts": material_counts,
                "materialPlanStatusCounts": status_counts,
                "planDetails": plan_summaries,
            },
            allowedRoles=first_item.allowed_roles,
        )

    def _collect_material_shortage_plan_summaries(
        self,
        items: list[EvidenceItem],
    ) -> list[dict[str, Any]]:
        plan_summaries: list[dict[str, Any]] = []
        seen_plan_keys: set[str] = set()
        for item in items:
            if item.type != "MATERIAL":
                continue

            plan_id = self._get_data_value(item.data, "planId", "plan_id")
            material_code = self._get_data_value(
                item.data,
                "materialCode",
                "material_code",
            )
            plan_key = str(plan_id or item.reference_id or item.title)
            if plan_key in seen_plan_keys:
                continue

            seen_plan_keys.add(plan_key)
            plan_summaries.append(
                {
                    "planId": plan_id,
                    "orderNo": self._get_data_value(item.data, "orderNo", "order_no"),
                    "productCode": self._get_data_value(
                        item.data,
                        "productCode",
                        "product_code",
                    ),
                    "lineCode": self._get_data_value(
                        item.data,
                        "lineCode",
                        "line_code",
                    ),
                    "materialCode": material_code,
                    "materialName": self._get_data_value(
                        item.data,
                        "materialName",
                        "material_name",
                    ),
                    "shortageQuantity": self._get_data_value(
                        item.data,
                        "shortageQuantity",
                        "shortage_quantity",
                    ),
                    "unit": self._get_data_value(item.data, "unit"),
                    "materialPlanStatus": self._get_data_value(
                        item.data,
                        "materialPlanStatus",
                        "material_plan_status",
                    ),
                }
            )
        return plan_summaries

    def _is_material_shortage_impact_question(self, question: str) -> bool:
        compact_question = "".join(question.casefold().split())
        has_plan_term = any(term in compact_question for term in ("생산계획", "계획", "plan"))
        has_impact_term = any(
            term in compact_question
            for term in ("영향", "영향받", "차질", "문제", "부족")
        )
        return has_plan_term and has_impact_term

    def _material_shortage_plan_details(
        self,
        plan_summaries: list[dict[str, Any]],
    ) -> list[str]:
        details: list[str] = []
        for plan_summary in plan_summaries:
            plan_id = plan_summary.get("planId")
            if plan_id is None:
                continue

            detail_parts = [f"계획 {plan_id}"]
            order_no = plan_summary.get("orderNo")
            if order_no:
                detail_parts.append(str(order_no))
            material_code = plan_summary.get("materialCode")
            if material_code:
                detail_parts.append(str(material_code))
            shortage_quantity = self._to_decimal(plan_summary.get("shortageQuantity"))
            unit = plan_summary.get("unit")
            if shortage_quantity is not None:
                quantity_text = self._format_decimal(shortage_quantity)
                if unit:
                    quantity_text = f"{quantity_text}{unit}"
                detail_parts.append(f"부족 {quantity_text}")
            line_code = plan_summary.get("lineCode")
            if line_code:
                detail_parts.append(str(line_code))
            details.append(" / ".join(detail_parts))
        return details

    def _apply_urgent_order_impact_summary(
        self,
        request: ChatAnswerRequest,
        result: EvidenceResult,
    ) -> EvidenceResult:
        if not result.items:
            return result
        if result.items[0].title == URGENT_ORDER_IMPACT_SUMMARY_TITLE:
            return result
        if not self._is_urgent_order_overall_question(request.question):
            return result

        summary_item = self._build_urgent_order_impact_summary_item(result.items)
        if summary_item is None:
            return result
        return result.model_copy(update={"items": [summary_item]})

    def _build_urgent_order_impact_summary_item(
        self,
        items: list[EvidenceItem],
    ) -> EvidenceItem | None:
        impact_summaries = self._collect_urgent_order_impact_summaries(items)
        if not impact_summaries:
            return None

        order_nos = self._unique_sorted_values(
            impact_summary.get("orderNo") for impact_summary in impact_summaries
        )
        delayed_order_nos = self._unique_sorted_values(
            impact_summary.get("orderNo")
            for impact_summary in impact_summaries
            if impact_summary.get("afterIsDelayed") is True
        )
        line_changes = self._urgent_order_line_changes(impact_summaries)
        recommendation_grade_counts = self._count_values(
            impact_summary.get("recommendationGrade")
            for impact_summary in impact_summaries
        )
        simulation_types = self._unique_sorted_values(
            impact_summary.get("simulationType")
            for impact_summary in impact_summaries
        )
        delay_reduction_values = [
            value
            for value in (
                self._to_decimal(impact_summary.get("delayReductionHr"))
                for impact_summary in impact_summaries
            )
            if value is not None
        ]
        total_delay_reduction_hr = sum(delay_reduction_values, Decimal("0"))
        affected_order_count = len(order_nos) if order_nos else len(impact_summaries)

        summary_parts = [
            f"조회된 시뮬레이션 기준 긴급 주문 영향 대상은 총 {affected_order_count}건"
        ]
        if order_nos:
            summary_parts.append(f"영향 주문: {', '.join(order_nos[:5])}")
        if line_changes:
            summary_parts.append(f"라인 변경: {', '.join(line_changes[:5])}")
        if delayed_order_nos:
            summary_parts.append(
                f"변경 후 지연 예상: {len(delayed_order_nos)}건"
                f"({', '.join(delayed_order_nos[:5])})"
            )
        elif any(
            impact_summary.get("afterIsDelayed") is False
            for impact_summary in impact_summaries
        ):
            summary_parts.append("변경 후 지연 예상: 없음")
        if delay_reduction_values:
            summary_parts.append(
                "총 지연 감소: "
                f"{self._format_decimal(total_delay_reduction_hr)}시간"
            )
        if recommendation_grade_counts:
            summary_parts.append(
                "추천 등급: "
                f"{self._format_status_counts(recommendation_grade_counts)}"
            )
        if simulation_types:
            summary_parts.append(f"대응 유형: {', '.join(simulation_types[:3])}")

        first_item = items[0]
        return EvidenceItem(
            type="ORDER",
            title=URGENT_ORDER_IMPACT_SUMMARY_TITLE,
            summary=". ".join(summary_parts) + ".",
            url="/schedule-simulations?mode=read",
            source=first_item.source,
            data={
                "orderCount": affected_order_count,
                "affectedOrderNos": order_nos,
                "lineChanges": line_changes,
                "delayedOrderCount": len(delayed_order_nos),
                "delayedOrderNos": delayed_order_nos,
                "totalDelayReductionHr": float(total_delay_reduction_hr),
                "recommendationGradeCounts": recommendation_grade_counts,
                "simulationTypes": simulation_types,
            },
            allowedRoles=first_item.allowed_roles,
        )

    def _collect_urgent_order_impact_summaries(
        self,
        items: list[EvidenceItem],
    ) -> list[dict[str, Any]]:
        impact_summaries: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for item in items:
            if item.type != "ORDER":
                continue

            order_no = self._get_data_value(item.data, "orderNo", "order_no")
            simulation_detail_id = self._get_data_value(
                item.data,
                "simulationDetailId",
                "simulation_detail_id",
            )
            impact_key = str(simulation_detail_id or order_no or item.reference_id)
            if impact_key in seen_keys:
                continue

            seen_keys.add(impact_key)
            impact_summaries.append(
                {
                    "orderNo": order_no,
                    "productCode": self._get_data_value(
                        item.data,
                        "productCode",
                        "product_code",
                    ),
                    "beforeLineCode": self._get_data_value(
                        item.data,
                        "beforeLineCode",
                        "before_line_code",
                    ),
                    "afterLineCode": self._get_data_value(
                        item.data,
                        "afterLineCode",
                        "after_line_code",
                    ),
                    "afterIsDelayed": self._get_data_value(
                        item.data,
                        "afterIsDelayed",
                        "after_is_delayed",
                    ),
                    "delayReductionHr": self._get_data_value(
                        item.data,
                        "delayReductionHr",
                        "delay_reduction_hr",
                    ),
                    "recommendationGrade": self._get_data_value(
                        item.data,
                        "recommendationGrade",
                        "recommendation_grade",
                    ),
                    "simulationType": self._get_data_value(
                        item.data,
                        "simulationType",
                        "simulation_type",
                    ),
                }
            )
        return impact_summaries

    def _is_urgent_order_overall_question(self, question: str) -> bool:
        compact_question = "".join(question.casefold().split())
        return any(
            term in compact_question
            for term in (
                "전체생산계획",
                "생산계획",
                "전체계획",
                "전체영향",
                "주문영향",
            )
        )

    def _urgent_order_line_changes(
        self,
        impact_summaries: list[dict[str, Any]],
    ) -> list[str]:
        line_changes: list[str] = []
        seen_changes: set[str] = set()
        for impact_summary in impact_summaries:
            before_line_code = impact_summary.get("beforeLineCode")
            after_line_code = impact_summary.get("afterLineCode")
            if not before_line_code or not after_line_code:
                continue
            if str(before_line_code).strip() == str(after_line_code).strip():
                continue

            order_no = impact_summary.get("orderNo")
            change = f"{before_line_code}->{after_line_code}"
            if order_no:
                change = f"{order_no} {change}"
            if change in seen_changes:
                continue

            seen_changes.add(change)
            line_changes.append(change)
        return line_changes

    def _unique_sorted_values(self, values: Any) -> list[str]:
        return sorted(
            {
                str(value).strip()
                for value in values
                if value is not None and str(value).strip()
            }
        )

    def _select_line_summary_type(self, question: str) -> str | None:
        if self._is_running_line_question(question):
            return "RUNNING"
        if self._is_line_count_question(question):
            return "COUNT"
        if self._is_line_composition_question(question):
            return "COMPOSITION"
        return None

    def _build_line_count_summary_item(
        self,
        items: list[EvidenceItem],
    ) -> EvidenceItem | None:
        line_summaries = self._collect_line_summaries(items)
        if not line_summaries:
            return None

        line_count = len(line_summaries)
        line_codes = self._line_codes(line_summaries)
        status_counts = self._count_values(
            line_summary.get("operationStatus") for line_summary in line_summaries
        )
        latest_recorded_at = self._latest_recorded_at(line_summaries)
        summary_parts = [f"현재 조회된 RDB 기준 공정 라인 수: 총 {line_count}개"]
        if line_codes:
            summary_parts.append(f"라인 코드: {', '.join(line_codes)}")
        if status_counts:
            summary_parts.append(f"상태별 개수: {self._format_status_counts(status_counts)}")
        if latest_recorded_at:
            summary_parts.append(f"최신 기록 시각: {latest_recorded_at}")

        first_item = items[0]
        return EvidenceItem(
            type="LINE",
            title=LINE_COUNT_SUMMARY_TITLE,
            summary=". ".join(summary_parts) + ".",
            url="/production-lines?mode=read",
            source=first_item.source,
            data={
                "lineCount": line_count,
                "lineCodes": line_codes,
                "operationStatusCounts": status_counts,
                "latestRecordedAt": latest_recorded_at,
            },
            allowedRoles=first_item.allowed_roles,
        )

    def _build_line_composition_summary_item(
        self,
        items: list[EvidenceItem],
    ) -> EvidenceItem | None:
        line_summaries = self._collect_line_summaries(items)
        if not line_summaries:
            return None

        line_count = len(line_summaries)
        line_codes = self._line_codes(line_summaries)
        status_counts = self._count_values(
            line_summary.get("operationStatus") for line_summary in line_summaries
        )
        latest_recorded_at = self._latest_recorded_at(line_summaries)
        line_details = self._line_details(line_summaries)

        summary_parts = [f"현재 조회된 RDB 기준 생산 라인은 총 {line_count}개"]
        if line_details:
            summary_parts.append(f"라인 구성: {', '.join(line_details)}")
        elif line_codes:
            summary_parts.append(f"라인 코드: {', '.join(line_codes)}")
        if status_counts:
            summary_parts.append(f"상태별 개수: {self._format_status_counts(status_counts)}")
        if latest_recorded_at:
            summary_parts.append(f"최신 기록 시각: {latest_recorded_at}")

        first_item = items[0]
        return EvidenceItem(
            type="LINE",
            title=LINE_COMPOSITION_SUMMARY_TITLE,
            summary=". ".join(summary_parts) + ".",
            url="/production-lines?mode=read",
            source=first_item.source,
            data={
                "lineCount": line_count,
                "lineCodes": line_codes,
                "lineDetails": line_summaries,
                "operationStatusCounts": status_counts,
                "latestRecordedAt": latest_recorded_at,
            },
            allowedRoles=first_item.allowed_roles,
        )

    def _build_running_line_summary_item(
        self,
        items: list[EvidenceItem],
    ) -> EvidenceItem | None:
        line_summaries = self._collect_line_summaries(items)
        if not line_summaries:
            return None

        running_lines = [
            line_summary
            for line_summary in line_summaries
            if self._normalize_status(line_summary.get("operationStatus")) == RUNNING_STATUS
        ]
        running_lines = sorted(running_lines, key=self._line_sort_key)
        running_line_codes = self._line_codes(running_lines)
        status_counts = self._count_values(
            line_summary.get("operationStatus") for line_summary in line_summaries
        )
        latest_recorded_at = self._latest_recorded_at(line_summaries)

        summary_parts = [
            f"현재 조회된 RDB 기준 RUNNING 라인은 총 {len(running_lines)}개"
        ]
        if running_line_codes:
            summary_parts.append(f"RUNNING 라인 코드: {', '.join(running_line_codes)}")
        if status_counts:
            summary_parts.append(f"상태별 개수: {self._format_status_counts(status_counts)}")
        if latest_recorded_at:
            summary_parts.append(f"최신 기록 시각: {latest_recorded_at}")

        first_item = items[0]
        return EvidenceItem(
            type="LINE",
            title=RUNNING_LINE_SUMMARY_TITLE,
            summary=". ".join(summary_parts) + ".",
            url="/production-lines?mode=read",
            source=first_item.source,
            data={
                "lineCount": len(line_summaries),
                "runningLineCount": len(running_lines),
                "runningLineCodes": running_line_codes,
                "operationStatusCounts": status_counts,
                "latestRecordedAt": latest_recorded_at,
            },
            allowedRoles=first_item.allowed_roles,
        )

    def _collect_line_summaries(
        self,
        items: list[EvidenceItem],
    ) -> list[dict[str, Any]]:
        line_summaries: list[dict[str, Any]] = []
        seen_line_keys: set[str] = set()
        for item in items:
            if item.type != "LINE":
                continue

            line_id = self._get_data_value(item.data, "lineId", "line_id")
            line_code = self._get_data_value(item.data, "lineCode", "line_code")
            line_key = str(line_id or line_code or item.reference_id or item.title)
            if line_key in seen_line_keys:
                continue

            seen_line_keys.add(line_key)
            line_summaries.append(
                {
                    "lineId": line_id,
                    "lineCode": line_code,
                    "lineName": self._get_data_value(
                        item.data,
                        "lineName",
                        "line_name",
                    ),
                    "operationStatus": self._get_data_value(
                        item.data,
                        "operationStatus",
                        "operation_status",
                    ),
                    "recordedAt": self._get_data_value(
                        item.data,
                        "recordedAt",
                        "recorded_at",
                    ),
                }
            )
        return line_summaries

    def _is_running_line_question(self, question: str) -> bool:
        compact_question = "".join(question.casefold().split())
        has_line_term = any(term in compact_question for term in ("라인", "공정", "line"))
        has_running_term = any(
            term in compact_question
            for term in (
                "가동중",
                "가동중인",
                "현재가동",
                "운영중",
                "운영중인",
                "running",
            )
        )
        return has_line_term and has_running_term

    def _is_line_count_question(self, question: str) -> bool:
        compact_question = "".join(question.casefold().split())
        has_line_term = any(term in compact_question for term in ("라인", "공정", "line"))
        has_count_term = any(
            term in compact_question
            for term in (
                "몇개",
                "몇개의",
                "몇대",
                "개수",
                "총몇",
                "총수",
                "전체몇",
            )
        )
        return has_line_term and has_count_term

    def _is_line_composition_question(self, question: str) -> bool:
        compact_question = "".join(question.casefold().split())
        has_line_term = any(term in compact_question for term in ("라인", "공정", "line"))
        if not has_line_term:
            return False

        strong_composition_terms = (
            "구성",
            "목록",
            "리스트",
        )
        if any(term in compact_question for term in strong_composition_terms):
            return True

        broad_composition_terms = (
            "전체라인",
            "라인전체",
            "생산라인전체",
            "공정라인전체",
        )
        if (
            any(term in compact_question for term in broad_composition_terms)
            and not self._has_bottleneck_term(compact_question)
        ):
            return True

        status_terms = ("상태", "현황")
        return (
            any(term in compact_question for term in status_terms)
            and not self._has_bottleneck_term(compact_question)
        )

    def _has_bottleneck_term(self, compact_question: str) -> bool:
        return any(
            term in compact_question
            for term in (
                "병목",
                "대기",
                "지연",
                "위험",
                "문제",
                "원인",
                "이상",
            )
        )

    def _line_codes(self, line_summaries: list[dict[str, Any]]) -> list[str]:
        return sorted(
            {
                str(line_summary["lineCode"])
                for line_summary in line_summaries
                if line_summary.get("lineCode")
            }
        )

    def _line_details(self, line_summaries: list[dict[str, Any]]) -> list[str]:
        line_details: list[str] = []
        for line_summary in sorted(line_summaries, key=self._line_sort_key):
            line_code = line_summary.get("lineCode")
            if not line_code:
                continue

            line_name = line_summary.get("lineName")
            operation_status = self._normalize_status(line_summary.get("operationStatus"))
            details = [str(value) for value in (line_name, operation_status) if value]
            if details:
                line_details.append(f"{line_code}({', '.join(details)})")
            else:
                line_details.append(str(line_code))
        return line_details

    def _latest_recorded_at(self, line_summaries: list[dict[str, Any]]) -> str | None:
        return max(
            (
                str(line_summary["recordedAt"])
                for line_summary in line_summaries
                if line_summary.get("recordedAt")
            ),
            default=None,
        )

    def _line_sort_key(self, line_summary: dict[str, Any]) -> str:
        return str(
            line_summary.get("lineCode")
            or line_summary.get("lineId")
            or line_summary.get("lineName")
            or ""
        )

    def _format_status_counts(self, status_counts: dict[str, int]) -> str:
        return ", ".join(f"{status} {count}개" for status, count in status_counts.items())

    def _get_data_value(self, data: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in data:
                return data[key]
        return None

    def _normalize_status(self, value: Any) -> str | None:
        if value is None:
            return None
        normalized_value = str(value).strip().upper()
        return normalized_value or None

    def _count_values(self, values: Any) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            normalized_value = self._normalize_status(value)
            if not normalized_value:
                continue
            counts[normalized_value] = counts.get(normalized_value, 0) + 1
        return dict(sorted(counts.items()))

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
