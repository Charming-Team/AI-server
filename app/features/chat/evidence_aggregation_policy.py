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
