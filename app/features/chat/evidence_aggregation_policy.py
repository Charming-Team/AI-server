from typing import Any

from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatIntent,
    EvidenceItem,
    EvidenceResult,
)

LINE_COUNT_SUMMARY_TITLE = "공정 라인 전체 현황"


class EvidenceAggregationPolicy:
    def apply(
        self,
        request: ChatAnswerRequest,
        result: EvidenceResult,
    ) -> EvidenceResult:
        if (
            result.intent != ChatIntent.LINE_BOTTLENECK
            or not self._is_line_count_question(request.question)
            or not result.items
            or result.items[0].title == LINE_COUNT_SUMMARY_TITLE
        ):
            return result

        line_summary_item = self._build_line_count_summary_item(result.items)
        if line_summary_item is None:
            return result
        return result.model_copy(update={"items": [line_summary_item, *result.items]})

    def _build_line_count_summary_item(
        self,
        items: list[EvidenceItem],
    ) -> EvidenceItem | None:
        line_summaries = self._collect_line_summaries(items)
        if not line_summaries:
            return None

        line_count = len(line_summaries)
        line_codes = sorted(
            {
                line_summary["lineCode"]
                for line_summary in line_summaries
                if line_summary.get("lineCode")
            }
        )
        status_counts = self._count_values(
            line_summary.get("operationStatus") for line_summary in line_summaries
        )
        latest_recorded_at = max(
            (
                str(line_summary["recordedAt"])
                for line_summary in line_summaries
                if line_summary.get("recordedAt")
            ),
            default=None,
        )
        summary_parts = [f"현재 조회된 RDB 기준 공정 라인 수: 총 {line_count}개"]
        if line_codes:
            summary_parts.append(f"라인 코드: {', '.join(line_codes)}")
        if status_counts:
            status_summary = ", ".join(
                f"{status} {count}개" for status, count in status_counts.items()
            )
            summary_parts.append(f"상태별 개수: {status_summary}")
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

    def _get_data_value(self, data: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in data:
                return data[key]
        return None

    def _count_values(self, values: Any) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            if value is None:
                continue
            normalized_value = str(value).strip().upper()
            if not normalized_value:
                continue
            counts[normalized_value] = counts.get(normalized_value, 0) + 1
        return dict(sorted(counts.items()))
