from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Protocol

from app.core.config import Settings
from app.features.chat.query_filter_extractor import QueryFilterExtractor
from app.features.chat.rdb_evidence_providers import (
    build_default_rdb_evidence_providers,
)
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatIntent,
    EvidenceItem,
    EvidenceLookupFilters,
    EvidenceResult,
)

_COMPANY_INFO_KEYWORDS = (
    "s-map",
    "smap",
    "회사",
    "기업",
    "사업",
    "매출",
    "조직",
    "공장",
    "생산라인",
    "제품",
    "자재",
    "용어",
    "워크플로우",
)
_REPORT_LOOKUP_KEYWORDS = (
    "보고서",
    "리포트",
    "report",
    "월간",
    "수시",
)


class RdbEvidenceProvider(Protocol):
    intent: ChatIntent

    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        filters: EvidenceLookupFilters,
    ) -> list[EvidenceItem]:
        """사전에 정의된 RDB read-only View 조회 결과를 Evidence로 변환한다."""


class RdbEvidenceService:
    def __init__(
        self,
        settings: Settings,
        providers: Sequence[RdbEvidenceProvider] | None = None,
        query_filter_extractor: QueryFilterExtractor | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.providers = {
            provider.intent: provider
            for provider in (
                providers
                if providers is not None
                else build_default_rdb_evidence_providers(settings)
            )
        }
        self.query_filter_extractor = query_filter_extractor or QueryFilterExtractor()
        self.clock = clock or self._default_clock

    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        if not self.settings.rdb_evidence_enabled:
            return self._empty_result(request, intent)

        provider = self.providers.get(intent)
        if provider is None:
            return self._empty_result(request, intent)

        if self._should_skip_report_metadata_evidence(request, intent):
            return self._empty_result(request, intent)

        filters = self._build_filters(request)
        items = await provider.get_evidence(request, filters)
        return EvidenceResult(
            intent=intent,
            basis_time=self.clock(),
            items=items,
        )

    def _build_filters(self, request: ChatAnswerRequest) -> EvidenceLookupFilters:
        extracted_filters = self.query_filter_extractor.extract_filters(
            request.question,
            request.requested_at,
        )
        filters = EvidenceLookupFilters.model_validate(extracted_filters)
        return filters.model_copy(
            update={
                "limit": min(
                    max(1, filters.limit),
                    self.settings.rdb_evidence_max_limit,
                )
            }
        )

    def _should_skip_report_metadata_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> bool:
        if intent != ChatIntent.REPORT_LOOKUP:
            return False

        normalized_question = request.question.strip().lower().replace(" ", "")
        asks_report = any(
            keyword in normalized_question
            for keyword in _REPORT_LOOKUP_KEYWORDS
        )
        asks_company_info = any(
            keyword in normalized_question
            for keyword in _COMPANY_INFO_KEYWORDS
        )
        return asks_company_info and not asks_report

    def _empty_result(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        return EvidenceResult(
            intent=intent,
            basis_time=request.requested_at,
            items=[],
        )

    def _default_clock(self) -> datetime:
        return datetime.now(UTC)
