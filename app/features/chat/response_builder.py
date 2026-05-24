from datetime import UTC, datetime

from app.features.chat.schemas import (
    ChatErrorCode,
    ChatSource,
    ChatUrl,
    DocumentSearchResult,
    EvidenceItem,
    EvidenceResult,
    SecurityResult,
    SecurityStatus,
)
from app.features.chat.source_url_policy import normalize_internal_url


class ChatResponseBuilder:
    _max_rdb_sources = 3
    _max_document_sources = 2
    _max_source_summary_chars = 300
    _max_urls = 3

    def build_sources(
        self,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> list[ChatSource]:
        evidence_sources = [
            self._evidence_item_to_source(item, evidence_result.basis_time)
            for item in evidence_result.items
        ]
        document_sources = [
            self._sanitize_source_url(source) for source in document_result.sources
        ]
        return [
            *self._deduplicate_sources(evidence_sources)[: self._max_rdb_sources],
            *self._deduplicate_sources(document_sources)[: self._max_document_sources],
        ]

    def build_urls(self, sources: list[ChatSource]) -> list[ChatUrl]:
        urls: list[ChatUrl] = []
        seen_urls: set[str] = set()
        for source in sources:
            url = self._safe_internal_url(source.url)
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            urls.append(
                ChatUrl(
                    label=source.title,
                    url=url,
                    type=source.source_type,
                )
            )
            if len(urls) >= self._max_urls:
                break
        return urls

    def build_basis_time(
        self,
        sources: list[ChatSource],
        fallback: datetime,
    ) -> datetime:
        source_basis_times = [
            source.basis_time
            for source in sources
            if source.basis_time is not None
        ]
        if not source_basis_times:
            return fallback
        return max(source_basis_times, key=self._basis_time_sort_key)

    def build_security_result(
        self,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> SecurityResult:
        if evidence_result.has_evidence or document_result.sources:
            return SecurityResult(
                status=SecurityStatus.PASSED,
                reason="보안 필터를 통과했고 내부 근거가 확인되었습니다.",
            )

        return SecurityResult(
            status=SecurityStatus.INSUFFICIENT_EVIDENCE,
            code=ChatErrorCode.CHAT_EVIDENCE_001,
            reason=self._build_insufficient_evidence_reason(document_result),
        )

    def _build_insufficient_evidence_reason(
        self,
        document_result: DocumentSearchResult,
    ) -> str:
        reason = "조회된 RDB Evidence가 없고 Qdrant 문서 근거도 확인되지 않았습니다."
        if document_result.skipped_reason:
            return f"{reason} Qdrant 사유: {document_result.skipped_reason}"
        if not document_result.was_searched:
            return f"{reason} Qdrant 검색은 수행되지 않았습니다."
        return reason

    def _evidence_item_to_source(
        self,
        item: EvidenceItem,
        basis_time: datetime,
    ) -> ChatSource:
        return ChatSource(
            source_type=item.type,
            title=item.title,
            summary=self._truncate_source_summary(item.summary),
            url=self._safe_internal_url(item.url),
            reference_id=item.reference_id,
            source=item.source,
            basis_time=basis_time,
            source_origin="RDB",
        )

    def _sanitize_source_url(self, source: ChatSource) -> ChatSource:
        safe_url = self._safe_internal_url(source.url)
        if source.url == safe_url:
            return source
        return source.model_copy(update={"url": safe_url})

    def _safe_internal_url(self, url: str | None) -> str | None:
        return normalize_internal_url(url)

    def _truncate_source_summary(self, summary: str) -> str:
        if len(summary) <= self._max_source_summary_chars:
            return summary
        return f"{summary[: self._max_source_summary_chars - 3]}..."

    def _deduplicate_sources(self, sources: list[ChatSource]) -> list[ChatSource]:
        deduplicated_sources: list[ChatSource] = []
        seen_keys: set[tuple] = set()
        for source in sources:
            key = self._source_key(source)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduplicated_sources.append(source)
        return deduplicated_sources

    def _source_key(self, source: ChatSource) -> tuple:
        if source.url:
            return (
                "url",
                source.source_origin,
                source.source_type,
                source.url,
            )
        return (
            "source",
            source.source_type,
            source.reference_id,
            source.source,
            source.title,
            source.basis_time,
            source.source_origin,
        )

    def _basis_time_sort_key(self, basis_time: datetime) -> float:
        if basis_time.tzinfo is not None and basis_time.utcoffset() is not None:
            return basis_time.timestamp()
        return basis_time.replace(tzinfo=UTC).timestamp()
