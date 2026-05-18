from datetime import datetime

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


class ChatResponseBuilder:
    _max_source_summary_chars = 300

    def build_sources(
        self,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> list[ChatSource]:
        sources = [
            self._evidence_item_to_source(item, evidence_result.basis_time)
            for item in evidence_result.items
        ]
        sources.extend(document_result.sources)
        return self._deduplicate_sources(sources)

    def build_urls(self, sources: list[ChatSource]) -> list[ChatUrl]:
        urls: list[ChatUrl] = []
        seen_urls: set[str] = set()
        for source in sources:
            if not source.url or source.url in seen_urls:
                continue
            seen_urls.add(source.url)
            urls.append(
                ChatUrl(
                    label=source.title,
                    url=source.url,
                    type=source.source_type,
                )
            )
        return urls

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
            reason="조회된 RDB Evidence가 없고 Qdrant 검색 결과가 없습니다.",
        )

    def _evidence_item_to_source(
        self,
        item: EvidenceItem,
        basis_time: datetime,
    ) -> ChatSource:
        return ChatSource(
            source_type=item.type,
            title=item.title,
            summary=self._truncate_source_summary(item.summary),
            url=item.url,
            reference_id=item.reference_id,
            source=item.source,
            basis_time=basis_time,
            source_origin="RDB",
        )

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
        return (
            source.source_type,
            source.reference_id,
            source.url,
            source.source,
            source.title,
            source.basis_time,
            source.source_origin,
        )
