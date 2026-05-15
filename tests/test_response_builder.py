from datetime import datetime

from app.features.chat.response_builder import ChatResponseBuilder
from app.features.chat.schemas import (
    ChatErrorCode,
    ChatIntent,
    ChatSource,
    DocumentSearchResult,
    EvidenceItem,
    EvidenceResult,
    SecurityStatus,
)


def _build_evidence_result(items: list[EvidenceItem]) -> EvidenceResult:
    return EvidenceResult(
        intent=ChatIntent.DELIVERY_RISK,
        basisTime=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
        items=items,
    )


def test_response_builder_merges_evidence_and_document_sources() -> None:
    builder = ChatResponseBuilder()
    evidence_result = _build_evidence_result(
        [
            EvidenceItem(
                type="ORDER",
                title="ORD-202605-001 납기 지연 위험",
                summary="납기일이 임박했고 현재 계획 상태는 DELAYED입니다.",
                url="/orders/1001",
                source="customer_orders",
                referenceId=1001,
            )
        ]
    )
    document_result = DocumentSearchResult(
        sources=[
            ChatSource(
                sourceType="REPORT",
                title="2026년 5월 생산 리스크 보고서",
                summary="자재 부족과 LINE-A01 병목이 주요 리스크입니다.",
                url="/reports/20",
                referenceId=20,
                source="report-202605:summary",
            )
        ]
    )

    sources = builder.build_sources(evidence_result, document_result)

    assert len(sources) == 2
    assert sources[0].source_type == "ORDER"
    assert sources[0].reference_id == 1001
    assert sources[1].source_type == "REPORT"
    assert sources[1].source == "report-202605:summary"


def test_response_builder_builds_unique_urls_from_sources() -> None:
    builder = ChatResponseBuilder()
    sources = [
        ChatSource(
            sourceType="REPORT",
            title="2026년 5월 생산 리스크 보고서",
            summary="요약",
            url="/reports/20",
        ),
        ChatSource(
            sourceType="REPORT",
            title="중복 보고서",
            summary="요약",
            url="/reports/20",
        ),
        ChatSource(
            sourceType="MATERIAL",
            title="자재 상세",
            summary="요약",
            url="/materials/11",
        ),
    ]

    urls = builder.build_urls(sources)

    assert [url.url for url in urls] == ["/reports/20", "/materials/11"]
    assert urls[0].label == "2026년 5월 생산 리스크 보고서"
    assert urls[0].type == "REPORT"


def test_response_builder_returns_insufficient_evidence_without_sources() -> None:
    builder = ChatResponseBuilder()
    evidence_result = _build_evidence_result([])
    document_result = DocumentSearchResult(sources=[])

    result = builder.build_security_result(evidence_result, document_result)

    assert result.status == SecurityStatus.INSUFFICIENT_EVIDENCE
    assert result.code == ChatErrorCode.CHAT_EVIDENCE_001


def test_response_builder_returns_passed_when_grounding_exists() -> None:
    builder = ChatResponseBuilder()
    evidence_result = _build_evidence_result(
        [
            EvidenceItem(
                type="ORDER",
                title="ORD-202605-001 납기 지연 위험",
                summary="납기일이 임박했습니다.",
                source="customer_orders",
            )
        ]
    )
    document_result = DocumentSearchResult(sources=[])

    result = builder.build_security_result(evidence_result, document_result)

    assert result.status == SecurityStatus.PASSED
    assert result.code is None
