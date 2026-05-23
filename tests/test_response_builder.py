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
                basisTime=datetime.fromisoformat("2026-05-12T11:00:00+09:00"),
                sourceOrigin="QDRANT",
            )
        ]
    )

    sources = builder.build_sources(evidence_result, document_result)

    assert len(sources) == 2
    assert sources[0].source_type == "ORDER"
    assert sources[0].reference_id == 1001
    assert sources[0].basis_time == datetime.fromisoformat("2026-05-12T10:30:00+09:00")
    assert sources[0].source_origin == "RDB"
    assert sources[1].source_type == "REPORT"
    assert sources[1].source == "report-202605:summary"
    assert sources[1].basis_time == datetime.fromisoformat("2026-05-12T11:00:00+09:00")
    assert sources[1].source_origin == "QDRANT"


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


def test_response_builder_builds_latest_basis_time_from_sources() -> None:
    builder = ChatResponseBuilder()
    sources = [
        ChatSource(
            sourceType="MATERIAL",
            title="자재 부족 현황",
            summary="MAT-001 부족",
            basisTime=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
            sourceOrigin="RDB",
        ),
        ChatSource(
            sourceType="REPORT",
            title="월간 생산 리스크 보고서",
            summary="자재 부족과 라인 병목 분석",
            basisTime=datetime.fromisoformat("2026-05-12T11:00:00+09:00"),
            sourceOrigin="QDRANT",
        ),
    ]

    basis_time = builder.build_basis_time(
        sources,
        fallback=datetime.fromisoformat("2026-05-12T09:00:00+09:00"),
    )

    assert basis_time == datetime.fromisoformat("2026-05-12T11:00:00+09:00")


def test_response_builder_uses_fallback_basis_time_without_source_times() -> None:
    builder = ChatResponseBuilder()
    fallback = datetime.fromisoformat("2026-05-12T10:30:00+09:00")

    basis_time = builder.build_basis_time(
        [
            ChatSource(
                sourceType="REPORT",
                title="기준 시각 없는 보고서",
                summary="요약",
            )
        ],
        fallback=fallback,
    )

    assert basis_time == fallback


def test_response_builder_keeps_only_safe_internal_urls() -> None:
    builder = ChatResponseBuilder()
    evidence_result = _build_evidence_result(
        [
            EvidenceItem(
                type="ORDER",
                title="외부 URL 근거",
                summary="요약",
                url="https://evil.example/orders/1001",
                source="customer_orders",
            )
        ]
    )
    document_result = DocumentSearchResult(
        sources=[
            ChatSource(
                sourceType="REPORT",
                title="프로토콜 상대 URL 문서",
                summary="요약",
                url="//evil.example/reports/20",
            ),
            ChatSource(
                sourceType="MATERIAL",
                title="안전한 내부 URL 문서",
                summary="요약",
                url=" /materials/11 ",
            ),
            ChatSource(
                sourceType="LINE",
                title="스크립트 URL 문서",
                summary="요약",
                url="javascript:alert(1)",
            ),
        ]
    )

    sources = builder.build_sources(evidence_result, document_result)
    urls = builder.build_urls(sources)

    assert [source.url for source in sources] == [None, None, "/materials/11", None]
    assert [url.url for url in urls] == ["/materials/11"]
    assert urls[0].label == "안전한 내부 URL 문서"


def test_response_builder_truncates_long_evidence_source_summary() -> None:
    builder = ChatResponseBuilder()
    evidence_result = _build_evidence_result(
        [
            EvidenceItem(
                type="REPORT",
                title="월간 생산 리스크 보고서",
                summary="A" * 400,
                source="reports",
            )
        ]
    )

    sources = builder.build_sources(evidence_result, DocumentSearchResult(sources=[]))

    assert len(sources[0].summary) == 300
    assert sources[0].summary == f"{'A' * 297}..."


def test_response_builder_returns_insufficient_evidence_without_sources() -> None:
    builder = ChatResponseBuilder()
    evidence_result = _build_evidence_result([])
    document_result = DocumentSearchResult(sources=[])

    result = builder.build_security_result(evidence_result, document_result)

    assert result.status == SecurityStatus.INSUFFICIENT_EVIDENCE
    assert result.code == ChatErrorCode.CHAT_EVIDENCE_001
    assert result.reason == (
        "조회된 RDB Evidence가 없고 Qdrant 문서 근거도 확인되지 않았습니다. "
        "Qdrant 검색은 수행되지 않았습니다."
    )


def test_response_builder_includes_qdrant_skip_reason_for_insufficient_evidence() -> None:
    builder = ChatResponseBuilder()
    evidence_result = _build_evidence_result([])
    document_result = DocumentSearchResult(
        sources=[],
        was_searched=True,
        skipped_reason="Qdrant 검색 결과가 없습니다.",
    )

    result = builder.build_security_result(evidence_result, document_result)

    assert result.status == SecurityStatus.INSUFFICIENT_EVIDENCE
    assert result.code == ChatErrorCode.CHAT_EVIDENCE_001
    assert result.reason == (
        "조회된 RDB Evidence가 없고 Qdrant 문서 근거도 확인되지 않았습니다. "
        "Qdrant 사유: Qdrant 검색 결과가 없습니다."
    )


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
