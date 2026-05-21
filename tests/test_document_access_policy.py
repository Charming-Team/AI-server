from app.features.chat.document_access_policy import DocumentAccessPolicy
from app.features.chat.schemas import ChatSource, DocumentSearchResult
from app.features.chat.skip_reasons import QDRANT_OPERATOR_RESTRICTED_CONTENT


def _build_source(
    *,
    title: str = "월간 생산 리스크 보고서",
    summary: str = "자재 부족과 LINE-A01 병목이 주요 리스크입니다.",
) -> ChatSource:
    return ChatSource(
        sourceType="REPORT",
        title=title,
        summary=summary,
        url="/reports/20?mode=read",
        sourceOrigin="QDRANT",
    )


def test_document_access_policy_keeps_executive_financial_source() -> None:
    policy = DocumentAccessPolicy()
    source = _build_source(
        title="납기 지연 패널티 보고서",
        summary="계약 금액과 패널티 금액을 함께 검토합니다.",
    )

    result = policy.sanitize_search_result(
        DocumentSearchResult(was_searched=True, sources=[source]),
        "EXECUTIVE",
    )

    assert result.sources == [source]
    assert result.skipped_reason is None


def test_document_access_policy_filters_operator_financial_source() -> None:
    policy = DocumentAccessPolicy()
    financial_source = _build_source(
        title="납기 지연 패널티 보고서",
        summary="계약 금액과 패널티 금액을 함께 검토합니다.",
    )
    operational_source = _build_source()

    result = policy.sanitize_search_result(
        DocumentSearchResult(
            was_searched=True,
            sources=[financial_source, operational_source],
        ),
        "OPERATOR",
    )

    assert result.sources == [operational_source]
    assert result.skipped_reason is None


def test_document_access_policy_marks_reason_when_all_operator_sources_filtered() -> None:
    policy = DocumentAccessPolicy()
    financial_source = _build_source(
        title="납기 지연 패널티 보고서",
        summary="계약 금액과 패널티 금액을 함께 검토합니다.",
    )

    result = policy.sanitize_search_result(
        DocumentSearchResult(was_searched=True, sources=[financial_source]),
        "OPERATOR",
    )

    assert result.sources == []
    assert result.skipped_reason == QDRANT_OPERATOR_RESTRICTED_CONTENT
