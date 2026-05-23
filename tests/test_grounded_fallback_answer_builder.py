from datetime import datetime

from app.features.chat.grounded_fallback_answer_builder import (
    GroundedFallbackAnswerBuilder,
)
from app.features.chat.schemas import (
    ChatIntent,
    ChatSource,
    DocumentSearchResult,
    EvidenceItem,
    EvidenceResult,
)


def test_grounded_fallback_answer_builder_summarizes_internal_evidence() -> None:
    builder = GroundedFallbackAnswerBuilder()
    basis_time = datetime.fromisoformat("2026-05-12T10:30:00+09:00")
    evidence_result = EvidenceResult(
        intent=ChatIntent.MATERIAL_SHORTAGE,
        basisTime=basis_time,
        items=[
            EvidenceItem(
                type="MATERIAL",
                title="RM-AL-001 알루미늄 원자재 재고 부족",
                summary="생산계획 1001에서 부족 수량 60KG입니다.",
                source="production_plan_materials",
            )
        ],
    )
    document_result = DocumentSearchResult(
        sources=[
            ChatSource(
                sourceType="REPORT",
                title="5월 생산 리스크 보고서",
                summary="LINE-A01 병목과 자재 부족을 주요 리스크로 제시합니다.",
            )
        ]
    )

    answer = builder.build(evidence_result, document_result)

    assert "확인된 RDB 근거와 문서 검색 근거 기준으로 요약합니다." in answer
    assert "RDB 근거:" in answer
    assert "RM-AL-001 알루미늄 원자재 재고 부족" in answer
    assert "문서 검색 근거:" in answer
    assert "5월 생산 리스크 보고서" in answer
    assert "확인 필요" in answer


def test_grounded_fallback_answer_builder_limits_items_and_long_summaries() -> None:
    builder = GroundedFallbackAnswerBuilder()
    basis_time = datetime.fromisoformat("2026-05-12T10:30:00+09:00")
    evidence_result = EvidenceResult(
        intent=ChatIntent.REPORT_LOOKUP,
        basisTime=basis_time,
        items=[
            EvidenceItem(
                type="REPORT",
                title=f"{index}번째 RDB 근거",
                summary="A" * 240,
                source="reports",
            )
            for index in range(1, 5)
        ],
    )
    document_result = DocumentSearchResult(sources=[])

    answer = builder.build(evidence_result, document_result)

    assert "1번째 RDB 근거" in answer
    assert "2번째 RDB 근거" in answer
    assert "3번째 RDB 근거" in answer
    assert "4번째 RDB 근거" not in answer
    assert "A" * 200 not in answer
    assert "..." in answer


def test_grounded_fallback_answer_builder_explains_document_only_grounding() -> None:
    builder = GroundedFallbackAnswerBuilder()
    basis_time = datetime.fromisoformat("2026-05-12T10:30:00+09:00")
    evidence_result = EvidenceResult(
        intent=ChatIntent.REPORT_LOOKUP,
        basisTime=basis_time,
        items=[],
    )
    document_result = DocumentSearchResult(
        sources=[
            ChatSource(
                sourceType="COMPANY_INFO",
                title="S-Map 회사 개요",
                summary="S-Map은 B2B 화학공정 회사로 ABS, PP, PE 컴파운드를 생산합니다.",
            )
        ]
    )

    answer = builder.build(evidence_result, document_result)

    assert "확인된 문서 검색 근거 기준으로 요약합니다." in answer
    assert "RDB 근거:" not in answer
    assert "문서 검색 근거:" in answer
    assert "S-Map 회사 개요" in answer


def test_grounded_fallback_answer_builder_explains_rdb_only_grounding() -> None:
    builder = GroundedFallbackAnswerBuilder()
    basis_time = datetime.fromisoformat("2026-05-12T10:30:00+09:00")
    evidence_result = EvidenceResult(
        intent=ChatIntent.MATERIAL_SHORTAGE,
        basisTime=basis_time,
        items=[
            EvidenceItem(
                type="MATERIAL",
                title="MAT-FOAM-ADD 부족",
                summary="계획 필요량 대비 가용 재고가 부족합니다.",
                source="chat_material_shortage_evidence_view",
            )
        ],
    )
    document_result = DocumentSearchResult(sources=[])

    answer = builder.build(evidence_result, document_result)

    assert "확인된 RDB 근거 기준으로 요약합니다." in answer
    assert "RDB 근거:" in answer
    assert "문서 검색 근거:" not in answer
