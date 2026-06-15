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
    assert "RDB 근거:" not in answer
    assert "RDB 근거로는" in answer
    assert "RM-AL-001 알루미늄 원자재 재고 부족" in answer
    assert "문서 검색 근거:" not in answer
    assert "문서 근거로는" in answer
    assert "5월 생산 리스크 보고서" in answer
    assert "추가 확인이 필요" in answer


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
    assert "문서 검색 근거:" not in answer
    assert "문서 근거로는" in answer
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
    assert "RDB 근거:" not in answer
    assert "RDB 근거로는" in answer
    assert "문서 검색 근거:" not in answer


def test_grounded_fallback_answer_builder_uses_material_shortage_plan_summary() -> None:
    builder = GroundedFallbackAnswerBuilder()
    basis_time = datetime.fromisoformat("2026-05-12T10:30:00+09:00")
    evidence_result = EvidenceResult(
        intent=ChatIntent.MATERIAL_SHORTAGE,
        basisTime=basis_time,
        items=[
            EvidenceItem(
                type="PLAN",
                title="자재 부족 영향 생산계획",
                summary=(
                    "자재 부족으로 영향받는 생산계획은 총 3건. "
                    "영향 계획: 계획 425 / ORD-202605-028 / MAT-HDPE / "
                    "부족 1250KG / LINE-PE-02. "
                    "부족 자재: MAT-HDPE 2개, MAT-PP-BASE 1개."
                ),
                url="/production-plans?mode=read",
                source="chat_material_shortage_evidence_view",
            )
        ],
    )
    document_result = DocumentSearchResult(sources=[])

    answer = builder.build(evidence_result, document_result)

    assert "RDB 근거로는" not in answer
    assert "자재 부족 영향 생산계획에서는" not in answer
    assert "영향받는 생산계획은 총 3건" in answer
    assert "MAT-HDPE 2개" in answer
    assert "..." not in answer
    assert "생산계획 ID:" not in answer


def test_grounded_fallback_answer_builder_uses_urgent_order_impact_summary() -> None:
    builder = GroundedFallbackAnswerBuilder()
    basis_time = datetime.fromisoformat("2026-05-12T10:30:00+09:00")
    evidence_result = EvidenceResult(
        intent=ChatIntent.URGENT_ORDER_IMPACT,
        basisTime=basis_time,
        items=[
            EvidenceItem(
                type="ORDER",
                title="긴급 주문 전체 생산계획 영향",
                summary=(
                    "조회된 시뮬레이션 기준 긴급 주문 영향 대상은 총 3건. "
                    "변경 후 지연 예상: 2건(ORD-202605-020, ORD-202605-033). "
                    "총 지연 감소: 7시간."
                ),
                url="/schedule-simulations?mode=read",
                source="chat_urgent_order_impact_evidence_view",
            )
        ],
    )
    document_result = DocumentSearchResult(sources=[])

    answer = builder.build(evidence_result, document_result)

    assert "RDB 근거로는" not in answer
    assert "긴급 주문 전체 생산계획 영향에서는" not in answer
    assert "영향 대상은 총 3건" in answer
    assert "변경 후 지연 예상: 2건" in answer
    assert "총 지연 감소: 7시간" in answer
    assert "..." not in answer
    assert "simulation_id" not in answer
    assert "simulation_detail_id" not in answer
