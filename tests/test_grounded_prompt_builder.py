from datetime import datetime

from app.features.chat.grounded_prompt_builder import GroundedPromptBuilder
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatIntent,
    ChatSource,
    ChatUserContext,
    DocumentSearchResult,
    EvidenceItem,
    EvidenceResult,
)


def _build_request() -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=10,
        messageId=24,
        user=ChatUserContext(
            userId=1,
            role="EXECUTIVE",
            department="경영기획팀",
            companyName="S-MAP",
            status="ACTIVE",
        ),
        question="최근 납기 위험과 자재 부족 근거를 요약해줘",
        requestedAt=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
    )


def test_grounded_prompt_builder_includes_internal_grounding_rules() -> None:
    builder = GroundedPromptBuilder()
    request = _build_request()
    evidence_result = EvidenceResult(
        intent=ChatIntent.DELIVERY_RISK,
        basisTime=request.requested_at,
        items=[],
    )
    document_result = DocumentSearchResult(sources=[])

    prompt = builder.build(request, evidence_result, document_result)

    assert "제공된 내부 근거만 사용" in prompt.system_prompt
    assert "웹 검색" in prompt.system_prompt
    assert "일반 상식" in prompt.system_prompt
    assert "RDB 근거:\n없음" in prompt.user_prompt
    assert "문서 검색 근거:\n없음" in prompt.user_prompt


def test_grounded_prompt_builder_formats_evidence_and_document_sources() -> None:
    builder = GroundedPromptBuilder()
    request = _build_request()
    evidence_result = EvidenceResult(
        intent=ChatIntent.DELIVERY_RISK,
        basisTime=request.requested_at,
        items=[
            EvidenceItem(
                type="ORDER",
                title="ORD-202605-001 납기 지연 위험",
                summary="납기일이 임박했고 현재 계획 상태는 DELAYED입니다.",
                url="/orders/1001",
                source="customer_orders",
                referenceId=1001,
                data={"orderNo": "ORD-202605-001", "riskLevel": "WARNING"},
            )
        ],
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

    prompt = builder.build(request, evidence_result, document_result)

    assert "ORD-202605-001 납기 지연 위험" in prompt.user_prompt
    assert '"orderNo": "ORD-202605-001"' in prompt.user_prompt
    assert "2026년 5월 생산 리스크 보고서" in prompt.user_prompt
    assert "report-202605:summary" in prompt.user_prompt
