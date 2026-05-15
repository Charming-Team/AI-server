from datetime import datetime

import anyio

from app.core.config import Settings
from app.features.chat.answer_generation_service import AnswerGenerationService
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
        question="최근 보고서 요약해줘",
        requestedAt=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
    )


def test_answer_generation_returns_insufficient_evidence_without_grounding() -> None:
    service = AnswerGenerationService(Settings())
    request = _build_request()
    evidence_result = EvidenceResult(
        intent=ChatIntent.REPORT_LOOKUP,
        basisTime=request.requested_at,
        items=[],
    )
    document_result = DocumentSearchResult(sources=[])

    result = anyio.run(
        service.generate_answer,
        request,
        evidence_result,
        document_result,
    )

    assert result.was_generated is False
    assert "근거" in result.answer
    assert result.skipped_reason == "No RDB Evidence or document sources are available."


def test_answer_generation_does_not_call_llm_when_disabled() -> None:
    service = AnswerGenerationService(Settings(llm_enabled=False))
    request = _build_request()
    evidence_result = EvidenceResult(
        intent=ChatIntent.REPORT_LOOKUP,
        basisTime=request.requested_at,
        items=[],
    )
    document_result = DocumentSearchResult(
        sources=[
            ChatSource(
                sourceType="REPORT",
                title="2026년 5월 생산 리스크 보고서",
                summary="자재 부족과 LINE-A01 병목이 주요 리스크입니다.",
            )
        ]
    )

    result = anyio.run(
        service.generate_answer,
        request,
        evidence_result,
        document_result,
    )

    assert result.was_generated is False
    assert result.answer == "근거는 조회됐지만 LLM 답변 생성 기능이 아직 활성화되지 않았습니다."
    assert result.skipped_reason == "LLM is disabled."


def test_answer_generation_service_builds_grounded_prompt() -> None:
    service = AnswerGenerationService(Settings())
    request = _build_request()
    evidence_result = EvidenceResult(
        intent=ChatIntent.REPORT_LOOKUP,
        basisTime=request.requested_at,
        items=[
            EvidenceItem(
                type="REPORT",
                title="월간 생산 리스크",
                summary="자재 부족이 주요 리스크입니다.",
                source="reports",
            )
        ],
    )
    document_result = DocumentSearchResult(sources=[])

    prompt = service.build_prompt(request, evidence_result, document_result)

    assert "제공된 내부 근거만 사용" in prompt.system_prompt
    assert "월간 생산 리스크" in prompt.user_prompt
