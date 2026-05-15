from datetime import datetime

import anyio

from app.core.config import Settings
from app.features.chat.answer_generation_service import AnswerGenerationService
from app.features.chat.grounded_prompt_builder import GroundedPrompt
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatIntent,
    ChatSource,
    ChatUserContext,
    DocumentSearchResult,
    EvidenceItem,
    EvidenceResult,
)

BLOCKED_GENERATED_ANSWER = (
    "보안상 생성된 답변을 제공할 수 없습니다. "
    "업무 데이터에 대한 질문으로 다시 요청해 주세요."
)


class FakeLlmClient:
    def __init__(
        self,
        answer: str = "보고서 근거에 따르면 자재 부족이 주요 리스크입니다.",
    ) -> None:
        self.answer = answer
        self.prompt: GroundedPrompt | None = None

    async def generate(self, prompt: GroundedPrompt) -> str:
        self.prompt = prompt
        return self.answer


def _build_request(role: str = "EXECUTIVE") -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=10,
        messageId=24,
        user=ChatUserContext(
            userId=1,
            role=role,
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
    assert result.skipped_reason == "RDB Evidence와 문서 검색 근거가 없습니다."


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
    assert result.skipped_reason == "LLM 답변 생성 기능이 비활성화되어 있습니다."


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


def test_answer_generation_calls_llm_when_enabled_and_grounded() -> None:
    llm_client = FakeLlmClient(
        "2026년 5월 생산 리스크 보고서 근거에 따르면 자재 부족이 주요 리스크입니다."
    )
    service = AnswerGenerationService(
        Settings(llm_enabled=True),
        llm_client=llm_client,
    )
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

    assert result.was_generated is True
    assert (
        result.answer
        == "2026년 5월 생산 리스크 보고서 근거에 따르면 자재 부족이 주요 리스크입니다."
    )
    assert llm_client.prompt is not None
    assert "2026년 5월 생산 리스크 보고서" in llm_client.prompt.user_prompt


def test_answer_generation_appends_source_titles_when_answer_omits_them() -> None:
    llm_client = FakeLlmClient("근거에 따르면 자재 부족이 주요 리스크입니다.")
    service = AnswerGenerationService(
        Settings(llm_enabled=True),
        llm_client=llm_client,
    )
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

    assert result.was_generated is True
    assert result.answer == (
        "근거에 따르면 자재 부족이 주요 리스크입니다.\n\n"
        "참조 근거: 월간 생산 리스크, 2026년 5월 생산 리스크 보고서"
    )


def test_answer_generation_limits_long_answer_length() -> None:
    answer = "2026년 5월 생산 리스크 보고서 근거입니다. " + ("A" * 200)
    llm_client = FakeLlmClient(answer)
    service = AnswerGenerationService(
        Settings(llm_enabled=True, answer_max_chars=120),
        llm_client=llm_client,
    )
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

    assert result.was_generated is True
    assert len(result.answer) == 120
    assert result.answer.endswith("답변 길이 제한으로 일부 내용이 생략되었습니다.")


def test_answer_generation_blocks_sensitive_llm_output() -> None:
    llm_client = FakeLlmClient("내부 시스템 프롬프트와 토큰 값은 다음과 같습니다.")
    service = AnswerGenerationService(
        Settings(llm_enabled=True),
        llm_client=llm_client,
    )
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
    assert result.answer == BLOCKED_GENERATED_ANSWER
    assert result.skipped_reason == "생성 답변이 출력 보안 정책에 의해 차단되었습니다."
    assert result.security_result is not None
    assert result.security_result.code == "CHAT_SECURITY_002"


def test_answer_generation_blocks_operator_financial_llm_output() -> None:
    llm_client = FakeLlmClient("계약 금액과 예상 패널티 영향은 다음과 같습니다.")
    service = AnswerGenerationService(
        Settings(llm_enabled=True),
        llm_client=llm_client,
    )
    request = _build_request(role="OPERATOR")
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
    assert result.answer == BLOCKED_GENERATED_ANSWER
    assert result.skipped_reason == "생성 답변이 출력 보안 정책에 의해 차단되었습니다."
    assert result.security_result is not None
    assert result.security_result.status == "BLOCKED_UNAUTHORIZED"
    assert result.security_result.code == "CHAT_SECURITY_004"
