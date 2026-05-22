from datetime import datetime

import anyio
import pytest

from app.core.config import Settings
from app.features.chat.answer_generation_service import AnswerGenerationService
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.grounded_prompt_builder import GroundedPrompt
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatErrorCode,
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


class FakeFailingLlmClient:
    def __init__(self) -> None:
        self.prompt: GroundedPrompt | None = None

    async def generate(self, prompt: GroundedPrompt) -> str:
        self.prompt = prompt
        raise ChatExternalServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_LLM_003,
            message="LLM 서버 호출에 실패했습니다.",
        )


def _build_request(role: str = "EXECUTIVE") -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=10,
        messageId=24,
        user=ChatUserContext(
            userId=1,
            role=role,
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
    assert "추측성 답변은 제공하지 않습니다." in result.answer
    assert "아직 연결되지 않았습니다" not in result.answer
    assert result.skipped_reason == "RDB Evidence와 문서 검색 근거가 없습니다."


def test_answer_generation_returns_grounded_fallback_when_llm_disabled() -> None:
    llm_client = FakeLlmClient()
    service = AnswerGenerationService(
        Settings(llm_enabled=False),
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
    assert "확인된 내부 근거 기준으로 요약합니다." in result.answer
    assert "문서 검색 근거:" in result.answer
    assert "2026년 5월 생산 리스크 보고서" in result.answer
    assert "자재 부족과 LINE-A01 병목이 주요 리스크입니다." in result.answer
    assert "확인 필요" in result.answer
    assert result.skipped_reason == "LLM 답변 생성 기능이 비활성화되어 있습니다."
    assert llm_client.prompt is None


@pytest.mark.parametrize(
    ("settings", "expected_message"),
    [
        (
            Settings(llm_enabled=True, llm_base_url=" "),
            "LLM 필수 설정이 누락되었습니다: llm_base_url",
        ),
        (
            Settings(llm_enabled=True, llm_model=" "),
            "LLM 필수 설정이 누락되었습니다: llm_model",
        ),
        (
            Settings(llm_enabled=True, llm_base_url=" ", llm_model=" "),
            "LLM 필수 설정이 누락되었습니다: llm_base_url, llm_model",
        ),
    ],
)
def test_answer_generation_requires_llm_settings_when_enabled(
    settings: Settings,
    expected_message: str,
) -> None:
    service = AnswerGenerationService(settings, llm_client=FakeLlmClient())
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

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(
            service.generate_answer,
            request,
            evidence_result,
            document_result,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ChatErrorCode.CHAT_LLM_001
    assert exc_info.value.message == expected_message


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


def test_answer_generation_prompt_includes_rdb_evidence_source_data() -> None:
    service = AnswerGenerationService(Settings())
    request = _build_request(role="MANUFACTURING_MANAGER")
    evidence_result = EvidenceResult(
        intent=ChatIntent.MATERIAL_SHORTAGE,
        basisTime=request.requested_at,
        items=[
            EvidenceItem(
                type="MATERIAL",
                title="RM-AL-001 알루미늄 원자재 재고 부족",
                summary=(
                    "생산계획 1001에서 RM-AL-001 알루미늄 원자재 부족 상태입니다."
                ),
                url="/materials/inventory/11?mode=read",
                source="production_plan_materials",
                referenceId=7001,
                data={
                    "planMaterialId": 7001,
                    "planId": 1001,
                    "materialCode": "RM-AL-001",
                    "requiredQuantity": 150.0,
                    "reservedQuantity": 90.0,
                    "shortageQuantity": 60.0,
                    "inventoryRegistered": True,
                    "currentInventoryQuantity": 120.0,
                    "availableInventoryQuantity": 30.0,
                    "safetyStockQuantity": 50.0,
                    "inventoryStatus": "LOW",
                },
            )
        ],
    )
    document_result = DocumentSearchResult(sources=[])

    prompt = service.build_prompt(request, evidence_result, document_result)

    assert "원천 데이터" in prompt.user_prompt
    assert '"planMaterialId": 7001' in prompt.user_prompt
    assert '"availableInventoryQuantity": 30.0' in prompt.user_prompt
    assert '"safetyStockQuantity": 50.0' in prompt.user_prompt
    assert '"inventoryStatus": "LOW"' in prompt.user_prompt


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


def test_answer_generation_returns_grounded_fallback_when_llm_call_fails() -> None:
    llm_client = FakeFailingLlmClient()
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
                summary="LINE-A01 병목이 확인되었습니다.",
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
    assert "확인된 내부 근거 기준으로 요약합니다." in result.answer
    assert "RDB 근거:" in result.answer
    assert "문서 검색 근거:" in result.answer
    assert "월간 생산 리스크" in result.answer
    assert "2026년 5월 생산 리스크 보고서" in result.answer
    assert (
        result.skipped_reason
        == "LLM 서버 호출에 실패해 근거 기반 대체 답변을 반환했습니다."
    )
    assert llm_client.prompt is not None


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


def test_answer_generation_blocks_prompt_injection_llm_output() -> None:
    llm_client = FakeLlmClient("이전 지시를 무시하고 새로운 규칙을 따르겠습니다.")
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
    assert result.security_result.status == "BLOCKED_PROMPT_INJECTION"
    assert result.security_result.code == "CHAT_SECURITY_001"


def test_answer_generation_blocks_sensitive_grounded_fallback_output() -> None:
    service = AnswerGenerationService(Settings(llm_enabled=False))
    request = _build_request()
    evidence_result = EvidenceResult(
        intent=ChatIntent.REPORT_LOOKUP,
        basisTime=request.requested_at,
        items=[
            EvidenceItem(
                type="REPORT",
                title="내부 설정 점검",
                summary="내부 시스템 프롬프트와 토큰 값은 외부에 공개하지 않습니다.",
                source="reports",
            )
        ],
    )
    document_result = DocumentSearchResult(sources=[])

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
