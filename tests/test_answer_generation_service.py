from datetime import datetime

import anyio
import pytest

from app.core.config import Settings
from app.features.chat.answer_generation_service import AnswerGenerationService
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.grounded_prompt_builder import GroundedPrompt
from app.features.chat.llm_client import LlmCompletion
from app.features.chat.llm_response_cache import LlmResponseCache
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatErrorCode,
    ChatIntent,
    ChatSource,
    ChatUserContext,
    DocumentSearchResult,
    EvidenceItem,
    EvidenceResult,
    LlmUsage,
)

BLOCKED_GENERATED_ANSWER = (
    "보안상 생성된 답변을 제공할 수 없습니다. "
    "업무 데이터에 대한 질문으로 다시 요청해 주세요."
)


class FakeLlmClient:
    def __init__(
        self,
        answer: str = "보고서 근거에 따르면 자재 부족이 주요 리스크입니다.",
        usage: LlmUsage | None = None,
    ) -> None:
        self.answer = answer
        self.usage = usage
        self.prompt: GroundedPrompt | None = None
        self.call_count = 0

    async def generate(self, prompt: GroundedPrompt) -> str:
        completion = await self.generate_completion(prompt)
        return completion.answer

    async def generate_completion(self, prompt: GroundedPrompt) -> LlmCompletion:
        self.call_count += 1
        self.prompt = prompt
        return LlmCompletion(answer=self.answer, usage=self.usage)


class FakeFailingLlmClient:
    def __init__(self) -> None:
        self.prompt: GroundedPrompt | None = None

    async def generate_completion(self, prompt: GroundedPrompt) -> LlmCompletion:
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
    assert "확인된 문서 검색 근거 기준으로 요약합니다." in result.answer
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


def test_answer_generation_prompt_excludes_raw_rdb_evidence_source_data() -> None:
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

    assert "원천 데이터" not in prompt.user_prompt
    assert '"planMaterialId": 7001' not in prompt.user_prompt
    assert '"availableInventoryQuantity": 30.0' not in prompt.user_prompt
    assert '"safetyStockQuantity": 50.0' not in prompt.user_prompt
    assert '"inventoryStatus": "LOW"' not in prompt.user_prompt
    assert "RM-AL-001 알루미늄 원자재 재고 부족" in prompt.user_prompt
    assert "생산계획 1001에서 RM-AL-001 알루미늄 원자재 부족 상태입니다." in (
        prompt.user_prompt
    )


def test_answer_generation_calls_llm_when_enabled_and_grounded() -> None:
    llm_client = FakeLlmClient(
        "2026년 5월 생산 리스크 보고서 근거에 따르면 자재 부족이 주요 리스크입니다.",
        usage=LlmUsage(promptTokens=120, completionTokens=32, totalTokens=152),
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
    assert result.llm_cache_hit is False
    assert result.llm_usage is not None
    assert result.llm_usage.prompt_tokens == 120
    assert result.llm_usage.completion_tokens == 32
    assert result.llm_usage.total_tokens == 152
    assert result.answer.startswith(
        "핵심 답변:\n"
        "2026년 5월 생산 리스크 보고서 근거에 따르면 자재 부족이 주요 리스크입니다."
    )
    assert "근거:\n- [QDRANT] 2026년 5월 생산 리스크 보고서" in result.answer
    assert "확인 필요:" in result.answer
    assert llm_client.prompt is not None
    assert "2026년 5월 생산 리스크 보고서" in llm_client.prompt.user_prompt
    assert llm_client.call_count == 1


def test_answer_generation_reuses_cached_llm_answer_for_same_grounded_prompt() -> None:
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

    first_result = anyio.run(
        service.generate_answer,
        request,
        evidence_result,
        document_result,
    )
    second_result = anyio.run(
        service.generate_answer,
        request,
        evidence_result,
        document_result,
    )

    assert first_result.was_generated is True
    assert first_result.llm_cache_hit is False
    assert second_result.was_generated is True
    assert second_result.llm_cache_hit is True
    assert second_result.llm_usage is not None
    assert second_result.llm_usage.total_tokens == 0
    assert first_result.answer == second_result.answer
    assert llm_client.call_count == 1


def test_answer_generation_cache_key_changes_by_reasoning_effort() -> None:
    cache = LlmResponseCache(ttl_seconds=60.0, max_entries=10)
    first_llm_client = FakeLlmClient("minimal reasoning 답변입니다.")
    second_llm_client = FakeLlmClient("low reasoning 답변입니다.")
    first_service = AnswerGenerationService(
        Settings(llm_enabled=True, llm_reasoning_effort="minimal"),
        llm_client=first_llm_client,
        llm_response_cache=cache,
    )
    second_service = AnswerGenerationService(
        Settings(llm_enabled=True, llm_reasoning_effort="low"),
        llm_client=second_llm_client,
        llm_response_cache=cache,
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

    first_result = anyio.run(
        first_service.generate_answer,
        request,
        evidence_result,
        document_result,
    )
    second_result = anyio.run(
        second_service.generate_answer,
        request,
        evidence_result,
        document_result,
    )

    assert first_result.llm_cache_hit is False
    assert second_result.llm_cache_hit is False
    assert "minimal reasoning 답변입니다." in first_result.answer
    assert "low reasoning 답변입니다." in second_result.answer
    assert first_llm_client.call_count == 1
    assert second_llm_client.call_count == 1


def test_answer_generation_skips_llm_cache_when_disabled() -> None:
    llm_client = FakeLlmClient(
        "2026년 5월 생산 리스크 보고서 근거에 따르면 자재 부족이 주요 리스크입니다."
    )
    service = AnswerGenerationService(
        Settings(
            llm_enabled=True,
            llm_response_cache_enabled=False,
        ),
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

    anyio.run(
        service.generate_answer,
        request,
        evidence_result,
        document_result,
    )
    anyio.run(
        service.generate_answer,
        request,
        evidence_result,
        document_result,
    )

    assert llm_client.call_count == 2


def test_answer_generation_keeps_structured_llm_answer() -> None:
    structured_answer = (
        "핵심 답변:\n"
        "2026년 5월 생산 리스크 보고서 기준으로 자재 부족이 주요 리스크입니다.\n\n"
        "근거:\n"
        "- 2026년 5월 생산 리스크 보고서\n\n"
        "확인 필요:\n"
        "- 근거에 없는 원인은 추가 확인이 필요합니다."
    )
    llm_client = FakeLlmClient(structured_answer)
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
    assert result.answer == structured_answer


def test_answer_generation_normalizes_duplicated_llm_sections() -> None:
    duplicated_answer = (
        "핵심 답변:\n"
        "핵심 답변:\n"
        "LINE-ABS-01은 대기 수량과 대기 시간이 높아 병목 가능성이 있습니다.\n\n"
        "근거:\n"
        "- RDB 출처 제목: LINE-ABS-01 RUNNING — 대기 수량 3200건, 대기 시간 2.5시간\n"
        "- QDRANT 출처 제목: S-Map 생산 라인 구성 — 라인 역할 보조 설명\n\n"
        "근거:\n"
        "- [RDB] LINE-ABS-01 RUNNING\n"
        "- [QDRANT] S-Map 생산 라인 구성\n"
        "- [QDRANT] S-Map 도메인 용어 사전\n\n"
        "확인 필요:\n"
        "- 설비 구간별 상세 병목은 추가 확인이 필요합니다.\n"
        "- 자재 대기 여부는 추가 확인이 필요합니다."
    )
    llm_client = FakeLlmClient(duplicated_answer)
    service = AnswerGenerationService(
        Settings(llm_enabled=True),
        llm_client=llm_client,
    )
    request = _build_request(role="MANUFACTURING_MANAGER")
    evidence_result = EvidenceResult(
        intent=ChatIntent.LINE_BOTTLENECK,
        basisTime=request.requested_at,
        items=[
            EvidenceItem(
                type="LINE",
                title="LINE-ABS-01 RUNNING",
                summary="대기 수량 3200건, 대기 시간 2.5시간입니다.",
                source="chat_line_bottleneck_evidence_view",
            )
        ],
    )
    document_result = DocumentSearchResult(
        sources=[
            ChatSource(
                sourceType="COMPANY_INFO",
                title="S-Map 생산 라인 구성",
                summary="라인별 역할 설명입니다.",
                sourceOrigin="QDRANT",
            ),
            ChatSource(
                sourceType="COMPANY_INFO",
                title="S-Map 도메인 용어 사전",
                summary="병목 관련 용어 설명입니다.",
                sourceOrigin="QDRANT",
            ),
        ]
    )

    result = anyio.run(
        service.generate_answer,
        request,
        evidence_result,
        document_result,
    )

    assert result.was_generated is True
    assert result.answer.count("핵심 답변:") == 1
    assert result.answer.count("근거:") == 1
    assert result.answer.count("확인 필요:") == 1
    evidence_section = result.answer.split("근거:\n", 1)[1].split(
        "\n\n확인 필요:",
        1,
    )[0]
    follow_up_section = result.answer.split("확인 필요:\n", 1)[1]
    assert len(evidence_section.splitlines()) == 2
    assert len(follow_up_section.splitlines()) == 1
    assert "S-Map 도메인 용어 사전" not in evidence_section


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
    assert "확인된 RDB 근거와 문서 검색 근거 기준으로 요약합니다." in result.answer
    assert "RDB 근거:" in result.answer
    assert "문서 검색 근거:" in result.answer
    assert "월간 생산 리스크" in result.answer
    assert "2026년 5월 생산 리스크 보고서" in result.answer
    assert (
        result.skipped_reason
        == "LLM 서버 호출에 실패해 근거 기반 대체 답변을 반환했습니다."
    )
    assert llm_client.prompt is not None


def test_answer_generation_wraps_unstructured_llm_answer_with_source_titles() -> None:
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
        "핵심 답변:\n"
        "근거에 따르면 자재 부족이 주요 리스크입니다.\n\n"
        "근거:\n"
        "- [RDB] 월간 생산 리스크\n"
        "- [QDRANT] 2026년 5월 생산 리스크 보고서\n\n"
        "확인 필요:\n"
        "- 위 근거에 포함되지 않은 수치, 원인, 조치는 추가 확인이 필요합니다."
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
    assert llm_client.call_count == 1

    second_result = anyio.run(
        service.generate_answer,
        request,
        evidence_result,
        document_result,
    )

    assert second_result.was_generated is False
    assert second_result.security_result is not None
    assert llm_client.call_count == 2


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
