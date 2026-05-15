from datetime import datetime

import anyio

from app.core.config import Settings
from app.features.chat.schemas import (
    AnswerGenerationResult,
    ChatAnswerRequest,
    ChatIntent,
    ChatSource,
    ChatUserContext,
    DocumentSearchResult,
    EvidenceItem,
    EvidenceResult,
    SecurityResult,
    SecurityStatus,
)
from app.features.chat.service import ChatService

BLOCKED_GENERATED_ANSWER = (
    "보안상 생성된 답변을 제공할 수 없습니다. "
    "업무 데이터에 대한 질문으로 다시 요청해 주세요."
)
SENSITIVE_OUTPUT_REASON = (
    "생성 답변에 민감 정보 또는 내부 설정 정보가 포함된 것으로 판단되었습니다."
)


class FakeEvidenceService:
    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        return EvidenceResult(
            intent=intent,
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


class FakeDocumentSearchService:
    def __init__(
        self,
        sources: list[ChatSource] | None = None,
        was_searched: bool = False,
        skipped_reason: str | None = None,
    ) -> None:
        self.sources = sources or []
        self.was_searched = was_searched
        self.skipped_reason = skipped_reason

    async def search(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> DocumentSearchResult:
        return DocumentSearchResult(
            was_searched=self.was_searched,
            sources=self.sources,
            skipped_reason=self.skipped_reason,
        )


class FakeGeneratedAnswerGenerationService:
    async def generate_answer(
        self,
        request: ChatAnswerRequest,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> AnswerGenerationResult:
        return AnswerGenerationResult(
            answer="근거에 따르면 자재 부족이 주요 리스크입니다.",
            was_generated=True,
        )


class FakeBlockedAnswerGenerationService:
    async def generate_answer(
        self,
        request: ChatAnswerRequest,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
    ) -> AnswerGenerationResult:
        return AnswerGenerationResult(
            answer=BLOCKED_GENERATED_ANSWER,
            was_generated=False,
            skipped_reason="생성 답변이 출력 보안 정책에 의해 차단되었습니다.",
            security_result=SecurityResult(
                status=SecurityStatus.BLOCKED_SENSITIVE_REQUEST,
                reason="생성 답변에 민감 정보 또는 내부 설정 정보가 포함된 것으로 판단되었습니다.",
            ),
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


def test_chat_service_uses_answer_output_security_result() -> None:
    service = ChatService(Settings())
    service.evidence_service = FakeEvidenceService()
    service.document_search_service = FakeDocumentSearchService()
    service.answer_generation_service = FakeBlockedAnswerGenerationService()

    response = anyio.run(service.create_answer, _build_request())

    assert response.security_result.status == SecurityStatus.BLOCKED_SENSITIVE_REQUEST
    assert response.security_result.reason == SENSITIVE_OUTPUT_REASON
    assert response.answer == BLOCKED_GENERATED_ANSWER
    assert (
        response.model_result.llm_generation_skipped_reason
        == "생성 답변이 출력 보안 정책에 의해 차단되었습니다."
    )


def test_chat_service_builds_detailed_model_result_counts() -> None:
    service = ChatService(Settings())
    service.evidence_service = FakeEvidenceService()
    service.document_search_service = FakeDocumentSearchService(
        was_searched=True,
        sources=[
            ChatSource(
                sourceType="REPORT",
                title="월간 생산 리스크 보고서",
                summary="자재 부족이 주요 리스크입니다.",
            ),
            ChatSource(
                sourceType="COMPANY_INFO",
                title="회사 운영 기준",
                summary="생산계획 우선순위 기준입니다.",
            ),
        ],
    )
    service.answer_generation_service = FakeGeneratedAnswerGenerationService()

    response = anyio.run(service.create_answer, _build_request())

    assert response.model_result.used_vector_search is True
    assert response.model_result.used_rdb_evidence is True
    assert response.model_result.used_llm_generation is True
    assert response.model_result.rdb_evidence_count == 1
    assert response.model_result.document_source_count == 2
    assert response.model_result.evidence_count == 3
    assert response.model_result.vector_search_skipped_reason is None
    assert response.model_result.llm_generation_skipped_reason is None


def test_chat_service_builds_model_result_skipped_reasons() -> None:
    service = ChatService(Settings())
    service.evidence_service = FakeEvidenceService()
    service.document_search_service = FakeDocumentSearchService(
        skipped_reason="임베딩 기능이 비활성화되어 있습니다."
    )
    service.answer_generation_service = FakeBlockedAnswerGenerationService()

    response = anyio.run(service.create_answer, _build_request())

    assert response.model_result.used_vector_search is False
    assert response.model_result.used_llm_generation is False
    assert (
        response.model_result.vector_search_skipped_reason
        == "임베딩 기능이 비활성화되어 있습니다."
    )
    assert (
        response.model_result.llm_generation_skipped_reason
        == "생성 답변이 출력 보안 정책에 의해 차단되었습니다."
    )
