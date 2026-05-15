from datetime import datetime

import anyio

from app.core.config import Settings
from app.features.chat.schemas import (
    AnswerGenerationResult,
    ChatAnswerRequest,
    ChatIntent,
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
    async def search(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> DocumentSearchResult:
        return DocumentSearchResult(was_searched=False, sources=[])


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
            skipped_reason="Generated answer failed output safety policy.",
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
