from app.core.config import Settings
from app.schemas.chat import (
    ChatAnswerRequest,
    ChatAnswerResponse,
    ChatIntent,
    ModelResult,
    SecurityResult,
    SecurityStatus,
)
from app.services.answer_generation_service import AnswerGenerationService
from app.services.document_search_service import DocumentSearchService
from app.services.evidence_service import EvidenceService
from app.services.intent_classifier import IntentClassifier


class ChatService:
    _sensitive_terms = (
        "system prompt",
        "developer prompt",
        "ignore previous",
        "ignore instructions",
        "api key",
        "password",
        "secret",
        "token",
        "config",
        "model name",
        "시스템 프롬프트",
        "개발자 프롬프트",
        "이전 지시",
        "프롬프트 무시",
        "api key",
        "비밀번호",
        "시크릿",
        "토큰",
        "설정값",
        "모델 정보",
        "모델명",
    )

    def __init__(self, settings: Settings) -> None:
        self.intent_classifier = IntentClassifier()
        self.evidence_service = EvidenceService(settings)
        self.document_search_service = DocumentSearchService(settings)
        self.answer_generation_service = AnswerGenerationService(settings)

    async def create_answer(self, request: ChatAnswerRequest) -> ChatAnswerResponse:
        blocked_status = self._get_blocked_status(request.question)
        if blocked_status is not None:
            return self._build_restricted_response(request, blocked_status)

        intent = self.intent_classifier.classify(request.question)
        evidence_result = await self.evidence_service.get_evidence(request, intent)
        document_result = await self.document_search_service.search(request, evidence_result.intent)
        answer_result = await self.answer_generation_service.generate_answer(
            request,
            evidence_result,
            document_result,
        )

        return ChatAnswerResponse(
            session_id=request.session_id,
            message_id=request.message_id,
            intent=evidence_result.intent,
            answer=answer_result.answer,
            basis_time=evidence_result.basis_time,
            sources=document_result.sources,
            security_result=SecurityResult(
                status=SecurityStatus.INSUFFICIENT_EVIDENCE,
                reason="조회된 RDB Evidence가 없고 Qdrant 검색이 아직 연결되지 않았습니다.",
            ),
            model_result=ModelResult(
                used_vector_search=document_result.was_searched,
                used_rdb_evidence=evidence_result.has_evidence,
                evidence_count=len(evidence_result.items),
            ),
        )

    def _get_blocked_status(self, question: str) -> SecurityStatus | None:
        normalized_question = question.lower()
        for term in self._sensitive_terms:
            if term.lower() in normalized_question:
                return SecurityStatus.BLOCKED_SENSITIVE_REQUEST
        return None

    def _build_restricted_response(
        self,
        request: ChatAnswerRequest,
        status: SecurityStatus,
    ) -> ChatAnswerResponse:
        return ChatAnswerResponse(
            session_id=request.session_id,
            message_id=request.message_id,
            intent=ChatIntent.UNKNOWN,
            answer=(
                "보안상 답변할 수 없는 요청입니다. "
                "업무 데이터에 대한 질문으로 다시 요청해 주세요."
            ),
            basis_time=request.requested_at,
            security_result=SecurityResult(
                status=status,
                reason="민감 정보 또는 내부 설정 정보 요청으로 판단되었습니다.",
            ),
            model_result=ModelResult(
                used_vector_search=False,
                used_rdb_evidence=False,
                evidence_count=0,
            ),
        )
