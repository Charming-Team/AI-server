from app.core.config import Settings
from app.features.chat.answer_generation_service import AnswerGenerationService
from app.features.chat.document_search_service import DocumentSearchService
from app.features.chat.evidence_service import EvidenceService
from app.features.chat.intent_classifier import IntentClassifier
from app.features.chat.question_validator import QuestionValidator
from app.features.chat.response_builder import ChatResponseBuilder
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatAnswerResponse,
    ChatIntent,
    ModelResult,
    SecurityResult,
    SecurityStatus,
)
from app.features.chat.security_policy import SecurityPolicy


class ChatService:
    def __init__(self, settings: Settings) -> None:
        self.question_validator = QuestionValidator()
        self.security_policy = SecurityPolicy()
        self.response_builder = ChatResponseBuilder()
        self.intent_classifier = IntentClassifier()
        self.evidence_service = EvidenceService(settings)
        self.document_search_service = DocumentSearchService(settings)
        self.answer_generation_service = AnswerGenerationService(settings)

    async def create_answer(self, request: ChatAnswerRequest) -> ChatAnswerResponse:
        validation_result = self.question_validator.validate(request.question)
        if validation_result is not None:
            return self._build_restricted_response(request, validation_result)

        security_result = self.security_policy.evaluate(request.question)
        if security_result is not None:
            return self._build_restricted_response(request, security_result)

        intent = self.intent_classifier.classify(request.question)
        evidence_result = await self.evidence_service.get_evidence(request, intent)
        document_result = await self.document_search_service.search(request, evidence_result.intent)
        answer_result = await self.answer_generation_service.generate_answer(
            request,
            evidence_result,
            document_result,
        )
        sources = self.response_builder.build_sources(evidence_result, document_result)

        return ChatAnswerResponse(
            session_id=request.session_id,
            message_id=request.message_id,
            intent=evidence_result.intent,
            answer=answer_result.answer,
            basis_time=evidence_result.basis_time,
            urls=self.response_builder.build_urls(sources),
            sources=sources,
            security_result=self.response_builder.build_security_result(
                evidence_result,
                document_result,
            ),
            model_result=ModelResult(
                used_vector_search=document_result.was_searched,
                used_rdb_evidence=evidence_result.has_evidence,
                evidence_count=len(evidence_result.items),
            ),
        )

    def _build_restricted_response(
        self,
        request: ChatAnswerRequest,
        security_result: SecurityResult,
    ) -> ChatAnswerResponse:
        return ChatAnswerResponse(
            session_id=request.session_id,
            message_id=request.message_id,
            intent=ChatIntent.UNKNOWN,
            answer=self._build_restricted_answer(security_result),
            basis_time=request.requested_at,
            security_result=security_result,
            model_result=ModelResult(
                used_vector_search=False,
                used_rdb_evidence=False,
                evidence_count=0,
            ),
        )

    def _build_restricted_answer(self, security_result: SecurityResult) -> str:
        if security_result.status == SecurityStatus.INVALID_REQUEST:
            return "질문 내용을 확인한 뒤 업무 데이터에 대한 질문으로 다시 요청해 주세요."
        return (
            "보안상 답변할 수 없는 요청입니다. "
            "업무 데이터에 대한 질문으로 다시 요청해 주세요."
        )
