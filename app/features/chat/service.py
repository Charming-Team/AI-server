from app.core.config import Settings
from app.features.chat.answer_generation_service import AnswerGenerationService
from app.features.chat.audit_logger import ChatAuditLogger
from app.features.chat.document_access_policy import DocumentAccessPolicy
from app.features.chat.document_search_service import DocumentSearchService
from app.features.chat.evidence_access_policy import EvidenceAccessPolicy
from app.features.chat.evidence_service import EvidenceService
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.intent_classifier import IntentClassifier
from app.features.chat.question_validator import QuestionValidator
from app.features.chat.response_builder import ChatResponseBuilder
from app.features.chat.role_access_policy import RoleAccessPolicy
from app.features.chat.schemas import (
    AnswerGenerationResult,
    ChatAnswerRequest,
    ChatAnswerResponse,
    ChatErrorCode,
    ChatIntent,
    DocumentSearchResult,
    EvidenceResult,
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
        self.role_access_policy = RoleAccessPolicy()
        self.evidence_access_policy = EvidenceAccessPolicy()
        self.document_access_policy = DocumentAccessPolicy()
        self.evidence_service = EvidenceService(settings)
        self.document_search_service = DocumentSearchService(settings)
        self.answer_generation_service = AnswerGenerationService(settings)
        self.audit_logger = ChatAuditLogger()

    async def create_answer(self, request: ChatAnswerRequest) -> ChatAnswerResponse:
        user_status_result = self._validate_user_status(request.user.status)
        if user_status_result is not None:
            return self._finalize_response(
                request,
                self._build_restricted_response(request, user_status_result),
            )

        validation_result = self.question_validator.validate(request.question)
        if validation_result is not None:
            return self._finalize_response(
                request,
                self._build_restricted_response(request, validation_result),
            )

        security_result = self.security_policy.evaluate(request.question)
        if security_result is not None:
            return self._finalize_response(
                request,
                self._build_restricted_response(request, security_result),
            )

        intent = self.intent_classifier.classify(request.question)
        role_access_result = self.role_access_policy.evaluate(
            request.user.role,
            request.question,
            intent,
        )
        if role_access_result is not None:
            return self._finalize_response(
                request,
                self._build_restricted_response(
                    request,
                    role_access_result,
                    intent=intent,
                ),
            )

        evidence_result = await self.evidence_service.get_evidence(request, intent)
        evidence_result = self.evidence_access_policy.sanitize(
            request.user,
            evidence_result,
        )
        document_result = await self._search_documents_safely(
            request,
            evidence_result.intent,
        )
        document_result = self.document_access_policy.sanitize_search_result(
            document_result,
            request.user.role,
        )
        answer_result = await self.answer_generation_service.generate_answer(
            request,
            evidence_result,
            document_result,
        )
        sources = self.response_builder.build_sources(evidence_result, document_result)
        basis_time = self.response_builder.build_basis_time(
            sources,
            fallback=evidence_result.basis_time,
        )

        return self._finalize_response(
            request,
            ChatAnswerResponse(
                session_id=request.session_id,
                message_id=request.message_id,
                intent=evidence_result.intent,
                answer=answer_result.answer,
                basis_time=basis_time,
                urls=self.response_builder.build_urls(sources),
                sources=sources,
                security_result=(
                    answer_result.security_result
                    or self.response_builder.build_security_result(
                        evidence_result,
                        document_result,
                    )
                ),
                model_result=self._build_model_result(
                    evidence_result,
                    document_result,
                    answer_result,
                ),
            ),
        )

    def _finalize_response(
        self,
        request: ChatAnswerRequest,
        response: ChatAnswerResponse,
    ) -> ChatAnswerResponse:
        self.audit_logger.log_answer_response(request, response)
        return response

    def _validate_user_status(self, status: str) -> SecurityResult | None:
        if status.strip().upper() == "ACTIVE":
            return None

        return SecurityResult(
            status=SecurityStatus.BLOCKED_UNAUTHORIZED,
            code=ChatErrorCode.CHAT_SECURITY_004,
            reason="ACTIVE 상태 사용자만 챗봇을 사용할 수 있습니다.",
        )

    def _build_restricted_response(
        self,
        request: ChatAnswerRequest,
        security_result: SecurityResult,
        intent: ChatIntent = ChatIntent.UNKNOWN,
    ) -> ChatAnswerResponse:
        return ChatAnswerResponse(
            session_id=request.session_id,
            message_id=request.message_id,
            intent=intent,
            answer=self._build_restricted_answer(security_result),
            basis_time=request.requested_at,
            security_result=security_result,
            model_result=ModelResult(
                used_vector_search=False,
                used_rdb_evidence=False,
                used_llm_generation=False,
                llm_cache_hit=False,
                rdb_evidence_count=0,
                document_source_count=0,
                evidence_count=0,
            ),
        )

    def _build_model_result(
        self,
        evidence_result: EvidenceResult,
        document_result: DocumentSearchResult,
        answer_result: AnswerGenerationResult,
    ) -> ModelResult:
        rdb_evidence_count = len(evidence_result.items)
        document_source_count = len(document_result.sources)
        return ModelResult(
            used_vector_search=document_result.was_searched,
            used_rdb_evidence=evidence_result.has_evidence,
            used_llm_generation=answer_result.was_generated,
            llm_cache_hit=answer_result.llm_cache_hit,
            rdb_evidence_count=rdb_evidence_count,
            document_source_count=document_source_count,
            evidence_count=rdb_evidence_count + document_source_count,
            vector_search_skipped_reason=document_result.skipped_reason,
            llm_generation_skipped_reason=answer_result.skipped_reason,
        )

    async def _search_documents_safely(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> DocumentSearchResult:
        try:
            return await self.document_search_service.search(request, intent)
        except ChatExternalServiceError as exc:
            return DocumentSearchResult(
                was_searched=self._did_vector_search_start(exc),
                skipped_reason=exc.message,
            )

    def _did_vector_search_start(self, exc: ChatExternalServiceError) -> bool:
        return not exc.code.value.startswith("CHAT_EMBEDDING_")

    def _build_restricted_answer(self, security_result: SecurityResult) -> str:
        if security_result.status == SecurityStatus.INVALID_REQUEST:
            return "질문 내용을 확인한 뒤 업무 데이터에 대한 질문으로 다시 요청해 주세요."
        if security_result.status == SecurityStatus.BLOCKED_UNAUTHORIZED:
            if security_result.reason and "ACTIVE 상태" in security_result.reason:
                return "현재 계정 상태로는 챗봇을 사용할 수 없습니다."
            return "현재 역할 권한으로는 답변할 수 없는 요청입니다."
        return (
            "보안상 답변할 수 없는 요청입니다. "
            "업무 데이터에 대한 질문으로 다시 요청해 주세요."
        )
