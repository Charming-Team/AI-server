import logging
from typing import Any

from app.features.chat.schemas import ChatAnswerRequest, ChatAnswerResponse


class ChatAuditLogger:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("app.features.chat.audit")

    def log_answer_response(
        self,
        request: ChatAnswerRequest,
        response: ChatAnswerResponse,
    ) -> None:
        self.logger.info(
            "chat_answer_completed",
            extra={"chat_audit": self.build_answer_payload(request, response)},
        )

    def build_answer_payload(
        self,
        request: ChatAnswerRequest,
        response: ChatAnswerResponse,
    ) -> dict[str, Any]:
        security_code = response.security_result.code
        return {
            "event": "chat_answer_completed",
            "sessionId": request.session_id,
            "messageId": request.message_id,
            "userId": request.user.user_id,
            "role": request.user.role,
            "companyName": request.user.company_name,
            "intent": response.intent.value,
            "securityStatus": response.security_result.status.value,
            "securityCode": security_code.value if security_code else None,
            "usedVectorSearch": response.model_result.used_vector_search,
            "usedRdbEvidence": response.model_result.used_rdb_evidence,
            "usedLlmGeneration": response.model_result.used_llm_generation,
            "vectorSearchSkippedReason": (
                response.model_result.vector_search_skipped_reason
            ),
            "llmGenerationSkippedReason": (
                response.model_result.llm_generation_skipped_reason
            ),
            "rdbEvidenceCount": response.model_result.rdb_evidence_count,
            "documentSourceCount": response.model_result.document_source_count,
            "evidenceCount": response.model_result.evidence_count,
            "urlCount": len(response.urls),
            "sourceCount": len(response.sources),
        }
