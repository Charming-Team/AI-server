import logging
from typing import Any

from app.features.chat.document_payload import (
    InternalDocumentDeleteRequest,
    InternalDocumentInput,
)
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

    def log_document_index_result(
        self,
        document: InternalDocumentInput,
        result: Any,
    ) -> None:
        self.logger.info(
            "chat_document_index_completed",
            extra={"chat_audit": self.build_document_index_payload(document, result)},
        )

    def log_document_delete_result(
        self,
        request: InternalDocumentDeleteRequest,
        result: Any,
    ) -> None:
        self.logger.info(
            "chat_document_delete_completed",
            extra={"chat_audit": self.build_document_delete_payload(request, result)},
        )

    def log_document_index_failure(
        self,
        document: InternalDocumentInput,
        error: Any,
    ) -> None:
        self.logger.warning(
            "chat_document_index_failed",
            extra={"chat_audit": self.build_document_index_failure_payload(document, error)},
        )

    def log_document_delete_failure(
        self,
        request: InternalDocumentDeleteRequest,
        error: Any,
    ) -> None:
        self.logger.warning(
            "chat_document_delete_failed",
            extra={"chat_audit": self.build_document_delete_failure_payload(request, error)},
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

    def build_document_index_payload(
        self,
        document: InternalDocumentInput,
        result: Any,
    ) -> dict[str, Any]:
        return {
            "event": "chat_document_index_completed",
            "documentId": document.document_id,
            "documentType": document.document_type,
            "requestedByRole": document.requested_by_role,
            "allowedRoles": document.allowed_roles,
            "companyName": document.company_name,
            "intentTags": document.intent_tags,
            "hasUrl": bool(document.url),
            "hasSummary": bool(document.summary),
            "contentLength": len(document.content),
            "operationType": result.operation_type,
            "chunkCount": result.chunk_count,
            "indexedCount": result.indexed_count,
            "skippedReason": result.skipped_reason,
            "operationStatus": result.operation.get("status"),
            "operationId": result.operation.get("operation_id"),
        }

    def build_document_delete_payload(
        self,
        request: InternalDocumentDeleteRequest,
        result: Any,
    ) -> dict[str, Any]:
        return {
            "event": "chat_document_delete_completed",
            "documentId": request.document_id,
            "operationType": result.operation_type,
            "operationStatus": result.operation.get("status"),
            "operationId": result.operation.get("operation_id"),
        }

    def build_document_index_failure_payload(
        self,
        document: InternalDocumentInput,
        error: Any,
    ) -> dict[str, Any]:
        return {
            "event": "chat_document_index_failed",
            "documentId": document.document_id,
            "documentType": document.document_type,
            "requestedByRole": document.requested_by_role,
            "allowedRoles": document.allowed_roles,
            "companyName": document.company_name,
            "intentTags": document.intent_tags,
            "hasUrl": bool(document.url),
            "hasSummary": bool(document.summary),
            "contentLength": len(document.content),
            "statusCode": error.status_code,
            "errorCode": error.code.value,
        }

    def build_document_delete_failure_payload(
        self,
        request: InternalDocumentDeleteRequest,
        error: Any,
    ) -> dict[str, Any]:
        return {
            "event": "chat_document_delete_failed",
            "documentId": request.document_id,
            "statusCode": error.status_code,
            "errorCode": error.code.value,
        }
