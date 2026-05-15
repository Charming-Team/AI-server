from app.features.chat.document_payload import InternalDocumentInput
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode, ChatIntent


class DocumentIndexPolicy:
    allowed_document_types = {"REPORT", "COMPANY_INFO"}
    allowed_roles = {"OPERATOR", "EXECUTIVE", "MANUFACTURING_MANAGER"}
    allowed_intent_tags = {
        intent.value
        for intent in ChatIntent
        if intent != ChatIntent.UNKNOWN
    }

    def validate(self, document: InternalDocumentInput) -> None:
        if document.document_type not in self.allowed_document_types:
            raise ChatServiceError(
                status_code=400,
                code=ChatErrorCode.CHAT_DOCUMENT_001,
                message="문서 유형은 REPORT 또는 COMPANY_INFO만 허용됩니다.",
            )

        if not document.allowed_roles:
            raise ChatServiceError(
                status_code=400,
                code=ChatErrorCode.CHAT_DOCUMENT_002,
                message="문서 접근 가능 역할이 필요합니다.",
            )

        invalid_roles = set(document.allowed_roles) - self.allowed_roles
        if invalid_roles:
            raise ChatServiceError(
                status_code=400,
                code=ChatErrorCode.CHAT_DOCUMENT_002,
                message="문서 접근 가능 역할이 올바르지 않습니다.",
            )

        invalid_intent_tags = set(document.intent_tags) - self.allowed_intent_tags
        if invalid_intent_tags:
            raise ChatServiceError(
                status_code=400,
                code=ChatErrorCode.CHAT_DOCUMENT_003,
                message="문서 의도 태그가 올바르지 않습니다.",
            )
