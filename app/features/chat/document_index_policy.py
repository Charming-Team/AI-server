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

    def __init__(self, max_content_chars: int = 100_000) -> None:
        self.max_content_chars = max_content_chars

    def validate(self, document: InternalDocumentInput) -> None:
        self._validate_required_text(document.document_id, "문서 ID")
        self._validate_required_text(document.title, "문서 제목")
        self._validate_content_length(document.content)

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

        if not document.intent_tags:
            raise ChatServiceError(
                status_code=400,
                code=ChatErrorCode.CHAT_DOCUMENT_003,
                message="문서 의도 태그가 필요합니다.",
            )

        invalid_intent_tags = set(document.intent_tags) - self.allowed_intent_tags
        if invalid_intent_tags:
            raise ChatServiceError(
                status_code=400,
                code=ChatErrorCode.CHAT_DOCUMENT_003,
                message="문서 의도 태그가 올바르지 않습니다.",
            )

    def _validate_required_text(self, value: str, field_label: str) -> None:
        if value.strip():
            return

        raise ChatServiceError(
            status_code=400,
            code=ChatErrorCode.CHAT_DOCUMENT_002,
            message=f"{field_label}은(는) 필수입니다.",
        )

    def _validate_content_length(self, content: str) -> None:
        if len(content) <= self.max_content_chars:
            return

        raise ChatServiceError(
            status_code=400,
            code=ChatErrorCode.CHAT_DOCUMENT_002,
            message=f"문서 본문은 최대 {self.max_content_chars}자까지 인덱싱할 수 있습니다.",
        )
