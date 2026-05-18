from app.features.chat.access_control import (
    BUSINESS_ROLES,
    COMPANY_INFO_INDEXER_ROLES,
    QDRANT_DOCUMENT_TYPES,
)
from app.features.chat.document_access_policy import DocumentAccessPolicy
from app.features.chat.document_payload import InternalDocumentInput
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode, ChatIntent
from app.features.chat.source_url_policy import normalize_internal_url


class DocumentIndexPolicy:
    allowed_document_types = QDRANT_DOCUMENT_TYPES
    allowed_roles = BUSINESS_ROLES
    company_info_indexer_roles = COMPANY_INFO_INDEXER_ROLES
    allowed_intent_tags = {
        intent.value
        for intent in ChatIntent
        if intent != ChatIntent.UNKNOWN
    }

    def __init__(
        self,
        max_content_chars: int = 100_000,
        document_access_policy: DocumentAccessPolicy | None = None,
    ) -> None:
        self.max_content_chars = max_content_chars
        self.document_access_policy = document_access_policy or DocumentAccessPolicy()

    def validate(self, document: InternalDocumentInput) -> None:
        self._validate_required_text(document.document_id, "문서 ID")
        self._validate_required_text(document.title, "문서 제목")
        self._validate_content_length(document.content)
        self._validate_url(document.url)

        if document.document_type not in self.allowed_document_types:
            raise ChatServiceError(
                status_code=400,
                code=ChatErrorCode.CHAT_DOCUMENT_001,
                message="문서 유형은 REPORT 또는 COMPANY_INFO만 허용됩니다.",
            )

        if document.document_type == "COMPANY_INFO":
            self._validate_company_info_indexer_role(document.requested_by_role)

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

        if not self.document_access_policy.allows_document(document):
            raise ChatServiceError(
                status_code=400,
                code=ChatErrorCode.CHAT_DOCUMENT_002,
                message=(
                    "OPERATOR 접근 문서에는 금액, 계약, 패널티 등 "
                    "경영/재무성 내용을 포함할 수 없습니다."
                ),
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

    def _validate_company_info_indexer_role(
        self,
        requested_by_role: str | None,
    ) -> None:
        if requested_by_role in self.company_info_indexer_roles:
            return

        raise ChatServiceError(
            status_code=403,
            code=ChatErrorCode.CHAT_SECURITY_004,
            message=(
                "회사정보 문서 인덱싱은 ADMIN 또는 "
                "MANUFACTURING_MANAGER만 요청할 수 있습니다."
            ),
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

    def _validate_url(self, url: str | None) -> None:
        if not url:
            return

        if normalize_internal_url(url) is not None:
            return

        raise ChatServiceError(
            status_code=400,
            code=ChatErrorCode.CHAT_DOCUMENT_002,
            message="문서 URL은 내부 상대 경로만 허용됩니다.",
        )
