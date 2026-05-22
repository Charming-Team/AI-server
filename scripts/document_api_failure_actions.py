from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode

DOCUMENT_API_TOKEN_FAILURE_ACTIONS = (
    "DOCUMENT_INDEX_INTERNAL_TOKEN 값을 FastAPI와 호출 환경에 동일하게 설정하세요.",
    "호출 시 X-Internal-Token 헤더가 FastAPI 설정값과 같은지 확인하세요.",
)
DOCUMENT_API_NETWORK_FAILURE_ACTIONS = (
    "FastAPI base URL과 문서 API path가 실행 중인 서버를 가리키는지 확인하세요.",
    "로컬 점검이면 FastAPI 서버 실행 상태와 --base-url 값을 확인하세요.",
)
DOCUMENT_API_PAYLOAD_FAILURE_ACTIONS = (
    "문서 payload의 documentId, documentType, allowedRoles, intentTags, "
    "requestedByRole 값을 확인하세요.",
)
DOCUMENT_API_RESPONSE_FAILURE_ACTIONS = (
    "문서 API 응답이 documentId, operationType, operation 상태 계약을 따르는지 확인하세요.",
)
DOCUMENT_API_INDEX_RESULT_FAILURE_ACTIONS = (
    "EMBEDDING_ENABLED, EMBEDDING_BASE_URL, QDRANT_SEARCH_ENABLED, QDRANT_URL 설정을 확인하세요.",
    "운영 점검에서는 문서가 실제 Qdrant에 저장되도록 --allow-skipped 없이 실행하세요.",
)


def build_document_api_failure_actions(exc: ChatServiceError) -> list[str]:
    if exc.code == ChatErrorCode.CHAT_SECURITY_003:
        return list(DOCUMENT_API_TOKEN_FAILURE_ACTIONS)
    if exc.code == ChatErrorCode.CHAT_SERVER_001:
        return list(DOCUMENT_API_NETWORK_FAILURE_ACTIONS)
    if exc.code == ChatErrorCode.CHAT_DOCUMENT_002:
        return list(DOCUMENT_API_PAYLOAD_FAILURE_ACTIONS)
    if exc.code == ChatErrorCode.CHAT_DOCUMENT_003:
        return _build_document_contract_failure_actions(exc.message)
    return list(DOCUMENT_API_RESPONSE_FAILURE_ACTIONS)


def _build_document_contract_failure_actions(message: str) -> list[str]:
    if "문서 저장을 생략" in message or "indexedCount" in message:
        return list(DOCUMENT_API_INDEX_RESULT_FAILURE_ACTIONS)
    return list(DOCUMENT_API_RESPONSE_FAILURE_ACTIONS)
