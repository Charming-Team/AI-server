from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode

ANSWER_API_TOKEN_FAILURE_ACTIONS = (
    "CHAT_ANSWER_INTERNAL_TOKEN 값을 FastAPI와 호출 환경에 동일하게 설정하세요.",
    "호출 시 X-Internal-Token 헤더가 FastAPI 설정값과 같은지 확인하세요.",
)
ANSWER_API_NETWORK_FAILURE_ACTIONS = (
    "FastAPI base URL과 챗봇 답변 API path가 실행 중인 서버를 가리키는지 확인하세요.",
)
ANSWER_API_EVIDENCE_FAILURE_ACTIONS = (
    "RDB Evidence, Qdrant 문서 출처, 최소 Evidence 개수 조건을 확인하세요.",
)
ANSWER_API_INTENT_FAILURE_ACTIONS = (
    "질문 문구와 expected intent 조건이 일치하는지 확인하세요.",
    "IntentClassifier 키워드 규칙 또는 RAG 시나리오 질문을 확인하세요.",
)
ANSWER_API_RDB_EVIDENCE_FAILURE_ACTIONS = (
    "RDB_EVIDENCE_ENABLED, RDB_EVIDENCE_DSN, chat_evidence view 조회 결과를 확인하세요.",
)
ANSWER_API_VECTOR_EVIDENCE_FAILURE_ACTIONS = (
    "QDRANT_SEARCH_ENABLED, QDRANT_URL, Qdrant 문서 출처와 Vector Search 조건을 확인하세요.",
)
ANSWER_API_SECURITY_FAILURE_ACTIONS = (
    "Role 기반 접근 제어와 expected securityStatus/securityCode 조건을 확인하세요.",
)
ANSWER_API_LLM_FAILURE_ACTIONS = (
    "LLM_ENABLED, LLM_BASE_URL, LLM_MODEL, expected LLM skipped reason 조건을 확인하세요.",
)

RECOMMENDATION_API_TOKEN_FAILURE_ACTIONS = (
    "CHAT_RECOMMENDATION_INTERNAL_TOKEN 값을 FastAPI와 호출 환경에 동일하게 설정하세요.",
    "호출 시 X-Internal-Token 헤더가 FastAPI 설정값과 같은지 확인하세요.",
)
RECOMMENDATION_API_NETWORK_FAILURE_ACTIONS = (
    "FastAPI base URL과 추천 질문 API path가 실행 중인 서버를 가리키는지 확인하세요.",
)
RECOMMENDATION_API_ITEM_FAILURE_ACTIONS = (
    "Role/keyword 추천 질문 seed, fallback 사용 여부, min item count 조건을 확인하세요.",
)
RECOMMENDATION_API_ROLE_FAILURE_ACTIONS = (
    "ROLE_INTENT_MATRIX, OPERATOR read-only URL, 금액성 추천 질문 차단 규칙을 확인하세요.",
)


def build_answer_api_failure_actions(exc: ChatServiceError) -> list[str]:
    if exc.code == ChatErrorCode.CHAT_SECURITY_003:
        return list(ANSWER_API_TOKEN_FAILURE_ACTIONS)
    if exc.code == ChatErrorCode.CHAT_SERVER_001:
        return list(ANSWER_API_NETWORK_FAILURE_ACTIONS)
    if exc.code == ChatErrorCode.CHAT_EVIDENCE_001:
        return _build_answer_evidence_failure_actions(exc.message)
    if exc.code == ChatErrorCode.CHAT_SECURITY_001:
        return list(ANSWER_API_SECURITY_FAILURE_ACTIONS)
    if exc.code == ChatErrorCode.CHAT_LLM_004:
        return list(ANSWER_API_LLM_FAILURE_ACTIONS)
    return list(ANSWER_API_EVIDENCE_FAILURE_ACTIONS)


def build_recommendation_api_failure_actions(exc: ChatServiceError) -> list[str]:
    if exc.code == ChatErrorCode.CHAT_SECURITY_003:
        return list(RECOMMENDATION_API_TOKEN_FAILURE_ACTIONS)
    if exc.code == ChatErrorCode.CHAT_SERVER_001:
        return list(RECOMMENDATION_API_NETWORK_FAILURE_ACTIONS)
    if exc.code == ChatErrorCode.CHAT_RECOMMEND_002:
        return list(RECOMMENDATION_API_ROLE_FAILURE_ACTIONS)
    return list(RECOMMENDATION_API_ITEM_FAILURE_ACTIONS)


def _build_answer_evidence_failure_actions(message: str) -> list[str]:
    if "intent" in message:
        return list(ANSWER_API_INTENT_FAILURE_ACTIONS)
    if "RDB Evidence" in message:
        return list(ANSWER_API_RDB_EVIDENCE_FAILURE_ACTIONS)
    if "Qdrant" in message or "Vector Search" in message:
        return list(ANSWER_API_VECTOR_EVIDENCE_FAILURE_ACTIONS)
    return list(ANSWER_API_EVIDENCE_FAILURE_ACTIONS)
