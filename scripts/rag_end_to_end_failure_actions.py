from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode
from scripts import chat_api_failure_actions, document_api_failure_actions

RAG_END_TO_END_TOKEN_FAILURE_ACTIONS = (
    "CHAT_ANSWER_INTERNAL_TOKEN과 DOCUMENT_INDEX_INTERNAL_TOKEN 설정을 모두 확인하세요.",
)
RAG_END_TO_END_FLOW_FAILURE_ACTIONS = (
    "문서 등록, 챗봇 답변, 문서 삭제 API의 base URL, path, 내부 토큰을 함께 확인하세요.",
)


def build_rag_end_to_end_failure_actions(exc: ChatServiceError) -> list[str]:
    if exc.code == ChatErrorCode.CHAT_SECURITY_003:
        return _build_security_failure_actions(exc)
    if exc.code in {ChatErrorCode.CHAT_DOCUMENT_002, ChatErrorCode.CHAT_DOCUMENT_003}:
        return document_api_failure_actions.build_document_api_failure_actions(exc)
    if exc.code in {
        ChatErrorCode.CHAT_EVIDENCE_001,
        ChatErrorCode.CHAT_SECURITY_001,
        ChatErrorCode.CHAT_LLM_004,
    }:
        return chat_api_failure_actions.build_answer_api_failure_actions(exc)
    if exc.code == ChatErrorCode.CHAT_SERVER_001:
        return _build_server_failure_actions(exc)
    return list(RAG_END_TO_END_FLOW_FAILURE_ACTIONS)


def _build_security_failure_actions(exc: ChatServiceError) -> list[str]:
    message = exc.message.casefold()
    if "chat answer" in message or "챗봇 답변" in message:
        return chat_api_failure_actions.build_answer_api_failure_actions(exc)
    if "document index" in message or "문서" in message:
        return document_api_failure_actions.build_document_api_failure_actions(exc)
    return list(RAG_END_TO_END_TOKEN_FAILURE_ACTIONS)


def _build_server_failure_actions(exc: ChatServiceError) -> list[str]:
    message = exc.message.casefold()
    if "챗봇 답변" in message or "chat answer" in message:
        return chat_api_failure_actions.build_answer_api_failure_actions(exc)
    if "문서" in message or "document" in message:
        return document_api_failure_actions.build_document_api_failure_actions(exc)
    return list(RAG_END_TO_END_FLOW_FAILURE_ACTIONS)
