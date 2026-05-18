from app.features.chat.access_control import (
    BUSINESS_ROLES,
    OPERATOR_RESTRICTED_TERMS,
    OPERATOR_ROLE,
    ROLE_INTENT_MATRIX,
)
from app.features.chat.schemas import (
    ChatErrorCode,
    ChatIntent,
    SecurityResult,
    SecurityStatus,
)


class RoleAccessPolicy:
    allowed_business_roles = BUSINESS_ROLES
    role_intent_matrix = ROLE_INTENT_MATRIX
    operator_restricted_terms = OPERATOR_RESTRICTED_TERMS

    def evaluate(
        self,
        role: str,
        question: str,
        intent: ChatIntent,
    ) -> SecurityResult | None:
        normalized_role = role.strip().upper()
        if normalized_role not in self.allowed_business_roles:
            return SecurityResult(
                status=SecurityStatus.BLOCKED_UNAUTHORIZED,
                code=ChatErrorCode.CHAT_SECURITY_004,
                reason=(
                    "업무 챗봇은 OPERATOR, EXECUTIVE, "
                    "MANUFACTURING_MANAGER 역할만 사용할 수 있습니다."
                ),
            )

        if not self._is_intent_allowed(normalized_role, intent):
            return SecurityResult(
                status=SecurityStatus.BLOCKED_UNAUTHORIZED,
                code=ChatErrorCode.CHAT_SECURITY_004,
                reason=(
                    f"{normalized_role} 역할은 {intent.value} 질문 의도에 "
                    "접근할 수 없습니다."
                ),
            )

        if normalized_role == OPERATOR_ROLE and self._contains_restricted_term(question):
            return SecurityResult(
                status=SecurityStatus.BLOCKED_UNAUTHORIZED,
                code=ChatErrorCode.CHAT_SECURITY_004,
                reason=(
                    "OPERATOR 역할은 금액, 계약, 패널티 등 "
                    "경영/재무성 정보에 접근할 수 없습니다."
                ),
            )

        return None

    def _is_intent_allowed(self, role: str, intent: ChatIntent) -> bool:
        if intent == ChatIntent.UNKNOWN:
            return True
        return intent in self.role_intent_matrix.get(role, set())

    def _contains_restricted_term(self, question: str) -> bool:
        normalized_question = self._normalize(question)
        compact_question = self._compact(normalized_question)
        return any(
            self._contains_term(term, normalized_question, compact_question)
            for term in self.operator_restricted_terms
        )

    def _contains_term(
        self,
        term: str,
        normalized_question: str,
        compact_question: str,
    ) -> bool:
        normalized_term = self._normalize(term)
        compact_term = self._compact(normalized_term)
        return normalized_term in normalized_question or compact_term in compact_question

    def _normalize(self, value: str) -> str:
        return value.casefold()

    def _compact(self, value: str) -> str:
        return "".join(value.split())
