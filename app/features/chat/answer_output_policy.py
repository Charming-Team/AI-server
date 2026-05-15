from dataclasses import dataclass

from app.features.chat.role_access_policy import RoleAccessPolicy
from app.features.chat.schemas import ChatErrorCode, SecurityResult, SecurityStatus


@dataclass(frozen=True)
class AnswerOutputRule:
    code: ChatErrorCode
    reason: str
    terms: tuple[str, ...]


class AnswerOutputPolicy:
    _sensitive_rule = AnswerOutputRule(
        code=ChatErrorCode.CHAT_SECURITY_002,
        reason="생성 답변에 민감 정보 또는 내부 설정 정보가 포함된 것으로 판단되었습니다.",
        terms=(
            "system prompt",
            "developer prompt",
            "api key",
            "secret",
            "token",
            "config",
            "model name",
            "llm_model",
            "environment variable",
            "시스템 프롬프트",
            "개발자 프롬프트",
            "내부 프롬프트",
            "api key",
            "시크릿",
            "토큰",
            "설정값",
            "환경변수",
            "모델 정보",
            "모델명",
        ),
    )
    _operator_financial_rule = AnswerOutputRule(
        code=ChatErrorCode.CHAT_SECURITY_004,
        reason=(
            "OPERATOR 역할 답변에 금액, 계약, 패널티 등 "
            "경영/재무성 정보가 포함된 것으로 판단되었습니다."
        ),
        terms=RoleAccessPolicy.operator_restricted_terms,
    )

    def evaluate(self, answer: str, *, role: str | None = None) -> SecurityResult | None:
        normalized_answer = self._normalize(answer)
        compact_answer = self._compact(normalized_answer)

        if any(
            self._contains_term(term, normalized_answer, compact_answer)
            for term in self._sensitive_rule.terms
        ):
            return SecurityResult(
                status=SecurityStatus.BLOCKED_SENSITIVE_REQUEST,
                code=self._sensitive_rule.code,
                reason=self._sensitive_rule.reason,
            )
        if (
            role is not None
            and role.upper() == "OPERATOR"
            and self._matches_rule(
                self._operator_financial_rule,
                normalized_answer,
                compact_answer,
            )
        ):
            return SecurityResult(
                status=SecurityStatus.BLOCKED_UNAUTHORIZED,
                code=self._operator_financial_rule.code,
                reason=self._operator_financial_rule.reason,
            )
        return None

    def _matches_rule(
        self,
        rule: AnswerOutputRule,
        normalized_answer: str,
        compact_answer: str,
    ) -> bool:
        return any(
            self._contains_term(term, normalized_answer, compact_answer)
            for term in rule.terms
        )

    def _contains_term(
        self,
        term: str,
        normalized_answer: str,
        compact_answer: str,
    ) -> bool:
        normalized_term = self._normalize(term)
        compact_term = self._compact(normalized_term)
        return normalized_term in normalized_answer or compact_term in compact_answer

    def _normalize(self, text: str) -> str:
        return text.casefold()

    def _compact(self, text: str) -> str:
        return "".join(text.split())
