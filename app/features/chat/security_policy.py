from dataclasses import dataclass

from app.features.chat.schemas import ChatErrorCode, SecurityResult, SecurityStatus
from app.features.chat.sensitive_pattern_policy import SensitivePatternPolicy


@dataclass(frozen=True)
class SecurityRule:
    status: SecurityStatus
    code: ChatErrorCode
    reason: str
    terms: tuple[str, ...]


class SecurityPolicy:
    _rules: tuple[SecurityRule, ...] = (
        SecurityRule(
            status=SecurityStatus.BLOCKED_PROMPT_INJECTION,
            code=ChatErrorCode.CHAT_SECURITY_001,
            reason="프롬프트 인젝션 또는 시스템 지시 우회 요청으로 판단되었습니다.",
            terms=(
                "ignore previous",
                "ignore instructions",
                "disregard previous",
                "jailbreak",
                "developer mode",
                "act as",
                "이전 지시",
                "지시 무시",
                "규칙 무시",
                "보안 규칙 무시",
                "프롬프트 무시",
                "개발자 모드",
                "탈옥",
                "너는 이제",
            ),
        ),
        SecurityRule(
            status=SecurityStatus.BLOCKED_SENSITIVE_REQUEST,
            code=ChatErrorCode.CHAT_SECURITY_002,
            reason="민감 정보 또는 내부 설정 정보 요청으로 판단되었습니다.",
            terms=(
                "system prompt",
                "developer prompt",
                "api key",
                "password",
                "secret",
                "token",
                "config",
                "model name",
                "environment variable",
                "env",
                "시스템 프롬프트",
                "개발자 프롬프트",
                "내부 프롬프트",
                "api key",
                "비밀번호",
                "시크릿",
                "토큰",
                "설정값",
                "환경변수",
                "모델 정보",
                "모델명",
            ),
        ),
    )

    _sensitive_pattern_reason = (
        "민감 정보 또는 내부 설정 정보로 보이는 패턴이 포함된 것으로 판단되었습니다."
    )

    def __init__(
        self,
        sensitive_pattern_policy: SensitivePatternPolicy | None = None,
    ) -> None:
        self.sensitive_pattern_policy = (
            sensitive_pattern_policy or SensitivePatternPolicy()
        )

    def evaluate(self, question: str) -> SecurityResult | None:
        normalized_question = self._normalize(question)
        compact_question = self._compact(normalized_question)

        for rule in self._rules:
            if self._matches_rule(rule, normalized_question, compact_question):
                return SecurityResult(
                    status=rule.status,
                    code=rule.code,
                    reason=rule.reason,
                )
        if self.sensitive_pattern_policy.contains_sensitive_pattern(question):
            return SecurityResult(
                status=SecurityStatus.BLOCKED_SENSITIVE_REQUEST,
                code=ChatErrorCode.CHAT_SECURITY_002,
                reason=self._sensitive_pattern_reason,
            )
        return None

    def _matches_rule(
        self,
        rule: SecurityRule,
        normalized_question: str,
        compact_question: str,
    ) -> bool:
        return any(
            self._contains_term(term, normalized_question, compact_question)
            for term in rule.terms
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

    def _normalize(self, text: str) -> str:
        return text.casefold()

    def _compact(self, text: str) -> str:
        return "".join(text.split())
