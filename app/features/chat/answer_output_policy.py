from dataclasses import dataclass

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

    def evaluate(self, answer: str) -> SecurityResult | None:
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
        return None

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
