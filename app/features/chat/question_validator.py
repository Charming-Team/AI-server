from app.features.chat.schemas import ChatErrorCode, SecurityResult, SecurityStatus


class QuestionValidator:
    _max_question_length = 1000
    _invalid_control_characters = tuple(
        chr(value) for value in range(32) if value not in (9, 10, 13)
    )

    def validate(self, question: str) -> SecurityResult | None:
        normalized_question = question.strip()
        if not normalized_question:
            return self._invalid_result(
                code=ChatErrorCode.CHAT_INPUT_001,
                reason="질문 내용이 비어 있습니다.",
            )

        if len(normalized_question) > self._max_question_length:
            return self._invalid_result(
                code=ChatErrorCode.CHAT_INPUT_002,
                reason="질문 길이가 허용 범위를 초과했습니다.",
            )

        if self._contains_invalid_control_character(normalized_question):
            return self._invalid_result(
                code=ChatErrorCode.CHAT_INPUT_003,
                reason="질문에 허용되지 않는 제어 문자가 포함되어 있습니다.",
            )

        return None

    def _contains_invalid_control_character(self, question: str) -> bool:
        return any(character in question for character in self._invalid_control_characters)

    def _invalid_result(
        self,
        code: ChatErrorCode,
        reason: str,
    ) -> SecurityResult:
        return SecurityResult(
            status=SecurityStatus.INVALID_REQUEST,
            code=code,
            reason=reason,
        )
