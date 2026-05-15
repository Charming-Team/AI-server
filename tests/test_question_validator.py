import pytest

from app.features.chat.question_validator import QuestionValidator
from app.features.chat.schemas import SecurityStatus


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_question_validator_rejects_blank_question(question: str) -> None:
    validator = QuestionValidator()

    result = validator.validate(question)

    assert result is not None
    assert result.status == SecurityStatus.INVALID_REQUEST
    assert result.reason == "질문 내용이 비어 있습니다."


def test_question_validator_rejects_control_character() -> None:
    validator = QuestionValidator()

    result = validator.validate("납기 위험 알려줘\x00")

    assert result is not None
    assert result.status == SecurityStatus.INVALID_REQUEST
    assert result.reason == "질문에 허용되지 않는 제어 문자가 포함되어 있습니다."


def test_question_validator_allows_business_question() -> None:
    validator = QuestionValidator()

    result = validator.validate("현재 납기 위험이 높은 주문 알려줘")

    assert result is None
