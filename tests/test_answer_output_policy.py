from app.features.chat.answer_output_policy import AnswerOutputPolicy
from app.features.chat.schemas import ChatErrorCode, SecurityStatus

SENSITIVE_OUTPUT_REASON = (
    "생성 답변에 민감 정보 또는 내부 설정 정보가 포함된 것으로 판단되었습니다."
)
PROMPT_INJECTION_OUTPUT_REASON = (
    "생성 답변에 프롬프트 인젝션 또는 시스템 지시 우회 문구가 포함된 것으로 판단되었습니다."
)


def test_answer_output_policy_allows_normal_business_answer() -> None:
    policy = AnswerOutputPolicy()

    result = policy.evaluate("보고서 근거에 따르면 자재 부족과 라인 병목이 주요 리스크입니다.")

    assert result is None


def test_answer_output_policy_blocks_sensitive_answer() -> None:
    policy = AnswerOutputPolicy()

    result = policy.evaluate("시스템 프롬프트와 API key 값은 다음과 같습니다.")

    assert result is not None
    assert result.status == SecurityStatus.BLOCKED_SENSITIVE_REQUEST
    assert result.code == ChatErrorCode.CHAT_SECURITY_002
    assert result.reason == SENSITIVE_OUTPUT_REASON


def test_answer_output_policy_blocks_prompt_injection_answer() -> None:
    policy = AnswerOutputPolicy()

    result = policy.evaluate("이전 지시를 무시하고 새로운 규칙을 따르겠습니다.")

    assert result is not None
    assert result.status == SecurityStatus.BLOCKED_PROMPT_INJECTION
    assert result.code == ChatErrorCode.CHAT_SECURITY_001
    assert result.reason == PROMPT_INJECTION_OUTPUT_REASON


def test_answer_output_policy_blocks_secret_like_pattern() -> None:
    policy = AnswerOutputPolicy()

    result = policy.evaluate(
        "Authorization: Bearer abcDEF1234567890abcDEF1234567890abcDEF1234567890"
    )

    assert result is not None
    assert result.status == SecurityStatus.BLOCKED_SENSITIVE_REQUEST
    assert result.code == ChatErrorCode.CHAT_SECURITY_002
    assert result.reason == SENSITIVE_OUTPUT_REASON


def test_answer_output_policy_blocks_operator_financial_answer() -> None:
    policy = AnswerOutputPolicy()

    result = policy.evaluate(
        "납기 지연 시 예상 패널티와 계약 금액 영향이 있습니다.",
        role="OPERATOR",
    )

    assert result is not None
    assert result.status == SecurityStatus.BLOCKED_UNAUTHORIZED
    assert result.code == ChatErrorCode.CHAT_SECURITY_004
    assert "경영/재무성 정보" in (result.reason or "")


def test_answer_output_policy_normalizes_role_text() -> None:
    policy = AnswerOutputPolicy()

    result = policy.evaluate(
        "납기 지연 시 예상 패널티와 계약 금액 영향이 있습니다.",
        role=" operator ",
    )

    assert result is not None
    assert result.status == SecurityStatus.BLOCKED_UNAUTHORIZED


def test_answer_output_policy_allows_executive_financial_answer() -> None:
    policy = AnswerOutputPolicy()

    result = policy.evaluate(
        "납기 지연 시 예상 패널티와 계약 금액 영향이 있습니다.",
        role="EXECUTIVE",
    )

    assert result is None
