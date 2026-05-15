import pytest

from app.features.chat.role_access_policy import RoleAccessPolicy
from app.features.chat.schemas import ChatErrorCode, ChatIntent, SecurityStatus


@pytest.mark.parametrize(
    ("role", "question", "intent"),
    [
        (
            "OPERATOR",
            "현재 자재 재고 현황을 조회해줘",
            ChatIntent.MATERIAL_SHORTAGE,
        ),
        (
            "OPERATOR",
            "현재 납기 위험이 높은 주문 알려줘",
            ChatIntent.DELIVERY_RISK,
        ),
        (
            "MANUFACTURING_MANAGER",
            "자재 부족으로 영향받는 생산계획을 알려줘",
            ChatIntent.MATERIAL_SHORTAGE,
        ),
        (
            "EXECUTIVE",
            "납기 지연 시 예상 패널티와 계약 금액 영향을 알려줘",
            ChatIntent.DELIVERY_RISK,
        ),
        (
            "EXECUTIVE",
            "자재 재고 부족한 항목 알려줘",
            ChatIntent.MATERIAL_SHORTAGE,
        ),
        (
            "EXECUTIVE",
            "다음 주 생산계획 변경 일정 보여줘",
            ChatIntent.PRODUCTION_PLAN,
        ),
    ],
)
def test_role_access_policy_allows_business_roles(
    role: str,
    question: str,
    intent: ChatIntent,
) -> None:
    policy = RoleAccessPolicy()

    result = policy.evaluate(role, question, intent)

    assert result is None


@pytest.mark.parametrize("role", ["ADMIN", "UNKNOWN"])
def test_role_access_policy_blocks_non_business_roles(role: str) -> None:
    policy = RoleAccessPolicy()

    result = policy.evaluate(
        role,
        "현재 납기 위험이 높은 주문 알려줘",
        ChatIntent.DELIVERY_RISK,
    )

    assert result is not None
    assert result.status == SecurityStatus.BLOCKED_UNAUTHORIZED
    assert result.code == ChatErrorCode.CHAT_SECURITY_004
    assert "OPERATOR" in (result.reason or "")


def test_role_access_policy_normalizes_role_text() -> None:
    policy = RoleAccessPolicy()

    result = policy.evaluate(
        " operator ",
        "현재 자재 재고 현황을 조회해줘",
        ChatIntent.MATERIAL_SHORTAGE,
    )

    assert result is None


def test_role_access_policy_blocks_operator_financial_question() -> None:
    policy = RoleAccessPolicy()

    result = policy.evaluate(
        "OPERATOR",
        "납기 지연 시 예상 패널티와 계약 금액 영향을 알려줘",
        ChatIntent.DELIVERY_RISK,
    )

    assert result is not None
    assert result.status == SecurityStatus.BLOCKED_UNAUTHORIZED
    assert result.code == ChatErrorCode.CHAT_SECURITY_004
    assert "경영/재무성 정보" in (result.reason or "")


@pytest.mark.parametrize(
    ("role", "question", "intent"),
    [
        (
            "OPERATOR",
            "이번 달 월간 리포트 요약해줘",
            ChatIntent.REPORT_LOOKUP,
        ),
    ],
)
def test_role_access_policy_blocks_intent_outside_role_matrix(
    role: str,
    question: str,
    intent: ChatIntent,
) -> None:
    policy = RoleAccessPolicy()

    result = policy.evaluate(role, question, intent)

    assert result is not None
    assert result.status == SecurityStatus.BLOCKED_UNAUTHORIZED
    assert result.code == ChatErrorCode.CHAT_SECURITY_004
    assert "질문 의도에 접근할 수 없습니다" in (result.reason or "")


def test_role_access_policy_allows_unknown_intent_for_no_guess_response() -> None:
    policy = RoleAccessPolicy()

    result = policy.evaluate(
        "OPERATOR",
        "점심 메뉴 추천해줘",
        ChatIntent.UNKNOWN,
    )

    assert result is None
