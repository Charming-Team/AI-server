import pytest

from app.features.chat.exceptions import ChatServiceError
from app.features.chat.recommendation_service import (
    RecommendationService,
    RecommendedQuestionRule,
)
from app.features.chat.schemas import (
    ChatErrorCode,
    ChatIntent,
    ChatRecommendationRequest,
    ChatUserContext,
)


def _build_request(
    role: str,
    keyword: str | None = None,
    status: str = "ACTIVE",
) -> ChatRecommendationRequest:
    return ChatRecommendationRequest(
        user=ChatUserContext(
            userId=1,
            role=role,
            companyName="S-MAP",
            status=status,
        ),
        keyword=keyword,
    )


def test_recommendation_service_returns_role_based_questions() -> None:
    service = RecommendationService()

    response = service.get_recommendations(_build_request("EXECUTIVE"))

    questions = [item.question for item in response.items]
    assert len(response.items) == 6
    assert "현재 납기 위험이 높은 주문을 알려줘" in questions
    assert "최근 생산 리스크 보고서를 요약해줘" in questions
    assert "납기 지연 시 예상 패널티와 계약 금액 영향을 알려줘" in questions
    assert "이번 주 전체 생산 리스크를 요약해줘" in questions
    assert "라인 가동률과 병목 현황을 요약해줘" in questions
    assert "자재 부족으로 영향받는 생산계획을 알려줘" not in questions
    assert response.fallback_used is False


def test_recommendation_service_rejects_inactive_user() -> None:
    service = RecommendationService()

    with pytest.raises(ChatServiceError) as exc_info:
        service.get_recommendations(_build_request("EXECUTIVE", status="SUSPENDED"))

    assert exc_info.value.status_code == 403
    assert exc_info.value.code == ChatErrorCode.CHAT_SECURITY_004
    assert exc_info.value.message == "ACTIVE 상태 사용자만 추천 질문을 조회할 수 있습니다."


def test_recommendation_service_filters_by_keyword() -> None:
    service = RecommendationService()

    response = service.get_recommendations(_build_request("MANUFACTURING_MANAGER", "자재"))

    assert len(response.items) == 1
    assert response.items[0].question_id == "material-shortage-impact"
    assert response.items[0].intent == "MATERIAL_SHORTAGE"
    assert response.items[0].url == "/materials/inventories"
    assert response.fallback_used is False


def test_recommendation_service_falls_back_when_keyword_has_no_match() -> None:
    service = RecommendationService()

    response = service.get_recommendations(_build_request("OPERATOR", "보고서"))

    assert response.items
    assert response.fallback_used is True
    assert all(item.intent != "REPORT_LOOKUP" for item in response.items)


def test_recommendation_service_excludes_admin_from_business_questions() -> None:
    service = RecommendationService()

    with pytest.raises(ChatServiceError) as exc_info:
        service.get_recommendations(_build_request("ADMIN"))

    assert exc_info.value.status_code == 403
    assert exc_info.value.code == ChatErrorCode.CHAT_SECURITY_004
    assert exc_info.value.message == "현재 역할 권한으로는 추천 질문을 조회할 수 없습니다."


def test_recommendation_service_normalizes_role_text() -> None:
    service = RecommendationService()

    response = service.get_recommendations(_build_request(" operator "))

    assert len(response.items) == 6
    assert all("mode=read" in item.url for item in response.items)


def test_recommendation_service_operator_gets_read_only_urls_without_money_questions() -> None:
    service = RecommendationService()

    response = service.get_recommendations(_build_request("OPERATOR"))

    questions = [item.question for item in response.items]
    assert len(response.items) == 6
    assert "현재 자재 재고 현황을 조회해줘" in questions
    assert "오늘 배정된 생산계획을 조회해줘" in questions
    assert "내 담당 설비의 현재 상태를 조회해줘" in questions
    assert "오늘 처리 수량과 불량 수량을 조회해줘" in questions
    assert "납기 지연 시 예상 패널티와 계약 금액 영향을 알려줘" not in questions
    assert all("mode=read" in item.url for item in response.items)


def test_recommendation_service_filters_operator_restricted_rules_defensively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe_rule = RecommendedQuestionRule(
        question_id="safe-operator-plan",
        question="오늘 배정된 생산계획을 조회해줘",
        intent=ChatIntent.PRODUCTION_PLAN,
        category="생산 계획",
        url="/production-plans?mode=read",
        allowed_roles=("OPERATOR",),
    )
    unsafe_rule = RecommendedQuestionRule(
        question_id="unsafe-operator-money",
        question="계약 금액과 패널티 영향을 조회해줘",
        intent=ChatIntent.DELIVERY_RISK,
        category="금액 영향",
        url="/orders/financial-impact?mode=read",
        allowed_roles=("OPERATOR",),
    )
    monkeypatch.setattr(
        RecommendationService,
        "_rules",
        (safe_rule, unsafe_rule),
    )
    service = RecommendationService()

    response = service.get_recommendations(_build_request("OPERATOR"))

    assert [item.question_id for item in response.items] == ["safe-operator-plan"]


def test_recommendation_service_filters_operator_non_read_urls_defensively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe_rule = RecommendedQuestionRule(
        question_id="safe-operator-line",
        question="내 담당 라인의 현재 상태를 조회해줘",
        intent=ChatIntent.LINE_BOTTLENECK,
        category="라인 현황",
        url="/production-lines/status?mode=read",
        allowed_roles=("OPERATOR",),
    )
    unsafe_rule = RecommendedQuestionRule(
        question_id="unsafe-operator-edit-url",
        question="오늘 배정된 생산계획을 조회해줘",
        intent=ChatIntent.PRODUCTION_PLAN,
        category="생산 계획",
        url="/production-plans",
        allowed_roles=("OPERATOR",),
    )
    monkeypatch.setattr(
        RecommendationService,
        "_rules",
        (safe_rule, unsafe_rule),
    )
    service = RecommendationService()

    response = service.get_recommendations(_build_request("OPERATOR"))

    assert [item.question_id for item in response.items] == ["safe-operator-line"]


def test_recommendation_service_filters_rules_outside_role_intent_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe_rule = RecommendedQuestionRule(
        question_id="safe-operator-material",
        question="현재 자재 재고 현황을 조회해줘",
        intent=ChatIntent.MATERIAL_SHORTAGE,
        category="자재 현황",
        url="/materials/inventories?mode=read",
        allowed_roles=("OPERATOR",),
    )
    unsafe_rule = RecommendedQuestionRule(
        question_id="unsafe-operator-report",
        question="최근 생산 리스크 보고서를 요약해줘",
        intent=ChatIntent.REPORT_LOOKUP,
        category="보고서 조회",
        url="/reports?mode=read",
        allowed_roles=("OPERATOR",),
    )
    monkeypatch.setattr(
        RecommendationService,
        "_rules",
        (safe_rule, unsafe_rule),
    )
    service = RecommendationService()

    response = service.get_recommendations(_build_request("OPERATOR"))

    assert [item.question_id for item in response.items] == ["safe-operator-material"]


def test_recommendation_service_manager_gets_six_manufacturing_questions() -> None:
    service = RecommendationService()

    response = service.get_recommendations(_build_request("MANUFACTURING_MANAGER"))

    questions = [item.question for item in response.items]
    assert len(response.items) == 6
    assert "자재 부족으로 영향받는 생산계획을 알려줘" in questions
    assert "오늘 변경이 필요한 생산계획을 알려줘" in questions
    assert "현재 병목이 발생한 라인과 원인을 알려줘" in questions
    assert "긴급 주문이 전체 생산계획에 미치는 영향을 알려줘" in questions
    assert "오늘 먼저 처리해야 할 작업 우선순위를 알려줘" in questions
    assert "최근 제조 리스크 보고서를 요약해줘" in questions
