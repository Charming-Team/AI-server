from app.features.chat.recommendation_service import RecommendationService
from app.features.chat.schemas import (
    ChatRecommendationRequest,
    ChatUserContext,
)


def _build_request(role: str, keyword: str | None = None) -> ChatRecommendationRequest:
    return ChatRecommendationRequest(
        user=ChatUserContext(
            userId=1,
            role=role,
            companyName="S-MAP",
            status="ACTIVE",
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

    response = service.get_recommendations(_build_request("ADMIN"))

    assert response.items == []
    assert response.fallback_used is False


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
