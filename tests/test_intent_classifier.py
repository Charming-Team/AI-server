import pytest

from app.features.chat.intent_classifier import IntentClassifier
from app.features.chat.schemas import ChatIntent


@pytest.mark.parametrize(
    ("question", "expected_intent"),
    [
        ("납기 지연 가능성이 높은 주문 알려줘", ChatIntent.DELIVERY_RISK),
        ("자재 재고 부족한 항목 알려줘", ChatIntent.MATERIAL_SHORTAGE),
        ("라인 병목이 발생한 공정 알려줘", ChatIntent.LINE_BOTTLENECK),
        ("긴급 주문이 현재 생산계획에 미치는 영향 알려줘", ChatIntent.URGENT_ORDER_IMPACT),
        ("오늘 먼저 처리해야 할 작업 우선순위 알려줘", ChatIntent.WORK_PRIORITY),
        ("이번 달 월간 리포트 요약해줘", ChatIntent.REPORT_LOOKUP),
        ("다음 주 생산계획 변경 일정 보여줘", ChatIntent.PRODUCTION_PLAN),
        ("점심 메뉴 추천해줘", ChatIntent.UNKNOWN),
    ],
)
def test_intent_classifier_classifies_business_questions(
    question: str,
    expected_intent: ChatIntent,
) -> None:
    classifier = IntentClassifier()

    assert classifier.classify(question) == expected_intent


def test_intent_classifier_handles_spacing_variants() -> None:
    classifier = IntentClassifier()

    assert classifier.classify("생산 계획 변경 일정 알려줘") == ChatIntent.PRODUCTION_PLAN
