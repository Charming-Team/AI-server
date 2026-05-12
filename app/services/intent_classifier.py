from app.schemas.chat import ChatIntent


class IntentClassifier:
    _intent_keywords: tuple[tuple[ChatIntent, tuple[str, ...]], ...] = (
        (
            ChatIntent.MATERIAL_SHORTAGE,
            ("자재", "재고", "부족", "shortage", "material", "inventory"),
        ),
        (
            ChatIntent.LINE_BOTTLENECK,
            ("라인", "병목", "설비", "machine", "line", "bottleneck", "error"),
        ),
        (
            ChatIntent.DELIVERY_RISK,
            ("납기", "지연", "위험", "delay", "risk", "due date"),
        ),
        (
            ChatIntent.URGENT_ORDER_IMPACT,
            ("긴급", "영향", "대응안", "시뮬레이션", "urgent", "impact", "simulation"),
        ),
        (
            ChatIntent.WORK_PRIORITY,
            ("우선순위", "먼저", "급한", "priority", "urgent work"),
        ),
        (
            ChatIntent.REPORT_LOOKUP,
            ("보고서", "리포트", "요약", "report", "summary"),
        ),
        (
            ChatIntent.PRODUCTION_PLAN,
            ("생산계획", "계획", "일정", "plan", "schedule"),
        ),
    )

    def classify(self, question: str) -> ChatIntent:
        normalized_question = question.lower()
        for intent, keywords in self._intent_keywords:
            if any(keyword.lower() in normalized_question for keyword in keywords):
                return intent
        return ChatIntent.UNKNOWN
