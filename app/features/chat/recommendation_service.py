from dataclasses import dataclass

from app.features.chat.schemas import (
    ChatIntent,
    ChatRecommendationRequest,
    ChatRecommendationResponse,
    ChatRecommendedQuestion,
)


@dataclass(frozen=True)
class RecommendedQuestionRule:
    question_id: str
    question: str
    intent: ChatIntent
    category: str
    url: str
    allowed_roles: tuple[str, ...]


class RecommendationService:
    _max_items = 6

    _rules: tuple[RecommendedQuestionRule, ...] = (
        RecommendedQuestionRule(
            question_id="delivery-risk-high-orders",
            question="현재 납기 위험이 높은 주문을 알려줘",
            intent=ChatIntent.DELIVERY_RISK,
            category="납기 위험",
            url="/orders?riskLevel=WARNING",
            allowed_roles=("EXECUTIVE",),
        ),
        RecommendedQuestionRule(
            question_id="delivery-risk-financial-impact",
            question="납기 지연 시 예상 패널티와 계약 금액 영향을 알려줘",
            intent=ChatIntent.DELIVERY_RISK,
            category="금액 영향",
            url="/orders/financial-impact",
            allowed_roles=("EXECUTIVE",),
        ),
        RecommendedQuestionRule(
            question_id="executive-production-risk-overview",
            question="이번 주 전체 생산 리스크를 요약해줘",
            intent=ChatIntent.REPORT_LOOKUP,
            category="경영 리스크",
            url="/reports?type=production-risk",
            allowed_roles=("EXECUTIVE",),
        ),
        RecommendedQuestionRule(
            question_id="executive-line-utilization-summary",
            question="라인 가동률과 병목 현황을 요약해줘",
            intent=ChatIntent.LINE_BOTTLENECK,
            category="라인 현황",
            url="/production-lines/dashboard",
            allowed_roles=("EXECUTIVE",),
        ),
        RecommendedQuestionRule(
            question_id="material-shortage-impact",
            question="자재 부족으로 영향받는 생산계획을 알려줘",
            intent=ChatIntent.MATERIAL_SHORTAGE,
            category="자재 부족",
            url="/materials/inventories",
            allowed_roles=("MANUFACTURING_MANAGER",),
        ),
        RecommendedQuestionRule(
            question_id="operator-material-inventory-read",
            question="현재 자재 재고 현황을 조회해줘",
            intent=ChatIntent.MATERIAL_SHORTAGE,
            category="자재 현황",
            url="/materials/inventories?mode=read",
            allowed_roles=("OPERATOR",),
        ),
        RecommendedQuestionRule(
            question_id="production-plan-changes",
            question="오늘 변경이 필요한 생산계획을 알려줘",
            intent=ChatIntent.PRODUCTION_PLAN,
            category="생산 계획 변경",
            url="/production-plans",
            allowed_roles=("MANUFACTURING_MANAGER",),
        ),
        RecommendedQuestionRule(
            question_id="operator-production-plan-read",
            question="오늘 배정된 생산계획을 조회해줘",
            intent=ChatIntent.PRODUCTION_PLAN,
            category="생산 계획",
            url="/production-plans?mode=read",
            allowed_roles=("OPERATOR",),
        ),
        RecommendedQuestionRule(
            question_id="line-bottleneck-current",
            question="현재 병목이 발생한 라인과 원인을 알려줘",
            intent=ChatIntent.LINE_BOTTLENECK,
            category="라인 병목",
            url="/production-lines/status",
            allowed_roles=("MANUFACTURING_MANAGER",),
        ),
        RecommendedQuestionRule(
            question_id="operator-line-status-read",
            question="내 담당 라인의 현재 상태를 조회해줘",
            intent=ChatIntent.LINE_BOTTLENECK,
            category="라인 현황",
            url="/production-lines/status?mode=read",
            allowed_roles=("OPERATOR",),
        ),
        RecommendedQuestionRule(
            question_id="operator-machine-status-read",
            question="내 담당 설비의 현재 상태를 조회해줘",
            intent=ChatIntent.LINE_BOTTLENECK,
            category="설비 현황",
            url="/production-machines/status?mode=read",
            allowed_roles=("OPERATOR",),
        ),
        RecommendedQuestionRule(
            question_id="operator-production-result-read",
            question="오늘 처리 수량과 불량 수량을 조회해줘",
            intent=ChatIntent.PRODUCTION_PLAN,
            category="생산 실적",
            url="/production-results?mode=read",
            allowed_roles=("OPERATOR",),
        ),
        RecommendedQuestionRule(
            question_id="urgent-order-impact",
            question="긴급 주문이 전체 생산계획에 미치는 영향을 알려줘",
            intent=ChatIntent.URGENT_ORDER_IMPACT,
            category="긴급 주문 영향",
            url="/schedule-simulations",
            allowed_roles=("EXECUTIVE", "MANUFACTURING_MANAGER"),
        ),
        RecommendedQuestionRule(
            question_id="work-priority-today",
            question="오늘 먼저 처리해야 할 작업 우선순위를 알려줘",
            intent=ChatIntent.WORK_PRIORITY,
            category="작업 우선순위",
            url="/work-orders?mode=read",
            allowed_roles=("OPERATOR", "MANUFACTURING_MANAGER"),
        ),
        RecommendedQuestionRule(
            question_id="monthly-report-summary",
            question="최근 생산 리스크 보고서를 요약해줘",
            intent=ChatIntent.REPORT_LOOKUP,
            category="보고서 조회",
            url="/reports",
            allowed_roles=("EXECUTIVE",),
        ),
        RecommendedQuestionRule(
            question_id="manager-report-summary",
            question="최근 제조 리스크 보고서를 요약해줘",
            intent=ChatIntent.REPORT_LOOKUP,
            category="보고서 조회",
            url="/reports?type=manufacturing",
            allowed_roles=("MANUFACTURING_MANAGER",),
        ),
    )

    def get_recommendations(
        self,
        request: ChatRecommendationRequest,
    ) -> ChatRecommendationResponse:
        role_rules = self._filter_by_role(request.user.role)
        keyword = self._normalize(request.keyword or "")
        if not keyword:
            return self._build_response(role_rules, fallback_used=False)

        keyword_rules = [
            rule for rule in role_rules if self._matches_keyword(rule, keyword)
        ]
        if keyword_rules:
            return self._build_response(keyword_rules, fallback_used=False)

        return self._build_response(role_rules, fallback_used=True)

    def _filter_by_role(self, role: str) -> list[RecommendedQuestionRule]:
        normalized_role = role.strip().upper()
        return [
            rule
            for rule in self._rules
            if normalized_role in rule.allowed_roles
        ]

    def _matches_keyword(
        self,
        rule: RecommendedQuestionRule,
        keyword: str,
    ) -> bool:
        targets = (
            rule.question,
            rule.category,
            rule.intent.value,
        )
        return any(keyword in self._normalize(target) for target in targets)

    def _build_response(
        self,
        rules: list[RecommendedQuestionRule],
        fallback_used: bool,
    ) -> ChatRecommendationResponse:
        return ChatRecommendationResponse(
            items=[
                self._to_recommended_question(rule)
                for rule in rules[: self._max_items]
            ],
            fallback_used=fallback_used,
        )

    def _to_recommended_question(
        self,
        rule: RecommendedQuestionRule,
    ) -> ChatRecommendedQuestion:
        return ChatRecommendedQuestion(
            question_id=rule.question_id,
            question=rule.question,
            intent=rule.intent,
            category=rule.category,
            url=rule.url,
        )

    def _normalize(self, value: str) -> str:
        return "".join(value.casefold().split())
