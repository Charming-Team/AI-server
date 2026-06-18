from dataclasses import dataclass

from app.features.chat.access_control import (
    BUSINESS_ROLES,
    OPERATOR_RESTRICTED_TERMS,
    OPERATOR_ROLE,
    ROLE_INTENT_MATRIX,
)
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import (
    ChatErrorCode,
    ChatIntent,
    ChatRecommendationRequest,
    ChatRecommendationResponse,
    ChatRecommendedQuestion,
)
from app.features.chat.security_policy import SecurityPolicy
from app.features.chat.source_url_policy import normalize_internal_url


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
    _business_roles = BUSINESS_ROLES

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
            url="/production-plans?mode=read",
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
            question_id="operator-report-summary-read",
            question="최근 생산 리스크 보고서를 조회해줘",
            intent=ChatIntent.REPORT_LOOKUP,
            category="보고서 조회",
            url="/reports?mode=read",
            allowed_roles=("OPERATOR",),
        ),
        RecommendedQuestionRule(
            question_id="urgent-order-impact",
            question="긴급 주문이 전체 생산계획에 미치는 영향을 알려줘",
            intent=ChatIntent.URGENT_ORDER_IMPACT,
            category="긴급 주문 영향",
            url="/production-plans?mode=read",
            allowed_roles=("OPERATOR", "EXECUTIVE", "MANUFACTURING_MANAGER"),
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
            question_id="monthly-report-summary",
            question="최근 생산 리스크 보고서를 요약해줘",
            intent=ChatIntent.REPORT_LOOKUP,
            category="보고서 조회",
            url="/reports",
            allowed_roles=("EXECUTIVE",),
        ),
    )

    def __init__(self, security_policy: SecurityPolicy | None = None) -> None:
        self.security_policy = security_policy or SecurityPolicy()

    def get_recommendations(
        self,
        request: ChatRecommendationRequest,
    ) -> ChatRecommendationResponse:
        self._validate_active_user(request.user.status)
        self._validate_business_role(request.user.role)
        self._validate_keyword(request.keyword)

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

    def _validate_active_user(self, status: str) -> None:
        if status.strip().upper() == "ACTIVE":
            return

        raise ChatServiceError(
            status_code=403,
            code=ChatErrorCode.CHAT_SECURITY_004,
            message="ACTIVE 상태 사용자만 추천 질문을 조회할 수 있습니다.",
        )

    def _validate_business_role(self, role: str) -> None:
        if role.strip().upper() in self._business_roles:
            return

        raise ChatServiceError(
            status_code=403,
            code=ChatErrorCode.CHAT_SECURITY_004,
            message="현재 역할 권한으로는 추천 질문을 조회할 수 없습니다.",
        )

    def _validate_keyword(self, keyword: str | None) -> None:
        if keyword is None or not keyword.strip():
            return

        security_result = self.security_policy.evaluate(keyword)
        if security_result is None:
            return

        raise ChatServiceError(
            status_code=400,
            code=security_result.code or ChatErrorCode.CHAT_SECURITY_001,
            message="추천 질문 키워드에 보안 정책상 허용되지 않는 내용이 포함되어 있습니다.",
        )

    def _filter_by_role(self, role: str) -> list[RecommendedQuestionRule]:
        normalized_role = role.strip().upper()
        return [
            rule
            for rule in self._rules
            if self._is_rule_allowed_for_role(rule, normalized_role)
        ]

    def _is_rule_allowed_for_role(
        self,
        rule: RecommendedQuestionRule,
        role: str,
    ) -> bool:
        if self._safe_internal_url(rule.url) is None:
            return False

        if role not in rule.allowed_roles:
            return False

        if rule.intent not in ROLE_INTENT_MATRIX.get(role, frozenset()):
            return False

        if role != OPERATOR_ROLE:
            return True

        return (
            self._is_operator_read_only_rule(rule)
            and not self._has_operator_restricted_content(rule)
        )

    def _is_operator_read_only_rule(self, rule: RecommendedQuestionRule) -> bool:
        safe_url = self._safe_internal_url(rule.url)
        return safe_url is not None and "mode=read" in self._normalize(safe_url)

    def _has_operator_restricted_content(
        self,
        rule: RecommendedQuestionRule,
    ) -> bool:
        target = self._normalize(
            " ".join(
                (
                    rule.question,
                    rule.category,
                    rule.intent.value,
                    rule.url,
                )
            )
        )
        return any(
            self._normalize(term) in target
            for term in OPERATOR_RESTRICTED_TERMS
        )

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
            url=self._safe_internal_url(rule.url) or "",
        )

    def _normalize(self, value: str) -> str:
        return "".join(value.casefold().split())

    def _safe_internal_url(self, url: str | None) -> str | None:
        return normalize_internal_url(url)
