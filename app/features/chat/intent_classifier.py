from dataclasses import dataclass

from app.features.chat.schemas import ChatIntent


@dataclass(frozen=True)
class IntentRule:
    intent: ChatIntent
    phrases: tuple[str, ...]
    keywords: tuple[str, ...]


class IntentClassifier:
    _phrase_score = 3
    _keyword_score = 1

    _intent_rules: tuple[IntentRule, ...] = (
        IntentRule(
            intent=ChatIntent.MATERIAL_SHORTAGE,
            phrases=(
                "자재 부족",
                "재고 부족",
                "부족 자재",
                "가용 재고",
                "안전 재고",
                "입고 예정",
                "material shortage",
            ),
            keywords=(
                "자재",
                "재고",
                "부족",
                "입고",
                "bom",
                "material",
                "inventory",
                "shortage",
            ),
        ),
        IntentRule(
            intent=ChatIntent.LINE_BOTTLENECK,
            phrases=(
                "라인 병목",
                "공정 병목",
                "대기시간 증가",
                "설비 이상",
                "장비 이상",
                "line bottleneck",
            ),
            keywords=(
                "라인",
                "공정",
                "병목",
                "설비",
                "장비",
                "대기시간",
                "가동률",
                "machine",
                "line",
                "bottleneck",
                "error",
                "stopped",
                "maintenance",
            ),
        ),
        IntentRule(
            intent=ChatIntent.URGENT_ORDER_IMPACT,
            phrases=(
                "긴급 주문",
                "긴급 오더",
                "주문 영향",
                "긴급 주문 영향",
                "urgent order",
            ),
            keywords=(
                "긴급",
                "영향",
                "대응안",
                "시뮬레이션",
                "urgent",
                "impact",
                "simulation",
            ),
        ),
        IntentRule(
            intent=ChatIntent.WORK_PRIORITY,
            phrases=(
                "작업 우선순위",
                "생산 우선순위",
                "먼저 처리",
                "먼저 생산",
            ),
            keywords=("우선순위", "우선", "먼저", "급한", "priority"),
        ),
        IntentRule(
            intent=ChatIntent.DELIVERY_RISK,
            phrases=(
                "납기 지연",
                "납기 위험",
                "지연 위험",
                "납기 가능성",
                "납기 준수",
                "due date",
                "delivery risk",
                "delay risk",
            ),
            keywords=("납기", "지연", "위험", "delay", "delayed", "risk", "due"),
        ),
        IntentRule(
            intent=ChatIntent.REPORT_LOOKUP,
            phrases=("보고서 조회", "보고서 요약", "월간 리포트", "수시 리포트"),
            keywords=("보고서", "리포트", "요약", "report", "summary"),
        ),
        IntentRule(
            intent=ChatIntent.PRODUCTION_PLAN,
            phrases=(
                "생산계획",
                "생산 계획",
                "계획 변경",
                "스케줄 변경",
                "일정 변경",
                "production plan",
            ),
            keywords=("계획", "일정", "스케줄", "배정", "plan", "schedule"),
        ),
    )

    def classify(self, question: str) -> ChatIntent:
        normalized_question = self._normalize(question)
        compact_question = self._compact(normalized_question)
        best_intent = ChatIntent.UNKNOWN
        best_score = 0

        for rule in self._intent_rules:
            score = self._score_rule(rule, normalized_question, compact_question)
            if score > best_score:
                best_intent = rule.intent
                best_score = score

        return best_intent

    def _score_rule(
        self,
        rule: IntentRule,
        normalized_question: str,
        compact_question: str,
    ) -> int:
        phrase_score = self._score_terms(
            rule.phrases,
            normalized_question,
            compact_question,
            self._phrase_score,
        )
        keyword_score = self._score_terms(
            rule.keywords,
            normalized_question,
            compact_question,
            self._keyword_score,
        )
        return phrase_score + keyword_score

    def _score_terms(
        self,
        terms: tuple[str, ...],
        normalized_question: str,
        compact_question: str,
        score_per_term: int,
    ) -> int:
        matched_terms = {
            self._compact(self._normalize(term))
            for term in terms
            if self._contains_term(term, normalized_question, compact_question)
        }
        return len(matched_terms) * score_per_term

    def _contains_term(
        self,
        term: str,
        normalized_question: str,
        compact_question: str,
    ) -> bool:
        normalized_term = self._normalize(term)
        compact_term = self._compact(normalized_term)
        return normalized_term in normalized_question or compact_term in compact_question

    def _normalize(self, text: str) -> str:
        return text.casefold()

    def _compact(self, text: str) -> str:
        return "".join(text.split())
