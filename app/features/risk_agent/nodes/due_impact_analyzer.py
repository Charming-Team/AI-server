from __future__ import annotations

from app.features.risk_agent.nodes.analyzer_utils import (
    as_float,
    clamp,
    ml_cause_boost,
    ml_factor_evidence,
    unique_texts,
)
from app.features.risk_agent.schemas.common import AnalyzerName
from app.features.risk_agent.schemas.evidence import RiskAgentEvidence
from app.features.risk_agent.schemas.state import AnalyzerFinding


class DueImpactAnalyzer:
    name = AnalyzerName.DUE_IMPACT

    def analyze(
        self,
        context: RiskAgentEvidence,
    ) -> list[AnalyzerFinding]:
        probability = as_float(context.delay_probability)
        predicted_delay_days = max(
            as_float(context.predicted_delay_days),
            0.0,
        )
        days_until_due = context.days_until_due or 0

        order_quantity = max(context.order_quantity, 0)
        remaining_ratio = (
            context.remaining_quantity / order_quantity
            if order_quantity > 0
            else 0.0
        )

        plan_overdue_days = 0

        if (
            context.planned_end_at is not None
            and context.due_date is not None
        ):
            plan_overdue_days = max(
                (
                    context.planned_end_at.date()
                    - context.due_date
                ).days,
                0,
            )

        penalty_amount = max(
            as_float(context.late_penalty_amount),
            0.0,
        )

        delay_component = clamp(predicted_delay_days / 14.0)
        overdue_component = clamp(plan_overdue_days / 14.0)
        due_urgency_component = (
            1.0
            if days_until_due <= 0
            else clamp(1.0 - days_until_due / 30.0)
        )

        ml_boost = ml_cause_boost(
            context.ml_cause_detail_json,
            {
                "DUE_MARGIN_RISK",
                "LONG_DURATION",
                "SCHEDULE_PRESSURE",
            },
        )

        score = clamp(
            probability * 0.45
            + delay_component * 0.20
            + overdue_component * 0.15
            + remaining_ratio * 0.10
            + due_urgency_component * 0.05
            + (0.05 if penalty_amount > 0 else 0.0)
            + ml_boost
        )

        detected = (
            probability > 0.10
            or predicted_delay_days > 0
            or plan_overdue_days > 0
        )

        evidence = [
            f"ML 지연 확률={probability * 100:.2f}%",
            f"예상 지연 일수={predicted_delay_days:.2f}일",
            f"납기까지 남은 일수={days_until_due}일",
            f"생산계획 납기 초과={plan_overdue_days}일",
            f"잔여 생산 비율={remaining_ratio * 100:.2f}%",
            f"지연 패널티 금액={penalty_amount:.2f}",
            *ml_factor_evidence(
                context.ml_cause_detail_json,
                {
                    "DUE_MARGIN_RISK",
                    "LONG_DURATION",
                    "SCHEDULE_PRESSURE",
                },
            ),
        ]

        return [
            AnalyzerFinding(
                analyzer=self.name,
                detected=detected,
                cause_type=None,
                score=score,
                summary=(
                    "현재 주문의 납기 및 계약 영향도가 높습니다."
                    if score >= 0.60
                    else "현재 주문의 납기 영향도를 추적할 필요가 있습니다."
                ),
                reasoning=(
                    "지연 확률, 예상 지연 일수, 납기 여유, "
                    "잔여 수량 및 패널티 금액을 종합했습니다."
                ),
                evidence=unique_texts(evidence),
                metrics={
                    "delayProbability": probability,
                    "predictedDelayDays": predicted_delay_days,
                    "daysUntilDue": days_until_due,
                    "planOverdueDays": plan_overdue_days,
                    "remainingRatio": remaining_ratio,
                    "latePenaltyAmount": penalty_amount,
                },
            )
        ]