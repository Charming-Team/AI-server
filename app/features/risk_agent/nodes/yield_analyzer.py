from __future__ import annotations

from app.features.risk_agent.nodes.analyzer_utils import (
    as_float,
    clamp,
    ml_cause_boost,
    ml_factor_evidence,
    unique_texts,
)
from app.features.risk_agent.schemas.common import (
    AnalyzerName,
    DelayCauseType,
)
from app.features.risk_agent.schemas.evidence import RiskAgentEvidence
from app.features.risk_agent.schemas.state import AnalyzerFinding


class YieldAnalyzer:
    name = AnalyzerName.YIELD

    REFERENCE_YIELD_RATE = 0.95
    LOW_YIELD_THRESHOLD = 0.92
    CRITICAL_DEFECT_RATE = 0.04

    def analyze(
        self,
        context: RiskAgentEvidence,
    ) -> list[AnalyzerFinding]:
        completed_quantity = max(context.completed_quantity, 0)

        current_yield: float | None = None
        yield_source: str | None = None

        if context.actual_yield_rate is not None and completed_quantity > 0:
            current_yield = as_float(context.actual_yield_rate)
            yield_source = "주문 생산실적 수율"
        elif context.line_yield_rate is not None:
            current_yield = as_float(context.line_yield_rate)
            yield_source = "라인 현재 수율"

        ml_boost = ml_cause_boost(
            context.ml_cause_detail_json,
            {"YIELD_RISK"},
        )

        shap_evidence = ml_factor_evidence(
            context.ml_cause_detail_json,
            {"YIELD_RISK"},
        )

        if current_yield is None:
            return [
                AnalyzerFinding(
                    analyzer=self.name,
                    detected=False,
                    score=ml_boost,
                    summary="현재 수율 데이터를 확인할 수 없습니다.",
                    reasoning=(
                        "생산 실적 수율과 라인 현재 수율이 모두 없어 "
                        "수율 저하 여부를 확정하지 않았습니다."
                    ),
                    evidence=shap_evidence,
                    missing_fields=["yieldEvidence"],
                )
            ]

        defect_rate = (
            context.defect_quantity / completed_quantity
            if completed_quantity > 0
            else 0.0
        )

        yield_gap = max(
            self.REFERENCE_YIELD_RATE - current_yield,
            0.0,
        )

        score = clamp(
            clamp(yield_gap / 0.10) * 0.70
            + clamp(defect_rate / 0.08) * 0.20
            + ml_boost
        )

        detected = (
            current_yield < self.LOW_YIELD_THRESHOLD
            or defect_rate >= self.CRITICAL_DEFECT_RATE
            or (
                current_yield < self.REFERENCE_YIELD_RATE
                and ml_boost > 0
            )
        )

        evidence = [
            f"{yield_source}={current_yield * 100:.2f}%",
            f"기준 비교 수율={self.REFERENCE_YIELD_RATE * 100:.2f}%",
            f"수율 차이={yield_gap * 100:.2f}%p",
            f"불량률={defect_rate * 100:.2f}%",
            *shap_evidence,
        ]

        return [
            AnalyzerFinding(
                analyzer=self.name,
                detected=detected,
                cause_type=(
                    DelayCauseType.LOW_YIELD
                    if detected
                    else None
                ),
                score=score,
                summary=(
                    "기준 수율 대비 생산 수율이 낮습니다."
                    if detected
                    else "수율 저하 위험이 크지 않습니다."
                ),
                reasoning=(
                    "현재 수율 또는 불량률로 인해 주문 수량 충족을 위한 "
                    "추가 생산 시간이 필요할 수 있습니다."
                    if detected
                    else "현재 수율과 불량률이 허용 범위 내에 있습니다."
                ),
                evidence=unique_texts(evidence),
                metrics={
                    "currentYieldRate": current_yield,
                    "referenceYieldRate": self.REFERENCE_YIELD_RATE,
                    "yieldGap": yield_gap,
                    "defectRate": defect_rate,
                },
            )
        ]