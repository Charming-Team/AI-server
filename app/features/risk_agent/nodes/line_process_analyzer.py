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


class LineProcessAnalyzer:
    name = AnalyzerName.LINE_PROCESS

    STATUS_WEIGHT = {
        "RUNNING": 0.0,
        "IDLE": 0.25,
        "SETUP": 0.40,
        "MAINTENANCE": 0.80,
        "STOPPED": 0.95,
        "ERROR": 1.00,
    }

    def analyze(
        self,
        context: RiskAgentEvidence,
    ) -> list[AnalyzerFinding]:
        if context.line_id is None:
            return [
                AnalyzerFinding(
                    analyzer=self.name,
                    detected=False,
                    summary="생산 라인 배정 정보를 확인할 수 없습니다.",
                    reasoning="라인이 배정되지 않아 라인·공정 병목을 분석하지 않았습니다.",
                    missing_fields=["productionLine"],
                )
            ]

        load_ratio = as_float(context.line_load_ratio)
        utilization = as_float(context.line_utilization_rate)
        waiting_time = as_float(context.line_waiting_time_hr)
        waiting_quantity = max(context.line_waiting_quantity or 0, 0)
        capacity = max(context.line_max_capacity_per_day or 0, 0)

        waiting_quantity_ratio = (
            waiting_quantity / capacity
            if capacity > 0
            else 0.0
        )

        status = str(
            context.line_operation_status or "UNKNOWN"
        ).upper()

        status_component = self.STATUS_WEIGHT.get(status, 0.25)
        load_component = clamp((load_ratio - 0.80) / 0.70)
        utilization_component = clamp((utilization - 0.80) / 0.20)
        waiting_time_component = clamp(waiting_time / 8.0)
        waiting_quantity_component = clamp(waiting_quantity_ratio)
        competition_component = clamp(
            len(context.competing_orders) / 10.0
        )

        ml_boost = ml_cause_boost(
            context.ml_cause_detail_json,
            {
                "LINE_LOAD",
                "LINE_CAPACITY",
                "LINE_RISK",
                "LONG_DURATION",
                "SCHEDULE_PRESSURE",
            },
        )

        score = clamp(
            load_component * 0.30
            + utilization_component * 0.20
            + waiting_time_component * 0.15
            + waiting_quantity_component * 0.10
            + status_component * 0.15
            + competition_component * 0.10
            + ml_boost
        )

        detected = (
            score >= 0.45
            or load_ratio > 1.0
            or utilization >= 0.95
            or waiting_time >= 4.0
            or status in {"ERROR", "STOPPED", "MAINTENANCE"}
        )

        evidence = [
            f"라인={context.line_name or context.line_code or context.line_id}",
            f"라인 부하 비율={load_ratio:.4f}",
            f"라인 가동률={utilization * 100:.2f}%",
            f"대기시간={waiting_time:.2f}시간",
            f"대기 수량={waiting_quantity}",
            f"동일 라인 경쟁 계획={len(context.competing_orders)}건",
            f"라인 상태={status}",
            *ml_factor_evidence(
                context.ml_cause_detail_json,
                {
                    "LINE_LOAD",
                    "LINE_CAPACITY",
                    "LINE_RISK",
                    "LONG_DURATION",
                    "SCHEDULE_PRESSURE",
                },
            ),
        ]

        return [
            AnalyzerFinding(
                analyzer=self.name,
                detected=detected,
                cause_type=(
                    DelayCauseType.LINE_ABNORMAL
                    if detected
                    else None
                ),
                score=score,
                summary=(
                    "라인 부하 또는 공정 대기 증가로 병목 가능성이 있습니다."
                    if detected
                    else "라인·공정 병목 위험이 크지 않습니다."
                ),
                reasoning=(
                    "라인 부하, 가동률, 대기시간, 대기 수량, "
                    "경쟁 생산계획을 종합해 판단했습니다."
                ),
                evidence=unique_texts(evidence),
                metrics={
                    "lineLoadRatio": load_ratio,
                    "lineUtilizationRate": utilization,
                    "lineWaitingTimeHr": waiting_time,
                    "lineWaitingQuantity": waiting_quantity,
                    "competingOrderCount": len(context.competing_orders),
                    "lineOperationStatus": status,
                },
            )
        ]