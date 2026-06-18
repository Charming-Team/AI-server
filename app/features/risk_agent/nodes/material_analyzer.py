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


class MaterialAnalyzer:
    name = AnalyzerName.MATERIAL

    def analyze(
        self,
        context: RiskAgentEvidence,
    ) -> list[AnalyzerFinding]:
        if not context.materials:
            return [
                AnalyzerFinding(
                    analyzer=self.name,
                    detected=False,
                    summary="자재 근거 데이터가 없습니다.",
                    reasoning=(
                        "생산계획별 필요 자재 또는 재고 데이터를 "
                        "조회하지 못해 자재 원인을 확정하지 않았습니다."
                    ),
                    missing_fields=["materialEvidence"],
                )
            ]

        shortage_score = 0.0
        delay_score = 0.0
        shortage_evidence: list[str] = []
        delay_evidence: list[str] = []
        shortage_material_count = 0
        delayed_material_count = 0

        for material in context.materials:
            required = as_float(material.required_quantity)
            consumed = as_float(material.consumed_quantity)
            remaining_need = max(required - consumed, 0.0)

            available = as_float(
                material.available_inventory_quantity,
                as_float(material.current_inventory_quantity),
            )

            recorded_shortage = as_float(material.shortage_quantity)
            calculated_shortage = max(remaining_need - available, 0.0)
            shortage = max(recorded_shortage, calculated_shortage)

            material_name = (
                material.material_name
                or material.material_code
                or str(material.material_id)
            )

            status = str(material.material_plan_status or "").upper()

            if shortage > 0:
                shortage_material_count += 1

                shortage_ratio = (
                    shortage / remaining_need
                    if remaining_need > 0
                    else 1.0
                )

                local_score = clamp(
                    0.30
                    + clamp(shortage_ratio) * 0.60
                    + (0.10 if "SHORTAGE" in status else 0.0)
                )

                shortage_score = max(shortage_score, local_score)

                shortage_evidence.append(
                    f"{material_name}: 필요 잔량={remaining_need:.2f}, "
                    f"가용 재고={available:.2f}, 부족량={shortage:.2f}"
                )

            inbound_at = material.expected_inbound_at
            planned_start_at = context.planned_start_at

            if (
                shortage > 0
                and inbound_at is not None
                and planned_start_at is not None
                and inbound_at > planned_start_at
            ):
                delayed_material_count += 1

                delay_days = (
                    inbound_at - planned_start_at
                ).total_seconds() / 86400.0

                local_delay_score = clamp(
                    0.35 + clamp(delay_days / 7.0) * 0.65
                )
                delay_score = max(delay_score, local_delay_score)

                delay_evidence.append(
                    f"{material_name}: 입고 예정 시각이 생산 시작보다 "
                    f"{delay_days:.1f}일 늦습니다."
                )

        shortage_ml_boost = ml_cause_boost(
            context.ml_cause_detail_json,
            {
                "MATERIAL_SHORTAGE",
                "MATERIAL_NOT_READY",
                "PLAN_QTY_GAP",
            },
        )

        delay_ml_boost = ml_cause_boost(
            context.ml_cause_detail_json,
            {"MATERIAL_DELAY"},
        )

        shortage_score = clamp(shortage_score + shortage_ml_boost)
        delay_score = clamp(delay_score + delay_ml_boost)

        shortage_evidence.extend(
            ml_factor_evidence(
                context.ml_cause_detail_json,
                {
                    "MATERIAL_SHORTAGE",
                    "MATERIAL_NOT_READY",
                    "PLAN_QTY_GAP",
                },
            )
        )

        delay_evidence.extend(
            ml_factor_evidence(
                context.ml_cause_detail_json,
                {"MATERIAL_DELAY"},
            )
        )

        findings: list[AnalyzerFinding] = []

        if shortage_material_count > 0:
            findings.append(
                AnalyzerFinding(
                    analyzer=self.name,
                    detected=True,
                    cause_type=DelayCauseType.MATERIAL_SHORTAGE,
                    score=shortage_score,
                    summary="생산에 필요한 자재가 부족합니다.",
                    reasoning=(
                        f"{shortage_material_count}개 자재에서 생산 필요량보다 "
                        "가용 또는 예약 수량이 부족한 상태가 확인되었습니다."
                    ),
                    evidence=unique_texts(shortage_evidence),
                    metrics={
                        "shortageMaterialCount": shortage_material_count,
                    },
                )
            )

        if delayed_material_count > 0:
            findings.append(
                AnalyzerFinding(
                    analyzer=self.name,
                    detected=True,
                    cause_type=DelayCauseType.MATERIAL_DELAY,
                    score=delay_score,
                    summary="필요 자재의 입고가 생산 시점보다 늦습니다.",
                    reasoning=(
                        f"{delayed_material_count}개 자재의 입고 예정 시각이 "
                        "생산 시작 예정 시각 이후입니다."
                    ),
                    evidence=unique_texts(delay_evidence),
                    metrics={
                        "delayedMaterialCount": delayed_material_count,
                    },
                )
            )

        if findings:
            return findings

        return [
            AnalyzerFinding(
                analyzer=self.name,
                detected=False,
                score=max(shortage_score, delay_score),
                summary="유의미한 자재 부족 또는 입고 지연이 확인되지 않았습니다.",
                reasoning="현재 조회된 자재 계획과 재고 기준으로 생산 투입이 가능합니다.",
                evidence=unique_texts(
                    shortage_evidence + delay_evidence
                ),
            )
        ]