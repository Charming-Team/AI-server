from __future__ import annotations

from app.features.risk_agent.nodes.analyzer_utils import (
    as_float,
    clamp,
    unique_texts,
)
from app.features.risk_agent.schemas.common import (
    AnalyzerName,
    DelayCauseType,
)
from app.features.risk_agent.schemas.evidence import RiskAgentEvidence
from app.features.risk_agent.schemas.state import AnalyzerFinding


class MachineAnalyzer:
    name = AnalyzerName.MACHINE

    STATUS_WEIGHT = {
        "RUNNING": 0.0,
        "IDLE": 0.10,
        "SETUP": 0.20,
        "MAINTENANCE": 0.75,
        "STOPPED": 0.90,
        "ERROR": 1.00,
    }

    ABNORMAL_STATUSES = {
        "MAINTENANCE",
        "STOPPED",
        "ERROR",
    }

    def analyze(
        self,
        context: RiskAgentEvidence,
    ) -> list[AnalyzerFinding]:
        if not context.machines:
            return [
                AnalyzerFinding(
                    analyzer=self.name,
                    detected=False,
                    summary="설비 상태 데이터가 없습니다.",
                    reasoning=(
                        "배정 라인의 Machine 또는 최신 상태 기록이 없어 "
                        "설비 이상 여부를 확정하지 않았습니다."
                    ),
                    missing_fields=["machineEvidence"],
                )
            ]

        abnormal_count = 0
        max_status_weight = 0.0
        total_processed = 0
        total_defects = 0
        evidence: list[str] = []

        for machine in context.machines:
            status = str(machine.operation_status or "UNKNOWN").upper()
            weight = self.STATUS_WEIGHT.get(status, 0.25)
            max_status_weight = max(max_status_weight, weight)

            processed = max(machine.processed_quantity or 0, 0)
            defects = max(machine.defect_quantity or 0, 0)

            total_processed += processed
            total_defects += defects

            machine_name = (
                machine.machine_name
                or machine.machine_code
                or str(machine.machine_id)
            )

            if status in self.ABNORMAL_STATUSES:
                abnormal_count += 1
                evidence.append(
                    f"{machine_name}: 상태={status}, "
                    f"특이사항={machine.status_note or '없음'}"
                )

        defect_rate = (
            total_defects / total_processed
            if total_processed > 0
            else 0.0
        )

        abnormal_ratio = abnormal_count / len(context.machines)

        score = clamp(
            max_status_weight * 0.70
            + abnormal_ratio * 0.20
            + clamp(defect_rate / 0.10) * 0.10
        )

        detected = (
            abnormal_count > 0
            or defect_rate >= 0.05
        )

        if not evidence:
            evidence.append(
                f"전체 {len(context.machines)}개 설비가 "
                "중대 이상 상태 없이 운영 중입니다."
            )

        evidence.append(
            f"설비 처리량 기준 불량률={defect_rate * 100:.2f}%"
        )

        return [
            AnalyzerFinding(
                analyzer=self.name,
                detected=detected,
                cause_type=(
                    DelayCauseType.MACHINE_ABNORMAL
                    if detected
                    else None
                ),
                score=score,
                summary=(
                    "생산 흐름에 영향을 줄 수 있는 설비 이상이 확인되었습니다."
                    if detected
                    else "중대한 설비 이상이 확인되지 않았습니다."
                ),
                reasoning=(
                    f"전체 {len(context.machines)}개 설비 중 "
                    f"{abnormal_count}개가 비정상 상태입니다."
                ),
                evidence=unique_texts(evidence),
                metrics={
                    "machineCount": len(context.machines),
                    "abnormalMachineCount": abnormal_count,
                    "abnormalMachineRatio": abnormal_ratio,
                    "machineDefectRate": defect_rate,
                },
            )
        ]