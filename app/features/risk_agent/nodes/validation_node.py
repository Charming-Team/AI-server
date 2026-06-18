from __future__ import annotations

import re

from app.features.risk_agent.nodes.analyzer_utils import (
    as_float,
)
from app.features.risk_agent.schemas.common import (
    DelayCauseType,
    RiskLevel,
    WorkflowStatus,
)
from app.features.risk_agent.schemas.state import (
    RiskAgentWorkflowState,
)


class RiskAgentValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = list(dict.fromkeys(errors))
        super().__init__("; ".join(self.errors))


class RiskAgentValidationNode:
    CAUSE_LABELS = {
        DelayCauseType.MATERIAL_SHORTAGE: "자재 부족",
        DelayCauseType.MATERIAL_DELAY: "자재 입고 지연",
        DelayCauseType.LOW_YIELD: "수율 저하",
        DelayCauseType.MACHINE_ABNORMAL: "설비 상태 이상",
        DelayCauseType.LINE_ABNORMAL: "라인 상태 이상",
    }

    FORBIDDEN_TERMS = (
        "CRITICAL",
        "WARNING",
        "CAUTION",
        "SAFE",
        "MATERIAL_SHORTAGE",
        "MATERIAL_DELAY",
        "LOW_YIELD",
        "MACHINE_ABNORMAL",
        "LINE_ABNORMAL",
        "rankedCauses",
        "selectedCauseTypes",
        "missingFields",
        "materialEvidence",
        "machineEvidence",
        "mlCauseDetailJson",
    )

    def run(
        self,
        state: RiskAgentWorkflowState,
    ) -> RiskAgentWorkflowState:
        errors: list[str] = []

        if state.status != WorkflowStatus.GENERATED:
            errors.append(
                "Validation은 GENERATED 상태에서만 수행할 수 있습니다."
            )

        evidence = state.evidence

        if evidence is None:
            errors.append("Validation에 필요한 evidence가 없습니다.")
        else:
            if evidence.prediction_id != state.prediction_id:
                errors.append(
                    "Workflow와 Evidence의 predictionId가 일치하지 않습니다."
                )

            if evidence.order_id != state.order_id:
                errors.append(
                    "Workflow와 Evidence의 orderId가 일치하지 않습니다."
                )

        if state.risk_level is None:
            errors.append("riskLevel이 없습니다.")
        elif state.risk_level == RiskLevel.SAFE:
            errors.append(
                "SAFE 예측 결과에는 Agent 분석을 저장할 수 없습니다."
            )

        self._validate_probability_and_risk_level(
            state,
            errors,
        )

        selected = list(state.selected_cause_types)

        if not 1 <= len(selected) <= 3:
            errors.append(
                "최종 원인은 1개 이상 3개 이하여야 합니다."
            )

        if len(set(selected)) != len(selected):
            errors.append(
                "최종 원인에 중복 값이 존재합니다."
            )

        ranked_selected = [
            cause.cause_type
            for cause in state.ranked_causes[: len(selected)]
        ]

        if selected and selected != ranked_selected:
            errors.append(
                "selectedCauseTypes와 rankedCauses 순서가 일치하지 않습니다."
            )

        for cause_type in selected:
            ranked_cause = next(
                (
                    cause
                    for cause in state.ranked_causes
                    if cause.cause_type == cause_type
                ),
                None,
            )

            if ranked_cause is None:
                errors.append(
                    f"{cause_type.value} 원인의 Ranking 결과가 없습니다."
                )
                continue

            if ranked_cause.score < 0.25:
                errors.append(
                    f"{cause_type.value} 원인의 점수가 저장 기준보다 낮습니다."
                )

            if not ranked_cause.evidence:
                errors.append(
                    f"{cause_type.value} 원인의 근거가 없습니다."
                )

        summary = (state.analysis_summary or "").strip()
        action = (state.recommended_action or "").strip()

        if len(summary) < 30:
            errors.append("analysisSummary가 너무 짧습니다.")

        if len(action) < 20:
            errors.append("recommendedAction이 너무 짧습니다.")

        combined_text = f"{summary} {action}"

        exposed_terms = [
            term
            for term in self.FORBIDDEN_TERMS
            if term in combined_text
        ]

        if exposed_terms:
            errors.append(
                "사용자 문구에 내부 코드가 포함되었습니다: "
                + ", ".join(exposed_terms)
            )

        if evidence is not None:
            if evidence.order_no not in summary:
                errors.append(
                    "analysisSummary에 주문번호가 포함되지 않았습니다."
                )

        if "%" not in summary:
            errors.append(
                "analysisSummary에 지연 확률이 포함되지 않았습니다."
            )

        if "일" not in summary:
            errors.append(
                "analysisSummary에 예상 지연 일수가 포함되지 않았습니다."
            )

        if selected:
            primary_cause_label = self.CAUSE_LABELS[selected[0]]

            if primary_cause_label not in summary:
                errors.append(
                    "analysisSummary에 1순위 원인 표시명이 포함되지 않았습니다."
                )

        if state.missing_fields:
            missing_disclosure_terms = (
                "추가 확인",
                "확인이 필요",
                "근거 데이터가 부족",
                "판단을 유보",
                "정보가 누락",
            )

            if not any(
                term in summary
                for term in missing_disclosure_terms
            ):
                errors.append(
                    "누락 Evidence가 있지만 추가 확인 문구가 없습니다."
                )

        self._validate_numbered_actions(
            action,
            errors,
        )

        self._validate_action_scope(
            action=action,
            selected=set(selected),
            errors=errors,
        )

        if errors:
            raise RiskAgentValidationError(errors)

        return state.model_copy(
            update={
                "status": WorkflowStatus.VALIDATED,
                "validation_errors": [],
            }
        )

    def _validate_probability_and_risk_level(
        self,
        state: RiskAgentWorkflowState,
        errors: list[str],
    ) -> None:
        if state.delay_probability is None:
            errors.append("delayProbability가 없습니다.")
            return

        probability = as_float(state.delay_probability)

        if not 0.0 <= probability <= 1.0:
            errors.append(
                "delayProbability는 0 이상 1 이하여야 합니다."
            )
            return

        expected_level = self._risk_level_from_probability(
            probability
        )

        if (
            state.risk_level is not None
            and state.risk_level != expected_level
        ):
            errors.append(
                "delayProbability와 riskLevel 임계치가 일치하지 않습니다."
            )

    @staticmethod
    def _risk_level_from_probability(
        probability: float,
    ) -> RiskLevel:
        if probability <= 0.10:
            return RiskLevel.SAFE

        if probability <= 0.40:
            return RiskLevel.CAUTION

        if probability <= 0.70:
            return RiskLevel.WARNING

        return RiskLevel.CRITICAL

    @staticmethod
    def _validate_numbered_actions(
        action: str,
        errors: list[str],
    ) -> None:
        lines = [
            line.strip()
            for line in action.splitlines()
            if line.strip()
        ]

        numbers: list[int] = []

        for line in lines:
            matched = re.match(
                r"^(\d+)\)\s+\S",
                line,
            )

            if matched is None:
                errors.append(
                    "recommendedAction은 각 줄이 '1) ...' 형식이어야 합니다."
                )
                return

            numbers.append(int(matched.group(1)))

        if not 2 <= len(lines) <= 4:
            errors.append(
                "recommendedAction은 2개 이상 4개 이하의 조치여야 합니다."
            )

        expected_numbers = list(
            range(1, len(numbers) + 1)
        )

        if numbers != expected_numbers:
            errors.append(
                "recommendedAction의 번호 순서가 올바르지 않습니다."
            )

    @staticmethod
    def _validate_action_scope(
        *,
        action: str,
        selected: set[DelayCauseType],
        errors: list[str],
    ) -> None:
        unsupported_keywords: list[str] = []

        material_causes = {
            DelayCauseType.MATERIAL_SHORTAGE,
            DelayCauseType.MATERIAL_DELAY,
        }

        if not material_causes.intersection(selected):
            unsupported_keywords.extend(
                keyword
                for keyword in (
                    "자재",
                    "재고",
                    "원료",
                    "입고",
                    "조달",
                    "공급망",
                )
                if keyword in action
            )

        if DelayCauseType.LOW_YIELD not in selected:
            unsupported_keywords.extend(
                keyword
                for keyword in (
                    "수율",
                    "불량",
                )
                if keyword in action
            )

        if DelayCauseType.MACHINE_ABNORMAL not in selected:
            unsupported_keywords.extend(
                keyword
                for keyword in (
                    "설비",
                    "기계",
                    "정비",
                    "복구",
                )
                if keyword in action
            )

        if DelayCauseType.LINE_ABNORMAL not in selected:
            unsupported_keywords.extend(
                keyword
                for keyword in (
                    "라인",
                    "라인 부하",
                    "대기시간",
                    "대기 시간",
                    "생산 순서",
                )
                if keyword in action
            )

        if unsupported_keywords:
            errors.append(
                "선정 원인과 연결되지 않은 권고가 포함되었습니다: "
                + ", ".join(
                    dict.fromkeys(
                        unsupported_keywords
                    )
                )
            )