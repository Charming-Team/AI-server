from __future__ import annotations

from dataclasses import dataclass

from app.features.risk_agent.nodes.analyzer_utils import (
    as_float,
    clamp,
    parse_ml_factors,
    unique_texts,
)
from app.features.risk_agent.schemas.common import (
    AnalyzerName,
    ConfidenceLevel,
    DelayCauseType,
    WorkflowStatus,
)
from app.features.risk_agent.schemas.state import (
    AnalyzerFinding,
    RankedCause,
    RiskAgentWorkflowState,
)


@dataclass(frozen=True)
class _CauseCandidate:
    cause_type: DelayCauseType
    score: float
    reason: str
    evidence: list[str]


class CauseRankingNode:
    MAX_CAUSES = 3
    MIN_FINAL_SCORE = 0.25

    CONFIDENCE_FACTOR = {
        ConfidenceLevel.HIGH: 1.00,
        ConfidenceLevel.MEDIUM: 0.92,
        ConfidenceLevel.LOW: 0.80,
    }

    SHAP_TAG_TO_CAUSE = {
        "MATERIAL_SHORTAGE": DelayCauseType.MATERIAL_SHORTAGE,
        "MATERIAL_NOT_READY": DelayCauseType.MATERIAL_SHORTAGE,
        "MATERIAL_DELAY": DelayCauseType.MATERIAL_DELAY,
        "YIELD_RISK": DelayCauseType.LOW_YIELD,
        "MACHINE_ABNORMAL": DelayCauseType.MACHINE_ABNORMAL,
        "LINE_LOAD": DelayCauseType.LINE_ABNORMAL,
        "LINE_CAPACITY": DelayCauseType.LINE_ABNORMAL,
        "LINE_RISK": DelayCauseType.LINE_ABNORMAL,
        "LONG_DURATION": DelayCauseType.LINE_ABNORMAL,
        "SCHEDULE_PRESSURE": DelayCauseType.LINE_ABNORMAL,
    }

    CAUSE_LABEL = {
        DelayCauseType.MATERIAL_SHORTAGE: "자재 부족",
        DelayCauseType.MATERIAL_DELAY: "자재 입고 지연",
        DelayCauseType.LOW_YIELD: "수율 저하",
        DelayCauseType.MACHINE_ABNORMAL: "설비 상태 이상",
        DelayCauseType.LINE_ABNORMAL: "라인 상태 이상",
    }

    def run(
        self,
        state: RiskAgentWorkflowState,
    ) -> RiskAgentWorkflowState:
        if state.status != WorkflowStatus.ANALYZED:
            raise ValueError(
                "Cause Ranking은 ANALYZED 상태에서만 실행할 수 있습니다."
            )

        if state.evidence is None:
            raise ValueError(
                "Cause Ranking에 필요한 evidence가 없습니다."
            )

        probability = as_float(state.delay_probability)

        due_impact_score = max(
            (
                finding.score
                for finding in state.analyzer_findings
                if finding.analyzer == AnalyzerName.DUE_IMPACT
            ),
            default=0.0,
        )

        ml_scores, ml_evidence = self._extract_ml_confirmation(
            state.evidence.ml_cause_detail_json
        )

        findings_by_cause: dict[
            DelayCauseType,
            list[AnalyzerFinding],
        ] = {}

        for finding in state.analyzer_findings:
            if not finding.detected:
                continue

            if finding.cause_type is None:
                continue

            findings_by_cause.setdefault(
                finding.cause_type,
                [],
            ).append(finding)

        confidence_factor = self.CONFIDENCE_FACTOR[
            state.confidence_level
        ]

        candidates: list[_CauseCandidate] = []

        for cause_type, findings in findings_by_cause.items():
            base_score = max(
                finding.score
                for finding in findings
            )

            ml_score = ml_scores.get(cause_type, 0.0)

            final_score = clamp(
                (
                    base_score * 0.65
                    + due_impact_score * 0.15
                    + probability * 0.10
                    + ml_score * 0.10
                )
                * confidence_factor
            )

            evidence = unique_texts(
                [
                    *[
                        item
                        for finding in findings
                        for item in finding.evidence
                    ],
                    *ml_evidence.get(cause_type, []),
                ]
            )

            reason_parts = unique_texts(
                [
                    finding.summary
                    for finding in findings
                    if finding.summary
                ]
            )

            candidates.append(
                _CauseCandidate(
                    cause_type=cause_type,
                    score=final_score,
                    reason=" ".join(reason_parts),
                    evidence=evidence,
                )
            )

        # Analyzer가 직접 원인을 탐지하지 못했을 때,
        # ML SHAP 결과를 보조 원인 후보로 사용합니다.
        if not candidates:
            candidates.extend(
                self._build_ml_fallback_candidates(
                    ml_scores=ml_scores,
                    ml_evidence=ml_evidence,
                    due_impact_score=due_impact_score,
                    probability=probability,
                    confidence_factor=confidence_factor,
                )
            )

        candidates = [
            candidate
            for candidate in candidates
            if candidate.score >= self.MIN_FINAL_SCORE
        ]

        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.cause_type.value,
            )
        )

        selected = candidates[: self.MAX_CAUSES]

        if not selected:
            raise ValueError(
                "지연 원인 후보를 산정하지 못했습니다."
            )

        ranked_causes = [
            RankedCause(
                cause_type=candidate.cause_type,
                rank=index,
                score=round(candidate.score, 4),
                reason=candidate.reason,
                evidence=candidate.evidence,
            )
            for index, candidate in enumerate(
                selected,
                start=1,
            )
        ]

        return state.model_copy(
            update={
                "status": WorkflowStatus.RANKED,
                "ranked_causes": ranked_causes,
                "selected_cause_types": [
                    cause.cause_type
                    for cause in ranked_causes
                ],
            }
        )

    def _extract_ml_confirmation(
        self,
        raw_json: str | None,
    ) -> tuple[
        dict[DelayCauseType, float],
        dict[DelayCauseType, list[str]],
    ]:
        scores: dict[DelayCauseType, float] = {}
        evidence: dict[DelayCauseType, list[str]] = {}

        for factor in parse_ml_factors(raw_json):
            direction = str(
                factor.get("direction") or ""
            ).lower()

            if direction == "decrease":
                continue

            cause_tag = str(
                factor.get("cause_tag") or ""
            ).upper()

            cause_type = self.SHAP_TAG_TO_CAUSE.get(
                cause_tag
            )

            if cause_type is None:
                continue

            impact = as_float(
                factor.get("abs_impact"),
                abs(as_float(factor.get("impact"))),
            )

            normalized_score = clamp(impact / 2.0)

            scores[cause_type] = max(
                scores.get(cause_type, 0.0),
                normalized_score,
            )

            feature_name = (
                factor.get("feature_name_ko")
                or factor.get("feature")
                or cause_tag
            )
            feature_value = factor.get("feature_value")
            shap_impact = factor.get("impact")

            evidence.setdefault(
                cause_type,
                [],
            ).append(
                f"ML SHAP: {feature_name}={feature_value}, "
                f"impact={shap_impact}"
            )

        return scores, evidence

    def _build_ml_fallback_candidates(
        self,
        *,
        ml_scores: dict[DelayCauseType, float],
        ml_evidence: dict[DelayCauseType, list[str]],
        due_impact_score: float,
        probability: float,
        confidence_factor: float,
    ) -> list[_CauseCandidate]:
        candidates: list[_CauseCandidate] = []

        for cause_type, ml_score in ml_scores.items():
            if ml_score <= 0:
                continue

            final_score = clamp(
                (
                    ml_score * 0.55
                    + due_impact_score * 0.25
                    + probability * 0.20
                )
                * confidence_factor
            )

            candidates.append(
                _CauseCandidate(
                    cause_type=cause_type,
                    score=final_score,
                    reason=(
                        "Analyzer의 직접 탐지는 없었으나 "
                        "ML SHAP 결과에서 "
                        f"{self.CAUSE_LABEL[cause_type]} 관련 요인이 "
                        "지연 위험을 증가시키는 방향으로 확인되었습니다."
                    ),
                    evidence=unique_texts(
                        ml_evidence.get(cause_type, [])
                    ),
                )
            )

        return candidates
