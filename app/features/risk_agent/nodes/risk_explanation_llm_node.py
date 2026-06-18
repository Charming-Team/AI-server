from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from app.core.config import Settings
from app.features.chat.grounded_prompt_builder import GroundedPrompt
from app.features.chat.llm_client import LlmClient
from app.features.risk_agent.nodes.analyzer_utils import (
    as_float,
    parse_ml_factors,
)

from app.features.risk_agent.schemas.common import (
    AnalyzerName,
    DelayCauseType,
    RiskLevel,
    WorkflowStatus,
)

from app.features.risk_agent.schemas.explanation import (
    RiskExplanationDraft,
)
from app.features.risk_agent.schemas.state import (
    RiskAgentWorkflowState,
)



class RiskExplanationGenerationError(RuntimeError):
    pass


class RiskExplanationLlmNode:
    SYSTEM_PROMPT = """
너는 석유화학 생산계획의 납기 지연 원인을 설명하는 AI Risk Analysis Agent다.

반드시 제공된 주문, 생산계획, 자재, 수율, 설비, 라인, Analyzer 결과,
지연 확률 모델 결과 및 ML SHAP 근거만 사용한다.

규칙:
1. 근거에 없는 원인이나 수치를 추측하지 않는다.
2. 최종 원인은 selectedCauseTypes에 포함된 원인만 사용한다.
3. delayProbability와 predictedDelayDays를 서로 다른 의미로 구분한다.
4. 지연 확률이 높아도 예상 지연 일수는 작을 수 있음을 고려한다.
5. analysisSummary에는 위험 단계, 지연 확률, 예상 지연 일수,
   핵심 원인과 생산 흐름 영향을 포함한다.
6. recommendedAction에는 현장에서 실행 가능한 구체적 조치를 작성한다.
7. 자재, 수율, 설비, 라인 데이터가 누락된 경우 해당 원인을 단정하지 않는다.
8. missingFields가 있으면 추가 확인이 필요한 데이터가 있다는 점을 자연스럽게 알린다.
9. Markdown, 코드 블록, 제목, 불필요한 인사말을 사용하지 않는다.
10. 반드시 지정된 JSON 객체 하나만 반환한다.
11. CRITICAL, WARNING, CAUTION, SAFE 같은 내부 코드는 각각 매우 위험, 위험, 주의, 안전으로 변환한다.
12. LINE_ABNORMAL 등 원인 enum과 rankedCauses, selectedCauseTypes 같은 내부 필드명을 출력하지 않는다.
13. missingFields에 포함된 영역은 원인이나 조치로 단정하지 않고, 추가 확인이 필요하다고만 표현한다.
14. 권고 조치는 selectedCauseTypes 및 실제 탐지된 Analyzer 근거와 직접 연결된 항목만 생성한다.

출력 JSON:
{
  "analysisSummary": "분석 요약",
  "recommendedAction": "권고 조치"
}
""".strip()
    RISK_LEVEL_LABELS = {
        RiskLevel.SAFE: "안전",
        RiskLevel.CAUTION: "주의",
        RiskLevel.WARNING: "위험",
        RiskLevel.CRITICAL: "매우 위험",
    }

    CAUSE_LABELS = {
        DelayCauseType.MATERIAL_SHORTAGE: "자재 부족",
        DelayCauseType.MATERIAL_DELAY: "자재 입고 지연",
        DelayCauseType.LOW_YIELD: "수율 저하",
        DelayCauseType.MACHINE_ABNORMAL: "설비 상태 이상",
        DelayCauseType.LINE_ABNORMAL: "라인 상태 이상",
    }

    ANALYZER_LABELS = {
        AnalyzerName.MATERIAL: "자재 분석",
        AnalyzerName.YIELD: "수율 분석",
        AnalyzerName.MACHINE: "설비 분석",
        AnalyzerName.LINE_PROCESS: "라인 및 공정 분석",
        AnalyzerName.DUE_IMPACT: "납기 영향 분석",
    }

    ACTION_GUIDES = {
        DelayCauseType.MATERIAL_SHORTAGE: [
            "부족 자재의 가용 재고와 예약 수량 확인",
            "후순위 생산계획의 자재 예약 수량 조정 검토",
            "생산 시작 시점 또는 분할 생산 검토",
        ],
        DelayCauseType.MATERIAL_DELAY: [
            "자재 조기 입고 가능 여부 확인",
            "입고 전 선행 가능한 주문 배치",
            "생산 시작 시점 재조정",
        ],
        DelayCauseType.LOW_YIELD: [
            "최근 생산 수율과 불량 실적 확인",
            "예상 손실분을 반영한 계획 수량 보정",
            "수율이 높은 대체 라인 검토",
        ],
        DelayCauseType.MACHINE_ABNORMAL: [
            "이상 설비의 복구 예상 시간 확인",
            "대체 설비 또는 대체 라인 검토",
            "복구 일정에 맞춘 생산 순서 재조정",
        ],
        DelayCauseType.LINE_ABNORMAL: [
            "동일 라인의 생산 순서 조정",
            "납기 우선순위가 낮은 계획의 후순위 이동 검토",
            "대체 생산 라인 투입 가능 여부 확인",
            "라인 대기시간과 가동률 모니터링",
        ],
    }

    def __init__(
        self,
        *,
        settings: Settings,
        llm_client: LlmClient,
    ) -> None:
        self.settings = settings
        self.llm_client = llm_client

    async def run(
        self,
        state: RiskAgentWorkflowState,
    ) -> RiskAgentWorkflowState:
        if state.status != WorkflowStatus.RANKED:
            raise ValueError(
                "Risk Explanation 생성은 RANKED 상태에서만 실행할 수 있습니다."
            )

        if state.evidence is None:
            raise ValueError(
                "Risk Explanation 생성에 필요한 evidence가 없습니다."
            )

        if not state.ranked_causes:
            raise ValueError(
                "Risk Explanation 생성에 필요한 rankedCauses가 없습니다."
            )

        if not self.settings.llm_enabled:
            raise RiskExplanationGenerationError(
                "LLM_ENABLED가 false이므로 Risk Explanation을 생성할 수 없습니다."
            )

        max_retries = max(
            int(self.settings.risk_agent_max_retries),
            0,
        )

        last_error: Exception | None = None

        for retry_index in range(max_retries + 1):
            try:
                prompt = self._build_prompt(
                    state=state,
                    retry_index=retry_index,
                )

                raw_response = await self.llm_client.generate(prompt)

                draft = self._parse_response(raw_response)

                self._validate_draft(draft, state)

                return state.model_copy(
                    update={
                        "status": WorkflowStatus.GENERATED,
                        "analysis_summary": draft.analysis_summary,
                        "recommended_action": draft.recommended_action,
                        "retry_count": (
                            state.retry_count + retry_index
                        ),
                    }
                )

            except Exception as exc:
                last_error = exc

        raise RiskExplanationGenerationError(
            "Risk Explanation LLM 응답 생성 또는 파싱에 실패했습니다. "
            f"attempts={max_retries + 1}, "
            f"reason={last_error}"
        ) from last_error

    def _build_prompt(
        self,
        *,
        state: RiskAgentWorkflowState,
        retry_index: int,
    ) -> GroundedPrompt:
        context = self._build_context(state)

        context_json = json.dumps(
            context,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )

        retry_instruction = ""

        if retry_index > 0:
            retry_instruction = """
이전 응답이 JSON 형식 또는 필수 필드 검증을 통과하지 못했습니다.
이번에는 설명을 추가하지 말고 정확한 JSON 객체만 반환하세요.
""".strip()
        validation_feedback = ""

        if state.validation_errors:
            validation_feedback = (
                "\n\n이전 Validation 실패 사유:\n- "
                + "\n- ".join(state.validation_errors)
                + "\n위 항목을 모두 수정하여 다시 생성하세요."
            )

        user_prompt = f"""
다음 근거를 이용하여 납기 지연 원인 분석 요약과 권고 조치를 생성하세요.

근거:
{context_json}

작성 규칙:
- analysisSummary는 3~5문장으로 작성합니다.
- 주문번호, 현재 위험 단계, 지연 확률, 예상 지연 일수를 포함합니다.
- rankedCauses의 1순위 원인을 중심으로 작성합니다.
- 복합 원인이면 최대 3개 원인의 상호 영향을 설명합니다.
- Analyzer가 탐지하지 않은 원인은 새로 만들지 않습니다.
- recommendedAction은 각 원인에 대한 조치 내용을 "1) ... 2) ... 3) ..."처럼 실행 순서가 드러나게 작성합니다.
- context의 allowedActions에 포함된 조치만 권고합니다.
- allowedActions에 없는 자재, 재고, 수율, 설비, 공급망 조치를 새로 만들지 않습니다.
- missingFields에 포함된 영역은 원인 또는 조치로 사용하지 않고 추가 확인 필요로만 표현합니다.
- missingFields에 포함된 영역은 문제가 없다고 결론 내리지 말고, 근거 부족으로 판단을 유보한다고 표현합니다.
- missingFields, materialEvidence 같은 내부 필드명은 출력하지 않고 자연어로 표현합니다.
- CRITICAL, LINE_ABNORMAL 같은 내부 코드와 JSON 필드명은 출력하지 않습니다.
- 원천 시스템명, JSON, 내부 API, SHAP이라는 기술 용어를
  사용자 화면용 문구에 직접 노출하지 않습니다.
- selectedCauseTypes 외의 원인 유형을 추가하지 않습니다.
- JSON 외의 문장은 반환하지 않습니다.

반환 형식:
{{
  "analysisSummary": "...",
  "recommendedAction": "..."
}}

{retry_instruction}
{validation_feedback}
""".strip()

        return GroundedPrompt(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    def _build_context(
        self,
        state: RiskAgentWorkflowState,
    ) -> dict[str, Any]:
        evidence = state.evidence

        if evidence is None:
            raise ValueError("evidence가 없습니다.")

        material_shortage_count = sum(
            1
            for material in evidence.materials
            if as_float(material.shortage_quantity) > 0
        )

        delayed_material_count = sum(
            1
            for material in evidence.materials
            if (
                material.expected_inbound_at is not None
                and evidence.planned_start_at is not None
                and material.expected_inbound_at
                > evidence.planned_start_at
            )
        )

        abnormal_machine_statuses = {
            "ERROR",
            "STOPPED",
            "MAINTENANCE",
        }

        abnormal_machine_count = sum(
            1
            for machine in evidence.machines
            if str(machine.operation_status or "").upper()
            in abnormal_machine_statuses
        )

        ml_factors = []

        for factor in parse_ml_factors(
            evidence.ml_cause_detail_json
        )[:5]:
            ml_factors.append(
                {
                    "causeTag": factor.get("cause_tag"),
                    "featureName": (
                        factor.get("feature_name_ko")
                        or factor.get("feature")
                    ),
                    "featureValue": factor.get("feature_value"),
                    "impact": factor.get("impact"),
                    "direction": factor.get("direction"),
                }
            )

        analyzer_findings = [
            {
                "analysisArea": self.ANALYZER_LABELS[finding.analyzer],
                "detected": finding.detected,
                "causeLabel": (
                    self.CAUSE_LABELS[finding.cause_type]
                    if finding.cause_type is not None
                    else None
                ),
                "score": round(finding.score, 4),
                "summary": finding.summary,
                "reasoning": finding.reasoning,
                "evidence": [
                    self._truncate_text(item, 180)
                    for item in finding.evidence[:3]
                ],
                "missingFields": finding.missing_fields,
            }
            for finding in state.analyzer_findings
        ]

        ranked_causes = [
            {
                "causeLabel": self.CAUSE_LABELS[cause.cause_type],
                "rank": cause.rank,
                "score": round(cause.score, 4),
                "reason": cause.reason,
                "evidence": [
                    self._truncate_text(item, 180)
                    for item in cause.evidence[:4]
                ],
            }
            for cause in state.ranked_causes
        ]

        selected_cause_labels = [
            self.CAUSE_LABELS[cause_type]
            for cause_type in state.selected_cause_types
        ]

        allowed_actions = list(
            dict.fromkeys(
                action
                for cause_type in state.selected_cause_types
                for action in self.ACTION_GUIDES[cause_type]
            )
        )

        return {
            "workflowRunId": state.workflow_run_id,
            "order": {
                "orderId": evidence.order_id,
                "orderNo": evidence.order_no,
                "customerName": evidence.customer_name,
                "productName": evidence.product_name,
                "orderQuantity": evidence.order_quantity,
                "completedQuantity": evidence.completed_quantity,
                "remainingQuantity": evidence.remaining_quantity,
                "progressRatePercent": as_float(
                    evidence.progress_rate
                ),
                "dueDate": evidence.due_date.isoformat(),
                "daysUntilDue": evidence.days_until_due,
            },
            "prediction": {
                "riskLevelLabel": (
                    self.RISK_LEVEL_LABELS[state.risk_level]
                    if state.risk_level is not None
                    else "확인 필요"
                ),
                "delayProbabilityPercent": round(
                    as_float(state.delay_probability) * 100,
                    2,
                ),
                "predictedDelayDays": as_float(
                    state.predicted_delay_days
                ),
                "mlFactors": ml_factors,
            },
            "productionPlan": {
                "planId": evidence.plan_id,
                "planStatus": evidence.plan_status,
                "plannedStartAt": self._isoformat(
                    evidence.planned_start_at
                ),
                "plannedEndAt": self._isoformat(
                    evidence.planned_end_at
                ),
                "plannedQuantity": evidence.planned_quantity,
                "estimatedDurationHr": as_float(
                    evidence.estimated_duration_hr
                ),
                "planSequence": evidence.plan_sequence,
            },
            "line": {
                "lineId": evidence.line_id,
                "lineName": evidence.line_name,
                "operationStatus": evidence.line_operation_status,
                "loadRatio": as_float(
                    evidence.line_load_ratio
                ),
                "utilizationRate": as_float(
                    evidence.line_utilization_rate
                ),
                "waitingTimeHr": as_float(
                    evidence.line_waiting_time_hr
                ),
                "waitingQuantity": (
                    evidence.line_waiting_quantity
                ),
                "competingOrderCount": len(
                    evidence.competing_orders
                ),
            },
            "yield": {
                "actualYieldRate": (
                    as_float(evidence.actual_yield_rate)
                    if evidence.actual_yield_rate is not None
                    else None
                ),
                "lineYieldRate": (
                    as_float(evidence.line_yield_rate)
                    if evidence.line_yield_rate is not None
                    else None
                ),
                "defectQuantity": evidence.defect_quantity,
            },
            "material": {
                "materialCount": len(evidence.materials),
                "shortageMaterialCount": (
                    material_shortage_count
                ),
                "delayedMaterialCount": (
                    delayed_material_count
                ),
            },
            "machine": {
                "machineCount": len(evidence.machines),
                "abnormalMachineCount": (
                    abnormal_machine_count
                ),
            },
            "impact": {
                "contractAmount": (
                    str(evidence.contract_amount)
                    if evidence.contract_amount is not None
                    else None
                ),
                "latePenaltyAmount": (
                    str(evidence.late_penalty_amount)
                    if evidence.late_penalty_amount is not None
                    else None
                ),
            },
            "analyzerFindings": analyzer_findings,
            "rankedCauses": ranked_causes,
            "selectedCauseLabels": selected_cause_labels,
            "allowedActions": allowed_actions,
            "confidenceLevel": state.confidence_level.value,
            "missingFields": state.missing_fields,
        }

    def _parse_response(
        self,
        raw_response: str,
    ) -> RiskExplanationDraft:
        if not raw_response or not raw_response.strip():
            raise RiskExplanationGenerationError(
                "LLM 응답이 비어 있습니다."
            )

        cleaned = raw_response.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        start_index = cleaned.find("{")
        end_index = cleaned.rfind("}")

        if start_index < 0 or end_index < start_index:
            raise RiskExplanationGenerationError(
                "LLM 응답에서 JSON 객체를 찾을 수 없습니다."
            )

        json_text = cleaned[start_index : end_index + 1]

        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise RiskExplanationGenerationError(
                "LLM 응답 JSON 파싱에 실패했습니다."
            ) from exc

        if not isinstance(payload, dict):
            raise RiskExplanationGenerationError(
                "LLM 응답은 JSON 객체여야 합니다."
            )

        try:
            return RiskExplanationDraft.model_validate(payload)
        except ValidationError as exc:
            raise RiskExplanationGenerationError(
                "LLM 응답이 Risk Explanation 스키마와 일치하지 않습니다."
            ) from exc

    @staticmethod
    def _truncate_text(
        value: str,
        max_length: int,
    ) -> str:
        normalized = " ".join(value.split())

        if len(normalized) <= max_length:
            return normalized

        return normalized[: max_length - 3] + "..."

    @staticmethod
    def _isoformat(value: object) -> str | None:
        if value is None:
            return None

        isoformat = getattr(value, "isoformat", None)

        if callable(isoformat):
            return str(isoformat())

        return str(value)
    
    def _validate_draft(
        self,
        draft: RiskExplanationDraft,
        state: RiskAgentWorkflowState,
    ) -> None:
        combined_text = (
            draft.analysis_summary
            + " "
            + draft.recommended_action
        )

        forbidden_terms = (
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
            "materialEvidence",
            "forbidden_terms"
        )

        exposed_terms = [
            term
            for term in forbidden_terms
            if term in combined_text
        ]

        if exposed_terms:
            raise RiskExplanationGenerationError(
                "사용자 문구에 내부 코드가 포함되었습니다: "
                + ", ".join(exposed_terms)
            )

        selected = set(state.selected_cause_types)
        recommendation = draft.recommended_action

        unsupported_keywords: list[str] = []

        if not {
            DelayCauseType.MATERIAL_SHORTAGE,
            DelayCauseType.MATERIAL_DELAY,
        }.intersection(selected):
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
                if keyword in recommendation
            )

        if DelayCauseType.LOW_YIELD not in selected:
            unsupported_keywords.extend(
                keyword
                for keyword in (
                    "수율",
                    "불량",
                )
                if keyword in recommendation
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
                if keyword in recommendation
            )

        if DelayCauseType.LINE_ABNORMAL not in selected:
            unsupported_keywords.extend(
                keyword
                for keyword in (
                    "라인",
                    "대기시간",
                    "생산 순서",
                    "라인 부하",
                )
                if keyword in recommendation
            )

        if unsupported_keywords:
            raise RiskExplanationGenerationError(
                "선정 원인과 연결되지 않은 권고가 포함되었습니다: "
                + ", ".join(dict.fromkeys(unsupported_keywords))
            )