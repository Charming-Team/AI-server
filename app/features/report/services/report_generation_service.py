from decimal import Decimal
from typing import Any

from app.features.report.agents.rdb_data_collection_agent import RdbDataCollectionAgent
from app.features.report.schemas.request import ReportGenerateRequest
from app.features.report.schemas.response import (
    EvidenceType,
    ReportEvidence,
    ReportGenerateResponse,
    ReportStatus,
    ReportValidationResult,
)
from app.features.report.schemas.state import ReportAgentState


class ReportGenerationService:
    def __init__(self) -> None:
        self.rdb_data_collection_agent = RdbDataCollectionAgent()

    def generate_report(self, request: ReportGenerateRequest) -> ReportGenerateResponse:
        state = ReportAgentState(request=request)

        try:
            state = self.rdb_data_collection_agent.run(state)
            return self._build_response_from_state(state)

        except Exception as error:
            validation = ReportValidationResult(
                requiredSectionIncluded=False,
                groundednessPassed=False,
                missingFields=["rdb_data"],
            )

            return ReportGenerateResponse(
                reportJobId=request.report_job_id,
                status=ReportStatus.FAILED,
                title=None,
                reportType=request.report_type.value,
                markdown=None,
                sections=None,
                evidence=[],
                validation=validation,
                errorMessage=str(error),
            )

    def _build_response_from_state(
        self,
        state: ReportAgentState,
    ) -> ReportGenerateResponse:
        request = state.request
        raw_data = state.raw_data

        period_text = f"{request.period.start_date} ~ {request.period.end_date}"
        title = self._build_title(request)

        sections = self._build_sections(period_text=period_text, raw_data=raw_data)
        markdown = self._build_markdown(
            title=title,
            period_text=period_text,
            sections=sections,
        )

        evidence = [
            ReportEvidence(
                type=EvidenceType.RDB,
                source="customer_orders",
                description="보고서 기간과 겹치는 주문 데이터 집계 기준",
            ),
            ReportEvidence(
                type=EvidenceType.RDB,
                source="production_plans",
                description="보고서 기간과 겹치는 생산계획 데이터 집계 기준",
            ),
            ReportEvidence(
                type=EvidenceType.RDB,
                source="production_results",
                description="보고서 기간과 겹치는 생산실적 데이터 집계 기준",
            ),
            ReportEvidence(
                type=EvidenceType.RDB,
                source="material_inventories",
                description="현재 자재 재고 및 안전 재고 기준",
            ),
            ReportEvidence(
                type=EvidenceType.RDB,
                source="ai_prediction_results",
                description="보고서 기간 내 AI 지연 예측 결과 기준",
            ),
            ReportEvidence(
                type=EvidenceType.RDB,
                source="line_status, machine_statuses",
                description="보고서 기간 내 라인 및 설비 상태 기록 기준",
            ),
        ]

        validation = ReportValidationResult(
            requiredSectionIncluded=True,
            groundednessPassed=True,
            missingFields=[],
        )

        return ReportGenerateResponse(
            reportJobId=request.report_job_id,
            status=ReportStatus.COMPLETED,
            title=title,
            reportType=request.report_type.value,
            markdown=markdown,
            sections=sections,
            evidence=evidence,
            validation=validation,
            errorMessage=None,
        )

    def _build_sections(
        self,
        period_text: str,
        raw_data: dict[str, Any],
    ) -> dict[str, Any]:
        order_summary = raw_data.get("order_summary", {})
        production_plan_summary = raw_data.get("production_plan_summary", {})
        production_result_summary = raw_data.get("production_result_summary", {})
        material_summary = raw_data.get("material_summary", {})
        plan_material_summary = raw_data.get("plan_material_summary", {})
        risk_summary = raw_data.get("risk_summary", {})
        line_summary = raw_data.get("line_summary", {})
        machine_summary = raw_data.get("machine_summary", {})

        top_risk_orders = raw_data.get("top_risk_orders", [])
        top_material_shortages = raw_data.get("top_material_shortages", [])
        top_line_statuses = raw_data.get("top_line_statuses", [])
        top_machine_statuses = raw_data.get("top_machine_statuses", [])

        total_planned_quantity = self._to_float(
            production_plan_summary.get("total_planned_quantity", 0)
        )
        total_completed_quantity = self._to_float(
            production_result_summary.get("total_actual_quantity", 0)
        )
        total_defect_quantity = self._to_float(
            production_result_summary.get("total_defect_quantity", 0)
        )

        achievement_rate = 0.0
        if total_planned_quantity > 0:
            achievement_rate = round(
                total_completed_quantity / total_planned_quantity * 100,
                2,
            )

        defect_rate = 0.0
        if total_completed_quantity > 0:
            defect_rate = round(
                total_defect_quantity / total_completed_quantity * 100,
                2,
            )

        summary = {
            "period": period_text,
            "totalOrderCount": self._to_int(order_summary.get("total_order_count", 0)),
            "totalOrderQuantity": self._to_float(order_summary.get("total_order_quantity", 0)),
            "dueOrderCount": self._to_int(order_summary.get("due_order_count", 0)),
            "delayedOrderCount": self._to_int(order_summary.get("delayed_order_count", 0)),
            "totalPlanCount": self._to_int(production_plan_summary.get("total_plan_count", 0)),
            "totalPlannedQuantity": total_planned_quantity,
            "totalCompletedQuantity": total_completed_quantity,
            "achievementRate": achievement_rate,
            "defectQuantity": total_defect_quantity,
            "defectRate": defect_rate,
            "avgYieldRate": round(
                self._to_float(production_result_summary.get("avg_yield_rate", 0)) * 100,
                2,
            ),
            "totalDelayHours": self._to_float(
                production_result_summary.get("total_actual_delay_hr", 0)
            ),
            "delayRiskOrderCount": self._to_int(
                risk_summary.get("delay_risk_order_count", 0)
            ),
            "criticalRiskCount": self._to_int(risk_summary.get("critical_risk_count", 0)),
            "warningRiskCount": self._to_int(risk_summary.get("warning_risk_count", 0)),
            "avgDelayProbability": round(
                self._to_float(risk_summary.get("avg_delay_probability", 0)) * 100,
                2,
            ),
            "avgPredictedDelayDays": round(
                self._to_float(risk_summary.get("avg_predicted_delay_days", 0)),
                2,
            ),
            "materialRiskCount": self._to_int(material_summary.get("risk_material_count", 0)),
            "safetyStockShortageCount": self._to_int(
                material_summary.get("safety_stock_shortage_count", 0)
            ),
            "shortagePlanMaterialCount": self._to_int(
                plan_material_summary.get("shortage_plan_material_count", 0)
            ),
            "totalShortageQuantity": self._to_float(
                plan_material_summary.get("total_shortage_quantity", 0)
            ),
            "avgLineUtilizationRate": round(
                self._to_float(line_summary.get("avg_line_utilization_rate", 0)) * 100,
                2,
            ),
            "avgLineProgressRate": round(
                self._to_float(line_summary.get("avg_line_progress_rate", 0)) * 100,
                2,
            ),
            "abnormalMachineStatusCount": self._to_int(
                machine_summary.get("abnormal_machine_status_count", 0)
            ),
        }

        normalized_top_risk_orders = [
            self._normalize_row(row) for row in top_risk_orders
        ]
        normalized_top_material_shortages = [
            self._normalize_row(row) for row in top_material_shortages
        ]
        normalized_top_line_statuses = [
            self._normalize_row(row) for row in top_line_statuses
        ]
        normalized_top_machine_statuses = [
            self._normalize_row(row) for row in top_machine_statuses
        ]

        executive_summary = self._build_executive_summary(summary)

        return {
            "summary": summary,
            "linePerformance": {
                "observedLineCount": self._to_int(line_summary.get("observed_line_count", 0)),
                "avgLineUtilizationRate": summary["avgLineUtilizationRate"],
                "avgLineProgressRate": summary["avgLineProgressRate"],
                "avgWaitingTimeHour": round(
                    self._to_float(line_summary.get("avg_waiting_time_hr", 0)),
                    2,
                ),
                "totalLineProcessedQuantity": self._to_float(
                    line_summary.get("total_line_processed_quantity", 0)
                ),
                "totalLineDefectQuantity": self._to_float(
                    line_summary.get("total_line_defect_quantity", 0)
                ),
                "nonRunningLineStatusCount": self._to_int(
                    line_summary.get("non_running_line_status_count", 0)
                ),
            },
            "materialRisk": {
                "totalMaterialCount": self._to_int(material_summary.get("total_material_count", 0)),
                "riskMaterialCount": self._to_int(material_summary.get("risk_material_count", 0)),
                "safetyStockShortageCount": self._to_int(
                    material_summary.get("safety_stock_shortage_count", 0)
                ),
                "totalCurrentQuantity": self._to_float(
                    material_summary.get("total_current_quantity", 0)
                ),
                "totalAvailableQuantity": self._to_float(
                    material_summary.get("total_available_quantity", 0)
                ),
                "totalReservedQuantity": self._to_float(
                    material_summary.get("total_reserved_quantity", 0)
                ),
                "totalShortageQuantity": summary["totalShortageQuantity"],
            },
            "riskAnalysis": {
                "totalPredictionCount": self._to_int(
                    risk_summary.get("total_prediction_count", 0)
                ),
                "delayRiskOrderCount": summary["delayRiskOrderCount"],
                "criticalRiskCount": summary["criticalRiskCount"],
                "warningRiskCount": summary["warningRiskCount"],
                "avgDelayProbability": summary["avgDelayProbability"],
                "avgPredictedDelayDays": summary["avgPredictedDelayDays"],
            },
            "machineStatus": {
                "observedMachineCount": self._to_int(
                    machine_summary.get("observed_machine_count", 0)
                ),
                "totalMachineProcessedQuantity": self._to_float(
                    machine_summary.get("total_machine_processed_quantity", 0)
                ),
                "totalMachineDefectQuantity": self._to_float(
                    machine_summary.get("total_machine_defect_quantity", 0)
                ),
                "abnormalMachineStatusCount": summary["abnormalMachineStatusCount"],
            },

            "topRiskOrders": normalized_top_risk_orders,
            "topMaterialShortages": normalized_top_material_shortages,
            "topLineStatuses": normalized_top_line_statuses,
            "topMachineStatuses": normalized_top_machine_statuses,

            "executiveSummary": executive_summary,
            "issueHistory": {
                "riskIssues": normalized_top_risk_orders,
                "machineIssues": normalized_top_machine_statuses,
                "lineIssues": normalized_top_line_statuses,
            },
            "delayResponseAnalysis": {
                "orderImpact": normalized_top_risk_orders,
                "productionPlanImpact": normalized_top_line_statuses,
                "materialImpact": normalized_top_material_shortages,
            },
            "economicAnalysis": {
                "simulationResults": [],
                "bestScenario": None,
                "comment": "시뮬레이션 결과 연동 후 솔루션 적용 전후 비용 비교 분석을 제공합니다.",
            },
            "conclusion": {
                "priorityActions": executive_summary["keyFindings"],
                "finalComment": "납기 위험 주문, 자재 부족 계획, 비가동 라인 및 설비 상태를 우선 검토해야 합니다.",
            },
            "appendix": {
                "sources": [
                    "customer_orders",
                    "production_plans",
                    "production_results",
                    "ai_prediction_results",
                    "production_plan_materials",
                    "material_inventories",
                    "line_status",
                    "machine_statuses",
                ]
            },
            "recommendation": {
                "priority": "납기 위험 주문, 자재 부족 계획, 비가동 라인 및 설비 상태를 우선 검토해야 합니다."
            },
        }

    def _build_title(self, request: ReportGenerateRequest) -> str:
        if request.report_type.value == "MONTHLY":
            return f"{request.period.start_date.strftime('%Y년 %m월')} 생산 운영 보고서"

        return f"{request.period.start_date} ~ {request.period.end_date} 수시 생산 운영 보고서"
    
    def _build_executive_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        key_findings = []

        if summary["criticalRiskCount"] > 0:
            key_findings.append(
                f"CRITICAL 위험 주문 {summary['criticalRiskCount']}건이 확인되어 우선 대응이 필요합니다."
            )

        if summary["materialRiskCount"] > 0 or summary["safetyStockShortageCount"] > 0:
            key_findings.append(
                f"자재 위험 품목 {summary['materialRiskCount']}건, 안전 재고 미만 자재 {summary['safetyStockShortageCount']}건이 확인되었습니다."
            )

        if summary["abnormalMachineStatusCount"] > 0:
            key_findings.append(
                f"비정상 또는 확인 필요 설비 상태 {summary['abnormalMachineStatusCount']}건이 확인되었습니다."
            )

        if not key_findings:
            key_findings.append("보고서 기간 내 주요 고위험 이슈는 확인되지 않았습니다.")

        return {
            "period": summary["period"],
            "totalOrderCount": summary["totalOrderCount"],
            "totalPlanCount": summary["totalPlanCount"],
            "delayRiskOrderCount": summary["delayRiskOrderCount"],
            "criticalRiskCount": summary["criticalRiskCount"],
            "warningRiskCount": summary["warningRiskCount"],
            "materialRiskCount": summary["materialRiskCount"],
            "avgDelayProbability": summary["avgDelayProbability"],
            "avgPredictedDelayDays": summary["avgPredictedDelayDays"],
            "keyFindings": key_findings,
            "summaryMessage": "보고서 기간 동안 납기 위험, 자재 부족, 설비 상태를 중심으로 주요 운영 리스크가 확인되었습니다.",
        }

    def _build_markdown(
        self,
        title: str,
        period_text: str,
        sections: dict[str, Any],
    ) -> str:
        executive = sections["executiveSummary"]
        issue_history = sections["issueHistory"]
        delay_response = sections["delayResponseAnalysis"]
        economic = sections["economicAnalysis"]
        conclusion = sections["conclusion"]
        appendix = sections["appendix"]

        risk_issue_lines = self._build_top_risk_order_lines(
            issue_history.get("riskIssues", [])
        )
        machine_issue_lines = self._build_top_machine_status_lines(
            issue_history.get("machineIssues", [])
        )
        line_issue_lines = self._build_top_line_status_lines(
            issue_history.get("lineIssues", [])
        )
        material_impact_lines = self._build_top_material_shortage_lines(
            delay_response.get("materialImpact", [])
        )
        source_lines = self._build_source_lines(appendix.get("sources", []))
        key_finding_lines = self._build_key_finding_lines(
            executive.get("keyFindings", [])
        )
        priority_action_lines = self._build_key_finding_lines(
            conclusion.get("priorityActions", [])
        )

        return f"""# {title}

## 1. Executive Summary

- 보고서 기간: {period_text}
- 총 주문 수: {executive["totalOrderCount"]}건
- 총 생산계획 수: {executive["totalPlanCount"]}건
- 납기 위험 주문 수: {executive["delayRiskOrderCount"]}건
- CRITICAL 위험 수: {executive["criticalRiskCount"]}건
- WARNING 위험 수: {executive["warningRiskCount"]}건
- 자재 위험 품목 수: {executive["materialRiskCount"]}건
- 평균 지연 확률: {executive["avgDelayProbability"]}%
- 평균 예상 지연일: {executive["avgPredictedDelayDays"]}일

### 핵심 요약

{key_finding_lines}

## 2. 선택한 날짜 사이의 이슈 이력

### 2-1. 지연 예측 이슈

{risk_issue_lines}

### 2-2. 머신 이슈

{machine_issue_lines}

### 2-3. 라인 이슈

{line_issue_lines}

## 3. 지연 대응 Solution 분석

### 3-1. 주문 관점

{risk_issue_lines}

### 3-2. 생산계획 관점

{line_issue_lines}

### 3-3. 자재 관점

{material_impact_lines}

## 4. 경제성 분석

{economic["comment"]}

- 시뮬레이션 결과 수: {len(economic.get("simulationResults", []))}
- 최적 시나리오: {economic.get("bestScenario") or "시뮬레이션 결과 연동 필요"}

## 5. 결론

{priority_action_lines}

{conclusion["finalComment"]}

## 6. Appendix / 출처

{source_lines}
"""

    def _to_int(self, value: Any) -> int:
        if value is None:
            return 0
        return int(value)

    def _to_float(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, Decimal):
            return float(value)
        return float(value)
    
    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = {}

        for key, value in row.items():
            if isinstance(value, Decimal):
                normalized[key] = float(value)
            elif hasattr(value, "isoformat"):
                normalized[key] = value.isoformat()
            else:
                normalized[key] = value

        return normalized
    
    def _build_top_risk_order_lines(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "- 조회된 주요 납기 위험 주문이 없습니다."

        lines = []
        for index, row in enumerate(rows, start=1):
            cause = row.get("cause_detail") or row.get("analysis_summary") or "원인 확인 필요"
            action = row.get("recommended_action") or "추천 조치 확인 필요"
            evidence = (
                f"prediction_id={row.get('prediction_id')}, "
                f"predicted_at={row.get('predicted_at')}"
            )

            lines.append(
                f"- {index}. 위험 대상: 주문 {row.get('order_id')} / "
                f"고객사: {row.get('customer_name')} / "
                f"제품: {row.get('product_name')} / "
                f"위험도: {row.get('risk_level')} / "
                f"예상 지연 시간: {row.get('predicted_delay_days')}일 / "
                f"주요 원인: {cause} / "
                f"추천 조치: {action} / "
                f"근거 데이터: {evidence}"
            )

        return "\n".join(lines)

    def _build_top_material_shortage_lines(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "- 조회된 주요 자재 부족 계획이 없습니다."

        lines = []
        for index, row in enumerate(rows, start=1):
            lines.append(
                f"- {index}. 계획 {row.get('plan_id')} / "
                f"자재: {row.get('material_name')} / "
                f"필요 수량: {row.get('required_quantity')} / "
                f"예약 수량: {row.get('reserved_quantity')} / "
                f"부족 수량: {row.get('shortage_quantity')} / "
                f"상태: {row.get('material_plan_status')}"
            )

        return "\n".join(lines)

    def _build_top_line_status_lines(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "- 조회된 주요 라인 상태가 없습니다."

        lines = []
        for index, row in enumerate(rows, start=1):
            lines.append(
                f"- {index}. {row.get('line_code')} {row.get('line_name')} / "
                f"상태: {row.get('operation_status')} / "
                f"가동률: {row.get('utilization_rate')} / "
                f"진행률: {row.get('progress_rate')} / "
                f"대기 시간: {row.get('waiting_time_hr')}시간"
            )

        return "\n".join(lines)

    def _build_top_machine_status_lines(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "- 조회된 주요 설비 상태가 없습니다."

        lines = []
        for index, row in enumerate(rows, start=1):
            lines.append(
                f"- {index}. {row.get('machine_code')} {row.get('machine_name')} / "
                f"라인: {row.get('line_code')} / "
                f"상태: {row.get('operation_status')} / "
                f"처리 수량: {row.get('processed_quantity')} / "
                f"불량 수량: {row.get('defect_quantity')} / "
                f"비고: {row.get('status_note')}"
            )

        return "\n".join(lines)
    
    def _build_key_finding_lines(self, findings: list[str]) -> str:
        if not findings:
            return "- 주요 요약 내용이 없습니다."

        return "\n".join(f"- {finding}" for finding in findings)

    def _build_source_lines(self, sources: list[str]) -> str:
        if not sources:
            return "- 참조한 데이터 출처가 없습니다."

        return "\n".join(f"- {source}" for source in sources)