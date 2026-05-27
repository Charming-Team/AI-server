from decimal import Decimal
from typing import Any

from app.features.report.agents.llm_report_writing_agent import LlmReportWritingAgent
from app.features.report.agents.rdb_data_collection_agent import RdbDataCollectionAgent
from app.features.report.builders.report_markdown_builder import ReportMarkdownBuilder
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
        self.markdown_builder = ReportMarkdownBuilder()
        self.llm_report_writing_agent = LlmReportWritingAgent()

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
        raw_data = state.raw_data or {}

        period_text = f"{request.period.start_date} ~ {request.period.end_date}"
        title = self._build_title(request)

        sections = self._build_sections(
            period_text=period_text,
            raw_data=raw_data,
        )

        base_markdown = self.markdown_builder.build(
            title=title,
            period_text=period_text,
            sections=sections,
        )

        markdown = self.llm_report_writing_agent.run(
            title=title,
            period_text=period_text,
            sections=sections,
            base_markdown=base_markdown,
        )

        evidence = self._build_evidence()

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

        economic_analysis = raw_data.get(
            "economic_analysis",
            {
                "simulationResults": [],
                "bestScenario": None,
                "comment": "선택 기간 내 조회 가능한 시뮬레이션 결과가 없습니다.",
            },
        )

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
            "totalOrderQuantity": self._to_float(
                order_summary.get("total_order_quantity", 0)
            ),
            "dueOrderCount": self._to_int(order_summary.get("due_order_count", 0)),
            "delayedOrderCount": self._to_int(
                order_summary.get("delayed_order_count", 0)
            ),
            "totalPlanCount": self._to_int(
                production_plan_summary.get("total_plan_count", 0)
            ),
            "totalPlannedQuantity": total_planned_quantity,
            "totalCompletedQuantity": total_completed_quantity,
            "achievementRate": achievement_rate,
            "defectQuantity": total_defect_quantity,
            "defectRate": defect_rate,
            "avgYieldRate": round(
                self._to_float(production_result_summary.get("avg_yield_rate", 0))
                * 100,
                2,
            ),
            "totalDelayHours": self._to_float(
                production_result_summary.get("total_actual_delay_hr", 0)
            ),
            "delayRiskOrderCount": self._to_int(
                risk_summary.get("delay_risk_order_count", 0)
            ),
            "criticalRiskCount": self._to_int(
                risk_summary.get("critical_risk_count", 0)
            ),
            "warningRiskCount": self._to_int(
                risk_summary.get("warning_risk_count", 0)
            ),
            "avgDelayProbability": round(
                self._to_float(risk_summary.get("avg_delay_probability", 0)) * 100,
                2,
            ),
            "avgPredictedDelayDays": round(
                self._to_float(risk_summary.get("avg_predicted_delay_days", 0)),
                2,
            ),
            "materialRiskCount": self._to_int(
                material_summary.get("risk_material_count", 0)
            ),
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
                self._to_float(line_summary.get("avg_line_utilization_rate", 0))
                * 100,
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

        executive_summary = self._build_executive_summary(
            summary=summary,
            economic_analysis=economic_analysis,
        )

        return {
            "summary": summary,
            "linePerformance": {
                "observedLineCount": self._to_int(
                    line_summary.get("observed_line_count", 0)
                ),
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
                "totalMaterialCount": self._to_int(
                    material_summary.get("total_material_count", 0)
                ),
                "riskMaterialCount": self._to_int(
                    material_summary.get("risk_material_count", 0)
                ),
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
            "economicAnalysis": economic_analysis,
            "conclusion": {
                "priorityActions": executive_summary["keyFindings"],
                "finalComment": self._build_final_comment(economic_analysis),
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
                    "schedule_simulation_results",
                    "schedule_simulation_details",
                ]
            },
            "recommendation": {
                "priority": self._build_final_comment(economic_analysis)
            },
        }

    def _build_title(self, request: ReportGenerateRequest) -> str:
        if request.report_type.value == "MONTHLY":
            return f"{request.period.start_date.strftime('%Y년 %m월')} 생산 운영 보고서"

        return (
            f"{request.period.start_date} ~ {request.period.end_date} "
            "수시 생산 운영 보고서"
        )

    def _build_executive_summary(
        self,
        *,
        summary: dict[str, Any],
        economic_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        key_findings = []

        if summary["criticalRiskCount"] > 0:
            key_findings.append(
                f"CRITICAL 위험 주문 {summary['criticalRiskCount']}건이 확인되어 우선 대응이 필요합니다."
            )

        if summary["materialRiskCount"] > 0 or summary["safetyStockShortageCount"] > 0:
            key_findings.append(
                f"자재 위험 품목 {summary['materialRiskCount']}건, "
                f"안전 재고 미만 자재 {summary['safetyStockShortageCount']}건이 확인되었습니다."
            )

        if summary["abnormalMachineStatusCount"] > 0:
            key_findings.append(
                f"비정상 또는 확인 필요 설비 상태 {summary['abnormalMachineStatusCount']}건이 확인되었습니다."
            )

        economic_key_finding = self._build_economic_key_finding(economic_analysis)
        if economic_key_finding:
            key_findings.append(economic_key_finding)

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

    def _build_economic_key_finding(
        self,
        economic_analysis: dict[str, Any],
    ) -> str | None:
        best_scenario = economic_analysis.get("bestScenario")

        if not best_scenario:
            return None

        simulation_name = best_scenario.get("simulationName", "추천 대응안")
        delay_reduction_hr = best_scenario.get("delayReductionHr")
        cost_change_amount = best_scenario.get("costChangeAmount")

        if delay_reduction_hr is None:
            return None

        if cost_change_amount is None:
            return (
                f"{simulation_name} 적용 시 총 지연 시간을 "
                f"{delay_reduction_hr}시간 감소시킬 수 있는 것으로 분석되었습니다."
            )

        return (
            f"{simulation_name} 적용 시 총 지연 시간을 "
            f"{delay_reduction_hr}시간 감소시킬 수 있으며, "
            f"비용 변화 금액은 {float(cost_change_amount):,.0f}원으로 분석되었습니다."
        )

    def _build_final_comment(
        self,
        economic_analysis: dict[str, Any],
    ) -> str:
        best_scenario = economic_analysis.get("bestScenario")

        if not best_scenario:
            return "납기 위험 주문, 자재 부족 계획, 비가동 라인 및 설비 상태를 우선 검토해야 합니다."

        simulation_name = best_scenario.get("simulationName", "추천 대응안")
        delay_reduction_hr = best_scenario.get("delayReductionHr")
        cost_change_amount = best_scenario.get("costChangeAmount")

        if delay_reduction_hr is None:
            return (
                f"{simulation_name}을 우선 검토하고, 납기 위험 주문과 자재 부족 계획을 함께 확인해야 합니다."
            )

        if cost_change_amount is None:
            return (
                f"{simulation_name}을 우선 검토해야 합니다. 해당 대응안은 총 지연 시간을 "
                f"{delay_reduction_hr}시간 감소시키는 것으로 분석되었습니다."
            )

        return (
            f"{simulation_name}을 우선 검토해야 합니다. 해당 대응안은 총 지연 시간을 "
            f"{delay_reduction_hr}시간 감소시키며, 비용 변화 금액은 "
            f"{float(cost_change_amount):,.0f}원으로 분석되었습니다."
        )

    def _build_evidence(self) -> list[ReportEvidence]:
        return [
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
            ReportEvidence(
                type=EvidenceType.RDB,
                source="schedule_simulation_results, schedule_simulation_details",
                description="보고서 기간 내 생산계획 대응안 시뮬레이션 결과 및 상세 변경 내역 기준",
            ),
        ]

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

    def _extract_related_simulation_id(
        self,
        sections: dict[str, Any],
    ) -> int | None:
        economic_analysis = sections.get("economicAnalysis", {})
        best_scenario = economic_analysis.get("bestScenario")

        if not best_scenario:
            return None

        simulation_id = best_scenario.get("simulationId")

        if simulation_id is None:
            return None

        return int(simulation_id)