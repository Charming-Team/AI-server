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

        top_risk_orders = raw_data.get("top_risk_orders", [])
        top_material_shortages = raw_data.get("top_material_shortages", [])
        top_line_statuses = raw_data.get("top_line_statuses", [])
        top_machine_statuses = raw_data.get("top_machine_statuses", [])

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
            "recommendation": {
                "priority": "납기 위험 주문, 자재 부족 계획, 비가동 라인 및 설비 상태를 우선 검토해야 합니다."
            },

            "topRiskOrders": [
                self._normalize_row(row) for row in top_risk_orders
            ],
            "topMaterialShortages": [
                self._normalize_row(row) for row in top_material_shortages
            ],
            "topLineStatuses": [
                self._normalize_row(row) for row in top_line_statuses
            ],
            "topMachineStatuses": [
                self._normalize_row(row) for row in top_machine_statuses
            ],
        }

    def _build_title(self, request: ReportGenerateRequest) -> str:
        if request.report_type.value == "MONTHLY":
            return f"{request.period.start_date.strftime('%Y년 %m월')} 생산 운영 보고서"

        return f"{request.period.start_date} ~ {request.period.end_date} 수시 생산 운영 보고서"

    def _build_markdown(
        self,
        title: str,
        period_text: str,
        sections: dict[str, Any],
    ) -> str:
        summary = sections["summary"]
        line = sections["linePerformance"]
        material = sections["materialRisk"]
        risk = sections["riskAnalysis"]
        machine = sections["machineStatus"]

        top_risk_orders = sections.get("topRiskOrders", [])
        top_material_shortages = sections.get("topMaterialShortages", [])
        top_line_statuses = sections.get("topLineStatuses", [])
        top_machine_statuses = sections.get("topMachineStatuses", [])

        top_risk_order_lines = self._build_top_risk_order_lines(top_risk_orders)
        top_material_shortage_lines = self._build_top_material_shortage_lines(top_material_shortages)
        top_line_status_lines = self._build_top_line_status_lines(top_line_statuses)
        top_machine_status_lines = self._build_top_machine_status_lines(top_machine_statuses)

        return f"""# {title}

        
## 1. 주요 요약

- 보고서 기간: {period_text}
- 총 주문 수: {summary["totalOrderCount"]}건
- 총 주문 수량: {summary["totalOrderQuantity"]}
- 생산계획 건수: {summary["totalPlanCount"]}건
- 총 생산 계획 수량: {summary["totalPlannedQuantity"]}
- 총 생산 완료 수량: {summary["totalCompletedQuantity"]}
- 계획 대비 실적률: {summary["achievementRate"]}%
- 평균 수율: {summary["avgYieldRate"]}%
- 불량 수량: {summary["defectQuantity"]}
- 불량률: {summary["defectRate"]}%
- 총 지연 시간: {summary["totalDelayHours"]}시간
- 납기 위험 주문 수: {summary["delayRiskOrderCount"]}건
- 자재 위험 품목 수: {summary["materialRiskCount"]}건

## 2. 생산 실적 분석

보고서 기간 동안 총 {summary["totalPlanCount"]}건의 생산계획이 확인되었으며,
계획 수량 {summary["totalPlannedQuantity"]} 대비 실제 생산 수량은 {summary["totalCompletedQuantity"]}입니다.
계획 대비 실적률은 {summary["achievementRate"]}%입니다.

## 3. 라인별 성과

- 관측 라인 수: {line["observedLineCount"]}개
- 평균 라인 가동률: {line["avgLineUtilizationRate"]}%
- 평균 진행률: {line["avgLineProgressRate"]}%
- 평균 대기 시간: {line["avgWaitingTimeHour"]}시간
- 라인 처리 수량: {line["totalLineProcessedQuantity"]}
- 라인 불량 수량: {line["totalLineDefectQuantity"]}
- 비가동 또는 확인 필요 라인 상태 수: {line["nonRunningLineStatusCount"]}건

## 4. 자재 및 재고 리스크

- 전체 자재 수: {material["totalMaterialCount"]}건
- 위험 자재 수: {material["riskMaterialCount"]}건
- 안전 재고 미만 자재 수: {material["safetyStockShortageCount"]}건
- 현재 재고 총량: {material["totalCurrentQuantity"]}
- 가용 재고 총량: {material["totalAvailableQuantity"]}
- 예약 재고 총량: {material["totalReservedQuantity"]}
- 계획 기준 총 부족 수량: {material["totalShortageQuantity"]}

## 5. 리스크 분석

- 전체 예측 결과 수: {risk["totalPredictionCount"]}건
- 납기 위험 주문 수: {risk["delayRiskOrderCount"]}건
- CRITICAL 위험 수: {risk["criticalRiskCount"]}건
- WARNING 위험 수: {risk["warningRiskCount"]}건
- 평균 지연 확률: {risk["avgDelayProbability"]}%
- 평균 예상 지연일: {risk["avgPredictedDelayDays"]}일

## 6. 주요 설비 현황

- 관측 설비 수: {machine["observedMachineCount"]}개
- 설비 처리 수량: {machine["totalMachineProcessedQuantity"]}
- 설비 불량 수량: {machine["totalMachineDefectQuantity"]}
- 비정상 또는 확인 필요 설비 상태 수: {machine["abnormalMachineStatusCount"]}건

## 7. 주요 납기 위험 주문 TOP 5

{top_risk_order_lines}

## 8. 주요 자재 부족 계획 TOP 5

{top_material_shortage_lines}

## 9. 주요 라인 상태 TOP 5

{top_line_status_lines}

## 10. 주요 설비 상태 TOP 5

{top_machine_status_lines}

## 11. 종합 의견 및 제안

현재 보고서는 RDB 운영 데이터를 기반으로 생성된 1차 보고서입니다.
납기 위험 주문, 자재 부족 계획, 비가동 라인 및 설비 상태를 우선 검토해야 합니다.

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
    
    def _build_top_risk_order_lines(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "- 조회된 주요 납기 위험 주문이 없습니다."

        lines = []
        for index, row in enumerate(rows, start=1):
            lines.append(
                f"- {index}. 주문 {row.get('order_id')} / "
                f"고객사: {row.get('customer_name')} / "
                f"제품: {row.get('product_name')} / "
                f"위험도: {row.get('risk_level')} / "
                f"지연확률: {row.get('delay_probability')} / "
                f"예상 지연일: {row.get('predicted_delay_days')}"
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