from datetime import date
from typing import Any

from app.features.report.schemas.request import ReportGenerateRequest
from app.features.report.services.report_generation_service import ReportGenerationService


class FakeRdbDataCollectionAgent:
    def __init__(self, raw_data: dict[str, Any]) -> None:
        self.raw_data = raw_data

    def run(self, state: Any) -> Any:
        state.raw_data = self.raw_data
        return state


class FakeLlmReportWritingAgent:
    def __init__(self) -> None:
        self.sections: dict[str, Any] | None = None

    def run(
        self,
        *,
        title: str,
        period_text: str,
        sections: dict[str, Any],
        base_markdown: str,
    ) -> str:
        self.sections = sections
        return base_markdown


def test_report_generation_adds_frontend_table_sections_inside_sections() -> None:
    service = ReportGenerationService()
    service.rdb_data_collection_agent = FakeRdbDataCollectionAgent(_build_raw_data())
    fake_llm_report_writing_agent = FakeLlmReportWritingAgent()
    service.llm_report_writing_agent = fake_llm_report_writing_agent

    response = service.generate_report(
        ReportGenerateRequest(
            reportJobId=1,
            requestedBy=100,
            userRole="PRODUCTION_MANAGER",
            reportType="AD_HOC",
            period={
                "startDate": date(2026, 6, 1),
                "endDate": date(2026, 6, 14),
            },
        )
    )

    payload = response.model_dump(by_alias=True)
    sections = payload["sections"]

    assert "summaryRows" not in payload
    assert "lineRows" not in payload
    assert "equipmentRows" not in payload
    assert "analysis" not in payload
    assert fake_llm_report_writing_agent.sections is not None
    assert "summaryRows" not in fake_llm_report_writing_agent.sections
    assert payload["reportType"] == "ON_DEMAND"
    assert payload["markdown"].startswith("# 2026-06-01 ~ 2026-06-14")
    assert payload["evidence"]

    assert sections["summary"]["period"] == "2026-06-01 ~ 2026-06-14"
    assert sections["executiveSummary"]["totalOrderCount"] == 8
    assert sections["summaryRows"][:2] == [
        {
            "label": "보고서 기간",
            "value": "2026-06-01 ~ 2026-06-14",
            "change": "-",
        },
        {
            "label": "보고서 유형",
            "value": "수시",
            "change": "-",
        },
    ]
    assert sections["summaryRows"][2] == {
        "label": "총 주문 수",
        "value": "8",
        "change": "-",
    }

    assert sections["lineRows"] == [
        {
            "line": "PP-L1 PP 범용 생산 Line",
            "utilization": "91%",
            "completed": "12,000",
            "defectRate": "1.2%",
            "note": "정상",
        }
    ]
    assert sections["equipmentRows"] == [
        {
            "name": "EXT-1 압출기",
            "utilization": "확인 필요",
            "downTime": "확인 필요",
            "status": "정상",
        }
    ]
    assert sections["analysis"]["overview"] == (
        "보고서 기간 동안 납기 위험 주문 1건, 자재 위험 품목 1건, "
        "비정상 설비 상태 0건이 확인되었습니다."
    )
    assert sections["analysis"]["sections"][0] == {
        "title": "자재 부족 분석",
        "items": [
            "Foaming Agent 자재는 계획 3001에서 30 부족이 확인되었습니다.",
        ],
    }
    assert sections["analysis"]["recommendation"].startswith("납기 위험 주문")


def test_report_generation_keeps_frontend_table_rows_when_source_rows_are_empty() -> None:
    service = ReportGenerationService()
    service.rdb_data_collection_agent = FakeRdbDataCollectionAgent({})
    service.llm_report_writing_agent = FakeLlmReportWritingAgent()

    response = service.generate_report(
        ReportGenerateRequest(
            reportJobId=2,
            requestedBy=100,
            userRole="PRODUCTION_MANAGER",
            reportType="AD_HOC",
            period={
                "startDate": date(2026, 6, 1),
                "endDate": date(2026, 6, 1),
            },
        )
    )

    sections = response.model_dump(by_alias=True)["sections"]

    assert len(sections["summaryRows"]) >= 8
    assert [row["label"] for row in sections["summaryRows"]] == [
        "보고서 기간",
        "보고서 유형",
        "총 주문 수",
        "총 생산계획 수",
        "총 생산 계획 수량",
        "총 생산 완료 수량",
        "생산 계획 대비 실적",
        "라인 가동률",
        "불량 수량",
        "불량률",
        "납기 위험 주문 수",
        "자재 위험 품목 수",
        "비정상 설비 상태 수",
    ]
    assert sections["lineRows"] == [
        {
            "line": "확인 필요",
            "utilization": "-",
            "completed": "-",
            "defectRate": "-",
            "note": "확인 필요",
        }
    ]
    assert sections["equipmentRows"] == [
        {
            "name": "확인 필요",
            "utilization": "확인 필요",
            "downTime": "확인 필요",
            "status": "확인 필요",
        }
    ]
    assert sections["analysis"]["sections"] == [
        {
            "title": "종합 분석",
            "items": ["분석 내용이 없습니다."],
        }
    ]
    assert sections["analysis"]["overview"] == "분석 내용이 없습니다."
    assert sections["analysis"]["recommendation"] == "생성 필요"


def _build_raw_data() -> dict[str, Any]:
    return {
        "order_summary": {
            "total_order_count": 8,
            "total_order_quantity": 12000,
            "due_order_count": 2,
            "delayed_order_count": 1,
        },
        "production_plan_summary": {
            "total_plan_count": 4,
            "total_planned_quantity": 12000,
        },
        "production_result_summary": {
            "total_actual_quantity": 12000,
            "total_defect_quantity": 144,
            "avg_yield_rate": 0.988,
            "total_actual_delay_hr": 2.5,
        },
        "material_summary": {
            "total_material_count": 12,
            "risk_material_count": 1,
            "safety_stock_shortage_count": 1,
            "total_current_quantity": 1000,
            "total_available_quantity": 800,
            "total_reserved_quantity": 200,
        },
        "plan_material_summary": {
            "shortage_plan_material_count": 1,
            "total_shortage_quantity": 30,
        },
        "risk_summary": {
            "total_prediction_count": 3,
            "delay_risk_order_count": 1,
            "critical_risk_count": 0,
            "warning_risk_count": 1,
            "avg_delay_probability": 0.42,
            "avg_predicted_delay_days": 1.5,
        },
        "line_summary": {
            "observed_line_count": 1,
            "avg_line_utilization_rate": 0.91,
            "avg_line_progress_rate": 0.75,
            "avg_waiting_time_hr": 0,
            "total_line_processed_quantity": 12000,
            "total_line_defect_quantity": 144,
            "non_running_line_status_count": 0,
        },
        "machine_summary": {
            "observed_machine_count": 1,
            "total_machine_processed_quantity": 12000,
            "total_machine_defect_quantity": 144,
            "abnormal_machine_status_count": 0,
        },
        "top_risk_orders": [
            {
                "prediction_id": 10,
                "order_id": 2001,
                "customer_name": "Alpha",
                "product_name": "PP-FOAM",
                "order_quantity": 12000,
                "due_date": date(2026, 6, 14),
                "delay_probability": 0.42,
                "predicted_delay_days": 1.5,
                "risk_level": "WARNING",
                "analysis_summary": "자재 부족",
                "recommended_action": "입고 일정 확인",
                "predicted_at": date(2026, 6, 1),
            }
        ],
        "top_material_shortages": [
            {
                "plan_material_id": 7001,
                "plan_id": 3001,
                "order_id": 2001,
                "material_id": 5001,
                "material_name": "Foaming Agent",
                "required_quantity": 120,
                "reserved_quantity": 90,
                "shortage_quantity": 30,
                "material_plan_status": "SHORTAGE",
                "planned_start_at": date(2026, 6, 1),
                "planned_end_at": date(2026, 6, 14),
            }
        ],
        "top_line_statuses": [
            {
                "line_status_id": 1,
                "line_id": 1,
                "line_code": "PP-L1",
                "line_name": "PP 범용 생산 Line",
                "operation_status": "RUNNING",
                "utilization_rate": 0.91,
                "progress_rate": 0.75,
                "waiting_time_hr": 0,
                "processed_quantity": 12000,
                "defect_quantity": 144,
                "recorded_at": date(2026, 6, 1),
            }
        ],
        "top_machine_statuses": [
            {
                "machine_status_id": 1,
                "machine_id": 1,
                "machine_code": "EXT-1",
                "machine_name": "압출기",
                "line_id": 1,
                "line_code": "PP-L1",
                "operation_status": "RUNNING",
                "processed_quantity": 12000,
                "defect_quantity": 144,
                "status_note": "정상",
                "recorded_at": date(2026, 6, 1),
            }
        ],
        "economic_analysis": {
            "simulationResults": [],
            "bestScenario": None,
            "comment": "선택 기간 내 조회 가능한 시뮬레이션 결과가 없습니다.",
        },
    }
