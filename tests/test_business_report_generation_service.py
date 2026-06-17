import asyncio
import json
from datetime import UTC, date, datetime

from app.features.business_report.schemas.request import BusinessReportGenerateRequest
from app.features.business_report.schemas.source import BusinessReportSource
from app.features.business_report.services.business_report_generation_service import (
    BusinessReportGenerationService,
)


class FakeBusinessReportRepository:
    def __init__(self, source: BusinessReportSource) -> None:
        self.source = source

    def get_report_by_id(self, report_id: int) -> BusinessReportSource | None:
        if report_id != self.source.report_id:
            return None

        return self.source


class FakeBusinessReportTransformer:
    def run(self, source: BusinessReportSource) -> str:
        return json.dumps(
            {
                "report_id": 999,
                "report_type": "MONTHLY",
                "report_title": "LLM이 만든 잘못된 제목",
                "author_id": 999,
                "target_start_date": "2099-01-01",
                "target_end_date": "2099-01-31",
                "report_content": {
                    "markdown": "경영진용 markdown",
                    "sections": {
                        "summaryRows": [
                            {"label": "보고서 기간", "value": "2026-06-01 ~ 2026-06-14"}
                        ],
                        "lineRows": [
                            {
                                "line": "PP 범용 생산 Line",
                                "utilization": "91%",
                                "completed": "12,000",
                                "defectRate": "1.2%",
                                "note": "정상",
                            }
                        ],
                        "equipmentRows": [
                            {
                                "name": "압출기",
                                "utilization": "확인 필요",
                                "downTime": "확인 필요",
                                "status": "정상",
                            }
                        ],
                        "analysis": {
                            "overview": "경영진용 분석",
                            "sections": [
                                {
                                    "title": "납기 위험 분석",
                                    "items": ["납기 위험 주문을 우선 점검해야 합니다."],
                                }
                            ],
                            "recommendation": "생산 순서 조정 검토가 필요합니다.",
                        },
                    },
                },
                "report_evidence": [],
                "related_simulation_id": None,
                "created_at": "2099-01-01T00:00:00Z",
                "updated_at": "2099-01-01T00:00:00Z",
            },
            ensure_ascii=False,
        )


def test_business_report_generation_builds_business_report_from_source_report() -> None:
    source = BusinessReportSource(
        report_id=10,
        report_type="ON_DEMAND",
        report_title="일반 수시 보고서",
        author_id=7,
        target_start_date=date(2026, 6, 1),
        target_end_date=date(2026, 6, 14),
        report_content={"sections": [{"title": "원본", "content": "원본 내용"}]},
        report_evidence=[{"source": "production_plans"}],
        related_simulation_id=4,
        created_at=datetime(2026, 6, 1, 1, 0, tzinfo=UTC),
        updated_at=datetime(2026, 6, 1, 1, 0, tzinfo=UTC),
    )
    service = BusinessReportGenerationService(
        repository=FakeBusinessReportRepository(source),
        llm_business_report_transformer=FakeBusinessReportTransformer(),
    )

    response = asyncio.run(
        service.generate_business_report(BusinessReportGenerateRequest(report_id=10))
    )

    assert response.report_id == source.report_id
    assert response.report_type == "ON_DEMAND_BUSINESS"
    assert response.report_title == (
        "[2026-06-01 ~ 2026-06-14] 생산계획 리스크 비즈니스 보고서"
    )
    assert response.author_id == source.author_id
    assert response.target_start_date == source.target_start_date
    assert response.target_end_date == source.target_end_date
    assert response.report_evidence == source.report_evidence
    assert response.related_simulation_id == source.related_simulation_id
    assert response.report_content == {
        "markdown": "경영진용 markdown",
        "sections": {
            "summaryRows": [{"label": "보고서 기간", "value": "2026-06-01 ~ 2026-06-14"}],
            "lineRows": [
                {
                    "line": "PP 범용 생산 Line",
                    "utilization": "91%",
                    "completed": "12,000",
                    "defectRate": "1.2%",
                    "note": "정상",
                }
            ],
            "equipmentRows": [
                {
                    "name": "압출기",
                    "utilization": "확인 필요",
                    "downTime": "확인 필요",
                    "status": "정상",
                }
            ],
            "analysis": {
                "overview": "경영진용 분석",
                "sections": [
                    {
                        "title": "납기 위험 분석",
                        "items": ["납기 위험 주문을 우선 점검해야 합니다."],
                    }
                ],
                "recommendation": "생산 순서 조정 검토가 필요합니다.",
            },
        },
    }
    assert response.created_at != source.created_at
    assert response.updated_at != source.updated_at


def test_business_report_generation_prefers_source_report_request() -> None:
    service = BusinessReportGenerationService(
        repository=FailingBusinessReportRepository(),
        llm_business_report_transformer=FallbackBusinessReportTransformer(),
    )

    response = asyncio.run(
        service.generate_business_report(
            BusinessReportGenerateRequest(
                report_id=3,
                source_report={
                    "report_id": 3,
                    "report_title": "자재 부족 및 입고 지연 리스크 분석 보고서",
                    "report_type": "ON_DEMAND",
                    "author_id": 4,
                    "target_start_date": "2026-06-01",
                    "target_end_date": "2026-06-14",
                    "markdown": "일반 보고서 markdown",
                    "sections": {
                        "summaryRows": [
                            {
                                "label": "총 생산계획 수",
                                "value": "29",
                                "change": "-",
                            }
                        ],
                        "lineRows": [],
                        "equipmentRows": [],
                        "analysis": {},
                    },
                    "report_content": {},
                    "report_evidence": [{"source": "production_plans"}],
                    "related_simulation_id": None,
                },
            )
        )
    )

    assert response.report_id == 3
    assert response.report_type == "ON_DEMAND_BUSINESS"
    assert response.report_content["markdown"] == "일반 보고서 markdown"
    assert response.report_content["sections"]["summaryRows"] == [
        {
            "label": "총 생산계획 수",
            "value": "29",
            "change": "-",
        }
    ]
    assert response.report_content["sections"]["lineRows"] == [
        {
            "line": "확인 필요",
            "utilization": "-",
            "completed": "-",
            "defectRate": "-",
            "note": "확인 필요",
        }
    ]
    assert response.report_content["sections"]["equipmentRows"] == [
        {
            "name": "확인 필요",
            "utilization": "확인 필요",
            "downTime": "확인 필요",
            "status": "확인 필요",
        }
    ]
    assert response.report_content["sections"]["analysis"] == {
        "overview": "일반 보고서 markdown",
        "sections": [],
        "recommendation": "생성 필요",
    }
    assert response.report_evidence == [{"source": "production_plans"}]


def test_business_report_generation_uses_markdown_when_analysis_is_empty() -> None:
    service = BusinessReportGenerationService(
        repository=FailingBusinessReportRepository(),
        llm_business_report_transformer=FallbackBusinessReportTransformer(),
    )

    response = asyncio.run(
        service.generate_business_report(
            BusinessReportGenerateRequest(
                report_id=4,
                source_report={
                    "report_id": 4,
                    "report_title": "리스크 분석 보고서",
                    "report_type": "ON_DEMAND",
                    "author_id": 4,
                    "target_start_date": "2026-06-01",
                    "target_end_date": "2026-06-14",
                    "markdown": (
                        "# 리스크 분석 보고서\n\n"
                        "주요 납기 리스크를 우선 점검해야 합니다.\n\n"
                        "## 납기 위험 분석\n"
                        "- 주문 ORD-1의 납기 위험이 높습니다.\n\n"
                        "## 종합 의견 및 제안\n"
                        "- 생산 순서 조정 검토가 필요합니다."
                    ),
                    "sections": {"analysis": {}},
                    "report_content": {},
                    "report_evidence": [],
                    "related_simulation_id": None,
                },
            )
        )
    )

    assert response.report_content["sections"]["analysis"] == {
        "overview": "주요 납기 리스크를 우선 점검해야 합니다.",
        "sections": [
            {
                "title": "납기 위험 분석",
                "items": ["주문 ORD-1의 납기 위험이 높습니다."],
            }
        ],
        "recommendation": "생산 순서 조정 검토가 필요합니다.",
    }


class FailingBusinessReportRepository:
    def get_report_by_id(self, report_id: int) -> BusinessReportSource | None:
        raise AssertionError("source_report request should not query repository")


class FallbackBusinessReportTransformer:
    def run(self, source: BusinessReportSource) -> str:
        return json.dumps(
            {
                "report_content": {},
            },
            ensure_ascii=False,
        )
