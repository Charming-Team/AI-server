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
                    "sections": [
                        {
                            "title": "경영진 요약",
                            "content": "납기 위험과 자재 수급 현황을 우선 검토해야 합니다.",
                        }
                    ]
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
        "sections": [
            {
                "title": "경영진 요약",
                "content": "납기 위험과 자재 수급 현황을 우선 검토해야 합니다.",
            }
        ]
    }
    assert response.created_at != source.created_at
    assert response.updated_at != source.updated_at
