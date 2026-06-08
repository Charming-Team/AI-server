from fastapi import APIRouter, HTTPException, Query

from app.features.report.schemas.request import ReportGenerateRequest
from app.features.report.schemas.response import (
    ReportDetailResponse,
    ReportGenerateResponse,
    ReportIncludedItemsBackfillResponse,
    ReportListResponse,
)
from app.features.report.services.report_generation_service import ReportGenerationService
from app.features.report.services.report_query_service import ReportQueryService

router = APIRouter(prefix="/reports", tags=["Report Agent"])

report_generation_service = ReportGenerationService()
report_query_service = ReportQueryService()


@router.post("/generate", response_model=ReportGenerateResponse)
def generate_report(request: ReportGenerateRequest) -> ReportGenerateResponse:
    return report_generation_service.generate_report(request)


@router.get("/health")
def report_health() -> dict[str, str]:
    return {
        "status": "ok",
        "feature": "report-agent",
    }


@router.get(
    "",
    response_model=ReportListResponse,
)
def get_reports(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    report_type: str | None = Query(None, alias="reportType"),
) -> ReportListResponse:
    return report_query_service.get_reports(
        limit=limit,
        offset=offset,
        report_type=report_type,
    )


@router.post(
    "/{report_id}/included-items/backfill",
    response_model=ReportIncludedItemsBackfillResponse,
)
def backfill_report_included_items(
    report_id: int,
) -> ReportIncludedItemsBackfillResponse:
    try:
        report = report_query_service.get_report_detail(report_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    sections = report_generation_service.build_sections_for_period(
        start_date=report.target_start_date,
        end_date=report.target_end_date,
        report_type=report.report_type,
    )
    return ReportIncludedItemsBackfillResponse(
        reportId=report_id,
        sections=sections,
    )


@router.get(
    "/{report_id}",
    response_model=ReportDetailResponse,
)
def get_report_detail(
    report_id: int,
) -> ReportDetailResponse:
    try:
        return report_query_service.get_report_detail(report_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
