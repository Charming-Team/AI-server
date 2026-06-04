from fastapi import APIRouter
from pydantic import BaseModel

from app.features.business_report.schemas.request import BusinessReportGenerateRequest
from app.features.business_report.schemas.response import BusinessReportGenerateResponse
from app.features.business_report.services.business_report_generation_service import (
    BusinessReportGenerationService,
)

router = APIRouter(prefix="/business-reports", tags=["Business Report"])


class BusinessReportHealthResponse(BaseModel):
    status: str
    feature: str

class BusinessReportErrorResponse(BaseModel):
    code: str
    message: str


business_report_generation_service = BusinessReportGenerationService()

business_report_error_responses = {
    400: {"model": BusinessReportErrorResponse},
    404: {"model": BusinessReportErrorResponse},
    422: {"model": BusinessReportErrorResponse},
    500: {"model": BusinessReportErrorResponse},
    502: {"model": BusinessReportErrorResponse},
    503: {"model": BusinessReportErrorResponse},
}


@router.post(
    "/generate",
    response_model=BusinessReportGenerateResponse,
    responses=business_report_error_responses,
)
async def generate_business_report(
    request: BusinessReportGenerateRequest,
) -> BusinessReportGenerateResponse:
    return await business_report_generation_service.generate_business_report(request)


@router.get("/health", response_model=BusinessReportHealthResponse)
async def business_report_health() -> BusinessReportHealthResponse:
    return BusinessReportHealthResponse(
        status="ok",
        feature="business-report-agent",
    )
