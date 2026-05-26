from fastapi import APIRouter

from app.features.report.schemas.request import ReportGenerateRequest
from app.features.report.schemas.response import ReportGenerateResponse
from app.features.report.services.report_generation_service import ReportGenerationService

router = APIRouter(prefix="/reports", tags=["Report Agent"])

report_generation_service = ReportGenerationService()


@router.post("/generate", response_model=ReportGenerateResponse)
def generate_report(request: ReportGenerateRequest) -> ReportGenerateResponse:
    return report_generation_service.generate_report(request)


@router.get("/health")
def report_health() -> dict[str, str]:
    return {
        "status": "ok",
        "feature": "report-agent",
    }