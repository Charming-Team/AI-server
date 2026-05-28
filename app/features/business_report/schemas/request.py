from pydantic import BaseModel, Field


class BusinessReportGenerateRequest(BaseModel):
    report_id: int = Field(..., ge=1)

