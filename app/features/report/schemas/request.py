from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    PRODUCTION_MANAGER = "PRODUCTION_MANAGER"
    EXECUTIVE = "EXECUTIVE"
    WORKER = "WORKER"
    ADMIN = "ADMIN"


class ReportType(str, Enum):
    MONTHLY = "MONTHLY"
    AD_HOC = "AD_HOC"


class ReportPeriod(BaseModel):
    start_date: date = Field(..., alias="startDate")
    end_date: date = Field(..., alias="endDate")

    class Config:
        populate_by_name = True


class ReportGenerateRequest(BaseModel):
    report_job_id: int = Field(..., alias="reportJobId")
    requested_by: int = Field(..., alias="requestedBy")
    user_role: UserRole = Field(..., alias="userRole")
    report_type: ReportType = Field(..., alias="reportType")
    period: ReportPeriod
    include_executive_summary: bool = Field(default=False, alias="includeExecutiveSummary")
    include_evidence: bool = Field(default=True, alias="includeEvidence")

    class Config:
        populate_by_name = True