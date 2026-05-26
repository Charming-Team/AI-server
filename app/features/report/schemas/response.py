from enum import Enum
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class ReportStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    TIMEOUT = "TIMEOUT"


class EvidenceType(str, Enum):
    RDB = "RDB"
    QDRANT = "QDRANT"
    AGENT = "AGENT"


class ReportEvidence(BaseModel):
    type: EvidenceType
    source: str
    description: str


class ReportValidationResult(BaseModel):
    required_section_included: bool = Field(..., alias="requiredSectionIncluded")
    groundedness_passed: bool = Field(..., alias="groundednessPassed")
    missing_fields: list[str] = Field(default_factory=list, alias="missingFields")

    class Config:
        populate_by_name = True


class ReportGenerateResponse(BaseModel):
    report_job_id: int = Field(..., alias="reportJobId")
    status: ReportStatus
    title: str | None = None
    report_type: str | None = Field(default=None, alias="reportType")
    markdown: str | None = None
    sections: dict[str, Any] | None = None
    evidence: list[ReportEvidence] = Field(default_factory=list)
    validation: ReportValidationResult
    error_message: str | None = Field(default=None, alias="errorMessage")

    class Config:
        populate_by_name = True

class ReportListItemResponse(BaseModel):
    report_id: int = Field(..., alias="reportId")
    report_type: str = Field(..., alias="reportType")
    report_title: str = Field(..., alias="reportTitle")
    author_id: int | None = Field(None, alias="authorId")
    target_start_date: date = Field(..., alias="targetStartDate")
    target_end_date: date = Field(..., alias="targetEndDate")
    related_simulation_id: int | None = Field(None, alias="relatedSimulationId")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")

    model_config = {
        "populate_by_name": True,
    }


class ReportListResponse(BaseModel):
    reports: list[ReportListItemResponse]
    total_count: int = Field(..., alias="totalCount")

    model_config = {
        "populate_by_name": True,
    }


class ReportDetailResponse(BaseModel):
    report_id: int = Field(..., alias="reportId")
    report_type: str = Field(..., alias="reportType")
    report_title: str = Field(..., alias="reportTitle")
    author_id: int | None = Field(None, alias="authorId")
    target_start_date: date = Field(..., alias="targetStartDate")
    target_end_date: date = Field(..., alias="targetEndDate")
    included_items: dict[str, Any] | None = Field(None, alias="includedItems")
    report_content: str = Field(..., alias="reportContent")
    report_evidence: list[dict[str, Any]] | dict[str, Any] | None = Field(
        None,
        alias="reportEvidence",
    )
    related_simulation_id: int | None = Field(None, alias="relatedSimulationId")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")

    model_config = {
        "populate_by_name": True,
    }