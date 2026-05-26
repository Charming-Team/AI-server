from enum import Enum
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