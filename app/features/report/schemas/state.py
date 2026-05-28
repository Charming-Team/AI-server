from typing import Any

from pydantic import BaseModel, Field

from app.features.report.schemas.request import ReportGenerateRequest
from app.features.report.schemas.response import ReportEvidence, ReportValidationResult


class ReportAgentState(BaseModel):
    request: ReportGenerateRequest

    raw_data: dict[str, Any] = Field(default_factory=dict)
    retrieved_documents: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    title: str | None = None
    markdown: str | None = None
    sections: dict[str, Any] | None = None

    evidence: list[ReportEvidence] = Field(default_factory=list)
    validation: ReportValidationResult | None = None

    error_message: str | None = None