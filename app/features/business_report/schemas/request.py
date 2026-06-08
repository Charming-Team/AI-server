from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class BusinessReportSourceRequest(BaseModel):
    report_id: int
    report_title: str
    report_type: str
    author_id: int
    target_start_date: date
    target_end_date: date
    markdown: str | None = None
    sections: dict[str, Any] | None = None
    report_content: Any = None
    report_evidence: Any = None
    related_simulation_id: int | None = None


class BusinessReportGenerateRequest(BaseModel):
    report_id: int = Field(..., ge=1)
    source_report: BusinessReportSourceRequest | None = None
