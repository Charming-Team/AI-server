from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BusinessReportSourceRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    report_id: int = Field(alias="reportId")
    report_title: str = Field(alias="reportTitle")
    report_type: str = Field(alias="reportType")
    author_id: int = Field(alias="authorId")
    target_start_date: date = Field(alias="targetStartDate")
    target_end_date: date = Field(alias="targetEndDate")
    markdown: str | None = None
    sections: dict[str, Any] | None = None
    report_content: Any = Field(default=None, alias="reportContent")
    report_evidence: Any = Field(default=None, alias="reportEvidence")
    related_simulation_id: int | None = Field(default=None, alias="relatedSimulationId")


class BusinessReportGenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    report_id: int = Field(..., alias="reportId", ge=1)
    source_report: BusinessReportSourceRequest | None = Field(
        default=None,
        alias="sourceReport",
    )
