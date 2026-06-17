from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BusinessReportGenerateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    report_id: int = Field(alias="reportId")
    report_type: str = Field(alias="reportType")
    report_title: str = Field(alias="reportTitle")
    author_id: int = Field(alias="authorId")
    target_start_date: date = Field(alias="targetStartDate")
    target_end_date: date = Field(alias="targetEndDate")
    report_content: Any = Field(alias="reportContent")
    report_evidence: Any = Field(default=None, alias="reportEvidence")
    related_simulation_id: int | None = Field(default=None, alias="relatedSimulationId")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
