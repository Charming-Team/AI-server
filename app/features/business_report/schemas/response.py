from datetime import date, datetime
from typing import Any

from pydantic import BaseModel


class BusinessReportGenerateResponse(BaseModel):
    report_id: int
    report_type: str
    report_title: str
    author_id: int
    target_start_date: date
    target_end_date: date
    report_content: Any
    report_evidence: Any = None
    related_simulation_id: int | None = None
    created_at: datetime
    updated_at: datetime

