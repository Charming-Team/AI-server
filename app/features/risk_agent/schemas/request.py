from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.features.risk_agent.schemas.common import CamelModel


class RiskAgentContextLoadRequest(CamelModel):
    prediction_id: int = Field(gt=0)
    order_id: int = Field(gt=0)
    workflow_run_id: str | None = None
    triggered_at: datetime | None = None