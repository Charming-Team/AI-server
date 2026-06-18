from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field

from app.features.risk_agent.schemas.common import (
    AnalyzerName,
    CamelModel,
    ConfidenceLevel,
    DelayCauseType,
    RiskLevel,
    WorkflowStatus,
)
from app.features.risk_agent.schemas.evidence import RiskAgentEvidence


class AnalyzerFinding(CamelModel):
    analyzer: AnalyzerName
    detected: bool = False
    cause_type: DelayCauseType | None = None
    score: float = Field(default=0.0, ge=0.0, le=1.0)

    summary: str = ""
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | str | bool | None] = Field(
        default_factory=dict
    )
    missing_fields: list[str] = Field(default_factory=list)
    error_message: str | None = None


class RankedCause(CamelModel):
    cause_type: DelayCauseType
    rank: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    evidence: list[str] = Field(default_factory=list)


class RiskAgentWorkflowState(CamelModel):
    workflow_run_id: str
    prediction_id: int
    order_id: int

    status: WorkflowStatus = WorkflowStatus.CREATED
    retry_count: int = 0

    risk_level: RiskLevel | None = None
    delay_probability: Decimal | None = None
    predicted_delay_days: Decimal | None = None

    evidence: RiskAgentEvidence | None = None
    analyzer_findings: list[AnalyzerFinding] = Field(default_factory=list)
    ranked_causes: list[RankedCause] = Field(default_factory=list)

    analysis_summary: str | None = None
    recommended_action: str | None = None
    selected_cause_types: list[DelayCauseType] = Field(default_factory=list)

    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    missing_fields: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)

    triggered_at: datetime | None = None
    started_at: datetime
    finished_at: datetime | None = None
    error_message: str | None = None