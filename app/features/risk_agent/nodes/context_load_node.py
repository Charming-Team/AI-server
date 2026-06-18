from __future__ import annotations

from app.features.risk_agent.clients.spring_risk_agent_client import (
    SpringRiskAgentClient,
)
from app.features.risk_agent.schemas.common import (
    ConfidenceLevel,
    RiskLevel,
    WorkflowStatus,
)
from app.features.risk_agent.schemas.state import RiskAgentWorkflowState


class ContextLoadNode:
    def __init__(
        self,
        spring_client: SpringRiskAgentClient,
    ) -> None:
        self.spring_client = spring_client

    async def run(
        self,
        state: RiskAgentWorkflowState,
    ) -> RiskAgentWorkflowState:
        evidence = await self.spring_client.fetch_evidence(
            prediction_id=state.prediction_id,
            order_id=state.order_id,
        )

        if evidence.risk_level == RiskLevel.SAFE:
            return state.model_copy(
                update={
                    "status": WorkflowStatus.SKIPPED_SAFE,
                    "risk_level": evidence.risk_level,
                    "delay_probability": evidence.delay_probability,
                    "predicted_delay_days": evidence.predicted_delay_days,
                    "evidence": evidence,
                    "missing_fields": evidence.missing_fields,
                    "confidence_level": ConfidenceLevel.HIGH,
                }
            )

        missing_fields = list(dict.fromkeys(evidence.missing_fields))

        if not missing_fields:
            confidence_level = ConfidenceLevel.HIGH
        elif len(missing_fields) <= 2:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW

        return state.model_copy(
            update={
                "status": WorkflowStatus.CONTEXT_LOADED,
                "risk_level": evidence.risk_level,
                "delay_probability": evidence.delay_probability,
                "predicted_delay_days": evidence.predicted_delay_days,
                "evidence": evidence,
                "missing_fields": missing_fields,
                "confidence_level": confidence_level,
            }
        )