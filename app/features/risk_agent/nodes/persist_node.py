from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.features.risk_agent.clients.spring_risk_agent_client import (
    SpringRiskAgentClient,
    SpringRiskAgentClientError,
)
from app.features.risk_agent.schemas.common import WorkflowStatus
from app.features.risk_agent.schemas.persist import (
    RiskAgentPersistRequest,
)
from app.features.risk_agent.schemas.state import (
    RiskAgentWorkflowState,
)


class RiskAgentPersistError(RuntimeError):
    pass


class RiskAgentPersistNode:
    def __init__(
        self,
        *,
        spring_client: SpringRiskAgentClient,
        max_retries: int,
    ) -> None:
        self.spring_client = spring_client
        self.max_retries = max(
            min(max_retries, 2),
            0,
        )

    async def run(
        self,
        state: RiskAgentWorkflowState,
    ) -> RiskAgentWorkflowState:
        if state.status != WorkflowStatus.VALIDATED:
            raise ValueError(
                "Persist는 VALIDATED 상태에서만 수행할 수 있습니다."
            )

        if not state.analysis_summary:
            raise ValueError(
                "저장할 analysisSummary가 없습니다."
            )

        if not state.recommended_action:
            raise ValueError(
                "저장할 recommendedAction이 없습니다."
            )

        if not state.selected_cause_types:
            raise ValueError(
                "저장할 causeTypes가 없습니다."
            )

        request = RiskAgentPersistRequest(
            workflow_run_id=state.workflow_run_id,
            prediction_id=state.prediction_id,
            order_id=state.order_id,
            analysis_summary=state.analysis_summary,
            recommended_action=state.recommended_action,
            cause_types=state.selected_cause_types,
        )

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                await self.spring_client.persist_analysis(
                    request
                )

                return state.model_copy(
                    update={
                        "status": WorkflowStatus.PERSISTED,
                        "retry_count": (
                            state.retry_count + attempt
                        ),
                        "finished_at": datetime.now(
                            timezone.utc
                        ),
                        "error_message": None,
                    }
                )

            except SpringRiskAgentClientError as exc:
                last_error = exc

                if attempt < self.max_retries:
                    await asyncio.sleep(
                        0.5 * (2 ** attempt)
                    )

        raise RiskAgentPersistError(
            "Agent 분석 결과 저장에 실패했습니다. "
            f"attempts={self.max_retries + 1}, "
            f"reason={last_error}"
        ) from last_error