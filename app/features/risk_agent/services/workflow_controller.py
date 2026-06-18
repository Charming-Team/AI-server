from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.features.risk_agent.nodes.cause_ranking_node import (
    CauseRankingNode,
)
from app.features.risk_agent.nodes.context_load_node import ContextLoadNode
from app.features.risk_agent.nodes.persist_node import (
    RiskAgentPersistNode,
)
from app.features.risk_agent.nodes.risk_explanation_llm_node import (
    RiskExplanationLlmNode,
)
from app.features.risk_agent.nodes.validation_node import (
    RiskAgentValidationError,
    RiskAgentValidationNode,
)
from app.features.risk_agent.schemas.common import (
    ConfidenceLevel,
    WorkflowStatus,
)
from app.features.risk_agent.schemas.request import (
    RiskAgentContextLoadRequest,
)
from app.features.risk_agent.schemas.state import RiskAgentWorkflowState
from app.features.risk_agent.services.analyzer_executor import (
    AnalyzerExecutor,
)


class RiskAgentWorkflowController:
    def __init__(
        self,
        context_load_node: ContextLoadNode,
        analyzer_executor: AnalyzerExecutor,
        cause_ranking_node: CauseRankingNode,
        risk_explanation_node: RiskExplanationLlmNode,
        validation_node: RiskAgentValidationNode,
        persist_node: RiskAgentPersistNode,
        validation_max_retries: int,
    ) -> None:
        self.context_load_node = context_load_node
        self.analyzer_executor = analyzer_executor
        self.cause_ranking_node = cause_ranking_node
        self.risk_explanation_node = risk_explanation_node
        self.validation_node = validation_node
        self.persist_node = persist_node
        self.validation_max_retries = max(
            min(validation_max_retries, 2),
            0,
        )

    async def load_context(
        self,
        request: RiskAgentContextLoadRequest,
    ) -> RiskAgentWorkflowState:
        now = datetime.now(UTC)

        state = RiskAgentWorkflowState(
            workflow_run_id=(
                request.workflow_run_id
                or f"risk-agent-{uuid4()}"
            ),
            prediction_id=request.prediction_id,
            order_id=request.order_id,
            status=WorkflowStatus.CREATED,
            triggered_at=request.triggered_at,
            started_at=now,
        )

        try:
            return await self.context_load_node.run(state)

        except Exception as exc:
            return state.model_copy(
                update={
                    "status": WorkflowStatus.FAILED,
                    "finished_at": datetime.now(UTC),
                    "error_message": str(exc),
                }
            )

    async def analyze(
        self,
        request: RiskAgentContextLoadRequest,
    ) -> RiskAgentWorkflowState:
        state = await self.load_context(request)

        if state.status in {
            WorkflowStatus.FAILED,
            WorkflowStatus.SKIPPED_SAFE,
        }:
            return state

        if state.evidence is None:
            return state.model_copy(
                update={
                    "status": WorkflowStatus.FAILED,
                    "finished_at": datetime.now(UTC),
                    "error_message": "Analyzer 실행에 필요한 evidence가 없습니다.",
                }
            )

        analyzing_state = state.model_copy(
            update={
                "status": WorkflowStatus.ANALYZING,
            }
        )

        batch_result = await self.analyzer_executor.run(
            analyzing_state.evidence
        )

        if (
            batch_result.failed_analyzers
            and len(batch_result.failed_analyzers)
            == len(self.analyzer_executor.analyzers)
        ):
            return analyzing_state.model_copy(
                update={
                    "status": WorkflowStatus.FAILED,
                    "analyzer_findings": batch_result.findings,
                    "finished_at": datetime.now(UTC),
                    "error_message": "모든 Analyzer 실행에 실패했습니다.",
                }
            )

        missing_fields = list(
            dict.fromkeys(
                [
                    *analyzing_state.missing_fields,
                    *[
                        field
                        for finding in batch_result.findings
                        for field in finding.missing_fields
                    ],
                ]
            )
        )

        confidence_level = self._resolve_confidence(
            current=analyzing_state.confidence_level,
            missing_field_count=len(missing_fields),
            failed_analyzer_count=len(
                batch_result.failed_analyzers
            ),
        )

        return analyzing_state.model_copy(
            update={
                "status": WorkflowStatus.ANALYZED,
                "analyzer_findings": batch_result.findings,
                "missing_fields": missing_fields,
                "confidence_level": confidence_level,
            }
        )

    async def rank_causes(
            self,
            request: RiskAgentContextLoadRequest,
        ) -> RiskAgentWorkflowState:
            state = await self.analyze(request)

            if state.status in {
                WorkflowStatus.FAILED,
                WorkflowStatus.SKIPPED_SAFE,
            }:
                return state

            try:
                return self.cause_ranking_node.run(state)

            except Exception as exc:
                return state.model_copy(
                    update={
                        "status": WorkflowStatus.FAILED,
                        "finished_at": datetime.now(UTC),
                        "error_message": str(exc),
                    }
                )

    async def generate_explanation(
        self,
        request: RiskAgentContextLoadRequest,
    ) -> RiskAgentWorkflowState:
        state = await self.rank_causes(request)

        if state.status in {
            WorkflowStatus.FAILED,
            WorkflowStatus.SKIPPED_SAFE,
        }:
            return state

        try:
            return await self.risk_explanation_node.run(state)

        except Exception as exc:
            return state.model_copy(
                update={
                    "status": WorkflowStatus.FAILED,
                    "finished_at": datetime.now(UTC),
                    "error_message": str(exc),
                }
            )

    @staticmethod
    def _resolve_confidence(
        *,
        current: ConfidenceLevel,
        missing_field_count: int,
        failed_analyzer_count: int,
    ) -> ConfidenceLevel:
        if failed_analyzer_count >= 2 or missing_field_count >= 3:
            return ConfidenceLevel.LOW

        if failed_analyzer_count >= 1 or missing_field_count >= 1:
            return ConfidenceLevel.MEDIUM

        return current
    
    async def validate_explanation(
        self,
        request: RiskAgentContextLoadRequest,
    ) -> RiskAgentWorkflowState:
        ranked_state = await self.rank_causes(request)

        if ranked_state.status in {
            WorkflowStatus.FAILED,
            WorkflowStatus.SKIPPED_SAFE,
        }:
            return ranked_state

        validation_feedback: list[str] = []
        last_generated_state: RiskAgentWorkflowState | None = None

        for attempt in range(
            self.validation_max_retries + 1
        ):
            generation_input = ranked_state.model_copy(
                update={
                    "validation_errors": validation_feedback,
                    "retry_count": (
                        ranked_state.retry_count + attempt
                    ),
                }
            )

            try:
                generated_state = (
                    await self.risk_explanation_node.run(
                        generation_input
                    )
                )

                last_generated_state = generated_state

                return self.validation_node.run(
                    generated_state
                )

            except RiskAgentValidationError as exc:
                validation_feedback = exc.errors

            except Exception as exc:
                return ranked_state.model_copy(
                    update={
                        "status": WorkflowStatus.FAILED,
                        "finished_at": datetime.now(
                            UTC
                        ),
                        "error_message": str(exc),
                    }
                )

        failed_state = (
            last_generated_state
            if last_generated_state is not None
            else ranked_state
        )

        return failed_state.model_copy(
            update={
                "status": WorkflowStatus.FAILED,
                "validation_errors": validation_feedback,
                "finished_at": datetime.now(
                    UTC
                ),
                "error_message": (
                    "Risk Agent 결과가 Validation을 "
                    "통과하지 못했습니다."
                ),
            }
        )


    async def execute(
        self,
        request: RiskAgentContextLoadRequest,
    ) -> RiskAgentWorkflowState:
        state = await self.validate_explanation(request)

        if state.status in {
            WorkflowStatus.FAILED,
            WorkflowStatus.SKIPPED_SAFE,
        }:
            return state

        try:
            return await self.persist_node.run(state)

        except Exception as exc:
            return state.model_copy(
                update={
                    "status": WorkflowStatus.FAILED,
                    "finished_at": datetime.now(
                        UTC
                    ),
                    "error_message": str(exc),
                }
            )