from __future__ import annotations

import hmac
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.core.config import get_settings
from app.features.risk_agent.clients.spring_risk_agent_client import (
    SpringRiskAgentClient,
)
from app.features.risk_agent.nodes.context_load_node import ContextLoadNode
from app.features.risk_agent.schemas.request import (
    RiskAgentContextLoadRequest,
)
from app.features.risk_agent.schemas.state import RiskAgentWorkflowState
from app.features.risk_agent.services.workflow_controller import (
    RiskAgentWorkflowController,
)
from app.features.risk_agent.nodes.due_impact_analyzer import (
    DueImpactAnalyzer,
)
from app.features.risk_agent.nodes.line_process_analyzer import (
    LineProcessAnalyzer,
)
from app.features.risk_agent.nodes.machine_analyzer import (
    MachineAnalyzer,
)
from app.features.risk_agent.nodes.material_analyzer import (
    MaterialAnalyzer,
)
from app.features.risk_agent.nodes.yield_analyzer import (
    YieldAnalyzer,
)
from app.features.risk_agent.services.analyzer_executor import (
    AnalyzerExecutor,
)
from app.features.risk_agent.nodes.cause_ranking_node import (
    CauseRankingNode,
)
from app.features.chat.llm_client import LlmClient
from app.features.risk_agent.nodes.risk_explanation_llm_node import (
    RiskExplanationLlmNode,
)
from app.features.risk_agent.nodes.persist_node import (
    RiskAgentPersistNode,
)
from app.features.risk_agent.nodes.validation_node import (
    RiskAgentValidationNode,
)


router = APIRouter(
    prefix="/risk-agent",
    tags=["Risk Agent"],
)


def verify_internal_token(
    x_internal_token: Annotated[
        str | None,
        Header(alias="X-Internal-Token"),
    ] = None,
) -> None:
    settings = get_settings()
    expected = settings.risk_agent_internal_token

    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk Agent internal token is not configured.",
        )

    if (
        x_internal_token is None
        or not hmac.compare_digest(x_internal_token, expected)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal token.",
        )


@lru_cache
def get_workflow_controller() -> RiskAgentWorkflowController:
    settings = get_settings()

    spring_client = SpringRiskAgentClient(
        base_url=settings.risk_agent_spring_base_url,
        internal_token=settings.risk_agent_internal_token,
        connect_timeout_seconds=(
            settings.risk_agent_connect_timeout_seconds
        ),
        read_timeout_seconds=(
            settings.risk_agent_read_timeout_seconds
        ),
    )

    llm_client = LlmClient(settings=settings)

    return RiskAgentWorkflowController(
        context_load_node=ContextLoadNode(spring_client),
        analyzer_executor=AnalyzerExecutor(
            [
                MaterialAnalyzer(),
                YieldAnalyzer(),
                MachineAnalyzer(),
                LineProcessAnalyzer(),
                DueImpactAnalyzer(),
            ]
        ),
        cause_ranking_node=CauseRankingNode(),
        risk_explanation_node=RiskExplanationLlmNode(
            settings=settings,
            llm_client=llm_client,
        ),
        validation_node=RiskAgentValidationNode(),
        persist_node=RiskAgentPersistNode(
            spring_client=spring_client,
            max_retries=settings.risk_agent_max_retries,
        ),
        validation_max_retries=(
            settings.risk_agent_max_retries
        ),
    )


@router.post(
    "/context-load",
    response_model=RiskAgentWorkflowState,
    summary="Risk Agent 컨텍스트 로드",
    description=(
        "Spring에서 예측 결과와 주문·생산계획·자재·라인·수율·설비 "
        "근거 데이터를 조회하여 Risk Agent Workflow 상태를 생성합니다."
    ),
    operation_id="loadRiskAgentContext",
    dependencies=[Depends(verify_internal_token)],
)
async def load_risk_agent_context(
    request: RiskAgentContextLoadRequest,
    controller: Annotated[
        RiskAgentWorkflowController,
        Depends(get_workflow_controller),
    ],
) -> RiskAgentWorkflowState:
    return await controller.load_context(request)

@router.post(
    "/analyze",
    response_model=RiskAgentWorkflowState,
    summary="Risk Agent 원인 분석 실행",
    description=(
        "Spring Evidence API에서 컨텍스트를 조회한 뒤 "
        "자재·수율·설비·라인/공정·납기/영향 Analyzer를 병렬 실행합니다."
    ),
    operation_id="analyzeRiskAgentContext",
    dependencies=[Depends(verify_internal_token)],
)
async def analyze_risk_agent_context(
    request: RiskAgentContextLoadRequest,
    controller: Annotated[
        RiskAgentWorkflowController,
        Depends(get_workflow_controller),
    ],
) -> RiskAgentWorkflowState:
    return await controller.analyze(request)

@router.post(
    "/rank",
    response_model=RiskAgentWorkflowState,
    summary="Risk Agent 지연 원인 우선순위 산정",
    description=(
        "5개 Analyzer 결과, ML SHAP 근거, 지연 확률, "
        "납기 영향도 및 데이터 신뢰도를 종합하여 "
        "최종 지연 원인 1~3개를 선정합니다."
    ),
    operation_id="rankRiskAgentCauses",
    dependencies=[Depends(verify_internal_token)],
)
async def rank_risk_agent_causes(
    request: RiskAgentContextLoadRequest,
    controller: Annotated[
        RiskAgentWorkflowController,
        Depends(get_workflow_controller),
    ],
) -> RiskAgentWorkflowState:
    return await controller.rank_causes(request)

@router.post(
    "/generate",
    response_model=RiskAgentWorkflowState,
    summary="Risk Agent 설명 및 권고 조치 생성",
    description=(
        "Evidence 조회, 5개 Analyzer 실행, 원인 우선순위 산정 후 "
        "LLM을 이용해 최종 원인 설명과 권고 조치를 생성합니다."
    ),
    operation_id="generateRiskAgentExplanation",
    dependencies=[Depends(verify_internal_token)],
)
async def generate_risk_agent_explanation(
    request: RiskAgentContextLoadRequest,
    controller: Annotated[
        RiskAgentWorkflowController,
        Depends(get_workflow_controller),
    ],
) -> RiskAgentWorkflowState:
    return await controller.generate_explanation(request)

@router.post(
    "/validate",
    response_model=RiskAgentWorkflowState,
    summary="Risk Agent 생성 결과 검증",
    description=(
        "Agent 설명과 권고 조치가 스키마, 근거, "
        "원인 타입 및 권고 범위를 준수하는지 검증합니다."
    ),
    operation_id="validateRiskAgentExplanation",
    dependencies=[Depends(verify_internal_token)],
)
async def validate_risk_agent_explanation(
    request: RiskAgentContextLoadRequest,
    controller: Annotated[
        RiskAgentWorkflowController,
        Depends(get_workflow_controller),
    ],
) -> RiskAgentWorkflowState:
    return await controller.validate_explanation(request)


@router.post(
    "/execute",
    response_model=RiskAgentWorkflowState,
    summary="Risk Agent 전체 Workflow 실행",
    description=(
        "Context 조회, Analyzer 실행, 원인 순위 산정, "
        "LLM 설명 생성, Validation 및 Spring DB 저장을 "
        "순서대로 수행합니다."
    ),
    operation_id="executeRiskAgentWorkflow",
    dependencies=[Depends(verify_internal_token)],
)
async def execute_risk_agent_workflow(
    request: RiskAgentContextLoadRequest,
    controller: Annotated[
        RiskAgentWorkflowController,
        Depends(get_workflow_controller),
    ],
) -> RiskAgentWorkflowState:
    return await controller.execute(request)