from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from app.core.config import Settings, get_settings
from app.schemas.health import HealthResponse, ReadinessComponent, ReadinessResponse

router = APIRouter()
SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check(
    response: Response,
    settings: SettingsDep,
) -> ReadinessResponse:
    components = _build_readiness_components(settings)
    is_ready = all(component.configured for component in components)
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        app_name=settings.app_name,
        environment=settings.environment,
        components=components,
    )


def _build_readiness_components(settings: Settings) -> list[ReadinessComponent]:
    return [
        _required_component(
            name="chatAnswerInternalToken",
            configured=_has_text(settings.chat_answer_internal_token),
        ),
        _required_component(
            name="chatRecommendationInternalToken",
            configured=_has_text(settings.chat_recommendation_internal_token),
        ),
        _required_component(
            name="documentIndexInternalToken",
            configured=_has_text(settings.document_index_internal_token),
        ),
        _integration_component(
            name="evidenceLookup",
            enabled=settings.evidence_lookup_enabled,
            required_fields={
                "baseUrl": settings.evidence_lookup_base_url,
                "path": settings.evidence_lookup_path,
                "internalToken": settings.evidence_lookup_internal_token,
            },
        ),
        _integration_component(
            name="qdrantSearch",
            enabled=settings.qdrant_search_enabled,
            required_fields={
                "url": settings.qdrant_url,
                "collection": settings.qdrant_collection,
            },
        ),
        _integration_component(
            name="embedding",
            enabled=settings.embedding_enabled,
            required_fields={
                "baseUrl": settings.embedding_base_url,
                "path": settings.embedding_path,
                "model": settings.embedding_model,
                "dimension": str(settings.embedding_dimension),
            },
        ),
        _rag_search_pipeline_component(settings),
        _integration_component(
            name="llm",
            enabled=settings.llm_enabled,
            required_fields={
                "baseUrl": settings.llm_base_url,
                "model": settings.llm_model,
            },
        ),
    ]


def _rag_search_pipeline_component(settings: Settings) -> ReadinessComponent:
    if not settings.qdrant_search_enabled:
        return ReadinessComponent(
            name="ragSearchPipeline",
            enabled=False,
            configured=True,
            reason="Qdrant 검색이 비활성화되어 있습니다.",
        )

    if settings.embedding_enabled:
        return ReadinessComponent(
            name="ragSearchPipeline",
            enabled=True,
            configured=True,
        )

    return ReadinessComponent(
        name="ragSearchPipeline",
        enabled=True,
        configured=False,
        reason="Qdrant 검색에는 Embedding 기능 활성화가 필요합니다.",
    )


def _required_component(name: str, configured: bool) -> ReadinessComponent:
    return ReadinessComponent(
        name=name,
        enabled=True,
        configured=configured,
        reason=None if configured else "필수 설정이 누락되었습니다.",
    )


def _integration_component(
    name: str,
    enabled: bool,
    required_fields: dict[str, str | None],
) -> ReadinessComponent:
    if not enabled:
        return ReadinessComponent(
            name=name,
            enabled=False,
            configured=True,
            reason="비활성화되어 있습니다.",
        )

    missing_fields = [
        field_name
        for field_name, value in required_fields.items()
        if not _has_text(value)
    ]
    return ReadinessComponent(
        name=name,
        enabled=True,
        configured=not missing_fields,
        reason=(
            None
            if not missing_fields
            else f"필수 설정이 누락되었습니다: {', '.join(missing_fields)}"
        ),
    )


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())
