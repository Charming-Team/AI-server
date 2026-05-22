from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from app.core.config import Settings, get_settings
from app.features.chat.schemas import ChatErrorCode
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


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    response_model_exclude_none=True,
)
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
            code=ChatErrorCode.CHAT_SECURITY_003,
        ),
        _required_component(
            name="chatRecommendationInternalToken",
            configured=_has_text(settings.chat_recommendation_internal_token),
            code=ChatErrorCode.CHAT_SECURITY_003,
        ),
        _required_component(
            name="documentIndexInternalToken",
            configured=_has_text(settings.document_index_internal_token),
            code=ChatErrorCode.CHAT_SECURITY_003,
        ),
        _integration_component(
            name="evidenceLookup",
            enabled=settings.evidence_lookup_enabled,
            required_fields={
                "evidence_lookup_base_url": settings.evidence_lookup_base_url,
                "evidence_lookup_path": settings.evidence_lookup_path,
                "evidence_lookup_internal_token": (
                    settings.evidence_lookup_internal_token
                ),
            },
            field_error_codes={
                "evidence_lookup_base_url": ChatErrorCode.CHAT_EVIDENCE_004,
                "evidence_lookup_path": ChatErrorCode.CHAT_EVIDENCE_004,
                "evidence_lookup_internal_token": ChatErrorCode.CHAT_SECURITY_003,
            },
        ),
        _integration_component(
            name="qdrantSearch",
            enabled=settings.qdrant_search_enabled,
            required_fields={
                "qdrant_url": settings.qdrant_url,
                "qdrant_collection": settings.qdrant_collection,
            },
            field_error_codes={
                "qdrant_url": ChatErrorCode.CHAT_QDRANT_001,
                "qdrant_collection": ChatErrorCode.CHAT_QDRANT_001,
            },
        ),
        _integration_component(
            name="embedding",
            enabled=settings.embedding_enabled,
            required_fields={
                "embedding_base_url": settings.embedding_base_url,
                "embedding_path": settings.embedding_path,
                "embedding_model": settings.embedding_model,
            },
            field_error_codes={
                "embedding_base_url": ChatErrorCode.CHAT_EMBEDDING_001,
                "embedding_path": ChatErrorCode.CHAT_EMBEDDING_001,
                "embedding_model": ChatErrorCode.CHAT_EMBEDDING_001,
            },
        ),
        _rag_search_pipeline_component(settings),
        _integration_component(
            name="llm",
            enabled=settings.llm_enabled,
            required_fields={
                "llm_base_url": settings.llm_base_url,
                "llm_model": settings.llm_model,
            },
            field_error_codes={
                "llm_base_url": ChatErrorCode.CHAT_LLM_001,
                "llm_model": ChatErrorCode.CHAT_LLM_001,
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
        code=ChatErrorCode.CHAT_EMBEDDING_001,
        reason="Qdrant 검색에는 Embedding 기능 활성화가 필요합니다.",
    )


def _required_component(
    name: str,
    configured: bool,
    code: str,
) -> ReadinessComponent:
    return ReadinessComponent(
        name=name,
        enabled=True,
        configured=configured,
        code=None if configured else code,
        reason=None if configured else "필수 설정이 누락되었습니다.",
    )


def _integration_component(
    name: str,
    enabled: bool,
    required_fields: dict[str, str | None],
    field_error_codes: dict[str, str],
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
        code=None if not missing_fields else field_error_codes[missing_fields[0]],
        reason=(
            None
            if not missing_fields
            else f"필수 설정이 누락되었습니다: {', '.join(missing_fields)}"
        ),
    )


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())
