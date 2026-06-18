from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from app.core.config import Settings, get_settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.llm_client import (
    OPENAI_COMPATIBLE_PROVIDER,
    OPENAI_PROVIDER,
    SUPPORTED_LLM_PROVIDERS,
    normalize_llm_provider,
    validate_llm_settings,
)
from app.features.chat.runtime_mode import build_chat_runtime_mode
from app.features.chat.schemas import ChatErrorCode
from app.schemas.health import HealthResponse, ReadinessComponent, ReadinessResponse

router = APIRouter()
SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: SettingsDep) -> HealthResponse:
    """Return a lightweight process liveness response."""
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
    """Return lightweight dependency readiness without calling LLM generation."""
    components = build_readiness_components(settings)
    is_ready = all(component.configured for component in components)
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        app_name=settings.app_name,
        environment=settings.environment,
        runtime_mode=build_chat_runtime_mode(settings),
        components=components,
    )


def build_readiness_components(settings: Settings) -> list[ReadinessComponent]:
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
            name="rdbEvidence",
            enabled=settings.rdb_evidence_enabled,
            required_fields={
                "rdb_evidence_dsn": settings.rdb_evidence_dsn,
            },
            field_error_codes={
                "rdb_evidence_dsn": ChatErrorCode.CHAT_EVIDENCE_004,
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
        _chat_grounding_pipeline_component(settings),
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
        _llm_component(settings),
        _answer_generation_pipeline_component(settings),
    ]


def _chat_grounding_pipeline_component(settings: Settings) -> ReadinessComponent:
    if (
        settings.evidence_lookup_enabled
        or settings.rdb_evidence_enabled
        or settings.qdrant_search_enabled
    ):
        return ReadinessComponent(
            name="chatGroundingPipeline",
            enabled=True,
            configured=True,
        )

    return ReadinessComponent(
        name="chatGroundingPipeline",
        enabled=True,
        configured=False,
        code=ChatErrorCode.CHAT_EVIDENCE_001,
        reason=(
            "챗봇 답변에는 RDB Evidence View, Spring Evidence 또는 "
            "Qdrant 검색 중 하나가 필요합니다."
        ),
    )


def _answer_generation_pipeline_component(settings: Settings) -> ReadinessComponent:
    if settings.llm_enabled:
        return ReadinessComponent(
            name="answerGenerationPipeline",
            enabled=True,
            configured=True,
        )

    return ReadinessComponent(
        name="answerGenerationPipeline",
        enabled=True,
        configured=True,
        reason="LLM 기능이 비활성화되어 근거 기반 fallback 답변 생성을 사용합니다.",
    )


def _llm_component(settings: Settings) -> ReadinessComponent:
    if not settings.llm_enabled:
        return ReadinessComponent(
            name="llm",
            enabled=False,
            configured=True,
            reason="비활성화되어 있습니다.",
        )

    provider = normalize_llm_provider(settings.llm_provider)
    if provider not in SUPPORTED_LLM_PROVIDERS:
        return ReadinessComponent(
            name="llm",
            enabled=True,
            configured=False,
            code=ChatErrorCode.CHAT_LLM_001,
            reason=f"지원하지 않는 LLM provider입니다: {settings.llm_provider}",
        )

    required_fields = {"llm_model": settings.llm_model}
    if provider == OPENAI_PROVIDER:
        required_fields["llm_api_key"] = settings.llm_api_key
    elif provider == OPENAI_COMPATIBLE_PROVIDER:
        required_fields["llm_base_url"] = settings.llm_base_url

    component = _integration_component(
        name="llm",
        enabled=True,
        required_fields=required_fields,
        field_error_codes={
            "llm_api_key": ChatErrorCode.CHAT_LLM_001,
            "llm_base_url": ChatErrorCode.CHAT_LLM_001,
            "llm_model": ChatErrorCode.CHAT_LLM_001,
        },
    )
    if not component.configured:
        return component

    try:
        validate_llm_settings(settings)
    except ChatExternalServiceError as exc:
        return ReadinessComponent(
            name="llm",
            enabled=True,
            configured=False,
            code=exc.code,
            reason=exc.message,
        )
    return component


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
