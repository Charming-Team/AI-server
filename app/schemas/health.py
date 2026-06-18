from app.schemas.base import ApiSchema


class HealthResponse(ApiSchema):
    status: str
    app_name: str
    environment: str


class ReadinessComponent(ApiSchema):
    name: str
    enabled: bool
    configured: bool
    code: str | None = None
    reason: str | None = None


class ChatRuntimeMode(ApiSchema):
    api_prefix: str
    grounding_mode: str
    answer_mode: str
    rag_search_mode: str
    enabled_grounding_sources: list[str]
    expected_llm_skipped_reason: str | None = None


class ReadinessResponse(ApiSchema):
    status: str
    app_name: str
    environment: str
    runtime_mode: ChatRuntimeMode
    components: list[ReadinessComponent]
