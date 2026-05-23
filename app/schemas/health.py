from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str


class ReadinessComponent(BaseModel):
    name: str
    enabled: bool
    configured: bool
    code: str | None = None
    reason: str | None = None


class ChatRuntimeMode(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    api_prefix: str = Field(alias="apiPrefix")
    grounding_mode: str = Field(alias="groundingMode")
    answer_mode: str = Field(alias="answerMode")
    rag_search_mode: str = Field(alias="ragSearchMode")
    enabled_grounding_sources: list[str] = Field(alias="enabledGroundingSources")
    expected_llm_skipped_reason: str | None = Field(
        default=None,
        alias="expectedLlmSkippedReason",
    )


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: str
    app_name: str
    environment: str
    runtime_mode: ChatRuntimeMode = Field(alias="runtimeMode")
    components: list[ReadinessComponent]
