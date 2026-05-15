from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _normalize_upper_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return value.strip().upper()


class ChatIntent(StrEnum):
    DELIVERY_RISK = "DELIVERY_RISK"
    MATERIAL_SHORTAGE = "MATERIAL_SHORTAGE"
    PRODUCTION_PLAN = "PRODUCTION_PLAN"
    URGENT_ORDER_IMPACT = "URGENT_ORDER_IMPACT"
    WORK_PRIORITY = "WORK_PRIORITY"
    LINE_BOTTLENECK = "LINE_BOTTLENECK"
    REPORT_LOOKUP = "REPORT_LOOKUP"
    UNKNOWN = "UNKNOWN"


class SecurityStatus(StrEnum):
    PASSED = "PASSED"
    INVALID_REQUEST = "INVALID_REQUEST"
    BLOCKED_PROMPT_INJECTION = "BLOCKED_PROMPT_INJECTION"
    BLOCKED_SENSITIVE_REQUEST = "BLOCKED_SENSITIVE_REQUEST"
    BLOCKED_UNAUTHORIZED = "BLOCKED_UNAUTHORIZED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ERROR = "ERROR"


class ChatErrorCode(StrEnum):
    CHAT_REQUEST_001 = "CHAT_REQUEST_001"
    CHAT_REQUEST_002 = "CHAT_REQUEST_002"
    CHAT_SERVER_001 = "CHAT_SERVER_001"
    CHAT_INPUT_001 = "CHAT_INPUT_001"
    CHAT_INPUT_002 = "CHAT_INPUT_002"
    CHAT_INPUT_003 = "CHAT_INPUT_003"
    CHAT_SECURITY_001 = "CHAT_SECURITY_001"
    CHAT_SECURITY_002 = "CHAT_SECURITY_002"
    CHAT_SECURITY_003 = "CHAT_SECURITY_003"
    CHAT_SECURITY_004 = "CHAT_SECURITY_004"
    CHAT_EVIDENCE_001 = "CHAT_EVIDENCE_001"
    CHAT_EVIDENCE_002 = "CHAT_EVIDENCE_002"
    CHAT_EVIDENCE_003 = "CHAT_EVIDENCE_003"
    CHAT_EMBEDDING_001 = "CHAT_EMBEDDING_001"
    CHAT_EMBEDDING_002 = "CHAT_EMBEDDING_002"
    CHAT_EMBEDDING_003 = "CHAT_EMBEDDING_003"
    CHAT_EMBEDDING_004 = "CHAT_EMBEDDING_004"
    CHAT_QDRANT_001 = "CHAT_QDRANT_001"
    CHAT_QDRANT_002 = "CHAT_QDRANT_002"
    CHAT_QDRANT_003 = "CHAT_QDRANT_003"
    CHAT_LLM_001 = "CHAT_LLM_001"
    CHAT_LLM_002 = "CHAT_LLM_002"
    CHAT_LLM_003 = "CHAT_LLM_003"
    CHAT_RECOMMEND_001 = "CHAT_RECOMMEND_001"
    CHAT_RECOMMEND_002 = "CHAT_RECOMMEND_002"
    CHAT_DOCUMENT_001 = "CHAT_DOCUMENT_001"
    CHAT_DOCUMENT_002 = "CHAT_DOCUMENT_002"
    CHAT_DOCUMENT_003 = "CHAT_DOCUMENT_003"


class ChatUserContext(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: int = Field(alias="userId")
    role: str
    company_name: str | None = Field(default=None, alias="companyName")
    status: str

    @field_validator("role", "status", mode="before")
    @classmethod
    def normalize_access_text(cls, value: Any) -> Any:
        return _normalize_upper_text(value)


class ChatAnswerRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: int = Field(alias="sessionId")
    message_id: int = Field(alias="messageId")
    user: ChatUserContext
    question: str = Field(min_length=1, max_length=1000)
    requested_at: datetime = Field(alias="requestedAt")


class ChatRecommendationRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user: ChatUserContext
    keyword: str | None = Field(default=None, max_length=100)


class ChatRecommendedQuestion(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question_id: str = Field(alias="questionId")
    question: str
    intent: ChatIntent
    category: str
    url: str


class ChatRecommendationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    items: list[ChatRecommendedQuestion]
    fallback_used: bool = Field(alias="fallbackUsed")


class EvidenceItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: str
    title: str
    summary: str
    url: str | None = None
    source: str
    reference_id: int | None = Field(default=None, alias="referenceId")
    data: dict[str, Any] = Field(default_factory=dict)


class EvidenceResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    intent: ChatIntent
    basis_time: datetime = Field(alias="basisTime")
    items: list[EvidenceItem] = Field(default_factory=list)

    @property
    def has_evidence(self) -> bool:
        return bool(self.items)


class EvidenceLookupUser(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: int = Field(alias="userId")
    role: str
    company_name: str | None = Field(default=None, alias="companyName")

    @field_validator("role", mode="before")
    @classmethod
    def normalize_role(cls, value: Any) -> Any:
        return _normalize_upper_text(value)


class EvidenceLookupFilters(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    limit: int = 5
    from_date: str | None = Field(default=None, alias="fromDate")
    to_date: str | None = Field(default=None, alias="toDate")
    target_type: str | None = Field(default=None, alias="targetType")
    target_code: str | None = Field(default=None, alias="targetCode")


class EvidenceLookupRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: int = Field(alias="sessionId")
    message_id: int = Field(alias="messageId")
    intent: ChatIntent
    question: str
    user: EvidenceLookupUser
    filters: EvidenceLookupFilters


class ChatUrl(BaseModel):
    label: str
    url: str
    type: str


class ChatSource(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source_type: str = Field(alias="sourceType")
    title: str
    summary: str
    url: str | None = None
    reference_id: int | None = Field(default=None, alias="referenceId")
    source: str | None = None
    basis_time: datetime | None = Field(default=None, alias="basisTime")
    source_origin: str | None = Field(default=None, alias="sourceOrigin")
    relevance_score: float | None = Field(default=None, alias="relevanceScore")


class DocumentSearchResult(BaseModel):
    sources: list[ChatSource] = Field(default_factory=list)
    was_searched: bool = False
    skipped_reason: str | None = None


class EmbeddingResult(BaseModel):
    vector: list[float] = Field(default_factory=list)
    was_embedded: bool = False
    model: str | None = None
    skipped_reason: str | None = None


class SecurityResult(BaseModel):
    status: SecurityStatus
    code: ChatErrorCode | None = None
    reason: str | None = None


class AnswerGenerationResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    answer: str
    was_generated: bool = False
    skipped_reason: str | None = None
    security_result: SecurityResult | None = Field(default=None, alias="securityResult")


class ErrorResponse(BaseModel):
    code: ChatErrorCode | str
    message: str


class ModelResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    used_vector_search: bool = Field(alias="usedVectorSearch")
    used_rdb_evidence: bool = Field(alias="usedRdbEvidence")
    used_llm_generation: bool = Field(default=False, alias="usedLlmGeneration")
    rdb_evidence_count: int = Field(default=0, alias="rdbEvidenceCount")
    document_source_count: int = Field(default=0, alias="documentSourceCount")
    evidence_count: int = Field(alias="evidenceCount")
    vector_search_skipped_reason: str | None = Field(
        default=None,
        alias="vectorSearchSkippedReason",
    )
    llm_generation_skipped_reason: str | None = Field(
        default=None,
        alias="llmGenerationSkippedReason",
    )


class ChatAnswerResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: int = Field(alias="sessionId")
    message_id: int = Field(alias="messageId")
    intent: ChatIntent
    answer: str
    basis_time: datetime = Field(alias="basisTime")
    urls: list[ChatUrl] = Field(default_factory=list)
    sources: list[ChatSource] = Field(default_factory=list)
    security_result: SecurityResult = Field(alias="securityResult")
    model_result: ModelResult = Field(alias="modelResult")
