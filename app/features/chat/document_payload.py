from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.features.chat.schemas import ChatSource


def _normalize_upper_text(value: str) -> str:
    return value.strip().upper()


def _strip_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return value.strip()


def _strip_optional_text(value: Any) -> Any:
    stripped_value = _strip_text(value)
    if not isinstance(stripped_value, str):
        return stripped_value
    return stripped_value or None


def _normalize_upper_text_list(value: Any) -> Any:
    if value is None:
        return []
    if not isinstance(value, list):
        return value

    normalized_values: list[str] = []
    seen_values: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            normalized_values.append(item)
            continue
        normalized_item = _normalize_upper_text(item)
        if not normalized_item or normalized_item in seen_values:
            continue
        seen_values.add(normalized_item)
        normalized_values.append(normalized_item)
    return normalized_values


class InternalDocumentInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(alias="documentId")
    document_type: str = Field(alias="documentType")
    title: str
    content: str
    summary: str | None = None
    url: str | None = None
    reference_type: str | None = Field(default=None, alias="referenceType")
    reference_id: int | None = Field(default=None, alias="referenceId")
    basis_time: datetime | None = Field(default=None, alias="basisTime")
    allowed_roles: list[str] = Field(default_factory=list, alias="allowedRoles")
    company_name: str | None = Field(default=None, alias="companyName")
    intent_tags: list[str] = Field(default_factory=list, alias="intentTags")
    requested_by_role: str | None = Field(default=None, alias="requestedByRole")

    @field_validator("document_id", "title", mode="before")
    @classmethod
    def strip_required_text(cls, value: Any) -> Any:
        return _strip_text(value)

    @field_validator("url", mode="before")
    @classmethod
    def strip_optional_text(cls, value: Any) -> Any:
        return _strip_optional_text(value)

    @field_validator("document_type", mode="before")
    @classmethod
    def normalize_document_type(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return _normalize_upper_text(value)

    @field_validator("allowed_roles", "intent_tags", mode="before")
    @classmethod
    def normalize_access_metadata(cls, value: Any) -> Any:
        return _normalize_upper_text_list(value)

    @field_validator("requested_by_role", mode="before")
    @classmethod
    def normalize_requested_by_role(cls, value: Any) -> Any:
        if value is None or not isinstance(value, str):
            return value
        normalized_value = _normalize_upper_text(value)
        return normalized_value or None


class InternalDocumentDeleteRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(alias="documentId")

    @field_validator("document_id", mode="before")
    @classmethod
    def strip_document_id(cls, value: Any) -> Any:
        return _strip_text(value)


class InternalDocumentPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(alias="documentId")
    document_type: str = Field(alias="documentType")
    title: str
    chunk_text: str = Field(alias="chunkText")
    chunk_id: str | None = Field(default=None, alias="chunkId")
    summary: str | None = None
    url: str | None = None
    reference_type: str | None = Field(default=None, alias="referenceType")
    reference_id: int | None = Field(default=None, alias="referenceId")
    basis_time: datetime | None = Field(default=None, alias="basisTime")
    allowed_roles: list[str] = Field(default_factory=list, alias="allowedRoles")
    company_name: str | None = Field(default=None, alias="companyName")
    intent_tags: list[str] = Field(default_factory=list, alias="intentTags")

    @field_validator("document_id", "title", mode="before")
    @classmethod
    def strip_required_text(cls, value: Any) -> Any:
        return _strip_text(value)

    @field_validator("url", mode="before")
    @classmethod
    def strip_optional_text(cls, value: Any) -> Any:
        return _strip_optional_text(value)

    @field_validator("document_type", mode="before")
    @classmethod
    def normalize_document_type(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return _normalize_upper_text(value)

    @field_validator("allowed_roles", "intent_tags", mode="before")
    @classmethod
    def normalize_access_metadata(cls, value: Any) -> Any:
        return _normalize_upper_text_list(value)

    def to_chat_source(self, relevance_score: float | None = None) -> ChatSource:
        return ChatSource(
            source_type=self.document_type,
            title=self.title,
            summary=self._source_summary,
            url=self.url,
            reference_id=self.reference_id,
            source=self._source_name,
            basis_time=self.basis_time,
            source_origin="QDRANT",
            relevance_score=relevance_score,
        )

    @property
    def _source_summary(self) -> str:
        text = self.summary or self.chunk_text
        if len(text) <= 300:
            return text
        return f"{text[:297]}..."

    @property
    def _source_name(self) -> str:
        if not self.chunk_id:
            return self.document_id
        return f"{self.document_id}:{self.chunk_id}"


class QdrantSearchPoint(BaseModel):
    id: int | str
    score: float | None = None
    payload: InternalDocumentPayload

    def to_chat_source(self) -> ChatSource:
        return self.payload.to_chat_source(relevance_score=self.score)


class QdrantUpsertPoint(BaseModel):
    id: str
    vector: list[float]
    payload: InternalDocumentPayload

    def to_qdrant_point(self) -> dict:
        return {
            "id": self.id,
            "vector": self.vector,
            "payload": self.payload.model_dump(by_alias=True, exclude_none=True),
        }
