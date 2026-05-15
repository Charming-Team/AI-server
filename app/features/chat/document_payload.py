from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.features.chat.schemas import ChatSource


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
    departments: list[str] = Field(default_factory=list)
    company_name: str | None = Field(default=None, alias="companyName")
    intent_tags: list[str] = Field(default_factory=list, alias="intentTags")

    def to_chat_source(self) -> ChatSource:
        return ChatSource(
            source_type=self.document_type,
            title=self.title,
            summary=self._source_summary,
            url=self.url,
            reference_id=self.reference_id,
            source=self._source_name,
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
        return self.payload.to_chat_source()
