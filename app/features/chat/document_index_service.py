from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.features.chat.document_index_builder import DocumentIndexBuilder
from app.features.chat.document_index_policy import DocumentIndexPolicy
from app.features.chat.document_payload import InternalDocumentInput
from app.features.chat.embedding_client import EmbeddingClient
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.qdrant_client import QdrantDocumentIndexClient
from app.features.chat.schemas import ChatErrorCode


class DocumentIndexResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    document_id: str = Field(alias="documentId")
    chunk_count: int = Field(alias="chunkCount")
    indexed_count: int = Field(alias="indexedCount")
    operation: dict = Field(default_factory=dict)
    skipped_reason: str | None = Field(default=None, alias="skippedReason")


class DocumentIndexService:
    def __init__(
        self,
        settings: Settings,
        index_builder: DocumentIndexBuilder | None = None,
        index_policy: DocumentIndexPolicy | None = None,
        embedding_client: EmbeddingClient | None = None,
        qdrant_index_client: QdrantDocumentIndexClient | None = None,
    ) -> None:
        self.settings = settings
        self.index_builder = index_builder or DocumentIndexBuilder(settings)
        self.index_policy = index_policy or DocumentIndexPolicy()
        self.embedding_client = embedding_client or EmbeddingClient(settings)
        self.qdrant_index_client = qdrant_index_client or QdrantDocumentIndexClient(settings)

    async def index_document(self, document: InternalDocumentInput) -> DocumentIndexResult:
        self.index_policy.validate(document)

        payloads = self.index_builder.build_payloads(document)
        if not payloads:
            return self._skipped_result(document, "Document content is empty.", chunk_count=0)

        if not self.settings.embedding_enabled:
            return self._skipped_result(
                document,
                "Embedding is disabled.",
                chunk_count=len(payloads),
            )

        vectors = await self.embedding_client.embed_many(
            [payload.chunk_text for payload in payloads]
        )
        self._validate_vectors(vectors, expected_count=len(payloads))

        points = [
            self.index_builder.build_point(payload, vector)
            for payload, vector in zip(payloads, vectors, strict=True)
        ]
        await self.qdrant_index_client.delete_by_document_id(document.document_id)
        operation = await self.qdrant_index_client.upsert(points)

        return DocumentIndexResult(
            document_id=document.document_id,
            chunk_count=len(payloads),
            indexed_count=len(points),
            operation=operation,
        )

    def _validate_vectors(
        self,
        vectors: list[list[float]],
        expected_count: int,
    ) -> None:
        if len(vectors) != expected_count:
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_EMBEDDING_002,
                message="임베딩 응답 개수가 문서 청크 개수와 일치하지 않습니다.",
            )

        for vector in vectors:
            if len(vector) != self.settings.embedding_dimension:
                raise ChatExternalServiceError(
                    status_code=502,
                    code=ChatErrorCode.CHAT_EMBEDDING_003,
                    message="임베딩 벡터 차원이 설정값과 일치하지 않습니다.",
                )

    def _skipped_result(
        self,
        document: InternalDocumentInput,
        skipped_reason: str,
        chunk_count: int,
    ) -> DocumentIndexResult:
        return DocumentIndexResult(
            document_id=document.document_id,
            chunk_count=chunk_count,
            indexed_count=0,
            skipped_reason=skipped_reason,
        )
