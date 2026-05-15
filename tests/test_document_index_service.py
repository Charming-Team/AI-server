import anyio
import pytest

from app.core.config import Settings
from app.features.chat.document_index_service import DocumentIndexService
from app.features.chat.document_payload import InternalDocumentInput, QdrantUpsertPoint
from app.features.chat.exceptions import ChatExternalServiceError, ChatServiceError
from app.features.chat.schemas import ChatErrorCode


class FakeEmbeddingClient:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.texts: list[str] = []

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.texts = texts
        return self.vectors


class FakeQdrantIndexClient:
    def __init__(self) -> None:
        self.points: list[QdrantUpsertPoint] = []
        self.deleted_document_id: str | None = None
        self.calls: list[str] = []

    async def delete_by_document_id(self, document_id: str) -> dict:
        self.deleted_document_id = document_id
        self.calls.append("delete")
        return {
            "operation_id": 99,
            "status": "completed",
        }

    async def upsert(self, points: list[QdrantUpsertPoint]) -> dict:
        self.points = points
        self.calls.append("upsert")
        return {
            "operation_id": 100,
            "status": "completed",
        }


def _build_document(content: str = "AAAAAAAAAA\nBBBBBBBBBB") -> InternalDocumentInput:
    return InternalDocumentInput(
        documentId="report-202605",
        documentType="REPORT",
        title="2026년 5월 생산 리스크 보고서",
        content=content,
        url="/reports/20",
        allowedRoles=["EXECUTIVE", "MANUFACTURING_MANAGER"],
        companyName="S-MAP",
        intentTags=["REPORT_LOOKUP"],
    )


def test_document_index_service_indexes_document_chunks_with_batch_embedding() -> None:
    embedding_client = FakeEmbeddingClient([[0.1, 0.2], [0.3, 0.4]])
    qdrant_index_client = FakeQdrantIndexClient()
    service = DocumentIndexService(
        Settings(
            embedding_enabled=True,
            embedding_dimension=2,
            document_chunk_size=10,
            document_chunk_overlap=0,
        ),
        embedding_client=embedding_client,
        qdrant_index_client=qdrant_index_client,
    )

    result = anyio.run(service.index_document, _build_document())

    assert result.document_id == "report-202605"
    assert result.chunk_count == 2
    assert result.indexed_count == 2
    assert result.operation == {"operation_id": 100, "status": "completed"}
    assert embedding_client.texts == ["AAAAAAAAAA", "BBBBBBBBBB"]
    assert qdrant_index_client.deleted_document_id == "report-202605"
    assert qdrant_index_client.calls == ["delete", "upsert"]
    assert len(qdrant_index_client.points) == 2
    assert qdrant_index_client.points[0].payload.chunk_id == "chunk-0001"
    assert qdrant_index_client.points[1].vector == [0.3, 0.4]


def test_document_index_service_skips_empty_content() -> None:
    embedding_client = FakeEmbeddingClient([[0.1, 0.2]])
    qdrant_index_client = FakeQdrantIndexClient()
    service = DocumentIndexService(
        Settings(embedding_enabled=True, embedding_dimension=2),
        embedding_client=embedding_client,
        qdrant_index_client=qdrant_index_client,
    )

    result = anyio.run(service.index_document, _build_document(content="  \n  "))

    assert result.chunk_count == 0
    assert result.indexed_count == 0
    assert result.skipped_reason == "문서 본문이 비어 있습니다."
    assert embedding_client.texts == []
    assert qdrant_index_client.points == []


def test_document_index_service_skips_when_embedding_is_disabled() -> None:
    embedding_client = FakeEmbeddingClient([[0.1, 0.2]])
    qdrant_index_client = FakeQdrantIndexClient()
    service = DocumentIndexService(
        Settings(
            embedding_enabled=False,
            document_chunk_size=10,
            document_chunk_overlap=0,
        ),
        embedding_client=embedding_client,
        qdrant_index_client=qdrant_index_client,
    )

    result = anyio.run(service.index_document, _build_document())

    assert result.chunk_count == 2
    assert result.indexed_count == 0
    assert result.skipped_reason == "임베딩 기능이 비활성화되어 있습니다."
    assert embedding_client.texts == []
    assert qdrant_index_client.points == []


def test_document_index_service_rejects_embedding_count_mismatch() -> None:
    service = DocumentIndexService(
        Settings(
            embedding_enabled=True,
            embedding_dimension=2,
            document_chunk_size=10,
            document_chunk_overlap=0,
        ),
        embedding_client=FakeEmbeddingClient([[0.1, 0.2]]),
        qdrant_index_client=FakeQdrantIndexClient(),
    )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(service.index_document, _build_document())

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == ChatErrorCode.CHAT_EMBEDDING_002
    assert exc_info.value.message == "임베딩 응답 개수가 문서 청크 개수와 일치하지 않습니다."


def test_document_index_service_rejects_invalid_document_before_external_calls() -> None:
    embedding_client = FakeEmbeddingClient([[0.1, 0.2]])
    qdrant_index_client = FakeQdrantIndexClient()
    service = DocumentIndexService(
        Settings(
            embedding_enabled=True,
            embedding_dimension=2,
            document_chunk_size=10,
            document_chunk_overlap=0,
        ),
        embedding_client=embedding_client,
        qdrant_index_client=qdrant_index_client,
    )
    document = _build_document()
    document.allowed_roles = ["ADMIN"]

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(service.index_document, document)

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == ChatErrorCode.CHAT_DOCUMENT_002
    assert embedding_client.texts == []
    assert qdrant_index_client.calls == []


def test_document_index_service_rejects_large_content_before_external_calls() -> None:
    embedding_client = FakeEmbeddingClient([[0.1, 0.2]])
    qdrant_index_client = FakeQdrantIndexClient()
    service = DocumentIndexService(
        Settings(
            embedding_enabled=True,
            embedding_dimension=2,
            document_content_max_chars=1_000,
        ),
        embedding_client=embedding_client,
        qdrant_index_client=qdrant_index_client,
    )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(service.index_document, _build_document(content="A" * 1_001))

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == ChatErrorCode.CHAT_DOCUMENT_002
    assert exc_info.value.message == "문서 본문은 최대 1000자까지 인덱싱할 수 있습니다."
    assert embedding_client.texts == []
    assert qdrant_index_client.calls == []


def test_document_index_service_rejects_too_many_chunks_before_external_calls() -> None:
    embedding_client = FakeEmbeddingClient([[0.1, 0.2]])
    qdrant_index_client = FakeQdrantIndexClient()
    service = DocumentIndexService(
        Settings(
            embedding_enabled=True,
            embedding_dimension=2,
            document_chunk_size=1,
            document_chunk_overlap=0,
            document_max_chunks=1,
        ),
        embedding_client=embedding_client,
        qdrant_index_client=qdrant_index_client,
    )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(service.index_document, _build_document(content="A\nB"))

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == ChatErrorCode.CHAT_DOCUMENT_002
    assert exc_info.value.message == "문서 청크는 최대 1개까지 인덱싱할 수 있습니다."
    assert embedding_client.texts == []
    assert qdrant_index_client.calls == []


def test_document_index_service_rejects_embedding_dimension_mismatch() -> None:
    service = DocumentIndexService(
        Settings(
            embedding_enabled=True,
            embedding_dimension=3,
            document_chunk_size=10,
            document_chunk_overlap=0,
        ),
        embedding_client=FakeEmbeddingClient([[0.1, 0.2], [0.3, 0.4]]),
        qdrant_index_client=FakeQdrantIndexClient(),
    )

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(service.index_document, _build_document())

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == ChatErrorCode.CHAT_EMBEDDING_003
    assert exc_info.value.message == "임베딩 벡터 차원이 설정값과 일치하지 않습니다."
