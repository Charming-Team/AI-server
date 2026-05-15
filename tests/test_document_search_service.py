from datetime import datetime

import anyio

from app.core.config import Settings
from app.features.chat.document_search_service import DocumentSearchService
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatIntent,
    ChatUserContext,
    EmbeddingResult,
)


class FakeEmbeddingService:
    async def embed_query(self, request: ChatAnswerRequest) -> EmbeddingResult:
        return EmbeddingResult(
            vector=[0.1, 0.2, 0.3],
            was_embedded=True,
            model="test-embedding-model",
        )


class FakeQdrantClient:
    def __init__(self) -> None:
        self.search_payload: dict | None = None

    async def search(self, payload: dict) -> list[dict]:
        self.search_payload = payload
        return [
            {
                "id": "point-1",
                "score": 0.88,
                "payload": {
                    "documentId": "report-202605",
                    "chunkId": "summary",
                    "documentType": "REPORT",
                    "title": "2026년 5월 생산 리스크 보고서",
                    "chunkText": "자재 부족과 LINE-A01 병목이 주요 리스크입니다.",
                    "url": "/reports/20",
                    "referenceId": 20,
                    "allowedRoles": ["EXECUTIVE"],
                    "intentTags": ["REPORT_LOOKUP"],
                },
            }
        ]


def _build_request() -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=10,
        messageId=24,
        user=ChatUserContext(
            userId=1,
            role="EXECUTIVE",
            department="경영기획팀",
            companyName="S-MAP",
            status="ACTIVE",
        ),
        question="최근 보고서 요약해줘",
        requestedAt=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
    )


def test_document_search_service_skips_search_when_qdrant_is_disabled() -> None:
    service = DocumentSearchService(Settings(qdrant_search_enabled=False))
    request = _build_request()

    result = anyio.run(service.search, request, ChatIntent.REPORT_LOOKUP)

    assert result.was_searched is False
    assert result.sources == []


def test_document_search_service_marks_search_when_qdrant_is_enabled() -> None:
    service = DocumentSearchService(
        Settings(qdrant_search_enabled=True, embedding_enabled=False)
    )
    request = _build_request()

    result = anyio.run(service.search, request, ChatIntent.REPORT_LOOKUP)

    assert result.was_searched is False
    assert result.sources == []
    assert result.skipped_reason == "Embedding is disabled."


def test_document_search_service_builds_qdrant_search_payload() -> None:
    service = DocumentSearchService(Settings(qdrant_top_k=3))
    request = _build_request()

    payload = service._build_search_payload(
        [0.1, 0.2, 0.3],
        request,
        ChatIntent.REPORT_LOOKUP,
    )

    assert payload["vector"] == [0.1, 0.2, 0.3]
    assert payload["limit"] == 3
    assert payload["with_payload"] is True
    assert payload["filter"] == {
        "must": [
            {"key": "allowedRoles", "match": {"any": ["EXECUTIVE"]}},
            {"key": "companyName", "match": {"value": "S-MAP"}},
            {"key": "intentTags", "match": {"any": ["REPORT_LOOKUP"]}},
        ]
    }


def test_document_search_service_builds_sources_from_qdrant_points() -> None:
    service = DocumentSearchService(Settings())

    sources = service._build_sources(
        [
            {
                "id": "point-1",
                "score": 0.88,
                "payload": {
                    "documentId": "process-guide",
                    "chunkId": "line-a01",
                    "documentType": "PROCESS",
                    "title": "LINE-A01 병목 대응 가이드",
                    "chunkText": "대기시간이 증가하면 LINE-A01 작업 순서를 조정합니다.",
                    "url": "/process/line-a01",
                    "allowedRoles": ["MANUFACTURING_MANAGER"],
                    "intentTags": ["LINE_BOTTLENECK"],
                },
            }
        ]
    )

    assert len(sources) == 1
    assert sources[0].source_type == "PROCESS"
    assert sources[0].title == "LINE-A01 병목 대응 가이드"
    assert sources[0].source == "process-guide:line-a01"


def test_document_search_service_searches_qdrant_when_embedding_is_ready() -> None:
    qdrant_client = FakeQdrantClient()
    service = DocumentSearchService(
        Settings(qdrant_search_enabled=True),
        embedding_service=FakeEmbeddingService(),
        qdrant_client=qdrant_client,
    )
    request = _build_request()

    result = anyio.run(service.search, request, ChatIntent.REPORT_LOOKUP)

    assert result.was_searched is True
    assert qdrant_client.search_payload is not None
    assert qdrant_client.search_payload["vector"] == [0.1, 0.2, 0.3]
    assert result.sources[0].source_type == "REPORT"
    assert result.sources[0].title == "2026년 5월 생산 리스크 보고서"
    assert result.sources[0].url == "/reports/20"
