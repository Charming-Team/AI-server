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


class FakeUnexpectedEmbeddingService:
    async def embed_query(self, request: ChatAnswerRequest) -> EmbeddingResult:
        raise AssertionError("UNKNOWN 의도에서는 임베딩을 호출하면 안 됩니다.")


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


class FakeLowScoreQdrantClient:
    def __init__(self) -> None:
        self.search_payload: dict | None = None

    async def search(self, payload: dict) -> list[dict]:
        self.search_payload = payload
        return [
            {
                "id": "low-score-point",
                "score": 0.31,
                "payload": {
                    "documentId": "report-low",
                    "chunkId": "summary",
                    "documentType": "REPORT",
                    "title": "낮은 유사도 보고서",
                    "chunkText": "질문과 관련성이 낮은 보고서입니다.",
                    "allowedRoles": ["EXECUTIVE"],
                    "intentTags": ["REPORT_LOOKUP"],
                },
            },
            {
                "id": "missing-score-point",
                "payload": {
                    "documentId": "report-missing-score",
                    "chunkId": "summary",
                    "documentType": "REPORT",
                    "title": "점수 없는 보고서",
                    "chunkText": "점수가 없는 검색 결과입니다.",
                    "allowedRoles": ["EXECUTIVE"],
                    "intentTags": ["REPORT_LOOKUP"],
                },
            },
            {
                "id": "high-score-point",
                "score": 0.82,
                "payload": {
                    "documentId": "report-high",
                    "chunkId": "summary",
                    "documentType": "REPORT",
                    "title": "높은 유사도 보고서",
                    "chunkText": "질문과 관련성이 높은 보고서입니다.",
                    "allowedRoles": ["EXECUTIVE"],
                    "intentTags": ["REPORT_LOOKUP"],
                },
            },
        ]


class FakeOnlyLowScoreQdrantClient:
    def __init__(self) -> None:
        self.search_payload: dict | None = None

    async def search(self, payload: dict) -> list[dict]:
        self.search_payload = payload
        return [
            {
                "id": "low-score-point",
                "score": 0.31,
                "payload": {
                    "documentId": "report-low",
                    "chunkId": "summary",
                    "documentType": "REPORT",
                    "title": "낮은 유사도 보고서",
                    "chunkText": "질문과 관련성이 낮은 보고서입니다.",
                    "allowedRoles": ["EXECUTIVE"],
                    "intentTags": ["REPORT_LOOKUP"],
                },
            }
        ]


class FakeEmptyQdrantClient:
    def __init__(self) -> None:
        self.search_payload: dict | None = None

    async def search(self, payload: dict) -> list[dict]:
        self.search_payload = payload
        return []


def _build_request(company_name: str | None = "S-MAP") -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=10,
        messageId=24,
        user=ChatUserContext(
            userId=1,
            role="EXECUTIVE",
            department="경영기획팀",
            companyName=company_name,
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
    assert result.skipped_reason == "임베딩 기능이 비활성화되어 있습니다."


def test_document_search_service_skips_search_when_intent_is_unknown() -> None:
    service = DocumentSearchService(
        Settings(qdrant_search_enabled=True),
        embedding_service=FakeUnexpectedEmbeddingService(),
    )
    request = _build_request()

    result = anyio.run(service.search, request, ChatIntent.UNKNOWN)

    assert result.was_searched is False
    assert result.sources == []
    assert (
        result.skipped_reason
        == "질문 의도를 분류할 수 없어 Qdrant 문서 검색을 수행하지 않았습니다."
    )


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
            {"key": "intentTags", "match": {"any": ["REPORT_LOOKUP"]}},
        ]
    }


def test_document_search_service_does_not_require_company_name_for_qdrant_search() -> None:
    service = DocumentSearchService(Settings(qdrant_top_k=3))
    request = _build_request(company_name=None)

    payload = service._build_search_payload(
        [0.1, 0.2, 0.3],
        request,
        ChatIntent.REPORT_LOOKUP,
    )

    assert payload["filter"] == {
        "must": [
            {"key": "allowedRoles", "match": {"any": ["EXECUTIVE"]}},
            {"key": "intentTags", "match": {"any": ["REPORT_LOOKUP"]}},
        ]
    }


def test_document_search_service_adds_score_threshold_to_qdrant_payload() -> None:
    service = DocumentSearchService(Settings(qdrant_score_threshold=0.65))
    request = _build_request()

    payload = service._build_search_payload(
        [0.1, 0.2, 0.3],
        request,
        ChatIntent.REPORT_LOOKUP,
    )

    assert payload["score_threshold"] == 0.65


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
    assert sources[0].relevance_score == 0.88


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
    assert result.sources[0].relevance_score == 0.88


def test_document_search_service_filters_points_below_score_threshold() -> None:
    qdrant_client = FakeLowScoreQdrantClient()
    service = DocumentSearchService(
        Settings(qdrant_search_enabled=True, qdrant_score_threshold=0.65),
        embedding_service=FakeEmbeddingService(),
        qdrant_client=qdrant_client,
    )
    request = _build_request()

    result = anyio.run(service.search, request, ChatIntent.REPORT_LOOKUP)

    assert qdrant_client.search_payload is not None
    assert qdrant_client.search_payload["score_threshold"] == 0.65
    assert len(result.sources) == 1
    assert result.sources[0].title == "높은 유사도 보고서"
    assert result.sources[0].relevance_score == 0.82


def test_document_search_service_marks_no_result_reason_when_qdrant_is_empty() -> None:
    qdrant_client = FakeEmptyQdrantClient()
    service = DocumentSearchService(
        Settings(qdrant_search_enabled=True),
        embedding_service=FakeEmbeddingService(),
        qdrant_client=qdrant_client,
    )
    request = _build_request()

    result = anyio.run(service.search, request, ChatIntent.REPORT_LOOKUP)

    assert result.was_searched is True
    assert result.sources == []
    assert result.skipped_reason == "Qdrant 검색 결과가 없습니다."


def test_document_search_service_marks_threshold_reason_when_all_points_are_filtered() -> None:
    qdrant_client = FakeOnlyLowScoreQdrantClient()
    service = DocumentSearchService(
        Settings(qdrant_search_enabled=True, qdrant_score_threshold=0.65),
        embedding_service=FakeEmbeddingService(),
        qdrant_client=qdrant_client,
    )
    request = _build_request()

    result = anyio.run(service.search, request, ChatIntent.REPORT_LOOKUP)

    assert result.was_searched is True
    assert result.sources == []
    assert result.skipped_reason == "Qdrant 관련도 기준을 통과한 검색 결과가 없습니다."
