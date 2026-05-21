from argparse import Namespace
from io import StringIO
from typing import Any

import anyio
import pytest

from app.features.chat.document_payload import QdrantUpsertPoint
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode, ChatIntent
from scripts import chat_check_common, check_qdrant_vector_search


def _build_args(**overrides: Any) -> Namespace:
    values = {
        "qdrant_url": "http://localhost:6333",
        "collection": "smap_internal_documents",
        "api_key": None,
        "env_file": None,
        "embedding_dimension": 4,
        "timeout_seconds": 3.0,
        "document_id": "smoke-company-line-bottleneck",
        "title": "LINE-A01 병목 대응 기준",
        "content": "LINE-A01 병목 대응 기준입니다.",
        "url": "/lines/LINE-A01",
        "intent": ChatIntent.LINE_BOTTLENECK.value,
        "role": "MANUFACTURING_MANAGER",
        "company_name": "S-MAP",
        "keep_sample": False,
        "validate_only": False,
        "json": False,
        "question": "LINE-A01 병목 대응 기준 알려줘",
        "user_id": 1,
        "session_id": 1,
        "message_id": 1,
        "requested_at": chat_check_common.DEFAULT_REQUESTED_AT,
    }
    values.update(overrides)
    return Namespace(**values)


class FakeQdrantIndexClient:
    def __init__(self) -> None:
        self.deleted_document_ids: list[str] = []
        self.upserted_points: list[QdrantUpsertPoint] = []

    async def delete_by_document_id(self, document_id: str) -> dict:
        self.deleted_document_ids.append(document_id)
        return {"operation_id": 1, "status": "acknowledged"}

    async def upsert(self, points: list[QdrantUpsertPoint]) -> dict:
        self.upserted_points.extend(points)
        return {"operation_id": 2, "status": "completed"}


class FakeQdrantSearchClient:
    def __init__(self, index_client: FakeQdrantIndexClient) -> None:
        self.index_client = index_client
        self.search_payload: dict | None = None

    async def search(self, payload: dict) -> list[dict]:
        self.search_payload = payload
        point = self.index_client.upserted_points[0].to_qdrant_point()
        point["score"] = 0.99
        return [point]


class FakeEmptyQdrantSearchClient:
    def __init__(self) -> None:
        self.search_payload: dict | None = None

    async def search(self, payload: dict) -> list[dict]:
        self.search_payload = payload
        return []


def test_qdrant_vector_search_script_builds_settings_from_args() -> None:
    settings = check_qdrant_vector_search.build_settings(_build_args(api_key="secret"))

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "smap_internal_documents"
    assert settings.qdrant_api_key == "secret"
    assert settings.embedding_dimension == 4
    assert settings.qdrant_timeout_seconds == 3.0


def test_qdrant_vector_search_script_builds_sample_document() -> None:
    document = check_qdrant_vector_search.build_sample_document(
        _build_args(role="EXECUTIVE", intent=ChatIntent.REPORT_LOOKUP.value)
    )

    assert document.document_id == "smoke-company-line-bottleneck"
    assert document.document_type == "COMPANY_INFO"
    assert document.allowed_roles == ["EXECUTIVE"]
    assert document.intent_tags == ["REPORT_LOOKUP"]
    assert document.requested_by_role == "MANUFACTURING_MANAGER"


def test_qdrant_vector_search_script_validate_only_result() -> None:
    args = _build_args()
    settings = check_qdrant_vector_search.build_settings(args)
    document = check_qdrant_vector_search.build_sample_document(args)
    vector = check_qdrant_vector_search.build_static_vector(settings.embedding_dimension)
    point = check_qdrant_vector_search.build_sample_point(settings, document, vector)

    result = check_qdrant_vector_search.build_validate_only_result(
        settings,
        document,
        point,
        args,
    )

    assert result["checkStatus"] == "VALIDATED"
    assert result["collectionName"] == "smap_internal_documents"
    assert result["documentId"] == "smoke-company-line-bottleneck"
    assert result["embeddingDimension"] == 4
    assert result["networkChecked"] is False


def test_qdrant_vector_search_script_fails_invalid_dimension() -> None:
    with pytest.raises(ChatServiceError) as exc_info:
        check_qdrant_vector_search.build_static_vector(0)

    assert exc_info.value.code == ChatErrorCode.CHAT_EMBEDDING_003


def test_qdrant_vector_search_script_indexes_searches_and_cleans_sample() -> None:
    args = _build_args()
    settings = check_qdrant_vector_search.build_settings(args)
    document = check_qdrant_vector_search.build_sample_document(args)
    request = chat_check_common.build_chat_answer_request(args)
    index_client = FakeQdrantIndexClient()
    search_client = FakeQdrantSearchClient(index_client)

    async def run_check() -> dict[str, Any]:
        return await check_qdrant_vector_search.check_qdrant_vector_search(
            settings,
            document,
            request,
            ChatIntent.LINE_BOTTLENECK,
            qdrant_index_client=index_client,
            qdrant_search_client=search_client,
        )

    result = anyio.run(run_check)

    assert result["checkStatus"] == "PASS"
    assert result["matchedSourceCount"] == 1
    assert result["matchedTitles"] == ["LINE-A01 병목 대응 기준"]
    assert result["usedCleanup"] is True
    assert index_client.deleted_document_ids == [
        "smoke-company-line-bottleneck",
        "smoke-company-line-bottleneck",
    ]
    assert search_client.search_payload is not None
    assert search_client.search_payload["filter"]["must"] == [
        {"key": "allowedRoles", "match": {"any": ["MANUFACTURING_MANAGER"]}},
        {"key": "intentTags", "match": {"any": ["LINE_BOTTLENECK"]}},
    ]


def test_qdrant_vector_search_script_can_keep_sample() -> None:
    args = _build_args(keep_sample=True)
    settings = check_qdrant_vector_search.build_settings(args)
    document = check_qdrant_vector_search.build_sample_document(args)
    request = chat_check_common.build_chat_answer_request(args)
    index_client = FakeQdrantIndexClient()
    search_client = FakeQdrantSearchClient(index_client)

    async def run_check() -> dict[str, Any]:
        return await check_qdrant_vector_search.check_qdrant_vector_search(
            settings,
            document,
            request,
            ChatIntent.LINE_BOTTLENECK,
            keep_sample=True,
            qdrant_index_client=index_client,
            qdrant_search_client=search_client,
        )

    result = anyio.run(run_check)

    assert result["usedCleanup"] is False
    assert index_client.deleted_document_ids == ["smoke-company-line-bottleneck"]


def test_qdrant_vector_search_script_cleans_sample_on_search_failure() -> None:
    args = _build_args()
    settings = check_qdrant_vector_search.build_settings(args)
    document = check_qdrant_vector_search.build_sample_document(args)
    request = chat_check_common.build_chat_answer_request(args)
    index_client = FakeQdrantIndexClient()
    search_client = FakeEmptyQdrantSearchClient()

    async def run_check() -> dict[str, Any]:
        return await check_qdrant_vector_search.check_qdrant_vector_search(
            settings,
            document,
            request,
            ChatIntent.LINE_BOTTLENECK,
            qdrant_index_client=index_client,
            qdrant_search_client=search_client,
        )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run_check)

    assert exc_info.value.code == ChatErrorCode.CHAT_QDRANT_004
    assert index_client.deleted_document_ids == [
        "smoke-company-line-bottleneck",
        "smoke-company-line-bottleneck",
    ]


def test_qdrant_vector_search_script_validate_only_main_does_not_expose_api_key() -> None:
    stdout = StringIO()

    exit_code = check_qdrant_vector_search.main(
        [
            "--validate-only",
            "--api-key",
            "qdrant-secret-token",
            "--embedding-dimension",
            "4",
            "--json",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert '"checkStatus": "VALIDATED"' in stdout.getvalue()
    assert '"apiKeyConfigured": true' in stdout.getvalue()
    assert "qdrant-secret-token" not in stdout.getvalue()


def test_qdrant_vector_search_script_main_returns_one_on_service_error() -> None:
    stderr = StringIO()

    exit_code = check_qdrant_vector_search.main(
        ["--validate-only", "--embedding-dimension", "0"],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "Qdrant Vector 검색 점검 실패" in stderr.getvalue()
    assert "CHAT_EMBEDDING_003" in stderr.getvalue()
