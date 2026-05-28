from argparse import Namespace
from io import StringIO
from typing import Any

import anyio
import pytest

from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatIntent, EmbeddingResult
from scripts import chat_check_common, check_qdrant_readonly_search


def _build_args(**overrides: Any) -> Namespace:
    values = {
        "qdrant_url": "http://localhost:6333",
        "collection": "smap_internal_documents",
        "api_key": None,
        "embedding_base_url": "http://localhost:8002",
        "embedding_path": "/embed",
        "embedding_api_key": None,
        "embedding_model": "BAAI/bge-m3",
        "embedding_dimension": 4,
        "qdrant_top_k": 5,
        "qdrant_score_threshold": 0.0,
        "timeout_seconds": 3.0,
        "embedding_timeout_seconds": 3.0,
        "env_file": None,
        "intent": ChatIntent.LINE_BOTTLENECK.value,
        "min_source_count": 1,
        "validate_only": False,
        "json": False,
        "question": "LINE-A01 병목 대응 기준 알려줘",
        "role": "MANUFACTURING_MANAGER",
        "user_id": 1,
        "company_name": "S-MAP",
        "session_id": 1,
        "message_id": 1,
        "requested_at": chat_check_common.DEFAULT_REQUESTED_AT,
    }
    values.update(overrides)
    return Namespace(**values)


class FakeEmbeddingService:
    async def embed_query(self, request):
        return EmbeddingResult(
            vector=[0.1, 0.2, 0.3, 0.4],
            was_embedded=True,
            model="BAAI/bge-m3",
        )


class FakeSkippedEmbeddingService:
    async def embed_query(self, request):
        return EmbeddingResult(
            was_embedded=False,
            model="BAAI/bge-m3",
            skipped_reason="임베딩 기능이 비활성화되어 있습니다.",
        )


class FakeQdrantSearchClient:
    def __init__(self) -> None:
        self.search_payload: dict | None = None

    async def search(self, payload: dict) -> list[dict]:
        self.search_payload = payload
        return [
            {
                "id": "point-1",
                "score": 0.91,
                "payload": {
                    "documentId": "company-line-guide",
                    "chunkId": "line-a01",
                    "documentType": "COMPANY_INFO",
                    "title": "LINE-A01 병목 대응 기준",
                    "chunkText": "LINE-A01 대기 시간이 증가하면 대기 수량과 처리량을 확인합니다.",
                    "url": "/lines/LINE-A01",
                    "allowedRoles": ["MANUFACTURING_MANAGER"],
                    "intentTags": ["LINE_BOTTLENECK"],
                },
            }
        ]


class FakeEmptyQdrantSearchClient:
    async def search(self, payload: dict) -> list[dict]:
        return []


def test_qdrant_readonly_search_builds_settings_from_args() -> None:
    settings = check_qdrant_readonly_search.build_settings(
        _build_args(api_key="qdrant-secret", embedding_api_key="embedding-secret")
    )

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "smap_internal_documents"
    assert settings.qdrant_api_key == "qdrant-secret"
    assert settings.embedding_base_url == "http://localhost:8002"
    assert settings.embedding_api_key == "embedding-secret"
    assert settings.embedding_dimension == 4
    assert settings.qdrant_top_k == 5


def test_qdrant_readonly_search_validate_only_result() -> None:
    args = _build_args()
    settings = check_qdrant_readonly_search.build_settings(args)
    request = chat_check_common.build_chat_answer_request(args)

    result = check_qdrant_readonly_search.build_validate_only_result(
        settings,
        request,
        ChatIntent.LINE_BOTTLENECK,
        min_source_count=1,
    )

    assert result["checkStatus"] == "VALIDATED"
    assert result["mode"] == "VALIDATE_ONLY"
    assert result["collectionName"] == "smap_internal_documents"
    assert result["runtimeMode"] == {
        "apiPrefix": "/api/v1",
        "groundingMode": "QDRANT_ONLY",
        "answerMode": "FALLBACK",
        "ragSearchMode": "ENABLED",
        "enabledGroundingSources": ["QDRANT"],
        "expectedLlmSkippedReason": "LLM 답변 생성 기능이 비활성화되어 있습니다.",
    }
    assert result["endpointSummary"] == {
        "qdrantBaseUrl": "http://localhost:6333",
        "qdrantEndpointScope": "LOCALHOST",
        "embeddingBaseUrl": "http://localhost:8002",
        "embeddingEndpointScope": "LOCALHOST",
    }
    assert result["qdrantUrlConfigured"] is True
    assert result["embeddingBaseUrlConfigured"] is True
    assert result["intent"] == "LINE_BOTTLENECK"
    assert result["role"] == "MANUFACTURING_MANAGER"
    assert result["minSourceCount"] == 1


def test_qdrant_readonly_search_validate_only_requires_qdrant_settings() -> None:
    args = _build_args(qdrant_url=" ")
    settings = check_qdrant_readonly_search.build_settings(args)
    request = chat_check_common.build_chat_answer_request(args)

    with pytest.raises(ChatServiceError) as exc_info:
        check_qdrant_readonly_search.build_validate_only_result(
            settings,
            request,
            ChatIntent.LINE_BOTTLENECK,
            min_source_count=1,
        )

    assert exc_info.value.code.value == "CHAT_QDRANT_001"


def test_qdrant_readonly_search_validate_only_requires_embedding_settings() -> None:
    args = _build_args(embedding_base_url=" ")
    settings = check_qdrant_readonly_search.build_settings(args)
    request = chat_check_common.build_chat_answer_request(args)

    with pytest.raises(ChatServiceError) as exc_info:
        check_qdrant_readonly_search.build_validate_only_result(
            settings,
            request,
            ChatIntent.LINE_BOTTLENECK,
            min_source_count=1,
        )

    assert exc_info.value.code.value == "CHAT_EMBEDDING_001"


def test_qdrant_readonly_search_uses_embedding_and_existing_qdrant_points() -> None:
    args = _build_args()
    settings = check_qdrant_readonly_search.build_settings(args)
    request = chat_check_common.build_chat_answer_request(args)
    qdrant_client = FakeQdrantSearchClient()

    async def run_check() -> dict[str, Any]:
        return await check_qdrant_readonly_search.check_qdrant_readonly_search(
            settings,
            request,
            ChatIntent.LINE_BOTTLENECK,
            embedding_service=FakeEmbeddingService(),
            qdrant_client=qdrant_client,
        )

    result = anyio.run(run_check)

    assert result["checkStatus"] == "PASS"
    assert result["mode"] == "NETWORK"
    assert result["sourceCount"] == 1
    assert result["sourceTitles"] == ["LINE-A01 병목 대응 기준"]
    assert result["sourceTypes"] == ["COMPANY_INFO"]
    assert result["usedReadOnlySearch"] is True
    assert result["runtimeMode"]["groundingMode"] == "QDRANT_ONLY"
    assert result["runtimeMode"]["ragSearchMode"] == "ENABLED"
    assert result["endpointSummary"]["qdrantEndpointScope"] == "LOCALHOST"
    assert result["endpointSummary"]["embeddingEndpointScope"] == "LOCALHOST"
    assert qdrant_client.search_payload is not None
    assert qdrant_client.search_payload["filter"]["must"] == [
        {
            "key": "allowedRoles",
            "match": {"any": ["EXECUTIVE", "MANUFACTURING_MANAGER", "OPERATOR"]},
        },
        {"key": "intentTags", "match": {"any": ["LINE_BOTTLENECK"]}},
    ]


def test_qdrant_readonly_search_fails_when_embedding_is_skipped() -> None:
    args = _build_args()
    settings = check_qdrant_readonly_search.build_settings(args)
    request = chat_check_common.build_chat_answer_request(args)

    async def run_check() -> dict[str, Any]:
        return await check_qdrant_readonly_search.check_qdrant_readonly_search(
            settings,
            request,
            ChatIntent.LINE_BOTTLENECK,
            embedding_service=FakeSkippedEmbeddingService(),
            qdrant_client=FakeEmptyQdrantSearchClient(),
        )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run_check)

    assert "Qdrant read-only 검색이 수행되지 않았습니다." in exc_info.value.message


def test_qdrant_readonly_search_fails_when_no_sources_match() -> None:
    args = _build_args()
    settings = check_qdrant_readonly_search.build_settings(args)
    request = chat_check_common.build_chat_answer_request(args)

    async def run_check() -> dict[str, Any]:
        return await check_qdrant_readonly_search.check_qdrant_readonly_search(
            settings,
            request,
            ChatIntent.LINE_BOTTLENECK,
            embedding_service=FakeEmbeddingService(),
            qdrant_client=FakeEmptyQdrantSearchClient(),
        )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run_check)

    assert "Qdrant read-only 검색 결과가 기준보다 적습니다." in exc_info.value.message


def test_qdrant_readonly_search_main_validate_only_does_not_expose_secret() -> None:
    stdout = StringIO()

    exit_code = check_qdrant_readonly_search.main(
        [
            "--validate-only",
            "--api-key",
            "qdrant-secret-token",
            "--embedding-api-key",
            "embedding-secret-token",
            "--embedding-dimension",
            "4",
            "--json",
        ],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert '"checkStatus": "VALIDATED"' in output
    assert '"apiKeyConfigured": true' in output
    assert '"endpointSummary"' in output
    assert "qdrant-secret-token" not in output
    assert "embedding-secret-token" not in output


def test_qdrant_readonly_search_endpoint_summary_redacts_basic_auth() -> None:
    args = _build_args(
        qdrant_url="http://reader:password@qdrant.qdrant.svc.cluster.local:6333",
        embedding_base_url="http://embedding-service:8002",
    )
    settings = check_qdrant_readonly_search.build_settings(args)

    summary = check_qdrant_readonly_search.build_endpoint_summary(settings)

    assert summary == {
        "qdrantBaseUrl": "http://***:***@qdrant.qdrant.svc.cluster.local:6333",
        "qdrantEndpointScope": "KUBERNETES_SERVICE",
        "embeddingBaseUrl": "http://embedding-service:8002",
        "embeddingEndpointScope": "KUBERNETES_SERVICE",
    }


def test_qdrant_readonly_search_main_guides_match_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_qdrant_readonly_search(*args: Any, **kwargs: Any) -> dict:
        raise check_qdrant_readonly_search.ChatServiceError(
            status_code=500,
            code=check_qdrant_readonly_search.ChatErrorCode.CHAT_QDRANT_004,
            message="Qdrant read-only 검색 결과가 기준보다 적습니다.",
        )

    monkeypatch.setattr(
        check_qdrant_readonly_search,
        "check_qdrant_readonly_search",
        fake_check_qdrant_readonly_search,
    )
    stderr = StringIO()

    exit_code = check_qdrant_readonly_search.main([], stderr=stderr)

    assert exit_code == 1
    assert "code=CHAT_QDRANT_004" in stderr.getvalue()
    assert "nextAction=Qdrant에 질문과 관련된 보고서" in stderr.getvalue()


def test_qdrant_readonly_search_main_guides_local_k8s_embedding_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_qdrant_readonly_search(*args: Any, **kwargs: Any) -> dict:
        raise check_qdrant_readonly_search.ChatServiceError(
            status_code=503,
            code=check_qdrant_readonly_search.ChatErrorCode.CHAT_EMBEDDING_004,
            message="임베딩 서버 호출에 실패했습니다.",
        )

    monkeypatch.setattr(
        check_qdrant_readonly_search,
        "check_qdrant_readonly_search",
        fake_check_qdrant_readonly_search,
    )
    stderr = StringIO()

    exit_code = check_qdrant_readonly_search.main(
        [
            "--embedding-base-url",
            "http://embedding-service:8002",
        ],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "code=CHAT_EMBEDDING_004" in stderr.getvalue()
    assert "nextAction=로컬에서 실행 중이면 embedding-service" in stderr.getvalue()
