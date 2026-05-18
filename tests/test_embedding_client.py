import json

import anyio
import httpx
import pytest

from app.core.config import Settings
from app.features.chat.embedding_client import EmbeddingClient
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.schemas import ChatErrorCode


def test_embedding_client_builds_huggingface_payload_url_and_headers() -> None:
    client = EmbeddingClient(
        Settings(
            embedding_base_url="http://embedding.svc.cluster.local:8000/",
            embedding_path="embed",
            embedding_api_key="embedding-token",
        )
    )

    assert client._build_payload("납기 위험 알려줘") == {"inputs": ["납기 위험 알려줘"]}
    assert client._embedding_url == "http://embedding.svc.cluster.local:8000/embed"
    assert client._headers == {
        "Content-Type": "application/json",
        "Authorization": "Bearer embedding-token",
    }


def test_embedding_client_parses_huggingface_tei_response() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=[[0.1, 0.2, 0.3]])

    async def run_embed() -> list[float]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = EmbeddingClient(
                Settings(embedding_base_url="http://embedding.local"),
                http_client=http_client,
            )
            return await client.embed("최근 보고서 요약해줘")

    vector = anyio.run(run_embed)

    assert vector == [0.1, 0.2, 0.3]
    assert captured_request["url"] == "http://embedding.local/embed"
    assert captured_request["body"] == {"inputs": ["최근 보고서 요약해줘"]}


def test_embedding_client_embed_many_sends_single_batch_request() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=[[0.1, 0.2], [0.3, 0.4]])

    async def run_embed_many() -> list[list[float]]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = EmbeddingClient(
                Settings(embedding_base_url="http://embedding.local"),
                http_client=http_client,
            )
            return await client.embed_many(
                [
                    "첫 번째 보고서 청크",
                    "두 번째 보고서 청크",
                ]
            )

    vectors = anyio.run(run_embed_many)

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert captured_request["url"] == "http://embedding.local/embed"
    assert captured_request["body"] == {
        "inputs": [
            "첫 번째 보고서 청크",
            "두 번째 보고서 청크",
        ]
    }


def test_embedding_client_embed_many_deduplicates_repeated_texts() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=[[0.1, 0.2], [0.3, 0.4]])

    async def run_embed_many() -> list[list[float]]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = EmbeddingClient(
                Settings(embedding_base_url="http://embedding.local"),
                http_client=http_client,
            )
            return await client.embed_many(
                [
                    "반복 보고서 청크",
                    "다른 보고서 청크",
                    "반복 보고서 청크",
                ]
            )

    vectors = anyio.run(run_embed_many)

    assert captured_request["body"] == {
        "inputs": [
            "반복 보고서 청크",
            "다른 보고서 청크",
        ]
    }
    assert vectors == [[0.1, 0.2], [0.3, 0.4], [0.1, 0.2]]


def test_embedding_client_rejects_deduplicated_response_count_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[[0.1, 0.2]])

    async def run_embed_many() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = EmbeddingClient(Settings(), http_client=http_client)
            await client.embed_many(
                [
                    "반복 보고서 청크",
                    "다른 보고서 청크",
                    "반복 보고서 청크",
                ]
            )

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run_embed_many)

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == ChatErrorCode.CHAT_EMBEDDING_002
    assert exc_info.value.message == "임베딩 응답 개수가 요청 텍스트 개수와 일치하지 않습니다."


def test_embedding_client_embed_many_skips_empty_texts() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=[])

    async def run_embed_many() -> list[list[float]]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = EmbeddingClient(Settings(), http_client=http_client)
            return await client.embed_many([])

    vectors = anyio.run(run_embed_many)

    assert vectors == []
    assert called is False


def test_embedding_client_parses_openai_compatible_embedding_response() -> None:
    client = EmbeddingClient(Settings())
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://embedding.local/embed"),
        json={
            "data": [
                {
                    "embedding": [0.4, 0.5, 0.6],
                    "index": 0,
                }
            ]
        },
    )

    vector = client._parse_response(response)

    assert vector == [0.4, 0.5, 0.6]


def test_embedding_client_parses_openai_compatible_embedding_batch_response() -> None:
    client = EmbeddingClient(Settings())
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://embedding.local/embed"),
        json={
            "data": [
                {
                    "embedding": [0.1, 0.2],
                    "index": 0,
                },
                {
                    "embedding": [0.3, 0.4],
                    "index": 1,
                },
            ]
        },
    )

    vectors = client._parse_batch_response(response)

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_embedding_client_raises_external_error_on_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "embedding failed"})

    async def run_embed() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = EmbeddingClient(Settings(), http_client=http_client)
            await client.embed("최근 보고서 요약해줘")

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run_embed)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ChatErrorCode.CHAT_EMBEDDING_004
    assert exc_info.value.message == "임베딩 서버 호출에 실패했습니다."


def test_embedding_client_raises_external_error_on_invalid_embedding_value() -> None:
    client = EmbeddingClient(Settings())
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://embedding.local/embed"),
        json=[["not-a-number"]],
    )

    with pytest.raises(ChatExternalServiceError) as exc_info:
        client._parse_response(response)

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == ChatErrorCode.CHAT_EMBEDDING_002
    assert exc_info.value.message == "임베딩 응답 형식이 올바르지 않습니다."
