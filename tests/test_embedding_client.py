import json

import anyio
import httpx

from app.core.config import Settings
from app.features.chat.embedding_client import EmbeddingClient


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
