import anyio
import httpx

from app.core.config import Settings
from app.features.chat.qdrant_client import QdrantDocumentSearchClient


def test_qdrant_client_builds_search_url_and_headers() -> None:
    client = QdrantDocumentSearchClient(
        Settings(
            qdrant_url="http://qdrant.qdrant.svc.cluster.local:6333/",
            qdrant_collection="smap_documents",
            qdrant_api_key="qdrant-token",
        )
    )

    assert (
        client._search_url
        == "http://qdrant.qdrant.svc.cluster.local:6333/collections/smap_documents/points/search"
    )
    assert client._headers == {"api-key": "qdrant-token"}


def test_qdrant_client_search_returns_result_points() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["headers"] = dict(request.headers)
        captured_request["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "result": [
                    {
                        "id": "point-1",
                        "score": 0.91,
                        "payload": {
                            "documentId": "report-202605",
                            "documentType": "REPORT",
                            "title": "2026년 5월 생산 리스크 보고서",
                            "chunkText": "자재 부족과 라인 병목이 주요 리스크입니다.",
                        },
                    }
                ],
                "status": "ok",
            },
        )

    async def run_search() -> list[dict]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentSearchClient(
                Settings(qdrant_api_key="qdrant-token"),
                http_client=http_client,
            )
            return await client.search({"vector": [0.1, 0.2], "limit": 1})

    points = anyio.run(run_search)

    assert len(points) == 1
    assert points[0]["id"] == "point-1"
    assert captured_request["url"].endswith(
        "/collections/smap_internal_documents/points/search"
    )
    assert captured_request["headers"]["api-key"] == "qdrant-token"
