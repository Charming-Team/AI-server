import anyio
import httpx
import pytest

from app.core.config import Settings
from app.features.chat.document_payload import InternalDocumentPayload, QdrantUpsertPoint
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.qdrant_client import (
    QdrantDocumentIndexClient,
    QdrantDocumentSearchClient,
)
from app.features.chat.schemas import ChatErrorCode


def _build_upsert_point() -> QdrantUpsertPoint:
    return QdrantUpsertPoint(
        id="point-1",
        vector=[0.1, 0.2, 0.3],
        payload=InternalDocumentPayload(
            documentId="report-202605",
            documentType="REPORT",
            title="2026년 5월 생산 리스크 보고서",
            chunkText="자재 부족과 LINE-A01 병목이 주요 리스크입니다.",
            chunkId="chunk-0001",
            url="/reports/20",
            allowedRoles=["EXECUTIVE", "MANUFACTURING_MANAGER"],
            intentTags=["REPORT_LOOKUP"],
        ),
    )


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
    assert (
        client._collection_url
        == "http://qdrant.qdrant.svc.cluster.local:6333/collections/smap_documents"
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


def test_qdrant_client_checks_collection_dimension() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["method"] = request.method
        captured_request["url"] = str(request.url)
        captured_request["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "result": {
                    "status": "green",
                    "points_count": 12,
                    "config": {
                        "params": {
                            "vectors": {
                                "size": 1024,
                                "distance": "Cosine",
                            }
                        }
                    },
                },
                "status": "ok",
            },
        )

    async def run_check():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentSearchClient(
                Settings(qdrant_api_key="qdrant-token", embedding_dimension=1024),
                http_client=http_client,
            )
            return await client.check_collection()

    result = anyio.run(run_check)

    assert captured_request["method"] == "GET"
    assert captured_request["url"].endswith("/collections/smap_internal_documents")
    assert captured_request["headers"]["api-key"] == "qdrant-token"
    assert result.collection_name == "smap_internal_documents"
    assert result.status == "green"
    assert result.expected_dimension == 1024
    assert result.actual_dimension == 1024
    assert result.is_dimension_matched is True
    assert result.points_count == 12


def test_qdrant_client_marks_collection_dimension_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": {
                    "status": "green",
                    "points_count": 0,
                    "config": {
                        "params": {
                            "vectors": {
                                "size": 384,
                                "distance": "Cosine",
                            }
                        }
                    },
                },
                "status": "ok",
            },
        )

    async def run_check():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentSearchClient(
                Settings(embedding_dimension=1024),
                http_client=http_client,
            )
            return await client.check_collection()

    result = anyio.run(run_check)

    assert result.expected_dimension == 1024
    assert result.actual_dimension == 384
    assert result.is_dimension_matched is False


def test_qdrant_client_checks_single_named_vector_dimension() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": {
                    "config": {
                        "params": {
                            "vectors": {
                                "document": {
                                    "size": 1024,
                                    "distance": "Cosine",
                                }
                            }
                        }
                    }
                },
                "status": "ok",
            },
        )

    async def run_check():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentSearchClient(
                Settings(embedding_dimension=1024),
                http_client=http_client,
            )
            return await client.check_collection()

    result = anyio.run(run_check)

    assert result.actual_dimension == 1024
    assert result.is_dimension_matched is True


def test_qdrant_client_raises_external_error_on_collection_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": "error"})

    async def run_check() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentSearchClient(Settings(), http_client=http_client)
            await client.check_collection()

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run_check)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ChatErrorCode.CHAT_QDRANT_002
    assert exc_info.value.message == "Qdrant 컬렉션 조회에 실패했습니다."


def test_qdrant_client_marks_collection_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"status": "error"})

    async def run_check() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentSearchClient(Settings(), http_client=http_client)
            await client.check_collection()

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run_check)

    assert exc_info.value.status_code == 404
    assert exc_info.value.code == ChatErrorCode.CHAT_QDRANT_002
    assert exc_info.value.message == "Qdrant 컬렉션 조회에 실패했습니다."


@pytest.mark.parametrize(
    "body",
    [
        ["not-a-dict"],
        {"result": []},
    ],
)
def test_qdrant_client_raises_external_error_on_invalid_collection_shape(
    body: object,
) -> None:
    client = QdrantDocumentSearchClient(Settings())
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "http://qdrant.local/collections/docs"),
        json=body,
    )

    with pytest.raises(ChatExternalServiceError) as exc_info:
        client._parse_collection_check_response(response)

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == ChatErrorCode.CHAT_QDRANT_003
    assert exc_info.value.message == "Qdrant 응답 형식이 올바르지 않습니다."


def test_qdrant_client_raises_external_error_on_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": "error"})

    async def run_search() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentSearchClient(Settings(), http_client=http_client)
            await client.search({"vector": [0.1, 0.2], "limit": 1})

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run_search)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ChatErrorCode.CHAT_QDRANT_002
    assert exc_info.value.message == "Qdrant 검색에 실패했습니다."


@pytest.mark.parametrize(
    "body",
    [
        ["not-a-dict"],
        {"result": {"id": "point-1"}},
        {"result": ["not-a-dict"]},
    ],
)
def test_qdrant_client_raises_external_error_on_invalid_result_shape(
    body: object,
) -> None:
    client = QdrantDocumentSearchClient(Settings())
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://qdrant.local/search"),
        json=body,
    )

    with pytest.raises(ChatExternalServiceError) as exc_info:
        client._parse_response(response)

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == ChatErrorCode.CHAT_QDRANT_003
    assert exc_info.value.message == "Qdrant 응답 형식이 올바르지 않습니다."


def test_qdrant_index_client_builds_points_url_and_headers() -> None:
    client = QdrantDocumentIndexClient(
        Settings(
            qdrant_url="http://qdrant.qdrant.svc.cluster.local:6333/",
            qdrant_collection="smap_documents",
            qdrant_api_key="qdrant-token",
        )
    )

    assert (
        client._points_url
        == "http://qdrant.qdrant.svc.cluster.local:6333/collections/smap_documents/points?wait=true"
    )
    assert client._headers == {"api-key": "qdrant-token"}
    assert (
        client._points_delete_url
        == "http://qdrant.qdrant.svc.cluster.local:6333/collections/smap_documents/points/delete?wait=true"
    )
    assert (
        client._collection_url
        == "http://qdrant.qdrant.svc.cluster.local:6333/collections/smap_documents"
    )


def test_qdrant_index_client_upserts_points() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["headers"] = dict(request.headers)
        captured_request["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "result": {
                    "operation_id": 100,
                    "status": "completed",
                },
                "status": "ok",
            },
        )

    async def run_upsert() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentIndexClient(
                Settings(qdrant_api_key="qdrant-token"),
                http_client=http_client,
            )
            return await client.upsert([_build_upsert_point()])

    result = anyio.run(run_upsert)

    assert result == {"operation_id": 100, "status": "completed"}
    assert captured_request["url"].endswith(
        "/collections/smap_internal_documents/points?wait=true"
    )
    assert captured_request["headers"]["api-key"] == "qdrant-token"
    assert '"points":[{"id":"point-1","vector":[0.1,0.2,0.3]' in captured_request["body"]
    assert '"documentId":"report-202605"' in captured_request["body"]


def test_qdrant_index_client_skips_empty_points() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"result": {}, "status": "ok"})

    async def run_upsert() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentIndexClient(Settings(), http_client=http_client)
            return await client.upsert([])

    result = anyio.run(run_upsert)

    assert result == {}
    assert called is False


def test_qdrant_index_client_creates_collection() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["method"] = request.method
        captured_request["url"] = str(request.url)
        captured_request["headers"] = dict(request.headers)
        captured_request["body"] = request.content.decode()
        return httpx.Response(200, json={"result": True, "status": "ok"})

    async def run_create() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentIndexClient(
                Settings(qdrant_api_key="qdrant-token", embedding_dimension=1024),
                http_client=http_client,
            )
            return await client.create_collection(distance="Cosine")

    result = anyio.run(run_create)

    assert result == {"result": True, "status": "ok"}
    assert captured_request["method"] == "PUT"
    assert captured_request["url"].endswith("/collections/smap_internal_documents")
    assert captured_request["headers"]["api-key"] == "qdrant-token"
    assert '"size":1024' in captured_request["body"]
    assert '"distance":"Cosine"' in captured_request["body"]


def test_qdrant_index_client_raises_external_error_on_create_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": "error"})

    async def run_create() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentIndexClient(Settings(), http_client=http_client)
            await client.create_collection()

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run_create)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ChatErrorCode.CHAT_QDRANT_002
    assert exc_info.value.message == "Qdrant 컬렉션 생성에 실패했습니다."


def test_qdrant_index_client_deletes_points_by_document_id() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["method"] = request.method
        captured_request["url"] = str(request.url)
        captured_request["headers"] = dict(request.headers)
        captured_request["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={
                "result": {
                    "operation_id": 99,
                    "status": "completed",
                },
                "status": "ok",
            },
        )

    async def run_delete() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentIndexClient(
                Settings(qdrant_api_key="qdrant-token"),
                http_client=http_client,
            )
            return await client.delete_by_document_id("report-202605")

    result = anyio.run(run_delete)

    assert result == {"operation_id": 99, "status": "completed"}
    assert captured_request["method"] == "POST"
    assert captured_request["url"].endswith(
        "/collections/smap_internal_documents/points/delete?wait=true"
    )
    assert captured_request["headers"]["api-key"] == "qdrant-token"
    assert '"key":"documentId"' in captured_request["body"]
    assert '"value":"report-202605"' in captured_request["body"]


def test_qdrant_index_client_skips_delete_without_document_id() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"result": {}, "status": "ok"})

    async def run_delete() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentIndexClient(Settings(), http_client=http_client)
            return await client.delete_by_document_id("")

    result = anyio.run(run_delete)

    assert result == {}
    assert called is False


def test_qdrant_index_client_raises_external_error_on_delete_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": "error"})

    async def run_delete() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentIndexClient(Settings(), http_client=http_client)
            await client.delete_by_document_id("report-202605")

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run_delete)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ChatErrorCode.CHAT_QDRANT_002
    assert exc_info.value.message == "Qdrant 기존 문서 삭제에 실패했습니다."


def test_qdrant_index_client_raises_external_error_on_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"status": "error"})

    async def run_upsert() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = QdrantDocumentIndexClient(Settings(), http_client=http_client)
            await client.upsert([_build_upsert_point()])

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run_upsert)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ChatErrorCode.CHAT_QDRANT_002
    assert exc_info.value.message == "Qdrant 문서 저장에 실패했습니다."


def test_qdrant_index_client_raises_external_error_on_invalid_result_shape() -> None:
    client = QdrantDocumentIndexClient(Settings())
    response = httpx.Response(
        200,
        request=httpx.Request("PUT", "http://qdrant.local/points"),
        json={"result": "invalid"},
    )

    with pytest.raises(ChatExternalServiceError) as exc_info:
        client._parse_response(response)

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == ChatErrorCode.CHAT_QDRANT_003
    assert exc_info.value.message == "Qdrant 응답 형식이 올바르지 않습니다."
