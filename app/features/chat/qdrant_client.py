from dataclasses import dataclass

import httpx

from app.core.config import Settings
from app.features.chat.document_payload import QdrantUpsertPoint
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.schemas import ChatErrorCode


@dataclass(frozen=True)
class QdrantCollectionCheckResult:
    collection_name: str
    status: str | None
    expected_dimension: int
    actual_dimension: int | None
    is_dimension_matched: bool
    points_count: int | None


class QdrantDocumentSearchClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.http_client = http_client

    async def search(self, payload: dict) -> list[dict]:
        try:
            if self.http_client is not None:
                response = await self.http_client.post(
                    self._search_url,
                    json=payload,
                    headers=self._headers,
                )
                return self._parse_response(response)

            async with httpx.AsyncClient(timeout=self.settings.qdrant_timeout_seconds) as client:
                response = await client.post(
                    self._search_url,
                    json=payload,
                    headers=self._headers,
                )
                return self._parse_response(response)
        except httpx.HTTPError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_QDRANT_002,
                message="Qdrant 검색에 실패했습니다.",
            ) from exc

    async def check_collection(self) -> QdrantCollectionCheckResult:
        try:
            if self.http_client is not None:
                response = await self.http_client.get(
                    self._collection_url,
                    headers=self._headers,
                )
                return self._parse_collection_check_response(response)

            async with httpx.AsyncClient(timeout=self.settings.qdrant_timeout_seconds) as client:
                response = await client.get(
                    self._collection_url,
                    headers=self._headers,
                )
                return self._parse_collection_check_response(response)
        except httpx.HTTPError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_QDRANT_002,
                message="Qdrant 컬렉션 조회에 실패했습니다.",
            ) from exc

    def _parse_response(self, response: httpx.Response) -> list[dict]:
        try:
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_QDRANT_002,
                message="Qdrant 검색에 실패했습니다.",
            ) from exc
        except ValueError as exc:
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_QDRANT_003,
                message="Qdrant 응답 형식이 올바르지 않습니다.",
            ) from exc

        result = body.get("result", [])
        if not isinstance(result, list):
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_QDRANT_003,
                message="Qdrant 응답 형식이 올바르지 않습니다.",
            )
        return result

    def _parse_collection_check_response(
        self,
        response: httpx.Response,
    ) -> QdrantCollectionCheckResult:
        try:
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            status_code = 404 if exc.response.status_code == 404 else 503
            raise ChatExternalServiceError(
                status_code=status_code,
                code=ChatErrorCode.CHAT_QDRANT_002,
                message="Qdrant 컬렉션 조회에 실패했습니다.",
            ) from exc
        except ValueError as exc:
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_QDRANT_003,
                message="Qdrant 응답 형식이 올바르지 않습니다.",
            ) from exc

        result = body.get("result")
        if not isinstance(result, dict):
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_QDRANT_003,
                message="Qdrant 응답 형식이 올바르지 않습니다.",
            )

        actual_dimension = self._extract_vector_dimension(result)
        return QdrantCollectionCheckResult(
            collection_name=self.settings.qdrant_collection,
            status=self._optional_text(result.get("status")),
            expected_dimension=self.settings.embedding_dimension,
            actual_dimension=actual_dimension,
            is_dimension_matched=actual_dimension == self.settings.embedding_dimension,
            points_count=self._optional_int(result.get("points_count")),
        )

    def _extract_vector_dimension(self, result: dict) -> int | None:
        config = result.get("config")
        if not isinstance(config, dict):
            return None

        params = config.get("params")
        if not isinstance(params, dict):
            return None

        vectors = params.get("vectors")
        if not isinstance(vectors, dict):
            return None

        size = vectors.get("size")
        if isinstance(size, int):
            return size

        named_vector_sizes = [
            vector_config.get("size")
            for vector_config in vectors.values()
            if isinstance(vector_config, dict)
        ]
        if len(named_vector_sizes) == 1 and isinstance(named_vector_sizes[0], int):
            return named_vector_sizes[0]
        return None

    def _optional_text(self, value: object) -> str | None:
        if isinstance(value, str):
            return value
        return None

    def _optional_int(self, value: object) -> int | None:
        if isinstance(value, int):
            return value
        return None

    @property
    def _search_url(self) -> str:
        base_url = self.settings.qdrant_url.rstrip("/")
        collection = self.settings.qdrant_collection
        return f"{base_url}/collections/{collection}/points/search"

    @property
    def _collection_url(self) -> str:
        base_url = self.settings.qdrant_url.rstrip("/")
        collection = self.settings.qdrant_collection
        return f"{base_url}/collections/{collection}"

    @property
    def _headers(self) -> dict[str, str]:
        if not self.settings.qdrant_api_key:
            return {}
        return {"api-key": self.settings.qdrant_api_key}


class QdrantDocumentIndexClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.http_client = http_client

    async def upsert(self, points: list[QdrantUpsertPoint]) -> dict:
        if not points:
            return {}

        payload = {"points": [point.to_qdrant_point() for point in points]}
        return await self._send_request(
            method="put",
            url=self._points_url,
            payload=payload,
            failure_message="Qdrant 문서 저장에 실패했습니다.",
        )

    async def delete_by_document_id(self, document_id: str) -> dict:
        if not document_id:
            return {}

        payload = {
            "filter": {
                "must": [
                    {
                        "key": "documentId",
                        "match": {"value": document_id},
                    }
                ]
            }
        }
        return await self._send_request(
            method="post",
            url=self._points_delete_url,
            payload=payload,
            failure_message="Qdrant 기존 문서 삭제에 실패했습니다.",
        )

    async def create_collection(
        self,
        dimension: int | None = None,
        distance: str = "Cosine",
    ) -> dict:
        payload = {
            "vectors": {
                "size": dimension or self.settings.embedding_dimension,
                "distance": distance,
            }
        }
        try:
            if self.http_client is not None:
                response = await self.http_client.put(
                    self._collection_url,
                    json=payload,
                    headers=self._headers,
                )
                return self._parse_create_collection_response(response)

            async with httpx.AsyncClient(timeout=self.settings.qdrant_timeout_seconds) as client:
                response = await client.put(
                    self._collection_url,
                    json=payload,
                    headers=self._headers,
                )
                return self._parse_create_collection_response(response)
        except httpx.HTTPError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_QDRANT_002,
                message="Qdrant 컬렉션 생성에 실패했습니다.",
            ) from exc

    async def _send_request(
        self,
        method: str,
        url: str,
        payload: dict,
        failure_message: str,
    ) -> dict:
        try:
            if self.http_client is not None:
                response = await self.http_client.request(
                    method,
                    url,
                    json=payload,
                    headers=self._headers,
                )
                return self._parse_response(response, failure_message)

            async with httpx.AsyncClient(timeout=self.settings.qdrant_timeout_seconds) as client:
                response = await client.request(
                    method,
                    url,
                    json=payload,
                    headers=self._headers,
                )
                return self._parse_response(response, failure_message)
        except httpx.HTTPError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_QDRANT_002,
                message=failure_message,
            ) from exc

    def _parse_response(
        self,
        response: httpx.Response,
        failure_message: str = "Qdrant 문서 저장에 실패했습니다.",
    ) -> dict:
        try:
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_QDRANT_002,
                message=failure_message,
            ) from exc
        except ValueError as exc:
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_QDRANT_003,
                message="Qdrant 응답 형식이 올바르지 않습니다.",
            ) from exc

        if not isinstance(body, dict):
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_QDRANT_003,
                message="Qdrant 응답 형식이 올바르지 않습니다.",
            )

        result = body.get("result", {})
        if not isinstance(result, dict):
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_QDRANT_003,
                message="Qdrant 응답 형식이 올바르지 않습니다.",
            )
        return result

    def _parse_create_collection_response(self, response: httpx.Response) -> dict:
        try:
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_QDRANT_002,
                message="Qdrant 컬렉션 생성에 실패했습니다.",
            ) from exc
        except ValueError as exc:
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_QDRANT_003,
                message="Qdrant 응답 형식이 올바르지 않습니다.",
            ) from exc

        if not isinstance(body, dict):
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_QDRANT_003,
                message="Qdrant 응답 형식이 올바르지 않습니다.",
            )
        return {
            "result": body.get("result"),
            "status": body.get("status"),
        }

    @property
    def _points_url(self) -> str:
        base_url = self.settings.qdrant_url.rstrip("/")
        collection = self.settings.qdrant_collection
        return f"{base_url}/collections/{collection}/points?wait=true"

    @property
    def _points_delete_url(self) -> str:
        base_url = self.settings.qdrant_url.rstrip("/")
        collection = self.settings.qdrant_collection
        return f"{base_url}/collections/{collection}/points/delete?wait=true"

    @property
    def _collection_url(self) -> str:
        base_url = self.settings.qdrant_url.rstrip("/")
        collection = self.settings.qdrant_collection
        return f"{base_url}/collections/{collection}"

    @property
    def _headers(self) -> dict[str, str]:
        if not self.settings.qdrant_api_key:
            return {}
        return {"api-key": self.settings.qdrant_api_key}
