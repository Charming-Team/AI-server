import httpx

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.schemas import ChatErrorCode


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

    @property
    def _search_url(self) -> str:
        base_url = self.settings.qdrant_url.rstrip("/")
        collection = self.settings.qdrant_collection
        return f"{base_url}/collections/{collection}/points/search"

    @property
    def _headers(self) -> dict[str, str]:
        if not self.settings.qdrant_api_key:
            return {}
        return {"api-key": self.settings.qdrant_api_key}
