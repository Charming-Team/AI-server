import httpx

from app.core.config import Settings


class QdrantDocumentSearchClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.http_client = http_client

    async def search(self, payload: dict) -> list[dict]:
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

    def _parse_response(self, response: httpx.Response) -> list[dict]:
        response.raise_for_status()
        result = response.json().get("result", [])
        if not isinstance(result, list):
            return []
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
