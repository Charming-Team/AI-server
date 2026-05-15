import httpx

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.schemas import ChatErrorCode


class EmbeddingClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.http_client = http_client

    async def embed(self, text: str) -> list[float]:
        payload = self._build_payload(text)
        try:
            if self.http_client is not None:
                response = await self.http_client.post(
                    self._embedding_url,
                    json=payload,
                    headers=self._headers,
                )
                return self._parse_response(response)

            async with httpx.AsyncClient(timeout=self.settings.embedding_timeout_seconds) as client:
                response = await client.post(
                    self._embedding_url,
                    json=payload,
                    headers=self._headers,
                )
                return self._parse_response(response)
        except httpx.HTTPError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_EMBEDDING_004,
                message="임베딩 서버 호출에 실패했습니다.",
            ) from exc

    def _build_payload(self, text: str) -> dict:
        return {"inputs": [text]}

    def _parse_response(self, response: httpx.Response) -> list[float]:
        try:
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_EMBEDDING_004,
                message="임베딩 서버 호출에 실패했습니다.",
            ) from exc
        except ValueError as exc:
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_EMBEDDING_002,
                message="임베딩 응답 형식이 올바르지 않습니다.",
            ) from exc
        embedding = self._extract_embedding(body)
        try:
            return [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_EMBEDDING_002,
                message="임베딩 응답 형식이 올바르지 않습니다.",
            ) from exc

    def _extract_embedding(self, body: object) -> list:
        if isinstance(body, list):
            if not body:
                return []
            first_item = body[0]
            if isinstance(first_item, list):
                return first_item
            return body

        if isinstance(body, dict):
            embedding = body.get("embedding")
            if isinstance(embedding, list):
                return embedding

            embeddings = body.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                first_embedding = embeddings[0]
                if isinstance(first_embedding, list):
                    return first_embedding

            data = body.get("data")
            if isinstance(data, list) and data:
                first_data = data[0]
                if isinstance(first_data, dict):
                    openai_embedding = first_data.get("embedding")
                    if isinstance(openai_embedding, list):
                        return openai_embedding

        return []

    @property
    def _embedding_url(self) -> str:
        base_url = self.settings.embedding_base_url.rstrip("/")
        path = self.settings.embedding_path
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{base_url}{path}"

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.embedding_api_key:
            headers["Authorization"] = f"Bearer {self.settings.embedding_api_key}"
        return headers
