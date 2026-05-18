import httpx

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.schemas import ChatErrorCode


def validate_embedding_settings(settings: Settings) -> None:
    missing_fields: list[str] = []
    if not settings.embedding_base_url.strip():
        missing_fields.append("embedding_base_url")
    if not settings.embedding_path.strip():
        missing_fields.append("embedding_path")
    if not settings.embedding_model.strip():
        missing_fields.append("embedding_model")

    if not missing_fields:
        return

    raise ChatExternalServiceError(
        status_code=503,
        code=ChatErrorCode.CHAT_EMBEDDING_001,
        message=f"임베딩 필수 설정이 누락되었습니다: {', '.join(missing_fields)}",
    )


class EmbeddingClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.http_client = http_client

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embed_many([text])
        if not vectors:
            return []
        return vectors[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        validate_embedding_settings(self.settings)
        unique_texts = self._deduplicate_texts(texts)
        payload = self._build_payloads(unique_texts)
        try:
            if self.http_client is not None:
                response = await self.http_client.post(
                    self._embedding_url,
                    json=payload,
                    headers=self._headers,
                )
                vectors = self._parse_batch_response(response)
                return self._restore_duplicate_vectors(texts, unique_texts, vectors)

            async with httpx.AsyncClient(timeout=self.settings.embedding_timeout_seconds) as client:
                response = await client.post(
                    self._embedding_url,
                    json=payload,
                    headers=self._headers,
                )
                vectors = self._parse_batch_response(response)
                return self._restore_duplicate_vectors(texts, unique_texts, vectors)
        except httpx.HTTPError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_EMBEDDING_004,
                message="임베딩 서버 호출에 실패했습니다.",
            ) from exc

    def _deduplicate_texts(self, texts: list[str]) -> list[str]:
        unique_texts: list[str] = []
        seen_texts: set[str] = set()
        for text in texts:
            if text in seen_texts:
                continue
            seen_texts.add(text)
            unique_texts.append(text)
        return unique_texts

    def _restore_duplicate_vectors(
        self,
        original_texts: list[str],
        unique_texts: list[str],
        vectors: list[list[float]],
    ) -> list[list[float]]:
        if len(unique_texts) == len(original_texts):
            return vectors

        if len(vectors) != len(unique_texts):
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_EMBEDDING_002,
                message="임베딩 응답 개수가 요청 텍스트 개수와 일치하지 않습니다.",
            )

        vector_by_text = dict(zip(unique_texts, vectors, strict=True))
        return [vector_by_text[text] for text in original_texts]

    def _build_payload(self, text: str) -> dict:
        return {"inputs": [text]}

    def _build_payloads(self, texts: list[str]) -> dict:
        return {"inputs": texts}

    def _parse_response(self, response: httpx.Response) -> list[float]:
        embeddings = self._parse_batch_response(response)
        if not embeddings:
            return []
        return embeddings[0]

    def _parse_batch_response(self, response: httpx.Response) -> list[list[float]]:
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
        embeddings = self._extract_embeddings(body)
        return [self._coerce_embedding(embedding) for embedding in embeddings]

    def _coerce_embedding(self, embedding: list) -> list[float]:
        try:
            return [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_EMBEDDING_002,
                message="임베딩 응답 형식이 올바르지 않습니다.",
            ) from exc

    def _extract_embedding(self, body: object) -> list:
        embeddings = self._extract_embeddings(body)
        if not embeddings:
            return []
        return embeddings[0]

    def _extract_embeddings(self, body: object) -> list[list]:
        if isinstance(body, list):
            if not body:
                return []
            first_item = body[0]
            if isinstance(first_item, list):
                return body
            return [body]

        if isinstance(body, dict):
            embedding = body.get("embedding")
            if isinstance(embedding, list):
                return [embedding]

            embeddings = body.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                first_embedding = embeddings[0]
                if isinstance(first_embedding, list):
                    return embeddings
                return [embeddings]

            data = body.get("data")
            if isinstance(data, list) and data:
                data_embeddings = [
                    item.get("embedding")
                    for item in data
                    if isinstance(item, dict) and isinstance(item.get("embedding"), list)
                ]
                if data_embeddings:
                    return data_embeddings

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
