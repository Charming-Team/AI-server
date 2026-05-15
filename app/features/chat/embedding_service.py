from app.core.config import Settings
from app.features.chat.embedding_client import EmbeddingClient
from app.features.chat.schemas import ChatAnswerRequest, EmbeddingResult


class EmbeddingService:
    def __init__(
        self,
        settings: Settings,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client or EmbeddingClient(settings)

    async def embed_query(self, request: ChatAnswerRequest) -> EmbeddingResult:
        if not self.settings.embedding_enabled:
            return EmbeddingResult(
                was_embedded=False,
                model=self.settings.embedding_model,
                skipped_reason="Embedding is disabled.",
            )

        vector = await self.embedding_client.embed(request.question)
        if not vector:
            return EmbeddingResult(
                was_embedded=False,
                model=self.settings.embedding_model,
                skipped_reason="Embedding model returned an empty vector.",
            )

        if len(vector) != self.settings.embedding_dimension:
            return EmbeddingResult(
                vector=vector,
                was_embedded=False,
                model=self.settings.embedding_model,
                skipped_reason="Embedding dimension mismatch.",
            )

        return EmbeddingResult(
            vector=vector,
            was_embedded=True,
            model=self.settings.embedding_model,
        )
