from app.core.config import Settings
from app.features.chat.embedding_client import EmbeddingClient
from app.features.chat.schemas import ChatAnswerRequest, EmbeddingResult
from app.features.chat.skip_reasons import (
    EMBEDDING_DIMENSION_MISMATCH,
    EMBEDDING_DISABLED,
    EMBEDDING_EMPTY_VECTOR,
)


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
                skipped_reason=EMBEDDING_DISABLED,
            )

        vector = await self.embedding_client.embed(request.question)
        if not vector:
            return EmbeddingResult(
                was_embedded=False,
                model=self.settings.embedding_model,
                skipped_reason=EMBEDDING_EMPTY_VECTOR,
            )

        if len(vector) != self.settings.embedding_dimension:
            return EmbeddingResult(
                vector=vector,
                was_embedded=False,
                model=self.settings.embedding_model,
                skipped_reason=EMBEDDING_DIMENSION_MISMATCH,
            )

        return EmbeddingResult(
            vector=vector,
            was_embedded=True,
            model=self.settings.embedding_model,
        )
