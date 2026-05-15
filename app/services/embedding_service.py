from app.core.config import Settings
from app.schemas.chat import ChatAnswerRequest, EmbeddingResult


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def embed_query(self, request: ChatAnswerRequest) -> EmbeddingResult:
        if not self.settings.embedding_enabled:
            return EmbeddingResult(
                was_embedded=False,
                model=self.settings.embedding_model,
                skipped_reason="Embedding is disabled.",
            )

        return EmbeddingResult(
            was_embedded=False,
            model=self.settings.embedding_model,
            skipped_reason="Embedding model execution is not connected yet.",
        )
