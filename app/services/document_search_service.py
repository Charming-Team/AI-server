from app.core.config import Settings
from app.schemas.chat import ChatAnswerRequest, ChatIntent, DocumentSearchResult
from app.services.embedding_service import EmbeddingService


class DocumentSearchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding_service = EmbeddingService(settings)

    async def search(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> DocumentSearchResult:
        if not self.settings.qdrant_search_enabled:
            return DocumentSearchResult(was_searched=False)

        embedding_result = await self.embedding_service.embed_query(request)
        if not embedding_result.was_embedded:
            return DocumentSearchResult(
                was_searched=False,
                skipped_reason=embedding_result.skipped_reason,
            )

        return DocumentSearchResult(was_searched=True, sources=[])
