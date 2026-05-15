from app.core.config import Settings
from app.features.chat.document_payload import QdrantSearchPoint
from app.features.chat.embedding_service import EmbeddingService
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatIntent,
    ChatSource,
    DocumentSearchResult,
)


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

    def _build_search_payload(
        self,
        vector: list[float],
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> dict:
        return {
            "vector": vector,
            "limit": self.settings.qdrant_top_k,
            "with_payload": True,
            "filter": self._build_search_filter(request, intent),
        }

    def _build_search_filter(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> dict:
        must_conditions: list[dict] = [
            {"key": "allowedRoles", "match": {"any": [request.user.role]}},
        ]
        if request.user.company_name:
            must_conditions.append(
                {"key": "companyName", "match": {"value": request.user.company_name}}
            )
        if intent != ChatIntent.UNKNOWN:
            must_conditions.append(
                {"key": "intentTags", "match": {"any": [intent.value]}}
            )
        return {"must": must_conditions}

    def _build_sources(self, points: list[dict]) -> list[ChatSource]:
        return [
            QdrantSearchPoint.model_validate(point).to_chat_source()
            for point in points
        ]
