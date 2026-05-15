from app.core.config import Settings
from app.features.chat.document_payload import QdrantSearchPoint
from app.features.chat.embedding_service import EmbeddingService
from app.features.chat.qdrant_client import QdrantDocumentSearchClient
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatIntent,
    ChatSource,
    DocumentSearchResult,
)


class DocumentSearchService:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService | None = None,
        qdrant_client: QdrantDocumentSearchClient | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service or EmbeddingService(settings)
        self.qdrant_client = qdrant_client or QdrantDocumentSearchClient(settings)

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

        search_payload = self._build_search_payload(embedding_result.vector, request, intent)
        points = await self.qdrant_client.search(search_payload)
        filtered_points = self._filter_points_by_score(points)

        return DocumentSearchResult(
            was_searched=True,
            sources=self._build_sources(filtered_points),
        )

    def _build_search_payload(
        self,
        vector: list[float],
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> dict:
        payload = {
            "vector": vector,
            "limit": self.settings.qdrant_top_k,
            "with_payload": True,
            "filter": self._build_search_filter(request, intent),
        }
        if self.settings.qdrant_score_threshold > 0:
            payload["score_threshold"] = self.settings.qdrant_score_threshold
        return payload

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

    def _filter_points_by_score(self, points: list[dict]) -> list[dict]:
        threshold = self.settings.qdrant_score_threshold
        if threshold <= 0:
            return points

        return [
            point
            for point in points
            if self._is_score_above_threshold(point, threshold)
        ]

    def _is_score_above_threshold(self, point: dict, threshold: float) -> bool:
        score = point.get("score")
        return isinstance(score, (int, float)) and score >= threshold

    def _build_sources(self, points: list[dict]) -> list[ChatSource]:
        return [
            QdrantSearchPoint.model_validate(point).to_chat_source()
            for point in points
        ]
