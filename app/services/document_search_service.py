from app.core.config import Settings
from app.schemas.chat import ChatAnswerRequest, ChatIntent, DocumentSearchResult


class DocumentSearchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def search(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> DocumentSearchResult:
        if not self.settings.qdrant_search_enabled:
            return DocumentSearchResult(was_searched=False)

        return DocumentSearchResult(was_searched=True, sources=[])
