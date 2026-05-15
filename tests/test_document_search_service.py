from datetime import datetime

import anyio

from app.core.config import Settings
from app.features.chat.document_search_service import DocumentSearchService
from app.features.chat.schemas import ChatAnswerRequest, ChatIntent, ChatUserContext


def _build_request() -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=10,
        messageId=24,
        user=ChatUserContext(
            userId=1,
            role="EXECUTIVE",
            department="경영기획팀",
            companyName="S-MAP",
            status="ACTIVE",
        ),
        question="최근 보고서 요약해줘",
        requestedAt=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
    )


def test_document_search_service_skips_search_when_qdrant_is_disabled() -> None:
    service = DocumentSearchService(Settings(qdrant_search_enabled=False))
    request = _build_request()

    result = anyio.run(service.search, request, ChatIntent.REPORT_LOOKUP)

    assert result.was_searched is False
    assert result.sources == []


def test_document_search_service_marks_search_when_qdrant_is_enabled() -> None:
    service = DocumentSearchService(
        Settings(qdrant_search_enabled=True, embedding_enabled=False)
    )
    request = _build_request()

    result = anyio.run(service.search, request, ChatIntent.REPORT_LOOKUP)

    assert result.was_searched is False
    assert result.sources == []
    assert result.skipped_reason == "Embedding is disabled."
