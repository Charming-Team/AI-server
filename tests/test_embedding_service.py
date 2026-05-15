from datetime import datetime

import anyio

from app.core.config import Settings
from app.features.chat.embedding_service import EmbeddingService
from app.features.chat.schemas import ChatAnswerRequest, ChatUserContext


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


def test_embedding_service_returns_disabled_result_by_default() -> None:
    service = EmbeddingService(Settings())
    request = _build_request()

    result = anyio.run(service.embed_query, request)

    assert result.was_embedded is False
    assert result.vector == []
    assert result.model == "BAAI/bge-m3"
    assert result.skipped_reason == "Embedding is disabled."


def test_embedding_settings_use_bge_m3_defaults() -> None:
    settings = Settings()

    assert settings.embedding_provider == "huggingface"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_dimension == 1024
