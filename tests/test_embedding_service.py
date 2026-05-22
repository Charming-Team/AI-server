from datetime import datetime

import anyio
import pytest

from app.core.config import Settings
from app.features.chat.embedding_service import EmbeddingService
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.schemas import ChatAnswerRequest, ChatErrorCode, ChatUserContext


class FakeEmbeddingClient:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.text: str | None = None

    async def embed(self, text: str) -> list[float]:
        self.text = text
        return self.vector


def _build_request() -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=10,
        messageId=24,
        user=ChatUserContext(
            userId=1,
            role="EXECUTIVE",
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
    assert result.skipped_reason == "임베딩 기능이 비활성화되어 있습니다."


def test_embedding_settings_use_bge_m3_defaults() -> None:
    settings = Settings()

    assert settings.embedding_provider == "huggingface"
    assert settings.embedding_base_url == "http://localhost:8002"
    assert settings.embedding_path == "/embed"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_dimension == 1024


def test_embedding_service_returns_vector_when_enabled() -> None:
    embedding_client = FakeEmbeddingClient([0.1, 0.2, 0.3])
    service = EmbeddingService(
        Settings(embedding_enabled=True, embedding_dimension=3),
        embedding_client=embedding_client,
    )
    request = _build_request()

    result = anyio.run(service.embed_query, request)

    assert result.was_embedded is True
    assert result.vector == [0.1, 0.2, 0.3]
    assert result.model == "BAAI/bge-m3"
    assert embedding_client.text == "최근 보고서 요약해줘"


@pytest.mark.parametrize(
    ("settings", "expected_message"),
    [
        (
            Settings(embedding_enabled=True, embedding_base_url=" "),
            "임베딩 필수 설정이 누락되었습니다: embedding_base_url",
        ),
        (
            Settings(embedding_enabled=True, embedding_path=" "),
            "임베딩 필수 설정이 누락되었습니다: embedding_path",
        ),
        (
            Settings(embedding_enabled=True, embedding_model=" "),
            "임베딩 필수 설정이 누락되었습니다: embedding_model",
        ),
        (
            Settings(
                embedding_enabled=True,
                embedding_base_url=" ",
                embedding_path=" ",
                embedding_model=" ",
            ),
            (
                "임베딩 필수 설정이 누락되었습니다: "
                "embedding_base_url, embedding_path, embedding_model"
            ),
        ),
    ],
)
def test_embedding_service_requires_embedding_settings_when_enabled(
    settings: Settings,
    expected_message: str,
) -> None:
    embedding_client = FakeEmbeddingClient([0.1, 0.2, 0.3])
    service = EmbeddingService(settings, embedding_client=embedding_client)

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(service.embed_query, _build_request())

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ChatErrorCode.CHAT_EMBEDDING_001
    assert exc_info.value.message == expected_message
    assert embedding_client.text is None


def test_embedding_service_rejects_dimension_mismatch() -> None:
    service = EmbeddingService(
        Settings(embedding_enabled=True, embedding_dimension=1024),
        embedding_client=FakeEmbeddingClient([0.1, 0.2, 0.3]),
    )
    request = _build_request()

    result = anyio.run(service.embed_query, request)

    assert result.was_embedded is False
    assert result.vector == [0.1, 0.2, 0.3]
    assert result.skipped_reason == "임베딩 벡터 차원이 설정값과 일치하지 않습니다."
