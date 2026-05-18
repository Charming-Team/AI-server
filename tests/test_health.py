from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app

client = TestClient(app)
_MISSING_OVERRIDE = object()


def _override_settings(settings: Settings) -> object:
    previous_override = app.dependency_overrides.get(get_settings, _MISSING_OVERRIDE)
    app.dependency_overrides[get_settings] = lambda: settings
    return previous_override


def _restore_settings(previous_override: object) -> None:
    if previous_override is _MISSING_OVERRIDE:
        app.dependency_overrides.pop(get_settings, None)
    else:
        app.dependency_overrides[get_settings] = previous_override


def test_health_check() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_check_returns_not_ready_when_required_tokens_are_missing() -> None:
    previous_override = _override_settings(Settings())
    try:
        response = client.get("/api/v1/health/ready")
    finally:
        _restore_settings(previous_override)

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    components = {component["name"]: component for component in body["components"]}
    assert components["chatAnswerInternalToken"]["configured"] is False
    assert components["chatRecommendationInternalToken"]["configured"] is False
    assert components["documentIndexInternalToken"]["configured"] is False
    assert components["qdrantSearch"] == {
        "name": "qdrantSearch",
        "enabled": False,
        "configured": True,
        "reason": "비활성화되어 있습니다.",
    }


def test_readiness_check_returns_ready_without_exposing_secret_values() -> None:
    previous_override = _override_settings(
        Settings(
            chat_answer_internal_token="answer-secret",
            chat_recommendation_internal_token="recommendation-secret",
            document_index_internal_token="document-secret",
            evidence_lookup_enabled=True,
            evidence_lookup_base_url="http://spring.local",
            evidence_lookup_path="/internal/chat/evidence",
            evidence_lookup_internal_token="evidence-secret",
            qdrant_search_enabled=True,
            qdrant_url="http://qdrant.local:6333",
            qdrant_collection="document_embeddings",
            embedding_enabled=True,
            embedding_base_url="http://embedding.local",
            embedding_path="/embed",
            embedding_model="BAAI/bge-m3",
            llm_enabled=True,
            llm_base_url="http://llm.local/v1",
            llm_model="local-open-source-model",
        )
    )
    try:
        response = client.get("/api/v1/health/ready")
    finally:
        _restore_settings(previous_override)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert all(component["configured"] for component in body["components"])
    response_text = response.text
    assert "answer-secret" not in response_text
    assert "recommendation-secret" not in response_text
    assert "document-secret" not in response_text
    assert "evidence-secret" not in response_text


def test_readiness_check_requires_embedding_when_qdrant_search_is_enabled() -> None:
    previous_override = _override_settings(
        Settings(
            chat_answer_internal_token="answer-secret",
            chat_recommendation_internal_token="recommendation-secret",
            document_index_internal_token="document-secret",
            qdrant_search_enabled=True,
            embedding_enabled=False,
        )
    )
    try:
        response = client.get("/api/v1/health/ready")
    finally:
        _restore_settings(previous_override)

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    components = {component["name"]: component for component in body["components"]}
    assert components["qdrantSearch"]["configured"] is True
    assert components["embedding"]["configured"] is True
    assert components["ragSearchPipeline"] == {
        "name": "ragSearchPipeline",
        "enabled": True,
        "configured": False,
        "reason": "Qdrant 검색에는 Embedding 기능 활성화가 필요합니다.",
    }
