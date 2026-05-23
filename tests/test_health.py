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
    assert body["runtimeMode"] == {
        "apiPrefix": "/api/v1",
        "groundingMode": "NONE",
        "answerMode": "FALLBACK",
        "ragSearchMode": "DISABLED",
        "enabledGroundingSources": [],
        "expectedLlmSkippedReason": "LLM 답변 생성 기능이 비활성화되어 있습니다.",
    }
    components = {component["name"]: component for component in body["components"]}
    assert components["chatAnswerInternalToken"]["configured"] is False
    assert components["chatAnswerInternalToken"]["code"] == "CHAT_SECURITY_003"
    assert components["chatRecommendationInternalToken"]["configured"] is False
    assert components["chatRecommendationInternalToken"]["code"] == "CHAT_SECURITY_003"
    assert "documentIndexInternalToken" not in components
    assert "documentIndexPipeline" not in components
    assert components["qdrantSearch"] == {
        "name": "qdrantSearch",
        "enabled": False,
        "configured": True,
        "reason": "비활성화되어 있습니다.",
    }
    assert components["rdbEvidence"] == {
        "name": "rdbEvidence",
        "enabled": False,
        "configured": True,
        "reason": "비활성화되어 있습니다.",
    }
    assert components["chatGroundingPipeline"] == {
        "name": "chatGroundingPipeline",
        "enabled": True,
        "configured": False,
        "code": "CHAT_EVIDENCE_001",
        "reason": (
            "챗봇 답변에는 RDB Evidence View, Spring Evidence 또는 "
            "Qdrant 검색 중 하나가 필요합니다."
        ),
    }
    assert components["answerGenerationPipeline"] == {
        "name": "answerGenerationPipeline",
        "enabled": True,
        "configured": True,
        "reason": "LLM 기능이 비활성화되어 근거 기반 fallback 답변 생성을 사용합니다.",
    }


def test_readiness_check_returns_ready_without_exposing_secret_values() -> None:
    previous_override = _override_settings(
        Settings(
            chat_answer_internal_token="answer-secret",
            chat_recommendation_internal_token="recommendation-secret",
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
    assert body["runtimeMode"] == {
        "apiPrefix": "/api/v1",
        "groundingMode": "HYBRID",
        "answerMode": "LLM",
        "ragSearchMode": "ENABLED",
        "enabledGroundingSources": ["QDRANT", "SPRING_EVIDENCE"],
    }
    assert all(component["configured"] for component in body["components"])
    response_text = response.text
    assert "answer-secret" not in response_text
    assert "recommendation-secret" not in response_text
    assert "evidence-secret" not in response_text


def test_readiness_check_summarizes_k8s_configmap_chat_mode() -> None:
    previous_override = _override_settings(
        Settings(
            environment="prod",
            api_v1_prefix="/ai/api/v1",
            chat_answer_internal_token="answer-secret",
            chat_recommendation_internal_token="recommendation-secret",
            rdb_evidence_enabled=True,
            rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
            qdrant_search_enabled=True,
            qdrant_url="http://qdrant.qdrant.svc.cluster.local:6333",
            qdrant_collection="smap_internal_documents",
            embedding_enabled=True,
            embedding_base_url="http://embedding-service:8002",
            embedding_path="/embed",
            embedding_model="BAAI/bge-m3",
            llm_enabled=False,
        )
    )
    try:
        response = client.get("/api/v1/health/ready")
    finally:
        _restore_settings(previous_override)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["environment"] == "prod"
    assert body["runtimeMode"] == {
        "apiPrefix": "/ai/api/v1",
        "groundingMode": "RDB_QDRANT",
        "answerMode": "FALLBACK",
        "ragSearchMode": "ENABLED",
        "enabledGroundingSources": ["RDB_EVIDENCE", "QDRANT"],
        "expectedLlmSkippedReason": "LLM 답변 생성 기능이 비활성화되어 있습니다.",
    }
    assert "reader:secret" not in response.text


def test_readiness_check_returns_custom_codes_for_missing_integration_settings() -> None:
    previous_override = _override_settings(
        Settings(
            chat_answer_internal_token="answer-secret",
            chat_recommendation_internal_token="recommendation-secret",
            evidence_lookup_enabled=True,
            evidence_lookup_base_url=" ",
            evidence_lookup_internal_token="evidence-secret",
            rdb_evidence_enabled=True,
            rdb_evidence_dsn=" ",
            qdrant_search_enabled=True,
            qdrant_collection=" ",
            embedding_enabled=True,
            embedding_path=" ",
            llm_enabled=True,
            llm_model=" ",
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
    assert components["evidenceLookup"]["code"] == "CHAT_EVIDENCE_004"
    assert (
        components["evidenceLookup"]["reason"]
        == "필수 설정이 누락되었습니다: evidence_lookup_base_url"
    )
    assert components["rdbEvidence"]["code"] == "CHAT_EVIDENCE_004"
    assert (
        components["rdbEvidence"]["reason"]
        == "필수 설정이 누락되었습니다: rdb_evidence_dsn"
    )
    assert components["qdrantSearch"]["code"] == "CHAT_QDRANT_001"
    assert (
        components["qdrantSearch"]["reason"]
        == "필수 설정이 누락되었습니다: qdrant_collection"
    )
    assert components["embedding"]["code"] == "CHAT_EMBEDDING_001"
    assert (
        components["embedding"]["reason"]
        == "필수 설정이 누락되었습니다: embedding_path"
    )
    assert components["llm"]["code"] == "CHAT_LLM_001"
    assert components["llm"]["reason"] == "필수 설정이 누락되었습니다: llm_model"


def test_readiness_check_requires_openai_api_key_when_llm_provider_is_openai() -> None:
    previous_override = _override_settings(
        Settings(
            chat_answer_internal_token="answer-secret",
            chat_recommendation_internal_token="recommendation-secret",
            rdb_evidence_enabled=True,
            rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
            llm_enabled=True,
            llm_provider="openai",
            llm_model="gpt-test",
            llm_api_key=None,
        )
    )
    try:
        response = client.get("/api/v1/health/ready")
    finally:
        _restore_settings(previous_override)

    assert response.status_code == 503
    body = response.json()
    components = {component["name"]: component for component in body["components"]}
    assert components["llm"] == {
        "name": "llm",
        "enabled": True,
        "configured": False,
        "code": "CHAT_LLM_001",
        "reason": "필수 설정이 누락되었습니다: llm_api_key",
    }
    assert "gpt-test" not in response.text


def test_readiness_check_accepts_openai_provider_without_exposing_api_key() -> None:
    previous_override = _override_settings(
        Settings(
            chat_answer_internal_token="answer-secret",
            chat_recommendation_internal_token="recommendation-secret",
            rdb_evidence_enabled=True,
            rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
            llm_enabled=True,
            llm_provider="openai",
            llm_model="gpt-test",
            llm_api_key="openai-secret-token",
        )
    )
    try:
        response = client.get("/api/v1/health/ready")
    finally:
        _restore_settings(previous_override)

    assert response.status_code == 200
    body = response.json()
    components = {component["name"]: component for component in body["components"]}
    assert components["llm"] == {
        "name": "llm",
        "enabled": True,
        "configured": True,
    }
    assert "openai-secret-token" not in response.text
    assert "gpt-test" not in response.text


def test_readiness_check_requires_at_least_one_grounding_source() -> None:
    previous_override = _override_settings(
        Settings(
            chat_answer_internal_token="answer-secret",
            chat_recommendation_internal_token="recommendation-secret",
            evidence_lookup_enabled=False,
            rdb_evidence_enabled=False,
            qdrant_search_enabled=False,
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
    assert components["evidenceLookup"]["configured"] is True
    assert components["rdbEvidence"]["configured"] is True
    assert components["qdrantSearch"]["configured"] is True
    assert components["chatGroundingPipeline"] == {
        "name": "chatGroundingPipeline",
        "enabled": True,
        "configured": False,
        "code": "CHAT_EVIDENCE_001",
        "reason": (
            "챗봇 답변에는 RDB Evidence View, Spring Evidence 또는 "
            "Qdrant 검색 중 하나가 필요합니다."
        ),
    }


def test_readiness_check_accepts_rdb_evidence_view_as_grounding_source() -> None:
    previous_override = _override_settings(
        Settings(
            chat_answer_internal_token="answer-secret",
            chat_recommendation_internal_token="recommendation-secret",
            evidence_lookup_enabled=False,
            rdb_evidence_enabled=True,
            rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
            qdrant_search_enabled=False,
            embedding_enabled=False,
            llm_enabled=False,
        )
    )
    try:
        response = client.get("/api/v1/health/ready")
    finally:
        _restore_settings(previous_override)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    components = {component["name"]: component for component in body["components"]}
    assert components["evidenceLookup"]["configured"] is True
    assert components["rdbEvidence"]["configured"] is True
    assert components["qdrantSearch"]["configured"] is True
    assert components["chatGroundingPipeline"] == {
        "name": "chatGroundingPipeline",
        "enabled": True,
        "configured": True,
    }
    assert components["answerGenerationPipeline"] == {
        "name": "answerGenerationPipeline",
        "enabled": True,
        "configured": True,
        "reason": "LLM 기능이 비활성화되어 근거 기반 fallback 답변 생성을 사용합니다.",
    }
    assert "reader:secret" not in response.text


def test_readiness_check_accepts_fallback_answer_generation_when_llm_is_disabled() -> None:
    previous_override = _override_settings(
        Settings(
            chat_answer_internal_token="answer-secret",
            chat_recommendation_internal_token="recommendation-secret",
            evidence_lookup_enabled=True,
            evidence_lookup_internal_token="evidence-secret",
            llm_enabled=False,
        )
    )
    try:
        response = client.get("/api/v1/health/ready")
    finally:
        _restore_settings(previous_override)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    components = {component["name"]: component for component in body["components"]}
    assert components["chatGroundingPipeline"]["configured"] is True
    assert components["llm"] == {
        "name": "llm",
        "enabled": False,
        "configured": True,
        "reason": "비활성화되어 있습니다.",
    }
    assert components["answerGenerationPipeline"] == {
        "name": "answerGenerationPipeline",
        "enabled": True,
        "configured": True,
        "reason": "LLM 기능이 비활성화되어 근거 기반 fallback 답변 생성을 사용합니다.",
    }


def test_readiness_check_does_not_require_document_index_token_for_chat_runtime() -> None:
    previous_override = _override_settings(
        Settings(
            chat_answer_internal_token="answer-secret",
            chat_recommendation_internal_token="recommendation-secret",
            rdb_evidence_enabled=True,
            rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
        )
    )
    try:
        response = client.get("/api/v1/health/ready")
    finally:
        _restore_settings(previous_override)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    components = {component["name"]: component for component in body["components"]}
    assert "documentIndexInternalToken" not in components
    assert "documentIndexPipeline" not in components
    assert components["chatGroundingPipeline"]["configured"] is True


def test_readiness_check_requires_embedding_when_qdrant_search_is_enabled() -> None:
    previous_override = _override_settings(
        Settings(
            chat_answer_internal_token="answer-secret",
            chat_recommendation_internal_token="recommendation-secret",
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
        "code": "CHAT_EMBEDDING_001",
        "reason": "Qdrant 검색에는 Embedding 기능 활성화가 필요합니다.",
    }
