import os

import pytest

ALLOW_LIVE_LLM_ENV = "PYTEST_ALLOW_LIVE_LLM"
SAFE_TEST_ENV = {
    "API_V1_PREFIX": "/api/v1",
    "CHAT_ANSWER_INTERNAL_TOKEN": "",
    "CHAT_RECOMMENDATION_INTERNAL_TOKEN": "",
    "DOCUMENT_INDEX_INTERNAL_TOKEN": "",
    "EVIDENCE_LOOKUP_ENABLED": "false",
    "EVIDENCE_LOOKUP_INTERNAL_TOKEN": "",
    "EVIDENCE_LOOKUP_BASE_URL": "http://localhost:8080",
    "EVIDENCE_LOOKUP_PATH": "/internal/chat/evidence",
    "RDB_EVIDENCE_ENABLED": "false",
    "RDB_EVIDENCE_DSN": "",
    "REPORT_RDB_DSN": "",
    "PLANNING_RDB_DSN": "",
    "QDRANT_SEARCH_ENABLED": "false",
    "EMBEDDING_ENABLED": "false",
    "EMBEDDING_BASE_URL": "http://localhost:8002",
    "EMBEDDING_API_KEY": "",
    "LLM_ENABLED": "false",
    "LLM_PROVIDER": "openai_compatible",
    "LLM_BASE_URL": "http://localhost:8001/v1",
    "LLM_API_KEY": "",
    "OPENAI_API_KEY": "",
    "LLM_MODEL": "local-open-source-model",
    "LLM_ALLOWED_MODELS": "",
    "LLM_MAX_TOKENS": "1024",
    "PROMPT_MAX_TOTAL_CHARS": "3000",
}
CONFIG_ENV_EXAMPLE_TEST = (
    "tests/test_config.py::test_env_example_loads_as_valid_settings"
)


def _allow_live_llm() -> bool:
    return os.getenv(ALLOW_LIVE_LLM_ENV, "").strip().lower() in {"1", "true", "yes"}


def pytest_configure(config) -> None:
    if _allow_live_llm():
        return

    for name, value in SAFE_TEST_ENV.items():
        os.environ[name] = value
    try:
        from app.core.config import get_settings

        get_settings.cache_clear()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def block_live_llm_calls(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    if _allow_live_llm():
        return

    if request.node.nodeid != CONFIG_ENV_EXAMPLE_TEST:
        for name, value in SAFE_TEST_ENV.items():
            monkeypatch.setenv(name, value)

        from app.core.config import get_settings

        get_settings.cache_clear()

    from app.features.chat.llm_client import LlmClient

    original_generate_completion = LlmClient.generate_completion

    async def guarded_generate_completion(self, prompt):
        if self.http_client is not None:
            return await original_generate_completion(self, prompt)

        raise RuntimeError(
            "Live LLM calls are disabled during pytest. "
            f"Set {ALLOW_LIVE_LLM_ENV}=true only when intentionally spending tokens."
        )

    monkeypatch.setattr(
        LlmClient,
        "generate_completion",
        guarded_generate_completion,
    )
