from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_env_example_covers_all_settings_fields() -> None:
    env_example = Path(".env.example").read_text(encoding="utf-8")
    env_keys = {
        line.split("=", maxsplit=1)[0]
        for line in env_example.splitlines()
        if line and not line.startswith("#") and "=" in line
    }

    expected_keys = {field_name.upper() for field_name in Settings.model_fields}

    assert expected_keys - env_keys == set()


def test_env_example_loads_as_valid_settings() -> None:
    settings = Settings(_env_file=".env.example")

    assert settings.app_name == "S-MAP AI Server"
    assert settings.cors_origins == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    assert settings.evidence_lookup_enabled is False
    assert settings.rdb_evidence_enabled is False
    assert settings.qdrant_search_enabled is False
    assert settings.embedding_enabled is False
    assert settings.llm_enabled is False
    assert settings.llm_provider == "openai"
    assert settings.llm_base_url == "https://api.openai.com/v1"
    assert settings.llm_allowed_models == []
    assert settings.llm_reasoning_effort == "minimal"
    assert settings.qdrant_top_k == 5
    assert settings.document_chunk_size == 800
    assert settings.document_chunk_overlap == 80
    assert settings.embedding_dimension == 1024
    assert settings.llm_max_tokens == 1024
    assert settings.llm_response_cache_enabled is True
    assert settings.llm_response_cache_ttl_seconds == 60.0
    assert settings.llm_response_cache_max_entries == 128
    assert settings.answer_max_chars == 1600
    assert settings.rdb_evidence_max_limit == 20
    assert settings.prompt_max_evidence_items == 3
    assert settings.prompt_max_document_sources == 3
    assert settings.prompt_max_summary_chars == 360
    assert settings.prompt_max_data_chars == 500
    assert settings.prompt_max_total_chars == 4000


def test_settings_accepts_comma_separated_cors_origins_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    )

    settings = Settings(_env_file=None)

    assert settings.cors_origins == [
        "http://localhost:3000",
        "http://localhost:5173",
    ]


def test_settings_accepts_json_cors_origins_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CORS_ORIGINS",
        '["http://localhost:3000","http://localhost:5173"]',
    )

    settings = Settings(_env_file=None)

    assert settings.cors_origins == [
        "http://localhost:3000",
        "http://localhost:5173",
    ]


def test_settings_accepts_comma_separated_llm_allowed_models_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ALLOWED_MODELS", "gpt-a,gpt-b")

    settings = Settings(_env_file=None)

    assert settings.llm_allowed_models == ["gpt-a", "gpt-b"]


def test_settings_accepts_json_llm_allowed_models_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ALLOWED_MODELS", '["gpt-a","gpt-b"]')

    settings = Settings(_env_file=None)

    assert settings.llm_allowed_models == ["gpt-a", "gpt-b"]


def test_settings_accepts_chat_cost_guardrail_limits() -> None:
    settings = Settings(
        qdrant_top_k=20,
        qdrant_score_threshold=1.0,
        document_content_max_chars=1_000_000,
        document_max_chunks=1_000,
        document_chunk_size=5_000,
        document_chunk_overlap=4_999,
        llm_max_tokens=4096,
        llm_response_cache_ttl_seconds=3600.0,
        llm_response_cache_max_entries=10_000,
        answer_max_chars=5000,
        prompt_max_evidence_items=20,
        prompt_max_document_sources=20,
        prompt_max_summary_chars=2_000,
        prompt_max_data_chars=5_000,
        prompt_max_total_chars=20_000,
        rdb_evidence_max_limit=100,
    )

    assert settings.qdrant_top_k == 20
    assert settings.qdrant_score_threshold == 1.0
    assert settings.document_content_max_chars == 1_000_000
    assert settings.document_max_chunks == 1_000
    assert settings.document_chunk_size == 5_000
    assert settings.document_chunk_overlap == 4_999
    assert settings.llm_max_tokens == 4096
    assert settings.llm_response_cache_ttl_seconds == 3600.0
    assert settings.llm_response_cache_max_entries == 10_000
    assert settings.answer_max_chars == 5000
    assert settings.prompt_max_evidence_items == 20
    assert settings.prompt_max_document_sources == 20
    assert settings.prompt_max_summary_chars == 2_000
    assert settings.prompt_max_data_chars == 5_000
    assert settings.prompt_max_total_chars == 20_000
    assert settings.rdb_evidence_max_limit == 100


def test_settings_default_internal_tokens_are_optional() -> None:
    settings = Settings()

    assert settings.chat_answer_internal_token is None
    assert settings.chat_recommendation_internal_token is None
    assert settings.document_index_internal_token is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("qdrant_top_k", 0),
        ("qdrant_top_k", 21),
        ("qdrant_score_threshold", -0.1),
        ("qdrant_score_threshold", 1.1),
        ("document_content_max_chars", 999),
        ("document_content_max_chars", 1_000_001),
        ("document_max_chunks", 0),
        ("document_max_chunks", 1_001),
        ("document_chunk_size", 0),
        ("document_chunk_size", 5_001),
        ("document_chunk_overlap", -1),
        ("document_chunk_overlap", 5_001),
        ("llm_max_tokens", 0),
        ("llm_max_tokens", 4097),
        ("llm_response_cache_ttl_seconds", -0.1),
        ("llm_response_cache_ttl_seconds", 3600.1),
        ("llm_response_cache_max_entries", -1),
        ("llm_response_cache_max_entries", 10_001),
        ("answer_max_chars", 99),
        ("answer_max_chars", 5001),
        ("prompt_max_evidence_items", -1),
        ("prompt_max_evidence_items", 21),
        ("prompt_max_document_sources", -1),
        ("prompt_max_document_sources", 21),
        ("prompt_max_summary_chars", 0),
        ("prompt_max_summary_chars", 2_001),
        ("prompt_max_data_chars", 0),
        ("prompt_max_data_chars", 5_001),
        ("prompt_max_total_chars", 999),
        ("prompt_max_total_chars", 20_001),
        ("rdb_evidence_max_limit", 0),
        ("rdb_evidence_max_limit", 101),
    ],
)
def test_settings_rejects_chat_cost_guardrail_violations(
    field_name: str,
    value: int | float,
) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field_name: value})


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (100, 100),
        (100, 101),
    ],
)
def test_settings_rejects_invalid_document_chunk_overlap(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            document_chunk_size=chunk_size,
            document_chunk_overlap=chunk_overlap,
        )
