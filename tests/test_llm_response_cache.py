from app.features.chat.grounded_prompt_builder import GroundedPrompt
from app.features.chat.llm_response_cache import (
    LlmResponseCache,
    build_llm_response_cache_key,
)


def _build_prompt(user_prompt: str = "사용자 질문") -> GroundedPrompt:
    return GroundedPrompt(
        system_prompt="시스템 프롬프트",
        user_prompt=user_prompt,
    )


def test_llm_response_cache_returns_cached_answer_before_ttl_expires() -> None:
    cache = LlmResponseCache(ttl_seconds=60.0, max_entries=2)

    cache.put("cache-key", "답변")

    assert cache.get("cache-key") == "답변"


def test_llm_response_cache_ignores_disabled_cache() -> None:
    cache = LlmResponseCache(enabled=False, ttl_seconds=60.0, max_entries=2)

    cache.put("cache-key", "답변")

    assert cache.get("cache-key") is None


def test_llm_response_cache_trims_oldest_entry_when_max_entries_exceeded() -> None:
    cache = LlmResponseCache(ttl_seconds=60.0, max_entries=1)

    cache.put("first-key", "첫 번째 답변")
    cache.put("second-key", "두 번째 답변")

    assert cache.get("first-key") is None
    assert cache.get("second-key") == "두 번째 답변"


def test_llm_response_cache_key_changes_by_prompt_and_model_settings() -> None:
    key = build_llm_response_cache_key(
        _build_prompt(),
        provider="openai",
        model="gpt-test",
        max_tokens=512,
        temperature=0.1,
    )

    assert key != build_llm_response_cache_key(
        _build_prompt("다른 질문"),
        provider="openai",
        model="gpt-test",
        max_tokens=512,
        temperature=0.1,
    )
    assert key != build_llm_response_cache_key(
        _build_prompt(),
        provider="openai",
        model="another-model",
        max_tokens=512,
        temperature=0.1,
    )
    assert key != build_llm_response_cache_key(
        _build_prompt(),
        provider="openai",
        model="gpt-test",
        max_tokens=512,
        temperature=0.1,
        reasoning_effort="low",
    )
