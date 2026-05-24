import json

import anyio
import httpx
import pytest

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.grounded_prompt_builder import GroundedPrompt
from app.features.chat.llm_client import (
    LlmClient,
    resolve_llm_base_url,
    validate_llm_settings,
    validate_openai_cost_guardrails,
    validate_openai_model_allowlist,
)
from app.features.chat.schemas import ChatErrorCode


def _build_prompt() -> GroundedPrompt:
    return GroundedPrompt(
        system_prompt="내부 근거만 사용한다.",
        user_prompt="RDB 근거와 문서 검색 근거를 바탕으로 답변한다.",
    )


def test_llm_client_builds_openai_compatible_payload() -> None:
    client = LlmClient(
        Settings(
            llm_model="qwen-test",
            llm_temperature=0.2,
            llm_max_tokens=512,
        )
    )

    payload = client._build_payload(_build_prompt())

    assert payload == {
        "model": "qwen-test",
        "messages": [
            {"role": "system", "content": "내부 근거만 사용한다."},
            {"role": "user", "content": "RDB 근거와 문서 검색 근거를 바탕으로 답변한다."},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }


def test_llm_client_builds_url_and_optional_auth_header() -> None:
    client = LlmClient(
        Settings(
            llm_base_url="http://llm.svc.cluster.local:8000/v1/",
            llm_api_key="local-token",
        )
    )

    assert client._chat_completions_url == "http://llm.svc.cluster.local:8000/v1/chat/completions"
    assert client._headers == {
        "Content-Type": "application/json",
        "Authorization": "Bearer local-token",
    }


def test_llm_client_resolves_openai_base_url_and_requires_api_key() -> None:
    settings = Settings(
        llm_provider="openai",
        llm_model="gpt-test",
        llm_allowed_models=["gpt-test"],
        llm_api_key="openai-secret-token",
    )
    client = LlmClient(settings)

    assert resolve_llm_base_url(settings) == "https://api.openai.com/v1"
    assert client._chat_completions_url == "https://api.openai.com/v1/chat/completions"
    assert client._headers == {
        "Content-Type": "application/json",
        "Authorization": "Bearer openai-secret-token",
    }


@pytest.mark.parametrize(
    ("settings", "expected_message"),
    [
        (
            Settings(
                llm_provider="openai",
                llm_model="gpt-test",
                llm_api_key="openai-secret-token",
                llm_max_tokens=2048,
            ),
            "llm_max_tokens<=1024",
        ),
        (
            Settings(
                llm_provider="openai",
                llm_model="gpt-test",
                llm_api_key="openai-secret-token",
                prompt_max_total_chars=12_000,
            ),
            "prompt_max_total_chars<=8000",
        ),
        (
            Settings(
                llm_provider="openai",
                llm_model="gpt-test",
                llm_api_key="openai-secret-token",
                llm_response_cache_enabled=False,
            ),
            "llm_response_cache_enabled=true",
        ),
        (
            Settings(
                llm_provider="openai",
                llm_model="gpt-test",
                llm_api_key="openai-secret-token",
                llm_response_cache_ttl_seconds=0.0,
            ),
            "llm_response_cache_ttl_seconds>=1",
        ),
        (
            Settings(
                llm_provider="openai",
                llm_model="gpt-test",
                llm_api_key="openai-secret-token",
                llm_response_cache_max_entries=0,
            ),
            "llm_response_cache_max_entries>=1",
        ),
    ],
)
def test_llm_client_rejects_openai_cost_guardrail_violations(
    settings: Settings,
    expected_message: str,
) -> None:
    with pytest.raises(ChatExternalServiceError) as exc_info:
        validate_openai_cost_guardrails(settings)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ChatErrorCode.CHAT_LLM_001
    assert "OpenAI 비용 가드레일" in exc_info.value.message
    assert expected_message in exc_info.value.message


def test_llm_client_does_not_apply_openai_cost_guardrail_to_compatible_provider() -> None:
    validate_openai_cost_guardrails(
        Settings(
            llm_provider="openai_compatible",
            llm_model="qwen-test",
            llm_max_tokens=4096,
            prompt_max_total_chars=20_000,
            llm_response_cache_enabled=False,
            llm_response_cache_ttl_seconds=0.0,
            llm_response_cache_max_entries=0,
        )
    )


def test_llm_client_generate_parses_chat_completion_response() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "자재 부족과 라인 병목이 주요 위험입니다.",
                        }
                    }
                ]
            },
        )

    async def run_generate() -> str:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = LlmClient(Settings(llm_model="qwen-test"), http_client=http_client)
            return await client.generate(_build_prompt())

    answer = anyio.run(run_generate)

    assert answer == "자재 부족과 라인 병목이 주요 위험입니다."
    assert captured_request["url"].endswith("/chat/completions")
    assert captured_request["body"]["model"] == "qwen-test"


@pytest.mark.parametrize(
    ("settings", "expected_message"),
    [
        (
            Settings(llm_base_url=" "),
            "LLM 필수 설정이 누락되었습니다: llm_base_url",
        ),
        (
            Settings(llm_model=" "),
            "LLM 필수 설정이 누락되었습니다: llm_model",
        ),
        (
            Settings(llm_base_url=" ", llm_model=" "),
            "LLM 필수 설정이 누락되었습니다: llm_base_url, llm_model",
        ),
        (
            Settings(llm_provider="openai", llm_model="gpt-test", llm_api_key=None),
            "LLM 필수 설정이 누락되었습니다: llm_api_key",
        ),
        (
            Settings(
                llm_provider="openai",
                llm_model="gpt-test",
                llm_api_key="openai-secret-token",
            ),
            "OpenAI 모델 allowlist가 설정되지 않았습니다: llm_allowed_models",
        ),
        (
            Settings(
                llm_provider="openai",
                llm_model="gpt-test",
                llm_allowed_models=["gpt-other"],
                llm_api_key="openai-secret-token",
            ),
            "OpenAI 허용 모델이 아닙니다: llm_model",
        ),
        (
            Settings(llm_provider="unsupported", llm_model="gpt-test"),
            "지원하지 않는 LLM provider입니다: unsupported",
        ),
    ],
)
def test_llm_client_requires_settings_before_request(
    settings: Settings,
    expected_message: str,
) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"choices": []})

    async def run_generate() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = LlmClient(settings, http_client=http_client)
            await client.generate(_build_prompt())

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run_generate)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ChatErrorCode.CHAT_LLM_001
    assert exc_info.value.message == expected_message
    assert called is False


def test_llm_client_accepts_openai_model_when_allowlisted() -> None:
    validate_llm_settings(
        Settings(
            llm_provider="openai",
            llm_model="gpt-test",
            llm_allowed_models=["gpt-test", "gpt-fallback"],
            llm_api_key="openai-secret-token",
        )
    )


def test_llm_client_does_not_apply_openai_model_allowlist_to_compatible_provider() -> None:
    validate_openai_model_allowlist(
        Settings(
            llm_provider="openai_compatible",
            llm_model="qwen-test",
            llm_allowed_models=[],
        )
    )


def test_llm_client_raises_external_error_on_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "llm failed"})

    async def run_generate() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = LlmClient(Settings(), http_client=http_client)
            await client.generate(_build_prompt())

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run_generate)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ChatErrorCode.CHAT_LLM_003
    assert exc_info.value.message == "LLM 서버 호출에 실패했습니다."


def test_llm_client_raises_external_error_on_invalid_response_body() -> None:
    client = LlmClient(Settings())
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://llm.local/chat/completions"),
        content=b"not-json",
    )

    with pytest.raises(ChatExternalServiceError) as exc_info:
        client._parse_response(response)

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == ChatErrorCode.CHAT_LLM_002
    assert exc_info.value.message == "LLM 응답 형식이 올바르지 않습니다."


@pytest.mark.parametrize(
    "body",
    [
        ["not-a-dict"],
        {"choices": "not-a-list"},
        {"choices": ["not-a-dict"]},
        {"choices": [{"message": "not-a-dict"}]},
        {"choices": [{"message": {"content": ["not-a-string"]}}]},
    ],
)
def test_llm_client_raises_external_error_on_invalid_response_shape(
    body: object,
) -> None:
    client = LlmClient(Settings())
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://llm.local/chat/completions"),
        json=body,
    )

    with pytest.raises(ChatExternalServiceError) as exc_info:
        client._parse_response(response)

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == ChatErrorCode.CHAT_LLM_002
    assert exc_info.value.message == "LLM 응답 형식이 올바르지 않습니다."


def test_llm_client_returns_empty_answer_on_missing_content() -> None:
    client = LlmClient(Settings())
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://llm.local/chat/completions"),
        json={"choices": [{"message": {"content": None}}]},
    )

    assert client._parse_response(response) == ""
