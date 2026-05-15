import json

import anyio
import httpx
import pytest

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.grounded_prompt_builder import GroundedPrompt
from app.features.chat.llm_client import LlmClient
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
