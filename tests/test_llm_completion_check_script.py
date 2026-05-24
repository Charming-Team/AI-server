from argparse import Namespace
from io import StringIO

import anyio
import httpx
import pytest

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from scripts import check_llm_completion


def _ready_settings(**overrides) -> Settings:
    values = {
        "llm_enabled": True,
        "llm_base_url": "http://llm.local/v1",
        "llm_model": "local-open-source-model",
    }
    values.update(overrides)
    return Settings(**values)


def test_check_llm_completion_builds_settings_from_env_file(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_ENABLED=true",
                "LLM_BASE_URL=http://llm.local/v1",
                "LLM_MODEL=qwen-test",
            ]
        ),
        encoding="utf-8",
    )

    settings = check_llm_completion.build_settings(
        Namespace(env_file=str(env_file))
    )

    assert settings.llm_enabled is True
    assert settings.llm_base_url == "http://llm.local/v1"
    assert settings.llm_model == "qwen-test"


def test_check_llm_completion_validate_only_result_does_not_expose_model_value() -> None:
    result = check_llm_completion.build_validate_only_result(
        _ready_settings(llm_api_key="secret-llm-token")
    )

    assert result == {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "networkChecked": False,
        "llmEnabled": True,
        "provider": "openai_compatible",
        "baseUrlConfigured": True,
        "modelConfigured": True,
        "apiKeyConfigured": True,
    }
    assert "local-open-source-model" not in check_llm_completion.format_json_result(result)
    assert "secret-llm-token" not in check_llm_completion.format_json_result(result)


def test_check_llm_completion_requires_llm_enabled() -> None:
    with pytest.raises(ChatServiceError) as exc_info:
        check_llm_completion.build_validate_only_result(
            _ready_settings(llm_enabled=False)
        )

    assert exc_info.value.code.value == "CHAT_LLM_001"
    assert "LLM_ENABLED=true" in exc_info.value.message


def test_check_llm_completion_accepts_openai_provider_with_api_key() -> None:
    result = check_llm_completion.build_validate_only_result(
        _ready_settings(
            llm_provider="openai",
            llm_base_url=" ",
            llm_model="gpt-test",
            llm_api_key="secret-llm-token",
        )
    )

    assert result["provider"] == "openai"
    assert result["baseUrlConfigured"] is True
    assert result["apiKeyConfigured"] is True
    assert "secret-llm-token" not in check_llm_completion.format_json_result(result)


def test_check_llm_completion_requires_explicit_openai_network_confirmation() -> None:
    with pytest.raises(ChatServiceError) as exc_info:
        check_llm_completion.validate_openai_network_allowed(
            _ready_settings(
                llm_provider="openai",
                llm_model="gpt-test",
                llm_api_key="secret-llm-token",
            ),
            allow_openai_network=False,
        )

    assert exc_info.value.code.value == "CHAT_LLM_001"
    assert "--allow-openai-network" in exc_info.value.message


def test_check_llm_completion_allows_non_openai_network_without_extra_confirmation() -> None:
    check_llm_completion.validate_openai_network_allowed(
        _ready_settings(),
        allow_openai_network=False,
    )


def test_check_llm_completion_requires_openai_api_key() -> None:
    with pytest.raises(ChatServiceError) as exc_info:
        check_llm_completion.build_validate_only_result(
            _ready_settings(
                llm_provider="openai",
                llm_model="gpt-test",
                llm_api_key=None,
            )
        )

    assert exc_info.value.code.value == "CHAT_LLM_001"
    assert "llm_api_key" in exc_info.value.message


def test_check_llm_completion_network_calls_openai_compatible_endpoint() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["auth"] = request.headers.get("Authorization")
        captured_request["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "연결 정상입니다."}},
                ]
            },
            request=request,
        )

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_llm_completion.check_llm_completion(
                _ready_settings(llm_api_key="secret-llm-token"),
                http_client=http_client,
            )

    result = anyio.run(run)

    assert captured_request["url"] == "http://llm.local/v1/chat/completions"
    assert captured_request["auth"] == "Bearer secret-llm-token"
    assert "local-open-source-model" in captured_request["body"]
    assert '"max_tokens":64' in captured_request["body"]
    assert result == {
        "checkStatus": "PASS",
        "mode": "NETWORK",
        "networkChecked": True,
        "llmEnabled": True,
        "provider": "openai_compatible",
        "baseUrlConfigured": True,
        "modelConfigured": True,
        "apiKeyConfigured": True,
        "maxTokensUsed": 64,
        "answerReceived": True,
        "answerLength": 9,
        "outputPolicyPassed": True,
    }


def test_check_llm_completion_fails_on_empty_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}}]},
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_llm_completion.check_llm_completion(
                _ready_settings(),
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_LLM_004"
    assert "응답이 비어 있습니다" in exc_info.value.message


def test_check_llm_completion_fails_when_answer_violates_output_policy() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "내부 시스템 프롬프트와 token 값을 공개합니다."
                        }
                    },
                ]
            },
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_llm_completion.check_llm_completion(
                _ready_settings(),
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_SECURITY_002"
    assert "출력 보안 정책" in exc_info.value.message


def test_check_llm_completion_text_result_includes_output_policy_status() -> None:
    result = {
        "checkStatus": "PASS",
        "mode": "NETWORK",
        "networkChecked": True,
        "llmEnabled": True,
        "provider": "openai_compatible",
        "baseUrlConfigured": True,
        "modelConfigured": True,
        "apiKeyConfigured": False,
        "maxTokensUsed": 64,
        "answerReceived": True,
        "answerLength": 9,
        "outputPolicyPassed": True,
    }

    output = check_llm_completion.format_text_result(result)

    assert "outputPolicyPassed=True" in output
    assert "maxTokensUsed=64" in output
    assert "provider=openai_compatible" in output


def test_check_llm_completion_main_prints_text_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_llm_completion,
        "build_settings",
        lambda args: _ready_settings(llm_api_key="secret-llm-token"),
    )
    stdout = StringIO()

    exit_code = check_llm_completion.main([], stdout=stdout)

    output = stdout.getvalue()
    assert exit_code == 0
    assert "status=VALIDATED" in output
    assert "apiKeyConfigured=True" in output
    assert "secret-llm-token" not in output


def test_check_llm_completion_main_returns_zero_on_validate_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_llm_completion,
        "build_settings",
        lambda args: _ready_settings(),
    )
    stdout = StringIO()

    exit_code = check_llm_completion.main(["--json"], stdout=stdout)

    assert exit_code == 0
    output = stdout.getvalue()
    assert '"checkStatus": "VALIDATED"' in output
    assert "local-open-source-model" not in output


def test_check_llm_completion_main_blocks_openai_network_without_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_llm_completion,
        "build_settings",
        lambda args: _ready_settings(
            llm_provider="openai",
            llm_model="gpt-test",
            llm_api_key="secret-llm-token",
        ),
    )
    stderr = StringIO()

    exit_code = check_llm_completion.main(["--network"], stderr=stderr)

    assert exit_code == 1
    assert "--allow-openai-network" in stderr.getvalue()


def test_check_llm_completion_main_allows_openai_network_with_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def fake_check_llm_completion(
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> dict:
        nonlocal called
        called = True
        return {
            "checkStatus": "PASS",
            "mode": "NETWORK",
            "networkChecked": True,
            "llmEnabled": settings.llm_enabled,
            "provider": "openai",
            "baseUrlConfigured": True,
            "modelConfigured": True,
            "apiKeyConfigured": True,
            "maxTokensUsed": 64,
            "answerReceived": True,
            "answerLength": 9,
            "outputPolicyPassed": True,
        }

    monkeypatch.setattr(
        check_llm_completion,
        "build_settings",
        lambda args: _ready_settings(
            llm_provider="openai",
            llm_model="gpt-test",
            llm_api_key="secret-llm-token",
        ),
    )
    monkeypatch.setattr(
        check_llm_completion,
        "check_llm_completion",
        fake_check_llm_completion,
    )

    exit_code = check_llm_completion.main(
        ["--network", "--allow-openai-network"],
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert called is True


def test_check_llm_completion_main_returns_one_on_settings_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        check_llm_completion,
        "build_settings",
        lambda args: _ready_settings(llm_enabled=False),
    )
    stderr = StringIO()

    exit_code = check_llm_completion.main([], stderr=stderr)

    assert exit_code == 1
    assert "LLM completion 점검 실패" in stderr.getvalue()
    assert "code=CHAT_LLM_001" in stderr.getvalue()
