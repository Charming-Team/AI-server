from argparse import Namespace
from io import StringIO

import anyio
import httpx
import pytest

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from scripts import check_chat_recommendations


def _build_args(**overrides):
    values = {
        "base_url": "http://fastapi.local",
        "path": "/api/v1/chat/recommendations",
        "token": "recommendation-token",
        "env_file": None,
        "timeout_seconds": 10.0,
        "keyword": "라인",
        "role": "MANUFACTURING_MANAGER",
        "user_id": 1,
        "company_name": "S-MAP",
        "status": "ACTIVE",
        "min_item_count": 1,
        "expect_fallback": False,
        "json": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _recommendation_response(
    *,
    fallback_used: bool = False,
    url: str = "/production-lines/status",
    intent: str = "LINE_BOTTLENECK",
) -> dict:
    return {
        "items": [
            {
                "questionId": "line-bottleneck-current",
                "question": "현재 병목이 발생한 라인과 원인을 알려줘",
                "intent": intent,
                "category": "라인 병목",
                "url": url,
            }
        ],
        "fallbackUsed": fallback_used,
    }


def test_chat_recommendations_script_resolves_path_and_token() -> None:
    settings = Settings(
        api_v1_prefix="/ai/api/v1",
        chat_recommendation_internal_token="env-recommendation-token",
    )

    assert (
        check_chat_recommendations.resolve_recommendation_path(
            _build_args(path=None),
            settings,
        )
        == "/ai/api/v1/chat/recommendations"
    )
    assert (
        check_chat_recommendations.resolve_recommendation_token(
            _build_args(token=None),
            settings,
        )
        == "env-recommendation-token"
    )
    assert (
        check_chat_recommendations.build_recommendation_url(
            "http://fastapi.local/",
            "/api/v1/chat/recommendations",
        )
        == "http://fastapi.local/api/v1/chat/recommendations"
    )


def test_chat_recommendations_script_builds_normalized_request() -> None:
    request = check_chat_recommendations.build_request(
        _build_args(role=" operator ", keyword=" 자재 ")
    )

    assert request.user.role == "OPERATOR"
    assert request.user.user_id == 1
    assert request.keyword == " 자재 "


def test_chat_recommendations_script_calls_fastapi_recommendation_contract() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["token"] = request.headers.get("X-Internal-Token")
        captured_request["body"] = request.read().decode()
        return httpx.Response(200, json=_recommendation_response(), request=request)

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_chat_recommendations.check_chat_recommendations(
                base_url="http://fastapi.local",
                path="/api/v1/chat/recommendations",
                token="recommendation-token",
                request=check_chat_recommendations.build_request(_build_args()),
                timeout_seconds=10.0,
                min_item_count=1,
                http_client=http_client,
            )

    result = anyio.run(run)

    assert captured_request["url"] == "http://fastapi.local/api/v1/chat/recommendations"
    assert captured_request["token"] == "recommendation-token"
    assert '"role":"MANUFACTURING_MANAGER"' in captured_request["body"]
    assert result == {
        "checkStatus": "PASS",
        "url": "http://fastapi.local/api/v1/chat/recommendations",
        "role": "MANUFACTURING_MANAGER",
        "keywordConfigured": True,
        "itemCount": 1,
        "minItemCount": 1,
        "fallbackUsed": False,
        "expectFallback": False,
        "questionIds": ["line-bottleneck-current"],
        "intents": ["LINE_BOTTLENECK"],
        "urlCount": 1,
        "networkChecked": True,
    }


def test_chat_recommendations_script_fails_when_item_count_is_below_minimum() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [], "fallbackUsed": False}, request=request)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_recommendations.check_chat_recommendations(
                base_url="http://fastapi.local",
                path="/api/v1/chat/recommendations",
                token="recommendation-token",
                request=check_chat_recommendations.build_request(_build_args()),
                timeout_seconds=10.0,
                min_item_count=1,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_RECOMMEND_001"
    assert "expected>=1, actual=0" in exc_info.value.message


def test_chat_recommendations_script_fails_when_fallback_is_expected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_recommendation_response(), request=request)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_recommendations.check_chat_recommendations(
                base_url="http://fastapi.local",
                path="/api/v1/chat/recommendations",
                token="recommendation-token",
                request=check_chat_recommendations.build_request(_build_args()),
                timeout_seconds=10.0,
                expect_fallback=True,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_RECOMMEND_001"
    assert "fallback이 사용되지 않았습니다" in exc_info.value.message


def test_chat_recommendations_script_fails_on_role_forbidden_intent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_recommendation_response(
                intent="URGENT_ORDER_IMPACT",
                url="/schedule-simulations?mode=read",
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_recommendations.check_chat_recommendations(
                base_url="http://fastapi.local",
                path="/api/v1/chat/recommendations",
                token="recommendation-token",
                request=check_chat_recommendations.build_request(
                    _build_args(role="OPERATOR")
                ),
                timeout_seconds=10.0,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_RECOMMEND_002"
    assert "Role 권한 밖 intent" in exc_info.value.message


def test_chat_recommendations_script_fails_operator_without_read_only_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_recommendation_response(), request=request)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_recommendations.check_chat_recommendations(
                base_url="http://fastapi.local",
                path="/api/v1/chat/recommendations",
                token="recommendation-token",
                request=check_chat_recommendations.build_request(
                    _build_args(role="OPERATOR")
                ),
                timeout_seconds=10.0,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_RECOMMEND_002"
    assert "read-only mode" in exc_info.value.message


def test_chat_recommendations_script_allows_operator_read_only_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_recommendation_response(url="/production-lines/status?mode=read"),
            request=request,
        )

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_chat_recommendations.check_chat_recommendations(
                base_url="http://fastapi.local",
                path="/api/v1/chat/recommendations",
                token="recommendation-token",
                request=check_chat_recommendations.build_request(
                    _build_args(role="OPERATOR")
                ),
                timeout_seconds=10.0,
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["checkStatus"] == "PASS"
    assert result["role"] == "OPERATOR"


def test_chat_recommendations_script_uses_error_response_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"code": "CHAT_SECURITY_003", "message": "추천 질문 권한이 없습니다."},
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_recommendations.check_chat_recommendations(
                base_url="http://fastapi.local",
                path="/api/v1/chat/recommendations",
                token="wrong-token",
                request=check_chat_recommendations.build_request(_build_args()),
                timeout_seconds=10.0,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_SECURITY_003"
    assert "추천 질문 권한이 없습니다" in exc_info.value.message


def test_chat_recommendations_script_main_does_not_expose_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_chat_recommendations(**kwargs) -> dict:
        return {
            "checkStatus": "PASS",
            "url": "http://fastapi.local/api/v1/chat/recommendations",
            "role": "MANUFACTURING_MANAGER",
            "keywordConfigured": True,
            "itemCount": 1,
            "minItemCount": 1,
            "fallbackUsed": False,
            "expectFallback": False,
            "questionIds": ["line-bottleneck-current"],
            "intents": ["LINE_BOTTLENECK"],
            "urlCount": 1,
            "networkChecked": True,
        }

    monkeypatch.setattr(
        check_chat_recommendations,
        "check_chat_recommendations",
        fake_check_chat_recommendations,
    )
    stdout = StringIO()

    exit_code = check_chat_recommendations.main(
        [
            "--base-url",
            "http://fastapi.local",
            "--token",
            "secret-recommendation-token",
        ],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "status=PASS" in output
    assert "secret-recommendation-token" not in output


def test_chat_recommendations_script_main_returns_one_without_token() -> None:
    stderr = StringIO()

    exit_code = check_chat_recommendations.main(
        ["--base-url", "http://fastapi.local"],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "FastAPI 추천 질문 점검 실패" in stderr.getvalue()
    assert "code=CHAT_SECURITY_003" in stderr.getvalue()
