from argparse import Namespace
from io import StringIO

import anyio
import httpx
import pytest

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode
from scripts import check_document_delete_api


def _build_args(**overrides):
    values = {
        "base_url": "http://fastapi.local",
        "path": "/ai/api/v1/chat/internal/documents/delete",
        "token": "document-token",
        "env_file": None,
        "timeout_seconds": 10.0,
        "document_id": "smoke-document-api-contract",
        "expected_operation_status": None,
        "json": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _delete_response(
    document_id: str = "smoke-document-api-contract",
    status: str = "completed",
) -> dict:
    return {
        "documentId": document_id,
        "operationType": "DELETE",
        "operation": {"operation_id": 101, "status": status},
    }


def test_document_delete_api_script_resolves_path_and_token() -> None:
    settings = Settings(
        api_v1_prefix="/ai/api/v1",
        document_index_internal_token="env-document-token",
    )

    assert (
        check_document_delete_api.resolve_delete_path(_build_args(path=None), settings)
        == "/ai/api/v1/chat/internal/documents/delete"
    )
    assert (
        check_document_delete_api.resolve_delete_token(_build_args(token=None), settings)
        == "env-document-token"
    )
    assert (
        check_document_delete_api.build_delete_url(
            "http://fastapi.local/",
            "/ai/api/v1/chat/internal/documents/delete",
        )
        == "http://fastapi.local/ai/api/v1/chat/internal/documents/delete"
    )


def test_document_delete_api_script_builds_normalized_request() -> None:
    request = check_document_delete_api.build_delete_request(
        _build_args(document_id=" smoke-document-api-contract ")
    )

    assert request.document_id == "smoke-document-api-contract"


def test_document_delete_api_script_calls_fastapi_delete_contract() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["token"] = request.headers.get("X-Internal-Token")
        captured_request["body"] = request.read().decode()
        return httpx.Response(200, json=_delete_response(), request=request)

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_document_delete_api.check_document_delete_api(
                base_url="http://fastapi.local",
                path="/ai/api/v1/chat/internal/documents/delete",
                token="document-token",
                request=check_document_delete_api.build_delete_request(_build_args()),
                timeout_seconds=10.0,
                expected_operation_status="completed",
                http_client=http_client,
            )

    result = anyio.run(run)

    assert captured_request["url"] == (
        "http://fastapi.local/ai/api/v1/chat/internal/documents/delete"
    )
    assert captured_request["token"] == "document-token"
    assert captured_request["body"] == '{"documentId":"smoke-document-api-contract"}'
    assert result == {
        "checkStatus": "PASS",
        "url": "http://fastapi.local/ai/api/v1/chat/internal/documents/delete",
        "documentId": "smoke-document-api-contract",
        "operationType": "DELETE",
        "operationId": 101,
        "operationStatus": "completed",
        "expectedOperationStatus": "completed",
        "networkChecked": True,
    }


def test_document_delete_api_script_fails_on_operation_status_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_delete_response(status="acknowledged"), request=request)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_document_delete_api.check_document_delete_api(
                base_url="http://fastapi.local",
                path="/ai/api/v1/chat/internal/documents/delete",
                token="document-token",
                request=check_document_delete_api.build_delete_request(_build_args()),
                timeout_seconds=10.0,
                expected_operation_status="completed",
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_DOCUMENT_003"
    assert "expected=completed, actual=acknowledged" in exc_info.value.message


def test_document_delete_api_script_fails_on_invalid_response_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "documentId": "smoke-document-api-contract",
                "operationType": "DELETE",
                "operation": {"operation_id": 101},
            },
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_document_delete_api.check_document_delete_api(
                base_url="http://fastapi.local",
                path="/ai/api/v1/chat/internal/documents/delete",
                token="document-token",
                request=check_document_delete_api.build_delete_request(_build_args()),
                timeout_seconds=10.0,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_DOCUMENT_003"
    assert "operation status" in exc_info.value.message


def test_document_delete_api_script_uses_error_response_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"code": "CHAT_SECURITY_003", "message": "문서 삭제 권한이 없습니다."},
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_document_delete_api.check_document_delete_api(
                base_url="http://fastapi.local",
                path="/ai/api/v1/chat/internal/documents/delete",
                token="wrong-token",
                request=check_document_delete_api.build_delete_request(_build_args()),
                timeout_seconds=10.0,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_SECURITY_003"
    assert "문서 삭제 권한이 없습니다" in exc_info.value.message


def test_document_delete_api_script_main_does_not_expose_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_document_delete_api(**kwargs) -> dict:
        return {
            "checkStatus": "PASS",
            "url": "http://fastapi.local/ai/api/v1/chat/internal/documents/delete",
            "documentId": "smoke-document-api-contract",
            "operationType": "DELETE",
            "operationId": 101,
            "operationStatus": "completed",
            "expectedOperationStatus": None,
            "networkChecked": True,
        }

    monkeypatch.setattr(
        check_document_delete_api,
        "check_document_delete_api",
        fake_check_document_delete_api,
    )
    stdout = StringIO()

    exit_code = check_document_delete_api.main(
        [
            "--base-url",
            "http://fastapi.local",
            "--token",
            "secret-document-token",
        ],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "status=PASS" in output
    assert "secret-document-token" not in output


def test_document_delete_api_script_main_returns_one_without_token() -> None:
    stderr = StringIO()

    exit_code = check_document_delete_api.main(
        ["--base-url", "http://fastapi.local"],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "FastAPI 문서 삭제 API 점검 실패" in stderr.getvalue()
    assert "code=CHAT_SECURITY_003" in stderr.getvalue()
    assert "nextAction=DOCUMENT_INDEX_INTERNAL_TOKEN" in stderr.getvalue()


def test_document_delete_api_script_builds_failure_actions_by_error_type() -> None:
    payload_error = ChatServiceError(
        status_code=400,
        code=ChatErrorCode.CHAT_DOCUMENT_002,
        message="문서 삭제 API 점검 payload 필수 필드 또는 타입이 올바르지 않습니다.",
    )
    response_error = ChatServiceError(
        status_code=502,
        code=ChatErrorCode.CHAT_DOCUMENT_003,
        message="FastAPI 문서 삭제 API 응답 형식이 올바르지 않습니다.",
    )

    payload_actions = check_document_delete_api.build_document_api_failure_actions(
        payload_error
    )
    response_actions = check_document_delete_api.build_document_api_failure_actions(
        response_error
    )

    assert "documentId" in payload_actions[0]
    assert "응답" in response_actions[0]
