from argparse import Namespace
from io import StringIO

import anyio
import httpx
import pytest

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode
from scripts import check_document_index_api


def _build_args(**overrides):
    values = {
        "base_url": "http://fastapi.local",
        "path": "/api/v1/chat/internal/documents/index",
        "token": "document-token",
        "env_file": None,
        "timeout_seconds": 10.0,
        "document_id": "company-info-line-bottleneck",
        "document_type": "COMPANY_INFO",
        "title": "LINE-A01 병목 대응 기준",
        "content": "LINE-A01에서 대기 시간이 증가하면 처리량과 가동률을 확인합니다.",
        "summary": None,
        "url": "/lines/LINE-A01?mode=read",
        "reference_type": None,
        "reference_id": None,
        "basis_time": None,
        "roles": None,
        "intents": None,
        "requested_by_role": "MANUFACTURING_MANAGER",
        "company_name": "S-MAP",
        "min_indexed_count": 0,
        "allow_skipped": False,
        "json": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _index_response(
    indexed_count: int = 1,
    chunk_count: int = 1,
    skipped_reason: str | None = None,
) -> dict:
    return {
        "documentId": "company-info-line-bottleneck",
        "operationType": "INDEX",
        "chunkCount": chunk_count,
        "indexedCount": indexed_count,
        "operation": {"operation_id": 100, "status": "completed"},
        "skippedReason": skipped_reason,
    }


def test_document_index_api_script_resolves_path_and_token() -> None:
    settings = Settings(
        api_v1_prefix="/ai/api/v1",
        document_index_internal_token="env-document-token",
    )

    assert (
        check_document_index_api.resolve_index_path(_build_args(path=None), settings)
        == "/ai/api/v1/chat/internal/documents/index"
    )
    assert (
        check_document_index_api.resolve_index_token(_build_args(token=None), settings)
        == "env-document-token"
    )
    assert (
        check_document_index_api.build_index_url(
            "http://fastapi.local/",
            "/api/v1/chat/internal/documents/index",
        )
        == "http://fastapi.local/api/v1/chat/internal/documents/index"
    )


def test_document_index_api_script_builds_normalized_sample_document() -> None:
    document = check_document_index_api.build_sample_document(
        _build_args(
            roles=[" operator ", "EXECUTIVE", "operator"],
            intents=[" line_bottleneck ", "REPORT_LOOKUP"],
        )
    )

    assert document.document_id == "company-info-line-bottleneck"
    assert document.document_type == "COMPANY_INFO"
    assert document.allowed_roles == ["OPERATOR", "EXECUTIVE"]
    assert document.intent_tags == ["LINE_BOTTLENECK", "REPORT_LOOKUP"]
    assert document.requested_by_role == "MANUFACTURING_MANAGER"


def test_document_index_api_script_calls_fastapi_index_contract() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["token"] = request.headers.get("X-Internal-Token")
        captured_request["body"] = request.read().decode()
        return httpx.Response(200, json=_index_response(), request=request)

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_document_index_api.check_document_index_api(
                base_url="http://fastapi.local",
                path="/api/v1/chat/internal/documents/index",
                token="document-token",
                document=check_document_index_api.build_sample_document(_build_args()),
                timeout_seconds=10.0,
                min_indexed_count=1,
                http_client=http_client,
            )

    result = anyio.run(run)

    assert captured_request["url"] == (
        "http://fastapi.local/api/v1/chat/internal/documents/index"
    )
    assert captured_request["token"] == "document-token"
    assert '"documentId":"company-info-line-bottleneck"' in captured_request["body"]
    assert '"content":"LINE-A01' in captured_request["body"]
    assert result == {
        "checkStatus": "PASS",
        "url": "http://fastapi.local/api/v1/chat/internal/documents/index",
        "documentId": "company-info-line-bottleneck",
        "documentType": "COMPANY_INFO",
        "title": "LINE-A01 병목 대응 기준",
        "contentCharCount": 39,
        "requestedByRole": "MANUFACTURING_MANAGER",
        "allowedRoleCount": 1,
        "intentTagCount": 1,
        "chunkCount": 1,
        "indexedCount": 1,
        "minIndexedCount": 1,
        "allowSkipped": False,
        "skippedReason": None,
        "operationId": 100,
        "operationStatus": "completed",
        "networkChecked": True,
    }


def test_document_index_api_script_fails_when_indexed_count_is_below_minimum() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_index_response(indexed_count=0), request=request)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_document_index_api.check_document_index_api(
                base_url="http://fastapi.local",
                path="/api/v1/chat/internal/documents/index",
                token="document-token",
                document=check_document_index_api.build_sample_document(_build_args()),
                timeout_seconds=10.0,
                min_indexed_count=1,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_DOCUMENT_003"
    assert "expected>=1, actual=0" in exc_info.value.message


def test_document_index_api_script_fails_when_document_is_skipped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_index_response(indexed_count=0, skipped_reason="임베딩 기능 비활성화"),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_document_index_api.check_document_index_api(
                base_url="http://fastapi.local",
                path="/api/v1/chat/internal/documents/index",
                token="document-token",
                document=check_document_index_api.build_sample_document(_build_args()),
                timeout_seconds=10.0,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_DOCUMENT_003"
    assert "문서 저장을 생략했습니다" in exc_info.value.message


def test_document_index_api_script_can_allow_skipped_document() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_index_response(indexed_count=0, skipped_reason="임베딩 기능 비활성화"),
            request=request,
        )

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_document_index_api.check_document_index_api(
                base_url="http://fastapi.local",
                path="/api/v1/chat/internal/documents/index",
                token="document-token",
                document=check_document_index_api.build_sample_document(_build_args()),
                timeout_seconds=10.0,
                allow_skipped=True,
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["checkStatus"] == "PASS"
    assert result["indexedCount"] == 0
    assert result["allowSkipped"] is True
    assert result["skippedReason"] == "임베딩 기능 비활성화"


def test_document_index_api_script_uses_error_response_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"code": "CHAT_SECURITY_003", "message": "문서 인덱싱 권한이 없습니다."},
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_document_index_api.check_document_index_api(
                base_url="http://fastapi.local",
                path="/api/v1/chat/internal/documents/index",
                token="wrong-token",
                document=check_document_index_api.build_sample_document(_build_args()),
                timeout_seconds=10.0,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_SECURITY_003"
    assert "문서 인덱싱 권한이 없습니다" in exc_info.value.message


def test_document_index_api_script_main_does_not_expose_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_document_index_api(**kwargs) -> dict:
        return {
            "checkStatus": "PASS",
            "url": "http://fastapi.local/api/v1/chat/internal/documents/index",
            "documentId": "company-info-line-bottleneck",
            "documentType": "COMPANY_INFO",
            "title": "LINE-A01 병목 대응 기준",
            "contentCharCount": 38,
            "requestedByRole": "MANUFACTURING_MANAGER",
            "allowedRoleCount": 1,
            "intentTagCount": 1,
            "chunkCount": 1,
            "indexedCount": 1,
            "minIndexedCount": 1,
            "allowSkipped": False,
            "skippedReason": None,
            "operationId": 100,
            "operationStatus": "completed",
            "networkChecked": True,
        }

    monkeypatch.setattr(
        check_document_index_api,
        "check_document_index_api",
        fake_check_document_index_api,
    )
    stdout = StringIO()

    exit_code = check_document_index_api.main(
        [
            "--base-url",
            "http://fastapi.local",
            "--token",
            "secret-document-token",
            "--min-indexed-count",
            "1",
        ],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "status=PASS" in output
    assert "secret-document-token" not in output
    assert "LINE-A01에서" not in output


def test_document_index_api_script_main_returns_one_without_token() -> None:
    stderr = StringIO()

    exit_code = check_document_index_api.main(
        ["--base-url", "http://fastapi.local"],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "FastAPI 문서 인덱싱 API 점검 실패" in stderr.getvalue()
    assert "code=CHAT_SECURITY_003" in stderr.getvalue()
    assert "nextAction=DOCUMENT_INDEX_INTERNAL_TOKEN" in stderr.getvalue()


def test_document_index_api_script_builds_failure_actions_by_error_type() -> None:
    skipped_error = ChatServiceError(
        status_code=500,
        code=ChatErrorCode.CHAT_DOCUMENT_003,
        message="FastAPI 문서 인덱싱 API가 문서 저장을 생략했습니다.",
    )
    network_error = ChatServiceError(
        status_code=503,
        code=ChatErrorCode.CHAT_SERVER_001,
        message="FastAPI 문서 인덱싱 API 호출에 실패했습니다.",
    )

    skipped_actions = check_document_index_api.build_document_api_failure_actions(
        skipped_error
    )
    network_actions = check_document_index_api.build_document_api_failure_actions(
        network_error
    )

    assert "EMBEDDING_ENABLED" in skipped_actions[0]
    assert "FastAPI base URL" in network_actions[0]
