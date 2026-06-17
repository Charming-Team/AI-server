import json
from argparse import Namespace
from io import StringIO

import anyio
import httpx
import pytest

from app.core.config import Settings
from scripts import index_document_payloads


def _build_payload(**overrides):
    payload = {
        "documentId": "company-line-a01-guide",
        "documentType": "COMPANY_INFO",
        "title": "LINE-A01 운영 기준",
        "content": "LINE-A01에서 대기 시간이 증가하면 처리량과 설비 상태를 확인합니다.",
        "summary": "LINE-A01 운영 기준입니다.",
        "url": "/lines/LINE-A01?mode=read",
        "referenceType": "LINE",
        "allowedRoles": ["OPERATOR", "MANUFACTURING_MANAGER", "EXECUTIVE"],
        "companyName": "S-MAP",
        "intentTags": ["LINE_BOTTLENECK", "WORK_PRIORITY"],
        "requestedByRole": "MANUFACTURING_MANAGER",
    }
    payload.update(overrides)
    return payload


def _write_payload(tmp_path, payload, file_name="document-payload.json"):
    input_path = tmp_path / file_name
    input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return input_path


def _build_args(**overrides):
    values = {
        "input": None,
        "input_dir": None,
        "base_url": "http://fastapi.local",
        "path": "/ai/api/v1/chat/internal/documents/index",
        "token": "document-token",
        "env_file": None,
        "timeout_seconds": 10.0,
        "min_indexed_count": 0,
        "allow_skipped": False,
        "dry_run": False,
        "json": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _index_response(document_id="company-line-a01-guide", indexed_count=1):
    return {
        "documentId": document_id,
        "operationType": "INDEX",
        "chunkCount": 1,
        "indexedCount": indexed_count,
        "operation": {"operation_id": 101, "status": "completed"},
        "skippedReason": None,
    }


def test_index_document_payloads_dry_run_validates_without_token(tmp_path) -> None:
    input_path = _write_payload(tmp_path, _build_payload())
    stdout = StringIO()

    exit_code = index_document_payloads.main(
        [
            "--input",
            str(input_path),
            "--dry-run",
        ],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "status=VALIDATED" in output
    assert "phase=DRY_RUN" in output
    assert "networkChecked=False" in output
    assert "documentCount=1" in output
    assert "validationValidCount=1" in output
    assert "LINE-A01에서" not in output


def test_index_document_payloads_dry_run_reports_validation_failure(tmp_path) -> None:
    input_path = _write_payload(
        tmp_path,
        _build_payload(documentType="PROCESS"),
    )
    stdout = StringIO()

    exit_code = index_document_payloads.main(
        [
            "--input",
            str(input_path),
            "--dry-run",
        ],
        stdout=stdout,
    )

    assert exit_code == 1
    assert "status=FAIL" in stdout.getvalue()
    assert "phase=DRY_RUN" in stdout.getvalue()
    assert "validationInvalidCount=1" in stdout.getvalue()


def test_index_document_payloads_calls_fastapi_for_each_payload(tmp_path) -> None:
    first_input = _write_payload(
        tmp_path,
        _build_payload(documentId="company-line-a01-guide"),
        "company-a.json",
    )
    second_input = _write_payload(
        tmp_path,
        _build_payload(documentId="company-line-b02-guide", title="LINE-B02 운영 기준"),
        "company-b.json",
    )
    captured_requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        captured_requests.append(
            {
                "url": str(request.url),
                "token": request.headers.get("X-Internal-Token"),
                "documentId": body["documentId"],
                "content": body["content"],
            }
        )
        return httpx.Response(
            200,
            json=_index_response(document_id=body["documentId"]),
            request=request,
        )

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await index_document_payloads.index_document_payloads(
                base_url="http://fastapi.local",
                path="/ai/api/v1/chat/internal/documents/index",
                token="document-token",
                input_paths=[str(first_input), str(second_input)],
                settings=Settings(),
                timeout_seconds=10.0,
                min_indexed_count=1,
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["checkStatus"] == "PASS"
    assert result["networkChecked"] is True
    assert result["documentCount"] == 2
    assert result["indexedDocumentCount"] == 2
    assert result["totalIndexedCount"] == 2
    assert [request["documentId"] for request in captured_requests] == [
        "company-line-a01-guide",
        "company-line-b02-guide",
    ]
    assert all(request["token"] == "document-token" for request in captured_requests)


def test_index_document_payloads_returns_partial_failure(tmp_path) -> None:
    first_input = _write_payload(
        tmp_path,
        _build_payload(documentId="company-line-a01-guide"),
        "company-a.json",
    )
    second_input = _write_payload(
        tmp_path,
        _build_payload(documentId="company-line-b02-guide", title="LINE-B02 운영 기준"),
        "company-b.json",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode())
        if body["documentId"] == "company-line-b02-guide":
            return httpx.Response(
                403,
                json={
                    "code": "CHAT_SECURITY_003",
                    "message": "문서 인덱싱 권한이 없습니다.",
                },
                request=request,
            )
        return httpx.Response(
            200,
            json=_index_response(document_id=body["documentId"]),
            request=request,
        )

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await index_document_payloads.index_document_payloads(
                base_url="http://fastapi.local",
                path="/ai/api/v1/chat/internal/documents/index",
                token="document-token",
                input_paths=[str(first_input), str(second_input)],
                settings=Settings(),
                timeout_seconds=10.0,
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["checkStatus"] == "FAIL"
    assert result["indexedDocumentCount"] == 1
    assert result["failedDocumentCount"] == 1
    assert result["results"][1]["error"]["code"] == "CHAT_SECURITY_003"


def test_index_document_payloads_main_does_not_expose_token_or_content(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = _write_payload(tmp_path, _build_payload(content="보안상 출력되면 안 되는 원문"))

    async def fake_index_document_payloads(**kwargs) -> dict:
        assert kwargs["token"] == "secret-document-token"
        return {
            "checkStatus": "PASS",
            "phase": "INDEX",
            "networkChecked": True,
            "url": "http://fastapi.local/ai/api/v1/chat/internal/documents/index",
            "documentCount": 1,
            "indexedDocumentCount": 1,
            "failedDocumentCount": 0,
            "totalChunkCount": 1,
            "totalIndexedCount": 1,
            "minIndexedCount": 0,
            "allowSkipped": False,
            "results": [
                {
                    "inputPath": str(input_path),
                    "status": "PASS",
                    "documentId": "company-line-a01-guide",
                    "documentType": "COMPANY_INFO",
                    "chunkCount": 1,
                    "indexedCount": 1,
                    "skippedReason": None,
                    "operationId": 101,
                    "operationStatus": "completed",
                }
            ],
        }

    monkeypatch.setattr(
        index_document_payloads,
        "index_document_payloads",
        fake_index_document_payloads,
    )
    stdout = StringIO()

    exit_code = index_document_payloads.main(
        [
            "--input",
            str(input_path),
            "--token",
            "secret-document-token",
        ],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "status=PASS" in output
    assert "company-line-a01-guide" in output
    assert "secret-document-token" not in output
    assert "보안상 출력되면 안 되는 원문" not in output


def test_index_document_payloads_main_requires_token_for_real_index(tmp_path) -> None:
    input_path = _write_payload(tmp_path, _build_payload())
    stderr = StringIO()

    exit_code = index_document_payloads.main(
        [
            "--input",
            str(input_path),
        ],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "문서 payload 일괄 인덱싱 실패" in stderr.getvalue()
    assert "CHAT_SECURITY_003" in stderr.getvalue()
