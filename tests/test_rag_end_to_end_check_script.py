import json
from argparse import Namespace
from io import StringIO
from typing import Any

import anyio
import httpx
import pytest

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode
from scripts import check_rag_end_to_end


def _build_args(**overrides: Any) -> Namespace:
    values = {
        "base_url": "http://fastapi.local",
        "answer_token": "answer-token",
        "document_token": "document-token",
        "env_file": None,
        "timeout_seconds": 10.0,
        "document_id": "smoke-company-line-bottleneck",
        "title": "LINE-A01 병목 대응 기준",
        "content": "LINE-A01에서 대기 시간이 증가하면 처리량과 설비 상태를 확인합니다.",
        "url": "/lines/LINE-A01",
        "question": "LINE-A01 병목 대응 기준 알려줘",
        "role": "MANUFACTURING_MANAGER",
        "user_id": 1,
        "company_name": "S-MAP",
        "session_id": 10,
        "message_id": 24,
        "requested_at": "2026-05-12T10:30:00+09:00",
        "min_indexed_count": 1,
        "min_document_source_count": 1,
        "min_evidence_count": 1,
        "require_rdb_evidence": False,
        "max_llm_total_tokens": None,
        "keep_document": False,
        "validate_only": False,
        "json": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _index_response(document_id: str = "smoke-company-line-bottleneck") -> dict:
    return {
        "documentId": document_id,
        "operationType": "INDEX",
        "chunkCount": 1,
        "indexedCount": 1,
        "operation": {"operation_id": 100, "status": "completed"},
        "skippedReason": None,
    }


def _answer_response(
    document_source_count: int = 1,
    used_vector_search: bool = True,
    rdb_evidence_count: int = 0,
    llm_usage: dict[str, int] | None = None,
) -> dict:
    evidence_count = document_source_count + rdb_evidence_count
    sources = []
    urls = []
    if document_source_count:
        sources.append(
            {
                "sourceType": "COMPANY_INFO",
                "title": "LINE-A01 병목 대응 기준",
                "summary": "Qdrant 문서 근거입니다.",
                "url": "/lines/LINE-A01",
                "referenceId": None,
                "source": "smoke-company-line-bottleneck:chunk-0001",
                "basisTime": "2026-05-12T10:35:00+09:00",
                "sourceOrigin": "QDRANT",
                "relevanceScore": 0.92,
            }
        )
        urls.append(
            {
                "label": "LINE-A01 병목 대응 기준",
                "url": "/lines/LINE-A01",
                "type": "COMPANY_INFO",
            }
        )
    if rdb_evidence_count:
        sources.append(
            {
                "sourceType": "LINE",
                "title": "LINE-A01 병목 현황",
                "summary": "RDB View에서 조회한 라인 병목 근거입니다.",
                "url": "/lines/LINE-A01?mode=read",
                "referenceId": 1,
                "source": "chat_line_bottleneck_evidence_view",
                "basisTime": "2026-05-12T10:35:00+09:00",
                "sourceOrigin": "RDB",
            }
        )

    return {
        "sessionId": 10,
        "messageId": 24,
        "intent": "LINE_BOTTLENECK",
        "answer": "Qdrant 문서 근거를 확인했습니다.",
        "basisTime": "2026-05-12T10:35:00+09:00",
        "urls": urls,
        "sources": sources,
        "securityResult": {
            "status": "PASSED",
            "code": None,
            "reason": "보안 필터를 통과했습니다.",
        },
        "modelResult": {
            "usedVectorSearch": used_vector_search,
            "usedRdbEvidence": rdb_evidence_count > 0,
            "usedLlmGeneration": False,
            "llmCacheHit": False,
            "llmUsage": llm_usage,
            "rdbEvidenceCount": rdb_evidence_count,
            "documentSourceCount": document_source_count,
            "evidenceCount": evidence_count,
            "vectorSearchSkippedReason": None if used_vector_search else "Qdrant 미사용",
            "llmGenerationSkippedReason": "LLM 답변 생성 기능이 비활성화되어 있습니다.",
        },
    }


def _delete_response(document_id: str = "smoke-company-line-bottleneck") -> dict:
    return {
        "documentId": document_id,
        "operationType": "DELETE",
        "operation": {"operation_id": 101, "status": "completed"},
    }


def test_rag_end_to_end_validate_only_checks_tokens_and_paths() -> None:
    args = _build_args(validate_only=True)
    settings = Settings(api_v1_prefix="/ai/api/v1")

    result = check_rag_end_to_end.build_validate_only_result(
        args,
        settings,
        answer_token="answer-token",
        document_token="document-token",
    )

    assert result == {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "networkChecked": False,
        "baseUrl": "http://fastapi.local",
        "answerPath": "/ai/api/v1/chat/answer",
        "indexPath": "/ai/api/v1/chat/internal/documents/index",
        "deletePath": "/ai/api/v1/chat/internal/documents/delete",
        "documentId": "smoke-company-line-bottleneck",
        "documentType": "COMPANY_INFO",
        "question": "LINE-A01 병목 대응 기준 알려줘",
        "role": "MANUFACTURING_MANAGER",
        "answerTokenConfigured": True,
        "documentTokenConfigured": True,
        "minIndexedCount": 1,
        "minEvidenceCount": 1,
        "minDocumentSourceCount": 1,
        "requireRdbEvidence": False,
        "maxLlmTotalTokens": None,
        "keepDocument": False,
    }


def test_rag_end_to_end_calls_index_answer_delete_in_order() -> None:
    calls: list[str] = []
    captured_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.read().decode())
        captured_headers.append(request.headers.get("X-Internal-Token"))
        if path.endswith("/documents/index"):
            calls.append("index")
            assert body["documentId"] == "smoke-company-line-bottleneck"
            return httpx.Response(200, json=_index_response(), request=request)
        if path.endswith("/chat/answer"):
            calls.append("answer")
            assert body["question"] == "LINE-A01 병목 대응 기준 알려줘"
            return httpx.Response(200, json=_answer_response(), request=request)
        if path.endswith("/documents/delete"):
            calls.append("delete")
            assert body["documentId"] == "smoke-company-line-bottleneck"
            return httpx.Response(200, json=_delete_response(), request=request)
        return httpx.Response(404, request=request)

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_rag_end_to_end.check_rag_end_to_end(
                args=_build_args(),
                settings=Settings(),
                answer_token="answer-token",
                document_token="document-token",
                http_client=http_client,
            )

    result = anyio.run(run)

    assert calls == ["index", "answer", "delete"]
    assert captured_headers == ["document-token", "answer-token", "document-token"]
    assert result["checkStatus"] == "PASS"
    assert result["answer"]["documentSourceCount"] == 1
    assert result["answer"]["usedVectorSearch"] is True
    assert result["usedCleanup"] is True


def test_rag_end_to_end_applies_llm_total_token_limit_and_cleans_up() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        if path.endswith("/documents/index"):
            return httpx.Response(200, json=_index_response(), request=request)
        if path.endswith("/chat/answer"):
            return httpx.Response(
                200,
                json=_answer_response(
                    llm_usage={
                        "promptTokens": 120,
                        "completionTokens": 32,
                        "totalTokens": 152,
                    }
                ),
                request=request,
            )
        if path.endswith("/documents/delete"):
            return httpx.Response(200, json=_delete_response(), request=request)
        return httpx.Response(404, request=request)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_rag_end_to_end.check_rag_end_to_end(
                args=_build_args(max_llm_total_tokens=100),
                settings=Settings(),
                answer_token="answer-token",
                document_token="document-token",
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_LLM_004"
    assert "expected<=100, actual=152" in exc_info.value.message
    assert calls[-1].endswith("/documents/delete")


def test_rag_end_to_end_can_keep_document() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        if path.endswith("/documents/index"):
            return httpx.Response(200, json=_index_response(), request=request)
        if path.endswith("/chat/answer"):
            return httpx.Response(200, json=_answer_response(), request=request)
        return httpx.Response(404, request=request)

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_rag_end_to_end.check_rag_end_to_end(
                args=_build_args(keep_document=True),
                settings=Settings(),
                answer_token="answer-token",
                document_token="document-token",
                http_client=http_client,
            )

    result = anyio.run(run)

    assert len(calls) == 2
    assert result["usedCleanup"] is False
    assert result["cleanup"] is None


def test_rag_end_to_end_cleans_up_when_answer_fails() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        if path.endswith("/documents/index"):
            return httpx.Response(200, json=_index_response(), request=request)
        if path.endswith("/chat/answer"):
            return httpx.Response(
                200,
                json=_answer_response(document_source_count=0, used_vector_search=False),
                request=request,
            )
        if path.endswith("/documents/delete"):
            return httpx.Response(200, json=_delete_response(), request=request)
        return httpx.Response(404, request=request)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_rag_end_to_end.check_rag_end_to_end(
                args=_build_args(),
                settings=Settings(),
                answer_token="answer-token",
                document_token="document-token",
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_001"
    assert calls[-1].endswith("/documents/delete")


def test_rag_end_to_end_format_text_does_not_expose_tokens() -> None:
    output = check_rag_end_to_end.format_text_result(
        {
            "checkStatus": "PASS",
            "mode": "NETWORK",
            "networkChecked": True,
            "baseUrl": "http://fastapi.local",
            "answerPath": "/api/v1/chat/answer",
            "indexPath": "/api/v1/chat/internal/documents/index",
            "deletePath": "/api/v1/chat/internal/documents/delete",
            "documentId": "smoke-company-line-bottleneck",
            "question": "LINE-A01 병목 대응 기준 알려줘",
            "role": "MANUFACTURING_MANAGER",
            "index": {"indexedCount": 1},
            "answer": {
                "intent": "LINE_BOTTLENECK",
                "evidenceCount": 1,
                "rdbEvidenceCount": 0,
                "documentSourceCount": 1,
                "usedVectorSearch": True,
            },
            "cleanup": {"operationStatus": "completed"},
            "maxLlmTotalTokens": 200,
            "usedCleanup": True,
            "keepDocument": False,
        }
    )

    assert "status=PASS" in output
    assert "answerDocumentSourceCount=1" in output
    assert "maxLlmTotalTokens=200" in output
    assert "answer-token" not in output
    assert "document-token" not in output


def test_rag_end_to_end_main_returns_one_without_tokens() -> None:
    stderr = StringIO()

    exit_code = check_rag_end_to_end.main(
        ["--base-url", "http://fastapi.local"],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "RAG end-to-end 점검 실패" in stderr.getvalue()
    assert "CHAT_SECURITY_003" in stderr.getvalue()
    assert "nextAction=CHAT_ANSWER_INTERNAL_TOKEN" in stderr.getvalue()


@pytest.mark.parametrize(
    ("error", "expected_action_text"),
    [
        (
            ChatServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_SECURITY_003,
                message="FastAPI chat answer internal token이 설정되지 않았습니다.",
            ),
            "CHAT_ANSWER_INTERNAL_TOKEN",
        ),
        (
            ChatServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_SECURITY_003,
                message="FastAPI document index internal token이 설정되지 않았습니다.",
            ),
            "DOCUMENT_INDEX_INTERNAL_TOKEN",
        ),
        (
            ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_DOCUMENT_003,
                message="FastAPI 문서 인덱싱 API indexedCount가 기준보다 적습니다.",
            ),
            "EMBEDDING_ENABLED",
        ),
        (
            ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_EVIDENCE_001,
                message="FastAPI 챗봇 응답에 Qdrant Vector Search가 사용되지 않았습니다.",
            ),
            "QDRANT_SEARCH_ENABLED",
        ),
        (
            ChatServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_SERVER_001,
                message="FastAPI 문서 삭제 API 호출에 실패했습니다.",
            ),
            "FastAPI base URL",
        ),
    ],
)
def test_rag_end_to_end_builds_failure_actions_by_error_type(
    error: ChatServiceError,
    expected_action_text: str,
) -> None:
    actions = (
        check_rag_end_to_end.rag_end_to_end_failure_actions
        .build_rag_end_to_end_failure_actions(error)
    )

    assert expected_action_text in actions[0]
