from argparse import Namespace
from io import StringIO

import anyio
import httpx
import pytest

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode, ChatIntent
from scripts import check_chat_answer


def _build_args(**overrides):
    values = {
        "base_url": "http://fastapi.local",
        "path": "/api/v1/chat/answer",
        "token": "answer-token",
        "env_file": None,
        "timeout_seconds": 10.0,
        "question": "자재 부족 현황 알려줘",
        "role": "MANUFACTURING_MANAGER",
        "user_id": 1,
        "company_name": "S-MAP",
        "session_id": 10,
        "message_id": 24,
        "requested_at": "2026-05-12T10:30:00+09:00",
        "min_evidence_count": 0,
        "require_rdb_evidence": False,
        "min_document_source_count": 0,
        "require_vector_search": False,
        "require_llm_generation": False,
        "require_llm_cache_miss": False,
        "max_llm_total_tokens": None,
        "expected_llm_skipped_reason": None,
        "expected_security_status": None,
        "expected_security_code": None,
        "expected_intent": None,
        "markdown": False,
        "json": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _answer_response(
    evidence_count: int = 1,
    rdb_evidence_count: int | None = None,
    document_source_count: int = 0,
    used_rdb_evidence: bool | None = None,
    used_vector_search: bool = False,
    used_llm_generation: bool = False,
    llm_skipped_reason: str | None = None,
    security_status: str = "PASSED",
    security_code: str | None = None,
    intent: str = "MATERIAL_SHORTAGE",
    answer: str = "근거는 조회됐지만 LLM 답변 생성 기능이 아직 활성화되지 않았습니다.",
    llm_usage: dict[str, int] | None = None,
    llm_cache_hit: bool = False,
) -> dict:
    resolved_rdb_evidence_count = (
        evidence_count if rdb_evidence_count is None else rdb_evidence_count
    )
    resolved_used_rdb_evidence = (
        resolved_rdb_evidence_count > 0
        if used_rdb_evidence is None
        else used_rdb_evidence
    )
    sources = []
    urls = []
    if evidence_count > 0:
        urls.append(
            {
                "label": "RM-AL-001 알루미늄 원자재 재고 부족",
                "url": "/materials/inventory/1?mode=read",
                "type": "MATERIAL",
            }
        )
        sources.append(
            {
                "sourceType": "MATERIAL",
                "title": "RM-AL-001 알루미늄 원자재 재고 부족",
                "summary": "생산계획 1001에서 부족 상태입니다.",
                "url": "/materials/inventory/1?mode=read",
                "referenceId": 1,
                "source": "production_plan_materials",
                "basisTime": "2026-05-12T10:35:00+09:00",
                "sourceOrigin": "RDB",
            }
        )
    if document_source_count > 0:
        urls.append(
            {
                "label": "LINE-A01 병목 대응 기준",
                "url": "/lines/LINE-A01",
                "type": "COMPANY_INFO",
            }
        )
        sources.append(
            {
                "sourceType": "COMPANY_INFO",
                "title": "LINE-A01 병목 대응 기준",
                "summary": "LINE-A01 대기 시간이 증가하면 처리량과 대기 수량을 확인합니다.",
                "url": "/lines/LINE-A01",
                "referenceId": None,
                "source": "line-bottleneck-guide:chunk-0001",
                "basisTime": "2026-05-12T10:35:00+09:00",
                "sourceOrigin": "QDRANT",
                "relevanceScore": 0.92,
            }
        )

    return {
        "sessionId": 10,
        "messageId": 24,
        "intent": intent,
        "answer": answer,
        "basisTime": "2026-05-12T10:35:00+09:00",
        "urls": urls,
        "sources": sources,
        "securityResult": {
            "status": security_status,
            "code": security_code,
            "reason": "보안 필터를 통과했고 내부 근거가 확인되었습니다.",
        },
        "modelResult": {
            "usedVectorSearch": used_vector_search,
            "usedRdbEvidence": resolved_used_rdb_evidence,
            "usedLlmGeneration": used_llm_generation,
            "llmCacheHit": llm_cache_hit,
            "llmUsage": llm_usage,
            "rdbEvidenceCount": resolved_rdb_evidence_count,
            "documentSourceCount": document_source_count,
            "evidenceCount": evidence_count,
            "vectorSearchSkippedReason": (
                None if used_vector_search else "Qdrant 검색이 비활성화되어 있습니다."
            ),
            "llmGenerationSkippedReason": (
                None
                if used_llm_generation
                else (
                    llm_skipped_reason
                    or "LLM 답변 생성 기능이 비활성화되어 있습니다."
                )
            ),
        },
    }


def test_check_chat_answer_script_builds_request() -> None:
    request = check_chat_answer.build_request(
        _build_args(role=" executive ", user_id=7)
    )

    assert request.session_id == 10
    assert request.message_id == 24
    assert request.user.user_id == 7
    assert request.user.role == "EXECUTIVE"
    assert request.question == "자재 부족 현황 알려줘"


def test_check_chat_answer_script_resolves_path_and_token() -> None:
    settings = Settings(
        api_v1_prefix="/ai/api/v1",
        chat_answer_internal_token="env-answer-token",
    )

    assert (
        check_chat_answer.resolve_answer_path(_build_args(path=None), settings)
        == "/ai/api/v1/chat/answer"
    )
    assert (
        check_chat_answer.resolve_answer_token(_build_args(token=None), settings)
        == "env-answer-token"
    )
    assert (
        check_chat_answer.build_answer_url(
            "http://fastapi.local/",
            "/api/v1/chat/answer",
        )
        == "http://fastapi.local/api/v1/chat/answer"
    )


def test_check_chat_answer_script_calls_fastapi_answer_contract() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["token"] = request.headers.get("X-Internal-Token")
        captured_request["body"] = request.read().decode()
        return httpx.Response(200, json=_answer_response(), request=request)

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                min_evidence_count=1,
                require_rdb_evidence=True,
                min_document_source_count=0,
                require_vector_search=False,
                http_client=http_client,
            )

    result = anyio.run(run)

    assert captured_request["url"] == "http://fastapi.local/api/v1/chat/answer"
    assert captured_request["token"] == "answer-token"
    assert '"question":"자재 부족 현황 알려줘"' in captured_request["body"]
    assert result == {
        "checkStatus": "PASS",
        "url": "http://fastapi.local/api/v1/chat/answer",
        "intent": ChatIntent.MATERIAL_SHORTAGE.value,
        "answer": "근거는 조회됐지만 LLM 답변 생성 기능이 아직 활성화되지 않았습니다.",
        "expectedIntent": None,
        "securityStatus": "PASSED",
        "expectedSecurityStatus": None,
        "securityCode": None,
        "expectedSecurityCode": None,
        "evidenceCount": 1,
        "minEvidenceCount": 1,
        "requireRdbEvidence": True,
        "rdbEvidenceCount": 1,
        "documentSourceCount": 0,
        "minDocumentSourceCount": 0,
        "usedRdbEvidence": True,
        "usedVectorSearch": False,
        "requireVectorSearch": False,
        "vectorSearchSkippedReason": "Qdrant 검색이 비활성화되어 있습니다.",
        "usedLlmGeneration": False,
        "llmCacheHit": False,
        "llmUsage": None,
        "maxLlmTotalTokens": None,
        "requireLlmGeneration": False,
        "requireLlmCacheMiss": False,
        "llmGenerationSkippedReason": "LLM 답변 생성 기능이 비활성화되어 있습니다.",
        "expectedLlmGenerationSkippedReason": None,
        "sourceCount": 1,
        "sourceDetails": [
            {
                "sourceType": "MATERIAL",
                "title": "RM-AL-001 알루미늄 원자재 재고 부족",
                "url": "/materials/inventory/1?mode=read",
                "sourceOrigin": "RDB",
            }
        ],
        "urlCount": 1,
        "urlDetails": [
            {
                "label": "RM-AL-001 알루미늄 원자재 재고 부족",
                "url": "/materials/inventory/1?mode=read",
                "type": "MATERIAL",
            }
        ],
    }


def test_check_chat_answer_script_exposes_llm_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                used_llm_generation=True,
                llm_usage={
                    "promptTokens": 120,
                    "completionTokens": 32,
                    "totalTokens": 152,
                },
            ),
            request=request,
        )

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                min_evidence_count=1,
                require_rdb_evidence=True,
                min_document_source_count=0,
                require_vector_search=False,
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["llmUsage"] == {
        "promptTokens": 120,
        "completionTokens": 32,
        "totalTokens": 152,
    }


def test_check_chat_answer_script_fails_when_llm_total_tokens_exceed_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                used_llm_generation=True,
                llm_usage={
                    "promptTokens": 120,
                    "completionTokens": 32,
                    "totalTokens": 152,
                },
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                max_llm_total_tokens=100,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_LLM_004"
    assert "expected<=100, actual=152" in exc_info.value.message


def test_check_chat_answer_script_fails_when_llm_token_limit_requires_missing_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                used_llm_generation=True,
                llm_usage=None,
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                max_llm_total_tokens=100,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_LLM_004"
    assert "LLM total token 사용량을 확인할 수 없습니다" in exc_info.value.message
    assert "llmUsage가 필요합니다" in exc_info.value.message


def test_check_chat_answer_script_fails_when_evidence_count_is_below_minimum() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_answer_response(evidence_count=0), request=request)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                min_evidence_count=1,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_001"
    assert "expected>=1, actual=0" in exc_info.value.message


def test_check_chat_answer_script_fails_when_rdb_evidence_is_required() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                evidence_count=1,
                rdb_evidence_count=0,
                document_source_count=1,
                used_rdb_evidence=False,
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                require_rdb_evidence=True,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_001"
    assert "RDB Evidence가 사용되지 않았습니다" in exc_info.value.message


def test_check_chat_answer_script_validates_expected_security_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                evidence_count=0,
                used_rdb_evidence=False,
                security_status="BLOCKED_UNAUTHORIZED",
                security_code="CHAT_SECURITY_004",
            ),
            request=request,
        )

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(
                    _build_args(
                        role="OPERATOR",
                        question="납기 지연 시 예상 패널티를 알려줘",
                    )
                ),
                timeout_seconds=10.0,
                expected_security_status="BLOCKED_UNAUTHORIZED",
                expected_security_code="CHAT_SECURITY_004",
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["securityStatus"] == "BLOCKED_UNAUTHORIZED"
    assert result["expectedSecurityStatus"] == "BLOCKED_UNAUTHORIZED"
    assert result["securityCode"] == "CHAT_SECURITY_004"
    assert result["expectedSecurityCode"] == "CHAT_SECURITY_004"


def test_check_chat_answer_script_validates_empty_security_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_answer_response(), request=request)

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                expected_security_status="PASSED",
                expected_security_code="NONE",
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["securityStatus"] == "PASSED"
    assert result["securityCode"] is None
    assert result["expectedSecurityCode"] == "NONE"


def test_check_chat_answer_script_validates_expected_intent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_answer_response(), request=request)

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                expected_intent="MATERIAL_SHORTAGE",
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["intent"] == "MATERIAL_SHORTAGE"
    assert result["expectedIntent"] == "MATERIAL_SHORTAGE"


def test_check_chat_answer_script_fails_when_intent_is_unexpected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(intent="UNKNOWN"),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                expected_intent="REPORT_LOOKUP",
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_001"
    assert "expected=REPORT_LOOKUP, actual=UNKNOWN" in exc_info.value.message


def test_check_chat_answer_script_fails_when_security_status_is_unexpected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_answer_response(), request=request)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                expected_security_status="BLOCKED_UNAUTHORIZED",
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_SECURITY_001"
    assert "expected=BLOCKED_UNAUTHORIZED, actual=PASSED" in exc_info.value.message


def test_check_chat_answer_script_fails_when_security_code_is_unexpected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                evidence_count=0,
                used_rdb_evidence=False,
                security_status="BLOCKED_UNAUTHORIZED",
                security_code="CHAT_SECURITY_004",
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                expected_security_status="BLOCKED_UNAUTHORIZED",
                expected_security_code="CHAT_SECURITY_001",
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_SECURITY_001"
    assert "expected=CHAT_SECURITY_001, actual=CHAT_SECURITY_004" in exc_info.value.message


def test_check_chat_answer_script_passes_when_vector_search_is_required() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                evidence_count=1,
                rdb_evidence_count=0,
                document_source_count=1,
                used_rdb_evidence=False,
                used_vector_search=True,
            ),
            request=request,
        )

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                min_document_source_count=1,
                require_vector_search=True,
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["usedVectorSearch"] is True
    assert result["requireVectorSearch"] is True
    assert result["documentSourceCount"] == 1
    assert result["minDocumentSourceCount"] == 1
    assert result["sourceCount"] == 2


def test_check_chat_answer_script_fails_when_document_source_count_is_below_minimum() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                evidence_count=1,
                document_source_count=0,
                used_vector_search=True,
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                min_document_source_count=1,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_001"
    assert "Qdrant 문서 출처 개수가 기준보다 적습니다" in exc_info.value.message
    assert "expected>=1, actual=0" in exc_info.value.message


def test_check_chat_answer_script_fails_when_vector_search_is_required() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                evidence_count=1,
                document_source_count=1,
                used_vector_search=False,
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                require_vector_search=True,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_001"
    assert "Qdrant Vector Search가 사용되지 않았습니다" in exc_info.value.message


def test_check_chat_answer_script_passes_when_llm_generation_is_required() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                evidence_count=1,
                used_llm_generation=True,
            ),
            request=request,
        )

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                require_llm_generation=True,
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["usedLlmGeneration"] is True
    assert result["requireLlmGeneration"] is True
    assert result["llmGenerationSkippedReason"] is None
    assert result["expectedLlmGenerationSkippedReason"] is None


def test_check_chat_answer_script_fails_when_llm_generation_is_required() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(evidence_count=1, used_llm_generation=False),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                require_llm_generation=True,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_LLM_004"
    assert "LLM 답변 생성이 사용되지 않았습니다" in exc_info.value.message


def test_check_chat_answer_script_passes_when_llm_cache_miss_is_required() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                evidence_count=1,
                used_llm_generation=True,
                llm_cache_hit=False,
            ),
            request=request,
        )

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                require_llm_generation=True,
                require_llm_cache_miss=True,
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["usedLlmGeneration"] is True
    assert result["llmCacheHit"] is False
    assert result["requireLlmCacheMiss"] is True


def test_check_chat_answer_script_fails_when_llm_cache_miss_requires_generation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                evidence_count=1,
                used_llm_generation=False,
                llm_cache_hit=False,
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                require_llm_cache_miss=True,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_LLM_004"
    assert "LLM 캐시 미스를 확인할 수 없습니다" in exc_info.value.message


def test_check_chat_answer_script_fails_when_llm_cache_miss_is_required_but_cache_hit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                evidence_count=1,
                used_llm_generation=True,
                llm_cache_hit=True,
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                require_llm_cache_miss=True,
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_LLM_004"
    assert "LLM 캐시를 사용했습니다" in exc_info.value.message


def test_check_chat_answer_script_validates_expected_llm_skipped_reason() -> None:
    expected_reason = "LLM 서버 호출에 실패해 근거 기반 대체 답변을 반환했습니다."

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                evidence_count=1,
                used_llm_generation=False,
                llm_skipped_reason=expected_reason,
            ),
            request=request,
        )

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                expected_llm_skipped_reason=expected_reason,
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["llmGenerationSkippedReason"] == expected_reason
    assert result["expectedLlmGenerationSkippedReason"] == expected_reason


def test_check_chat_answer_script_validates_no_llm_skipped_reason() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(evidence_count=1, used_llm_generation=True),
            request=request,
        )

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                expected_llm_skipped_reason="NONE",
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["llmGenerationSkippedReason"] is None
    assert result["expectedLlmGenerationSkippedReason"] == "NONE"


def test_check_chat_answer_script_formats_markdown_result() -> None:
    output = check_chat_answer.format_markdown_result(
        {
            "checkStatus": "PASS",
            "url": "http://fastapi.local/api/v1/chat/answer",
            "intent": "LINE_BOTTLENECK",
            "securityStatus": "PASSED",
            "securityCode": None,
            "evidenceCount": 2,
            "rdbEvidenceCount": 1,
            "documentSourceCount": 1,
            "usedLlmGeneration": False,
            "llmCacheHit": False,
            "llmUsage": {
                "promptTokens": 120,
                "completionTokens": 32,
                "totalTokens": 152,
            },
            "maxLlmTotalTokens": 200,
            "usedVectorSearch": True,
            "answer": "핵심 답변: LINE-PE-01 병목 근거를 확인했습니다.",
            "sourceDetails": [
                {
                    "sourceOrigin": "RDB",
                    "sourceType": "LINE",
                    "title": "LINE-PE-01 MAINTENANCE",
                    "url": "/production-lines/103?mode=read",
                },
                {
                    "sourceOrigin": "QDRANT",
                    "sourceType": "COMPANY_INFO",
                    "title": "LINE-PE-01 병목 대응 기준",
                    "url": "/company-info/line-pe-01-bottleneck-guide",
                },
            ],
            "urlDetails": [
                {
                    "type": "LINE",
                    "label": "LINE-PE-01 MAINTENANCE",
                    "url": "/production-lines/103?mode=read",
                }
            ],
        }
    )

    assert "# 챗봇 답변 점검 결과" in output
    assert "Intent: `LINE_BOTTLENECK`" in output
    assert "LLM Cache `False`" in output
    assert "Cache Miss 요구 `False`" in output
    assert "- LLM 토큰 사용량: `prompt=120, completion=32, total=152`" in output
    assert "- LLM 최대 토큰 기준: `200`" in output
    assert "```text\n핵심 답변: LINE-PE-01 병목 근거를 확인했습니다.\n```" in output
    assert "| `RDB` / `LINE` | LINE-PE-01 MAINTENANCE |" in output
    assert "| `LINE` | LINE-PE-01 MAINTENANCE | `/production-lines/103?mode=read` |" in output


def test_check_chat_answer_script_fails_when_llm_skipped_reason_is_unexpected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                evidence_count=1,
                used_llm_generation=False,
                llm_skipped_reason="LLM이 빈 답변을 반환했습니다.",
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_chat_answer.check_chat_answer(
                base_url="http://fastapi.local",
                path="/api/v1/chat/answer",
                token="answer-token",
                request=check_chat_answer.build_request(_build_args()),
                timeout_seconds=10.0,
                expected_llm_skipped_reason=(
                    "LLM 서버 호출에 실패해 근거 기반 대체 답변을 반환했습니다."
                ),
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_LLM_004"
    assert "LLM 생성 스킵 사유가 기대값과 다릅니다" in exc_info.value.message


def test_check_chat_answer_script_main_does_not_expose_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_chat_answer(**kwargs) -> dict:
        return {
            "checkStatus": "PASS",
            "url": "http://fastapi.local/api/v1/chat/answer",
            "intent": "MATERIAL_SHORTAGE",
            "expectedIntent": None,
            "securityStatus": "PASSED",
            "expectedSecurityStatus": None,
            "securityCode": None,
            "expectedSecurityCode": None,
            "evidenceCount": 1,
            "minEvidenceCount": 1,
            "requireRdbEvidence": True,
            "rdbEvidenceCount": 1,
            "documentSourceCount": 0,
            "minDocumentSourceCount": 0,
            "usedRdbEvidence": True,
            "usedVectorSearch": False,
            "requireVectorSearch": False,
            "vectorSearchSkippedReason": None,
            "usedLlmGeneration": False,
            "llmCacheHit": False,
            "requireLlmGeneration": False,
            "requireLlmCacheMiss": False,
            "llmGenerationSkippedReason": "LLM 답변 생성 기능이 비활성화되어 있습니다.",
            "expectedLlmGenerationSkippedReason": None,
            "sourceCount": 1,
            "urlCount": 1,
        }

    monkeypatch.setattr(check_chat_answer, "check_chat_answer", fake_check_chat_answer)
    stdout = StringIO()

    exit_code = check_chat_answer.main(
        [
            "--base-url",
            "http://fastapi.local",
            "--token",
            "secret-answer-token",
            "--min-evidence-count",
            "1",
            "--require-rdb-evidence",
            "--min-document-source-count",
            "1",
            "--require-vector-search",
        ],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "status=PASS" in output
    assert "secret-answer-token" not in output


def test_check_chat_answer_script_main_formats_markdown_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_chat_answer(**kwargs) -> dict:
        return {
            "checkStatus": "PASS",
            "url": "http://fastapi.local/api/v1/chat/answer",
            "intent": "MATERIAL_SHORTAGE",
            "answer": "핵심 답변: 자재 부족 근거를 확인했습니다.",
            "expectedIntent": None,
            "securityStatus": "PASSED",
            "expectedSecurityStatus": None,
            "securityCode": None,
            "expectedSecurityCode": None,
            "evidenceCount": 1,
            "minEvidenceCount": 1,
            "requireRdbEvidence": True,
            "rdbEvidenceCount": 1,
            "documentSourceCount": 0,
            "minDocumentSourceCount": 0,
            "usedRdbEvidence": True,
            "usedVectorSearch": False,
            "requireVectorSearch": False,
            "vectorSearchSkippedReason": None,
            "usedLlmGeneration": False,
            "llmCacheHit": False,
            "requireLlmGeneration": False,
            "requireLlmCacheMiss": False,
            "llmGenerationSkippedReason": "LLM 답변 생성 기능이 비활성화되어 있습니다.",
            "expectedLlmGenerationSkippedReason": None,
            "sourceCount": 1,
            "sourceDetails": [
                {
                    "sourceOrigin": "RDB",
                    "sourceType": "MATERIAL",
                    "title": "MAT-FOAM-ADD 발포 첨가제 SHORTAGE",
                    "url": "/materials/inventory/711?mode=read",
                }
            ],
            "urlCount": 1,
            "urlDetails": [
                {
                    "type": "MATERIAL",
                    "label": "MAT-FOAM-ADD 발포 첨가제 SHORTAGE",
                    "url": "/materials/inventory/711?mode=read",
                }
            ],
        }

    monkeypatch.setattr(check_chat_answer, "check_chat_answer", fake_check_chat_answer)
    stdout = StringIO()

    exit_code = check_chat_answer.main(
        [
            "--base-url",
            "http://fastapi.local",
            "--token",
            "secret-answer-token",
            "--min-evidence-count",
            "1",
            "--require-rdb-evidence",
            "--markdown",
        ],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "# 챗봇 답변 점검 결과" in output
    assert "핵심 답변: 자재 부족 근거를 확인했습니다." in output
    assert "| `RDB` / `MATERIAL` | MAT-FOAM-ADD 발포 첨가제 SHORTAGE |" in output
    assert "secret-answer-token" not in output


def test_check_chat_answer_script_main_returns_one_without_token() -> None:
    stderr = StringIO()

    exit_code = check_chat_answer.main(
        ["--base-url", "http://fastapi.local"],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "FastAPI 챗봇 답변 점검 실패" in stderr.getvalue()
    assert "code=CHAT_SECURITY_003" in stderr.getvalue()
    assert "nextAction=CHAT_ANSWER_INTERNAL_TOKEN" in stderr.getvalue()


@pytest.mark.parametrize(
    ("error", "expected_action_text"),
    [
        (
            ChatServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_SERVER_001,
                message="FastAPI 챗봇 답변 API 호출에 실패했습니다.",
            ),
            "FastAPI base URL",
        ),
        (
            ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_EVIDENCE_001,
                message="FastAPI 챗봇 응답 intent가 기대값과 다릅니다.",
            ),
            "expected intent",
        ),
        (
            ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_EVIDENCE_001,
                message="FastAPI 챗봇 응답에 RDB Evidence가 사용되지 않았습니다.",
            ),
            "RDB_EVIDENCE_ENABLED",
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
                status_code=500,
                code=ChatErrorCode.CHAT_SECURITY_001,
                message="FastAPI 챗봇 응답 보안 상태가 기대값과 다릅니다.",
            ),
            "securityStatus",
        ),
        (
            ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_LLM_004,
                message="FastAPI 챗봇 응답에 LLM 답변 생성이 사용되지 않았습니다.",
            ),
            "LLM_ENABLED",
        ),
    ],
)
def test_check_chat_answer_script_builds_failure_actions_by_error_type(
    error: ChatServiceError,
    expected_action_text: str,
) -> None:
    actions = check_chat_answer.build_answer_api_failure_actions(error)

    assert expected_action_text in actions[0]
