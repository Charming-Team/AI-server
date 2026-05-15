from datetime import datetime

import anyio
import httpx
import pytest

from app.core.config import Settings
from app.features.chat.evidence_service import EvidenceService
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatErrorCode,
    ChatIntent,
    ChatUserContext,
    EvidenceResult,
)


def _build_request() -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=10,
        messageId=24,
        user=ChatUserContext(
            userId=1,
            role="MANUFACTURING_MANAGER",
            department="생산관리팀",
            companyName="S-MAP",
            status="ACTIVE",
        ),
        question="자재 부족으로 영향받는 생산계획 알려줘",
        requestedAt=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
    )


def test_evidence_service_returns_empty_result_when_lookup_is_disabled() -> None:
    service = EvidenceService(Settings(evidence_lookup_enabled=False))
    request = _build_request()

    result = anyio.run(service.get_evidence, request, ChatIntent.MATERIAL_SHORTAGE)

    assert result.intent == ChatIntent.MATERIAL_SHORTAGE
    assert result.items == []
    assert result.has_evidence is False


def test_evidence_service_builds_internal_request_payload() -> None:
    service = EvidenceService(Settings(evidence_lookup_internal_token="internal-token"))
    request = _build_request().model_copy(
        update={"question": "LINE-A01 병목 원인을 알려줘"}
    )

    payload = service._build_payload(request, ChatIntent.LINE_BOTTLENECK)

    assert service._headers == {"X-Internal-Token": "internal-token"}
    assert payload["sessionId"] == 10
    assert payload["messageId"] == 24
    assert payload["intent"] == "LINE_BOTTLENECK"
    assert payload["user"] == {
        "userId": 1,
        "role": "MANUFACTURING_MANAGER",
        "companyName": "S-MAP",
    }
    assert payload["filters"] == {
        "limit": 5,
        "fromDate": None,
        "toDate": None,
        "targetType": "LINE",
        "targetCode": "LINE-A01",
    }


def test_evidence_service_calls_internal_endpoint_and_parses_response() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["token"] = request.headers.get("X-Internal-Token")
        captured_request["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "intent": "MATERIAL_SHORTAGE",
                "basisTime": "2026-05-12T10:35:00+09:00",
                "items": [
                    {
                        "type": "MATERIAL",
                        "title": "MAT-001 재고 부족",
                        "summary": "가용 재고가 안전 재고보다 낮습니다.",
                        "source": "material_inventories",
                        "referenceId": 11,
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)

    async def run() -> EvidenceResult:
        async with httpx.AsyncClient(transport=transport) as http_client:
            service = EvidenceService(
                Settings(
                    evidence_lookup_enabled=True,
                    evidence_lookup_base_url="http://spring.local",
                    evidence_lookup_internal_token="internal-token",
                ),
                http_client=http_client,
            )
            return await service.get_evidence(
                _build_request(),
                ChatIntent.MATERIAL_SHORTAGE,
            )

    result = anyio.run(run)

    assert captured_request["url"] == "http://spring.local/internal/chat/evidence"
    assert captured_request["token"] == "internal-token"
    assert '"intent":"MATERIAL_SHORTAGE"' in captured_request["body"]
    assert result.intent == ChatIntent.MATERIAL_SHORTAGE
    assert result.has_evidence is True
    assert result.items[0].title == "MAT-001 재고 부족"


def test_evidence_service_raises_custom_error_on_lookup_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"message": "spring error"},
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async def run() -> None:
        async with httpx.AsyncClient(transport=transport) as http_client:
            service = EvidenceService(
                Settings(
                    evidence_lookup_enabled=True,
                    evidence_lookup_base_url="http://spring.local",
                ),
                http_client=http_client,
            )
            await service.get_evidence(_build_request(), ChatIntent.MATERIAL_SHORTAGE)

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ChatErrorCode.CHAT_EVIDENCE_002
    assert exc_info.value.message == "RDB Evidence 조회에 실패했습니다."


def test_evidence_service_raises_custom_error_on_invalid_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"result": "invalid"},
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async def run() -> None:
        async with httpx.AsyncClient(transport=transport) as http_client:
            service = EvidenceService(
                Settings(
                    evidence_lookup_enabled=True,
                    evidence_lookup_base_url="http://spring.local",
                ),
                http_client=http_client,
            )
            await service.get_evidence(_build_request(), ChatIntent.MATERIAL_SHORTAGE)

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == ChatErrorCode.CHAT_EVIDENCE_003
    assert exc_info.value.message == "RDB Evidence 응답 형식이 올바르지 않습니다."
