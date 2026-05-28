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
    EvidenceItem,
    EvidenceResult,
)


class FakeRdbEvidenceService:
    def __init__(self) -> None:
        self.request: ChatAnswerRequest | None = None
        self.intent: ChatIntent | None = None

    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        self.request = request
        self.intent = intent
        return EvidenceResult(
            intent=intent,
            basisTime=request.requested_at,
            items=[
                {
                    "type": "MATERIAL",
                    "title": "RM-AL-001 알루미늄 원자재 재고 부족",
                    "summary": "read-only View에서 조회한 자재 부족 Evidence입니다.",
                    "source": "chat_material_shortage_evidence_view",
                    "referenceId": 7001,
                }
            ],
        )


class FakeLineRdbEvidenceService:
    def __init__(self) -> None:
        self.request: ChatAnswerRequest | None = None
        self.intent: ChatIntent | None = None

    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        self.request = request
        self.intent = intent
        return EvidenceResult(
            intent=intent,
            basisTime=request.requested_at,
            items=[
                _build_line_item(1, "LINE-ABS-01", "RUNNING"),
                _build_line_item(2, "LINE-ABS-02", "RUNNING"),
                _build_line_item(3, "LINE-PP-01", "RUNNING"),
                _build_line_item(4, "LINE-PP-02", "MAINTENANCE"),
                _build_line_item(5, "LINE-PE-01", "SETUP"),
                _build_line_item(6, "LINE-PE-02", "RUNNING"),
            ],
        )


def _build_request() -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=10,
        messageId=24,
        user=ChatUserContext(
            userId=1,
            role="MANUFACTURING_MANAGER",
            companyName="S-MAP",
            status="ACTIVE",
        ),
        question="자재 부족으로 영향받는 생산계획 알려줘",
        requestedAt=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
    )


def _build_line_item(
    line_id: int,
    line_code: str,
    operation_status: str,
) -> EvidenceItem:
    return EvidenceItem(
        type="LINE",
        title=f"{line_code} {operation_status}",
        summary=f"라인 코드: {line_code}, 가동 상태: {operation_status}",
        url=f"/production-lines/{line_id}?mode=read",
        source="chat_line_bottleneck_evidence_view",
        referenceId=line_id,
        data={
            "lineId": line_id,
            "lineCode": line_code,
            "operationStatus": operation_status,
            "recordedAt": "2026-06-01T00:00:00+00:00",
        },
        allowedRoles=["OPERATOR", "EXECUTIVE", "MANUFACTURING_MANAGER"],
    )


def test_evidence_service_returns_empty_result_when_lookup_is_disabled() -> None:
    service = EvidenceService(Settings(evidence_lookup_enabled=False))
    request = _build_request()

    result = anyio.run(service.get_evidence, request, ChatIntent.MATERIAL_SHORTAGE)

    assert result.intent == ChatIntent.MATERIAL_SHORTAGE
    assert result.items == []
    assert result.has_evidence is False


def test_evidence_service_uses_rdb_evidence_service_when_rdb_view_mode_is_enabled() -> None:
    rdb_evidence_service = FakeRdbEvidenceService()
    request = _build_request()
    service = EvidenceService(
        Settings(
            evidence_lookup_enabled=True,
            rdb_evidence_enabled=True,
        ),
        rdb_evidence_service=rdb_evidence_service,
    )

    result = anyio.run(service.get_evidence, request, ChatIntent.MATERIAL_SHORTAGE)

    assert rdb_evidence_service.request == request
    assert rdb_evidence_service.intent == ChatIntent.MATERIAL_SHORTAGE
    assert result.intent == ChatIntent.MATERIAL_SHORTAGE
    assert result.items[0].source == "chat_material_shortage_evidence_view"


def test_evidence_service_adds_line_count_summary_for_line_count_question() -> None:
    rdb_evidence_service = FakeLineRdbEvidenceService()
    request = _build_request().model_copy(
        update={"question": "우리 공정 라인은 몇개 있어?"}
    )
    service = EvidenceService(
        Settings(
            evidence_lookup_enabled=True,
            rdb_evidence_enabled=True,
        ),
        rdb_evidence_service=rdb_evidence_service,
    )

    result = anyio.run(service.get_evidence, request, ChatIntent.LINE_BOTTLENECK)

    assert result.items[0].title == "공정 라인 전체 현황"
    assert "공정 라인 수: 총 6개" in result.items[0].summary
    assert "LINE-ABS-01" in result.items[0].summary
    assert "RUNNING 4개" in result.items[0].summary
    assert "MAINTENANCE 1개" in result.items[0].summary
    assert "SETUP 1개" in result.items[0].summary
    assert result.items[0].url == "/production-lines?mode=read"
    assert result.items[0].data["lineCount"] == 6
    assert result.items[0].data["operationStatusCounts"] == {
        "MAINTENANCE": 1,
        "RUNNING": 4,
        "SETUP": 1,
    }


def test_evidence_service_builds_internal_request_payload() -> None:
    service = EvidenceService(Settings(evidence_lookup_internal_token="internal-token"))
    request = _build_request().model_copy(
        update={"question": "오늘 LINE-A01 병목 원인을 알려줘"}
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
        "fromDate": "2026-05-12",
        "toDate": "2026-05-12",
        "targetType": "LINE",
        "targetCode": "LINE-A01",
    }


def test_evidence_service_expands_internal_request_limit_for_count_question() -> None:
    service = EvidenceService(Settings(evidence_lookup_internal_token="internal-token"))
    request = _build_request().model_copy(
        update={"question": "우리 공정 라인은 몇개 있어?"}
    )

    payload = service._build_payload(request, ChatIntent.LINE_BOTTLENECK)

    assert payload["filters"]["limit"] == 50


def test_evidence_service_calls_internal_endpoint_and_parses_response() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["token"] = request.headers.get("X-Internal-Token")
        captured_request["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "success": True,
                "code": "COMMON200",
                "message": "요청 성공",
                "data": {
                    "intent": "MATERIAL_SHORTAGE",
                    "basisTime": "2026-05-12T10:35:00+09:00",
                    "items": [
                        {
                            "type": "MATERIAL",
                            "title": "RM-AL-001 알루미늄 원자재 재고 부족",
                            "summary": (
                                "생산계획 1001에서 RM-AL-001 알루미늄 원자재 "
                                "부족 상태입니다."
                            ),
                            "url": "/materials/inventory/11?mode=read",
                            "source": "production_plan_materials",
                            "referenceId": 7001,
                            "data": {
                                "planMaterialId": 7001,
                                "planId": 1001,
                                "materialId": 11,
                                "materialCode": "RM-AL-001",
                                "requiredQuantity": 150.0,
                                "reservedQuantity": 90.0,
                                "shortageQuantity": 60.0,
                                "inventoryRegistered": True,
                                "currentInventoryQuantity": 120.0,
                                "availableInventoryQuantity": 30.0,
                                "safetyStockQuantity": 50.0,
                                "inventoryStatus": "LOW",
                            },
                            "allowedRoles": [
                                "operator",
                                "executive",
                                "manufacturing_manager",
                            ],
                        }
                    ],
                },
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
    assert result.items[0].title == "RM-AL-001 알루미늄 원자재 재고 부족"
    assert result.items[0].source == "production_plan_materials"
    assert result.items[0].reference_id == 7001
    assert result.items[0].url == "/materials/inventory/11?mode=read"
    assert result.items[0].data["planMaterialId"] == 7001
    assert result.items[0].data["availableInventoryQuantity"] == 30.0
    assert result.items[0].data["safetyStockQuantity"] == 50.0
    assert result.items[0].data["inventoryStatus"] == "LOW"
    assert result.items[0].allowed_roles == [
        "OPERATOR",
        "EXECUTIVE",
        "MANUFACTURING_MANAGER",
    ]


def test_evidence_service_parses_legacy_raw_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "intent": "MATERIAL_SHORTAGE",
                "basisTime": "2026-05-12T10:35:00+09:00",
                "items": [],
            },
            request=request,
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

    assert result.intent == ChatIntent.MATERIAL_SHORTAGE
    assert result.items == []


def test_evidence_service_normalizes_lookup_path() -> None:
    service = EvidenceService(
        Settings(
            evidence_lookup_base_url="http://spring.local/",
            evidence_lookup_path="internal/chat/evidence",
            evidence_lookup_internal_token="internal-token",
        )
    )

    assert service._evidence_lookup_url == "http://spring.local/internal/chat/evidence"


@pytest.mark.parametrize(
    ("settings", "expected_message"),
    [
        (
            Settings(
                evidence_lookup_enabled=True,
                evidence_lookup_base_url=" ",
                evidence_lookup_internal_token="internal-token",
            ),
            "RDB Evidence 필수 설정이 누락되었습니다: evidence_lookup_base_url",
        ),
        (
            Settings(
                evidence_lookup_enabled=True,
                evidence_lookup_path=" ",
                evidence_lookup_internal_token="internal-token",
            ),
            "RDB Evidence 필수 설정이 누락되었습니다: evidence_lookup_path",
        ),
        (
            Settings(
                evidence_lookup_enabled=True,
                evidence_lookup_base_url=" ",
                evidence_lookup_path=" ",
                evidence_lookup_internal_token="internal-token",
            ),
            (
                "RDB Evidence 필수 설정이 누락되었습니다: "
                "evidence_lookup_base_url, evidence_lookup_path"
            ),
        ),
    ],
)
def test_evidence_service_requires_lookup_settings_when_enabled(
    settings: Settings,
    expected_message: str,
) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"items": []})

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            service = EvidenceService(settings, http_client=http_client)
            await service.get_evidence(_build_request(), ChatIntent.MATERIAL_SHORTAGE)

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ChatErrorCode.CHAT_EVIDENCE_004
    assert exc_info.value.message == expected_message
    assert called is False


def test_evidence_service_requires_internal_token_when_lookup_is_enabled() -> None:
    service = EvidenceService(
        Settings(
            evidence_lookup_enabled=True,
            evidence_lookup_base_url="http://spring.local",
            evidence_lookup_internal_token=None,
        )
    )

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(
            service.get_evidence,
            _build_request(),
            ChatIntent.MATERIAL_SHORTAGE,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == ChatErrorCode.CHAT_SECURITY_003
    assert exc_info.value.message == "RDB Evidence 내부 토큰이 설정되지 않았습니다."


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
                    evidence_lookup_internal_token="internal-token",
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
                    evidence_lookup_internal_token="internal-token",
                ),
                http_client=http_client,
            )
            await service.get_evidence(_build_request(), ChatIntent.MATERIAL_SHORTAGE)

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == ChatErrorCode.CHAT_EVIDENCE_003
    assert exc_info.value.message == "RDB Evidence 응답 형식이 올바르지 않습니다."


def test_evidence_service_raises_custom_error_on_failed_base_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "success": False,
                "code": "400-001",
                "message": "요청 값 검증에 실패했습니다.",
                "data": None,
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async def run() -> None:
        async with httpx.AsyncClient(transport=transport) as http_client:
            service = EvidenceService(
                Settings(
                    evidence_lookup_enabled=True,
                    evidence_lookup_base_url="http://spring.local",
                    evidence_lookup_internal_token="internal-token",
                ),
                http_client=http_client,
            )
            await service.get_evidence(_build_request(), ChatIntent.MATERIAL_SHORTAGE)

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == ChatErrorCode.CHAT_EVIDENCE_003
    assert exc_info.value.message == "RDB Evidence 응답 형식이 올바르지 않습니다."


def test_evidence_service_rejects_response_intent_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "intent": "REPORT_LOOKUP",
                "basisTime": "2026-05-12T10:35:00+09:00",
                "items": [],
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async def run() -> None:
        async with httpx.AsyncClient(transport=transport) as http_client:
            service = EvidenceService(
                Settings(
                    evidence_lookup_enabled=True,
                    evidence_lookup_base_url="http://spring.local",
                    evidence_lookup_internal_token="internal-token",
                ),
                http_client=http_client,
            )
            await service.get_evidence(_build_request(), ChatIntent.MATERIAL_SHORTAGE)

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == ChatErrorCode.CHAT_EVIDENCE_003
    assert (
        exc_info.value.message
        == "RDB Evidence 응답 의도가 요청 의도와 일치하지 않습니다."
    )
