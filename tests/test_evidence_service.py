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


class FakeMaterialShortageImpactRdbEvidenceService:
    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        return EvidenceResult(
            intent=intent,
            basisTime=request.requested_at,
            items=[
                _build_material_shortage_item(
                    425,
                    "ORD-202605-028",
                    "PE-PIPE",
                    "LINE-PE-02",
                    "MAT-HDPE",
                    "HDPE Resin",
                    1250,
                    "KG",
                    "SHORTAGE",
                ),
                _build_material_shortage_item(
                    423,
                    "ORD-202605-026",
                    "PE-PIPE",
                    "LINE-PE-02",
                    "MAT-HDPE",
                    "HDPE Resin",
                    410,
                    "KG",
                    "PARTIAL_RESERVED",
                ),
                _build_material_shortage_item(
                    420,
                    "ORD-202605-023",
                    "PP-HEAT",
                    "LINE-PP-01",
                    "MAT-PP-BASE",
                    "PP Base Resin",
                    860,
                    "KG",
                    "SHORTAGE",
                ),
            ],
        )


class FakeUrgentOrderImpactRdbEvidenceService:
    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        return EvidenceResult(
            intent=intent,
            basisTime=request.requested_at,
            items=[
                _build_urgent_order_impact_item(
                    101,
                    "ORD-202605-020",
                    "LINE-PE-01",
                    "LINE-PE-02",
                    True,
                    2.5,
                    "A",
                ),
                _build_urgent_order_impact_item(
                    102,
                    "ORD-202605-026",
                    "LINE-ABS-01",
                    "LINE-ABS-01",
                    False,
                    1.5,
                    "B",
                ),
                _build_urgent_order_impact_item(
                    103,
                    "ORD-202605-033",
                    "LINE-PP-01",
                    "LINE-PP-02",
                    True,
                    3,
                    "A",
                ),
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


def _build_material_shortage_item(
    plan_id: int,
    order_no: str,
    product_code: str,
    line_code: str,
    material_code: str,
    material_name: str,
    shortage_quantity: float,
    unit: str,
    material_plan_status: str,
) -> EvidenceItem:
    return EvidenceItem(
        type="MATERIAL",
        title=f"{material_code} {material_name} {material_plan_status}",
        summary=(
            f"생산계획 ID: {plan_id}, 주문 번호: {order_no}, "
            f"자재 코드: {material_code}, 부족 수량: {shortage_quantity}{unit}"
        ),
        url=f"/materials/inventory/{plan_id}?mode=read",
        source="chat_material_shortage_evidence_view",
        referenceId=plan_id,
        data={
            "planId": plan_id,
            "orderNo": order_no,
            "productCode": product_code,
            "lineCode": line_code,
            "materialCode": material_code,
            "materialName": material_name,
            "shortageQuantity": shortage_quantity,
            "unit": unit,
            "materialPlanStatus": material_plan_status,
        },
        allowedRoles=["OPERATOR", "EXECUTIVE", "MANUFACTURING_MANAGER"],
    )


def _build_urgent_order_impact_item(
    simulation_detail_id: int,
    order_no: str,
    before_line_code: str,
    after_line_code: str,
    after_is_delayed: bool,
    delay_reduction_hr: float,
    recommendation_grade: str,
) -> EvidenceItem:
    return EvidenceItem(
        type="ORDER",
        title=f"{order_no} Due-Date Optimal DUE_DATE_OPTIMIZATION",
        summary=(
            f"simulation_detail_id: {simulation_detail_id}, 대응안: Due-Date Optimal, "
            f"주문 번호: {order_no}, 변경 후 지연 여부: {after_is_delayed}"
        ),
        url=f"/orders/{simulation_detail_id}?mode=read",
        source="chat_urgent_order_impact_evidence_view",
        referenceId=simulation_detail_id,
        data={
            "simulationDetailId": simulation_detail_id,
            "orderNo": order_no,
            "beforeLineCode": before_line_code,
            "afterLineCode": after_line_code,
            "afterIsDelayed": after_is_delayed,
            "delayReductionHr": delay_reduction_hr,
            "recommendationGrade": recommendation_grade,
            "simulationType": "DUE_DATE_OPTIMIZATION",
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


def test_evidence_service_adds_running_line_summary_for_running_question() -> None:
    rdb_evidence_service = FakeLineRdbEvidenceService()
    request = _build_request().model_copy(
        update={"question": "현재 가동 중인 라인은 뭐야?"}
    )
    service = EvidenceService(
        Settings(
            evidence_lookup_enabled=True,
            rdb_evidence_enabled=True,
        ),
        rdb_evidence_service=rdb_evidence_service,
    )

    result = anyio.run(service.get_evidence, request, ChatIntent.LINE_BOTTLENECK)

    assert result.items[0].title == "가동 라인 전체 현황"
    assert "RUNNING 라인은 총 4개" in result.items[0].summary
    assert "RUNNING 라인 코드: LINE-ABS-01, LINE-ABS-02, LINE-PE-02, LINE-PP-01" in (
        result.items[0].summary
    )
    assert result.items[0].data["runningLineCount"] == 4
    assert result.items[0].data["runningLineCodes"] == [
        "LINE-ABS-01",
        "LINE-ABS-02",
        "LINE-PE-02",
        "LINE-PP-01",
    ]
    assert result.items[0].data["operationStatusCounts"] == {
        "MAINTENANCE": 1,
        "RUNNING": 4,
        "SETUP": 1,
    }


def test_evidence_service_adds_line_composition_summary_for_composition_question() -> None:
    rdb_evidence_service = FakeLineRdbEvidenceService()
    request = _build_request().model_copy(
        update={"question": "생산 라인 구성 알려줘"}
    )
    service = EvidenceService(
        Settings(
            evidence_lookup_enabled=True,
            rdb_evidence_enabled=True,
        ),
        rdb_evidence_service=rdb_evidence_service,
    )

    result = anyio.run(service.get_evidence, request, ChatIntent.LINE_BOTTLENECK)

    assert result.items[0].title == "생산 라인 구성 전체 현황"
    assert "생산 라인은 총 6개" in result.items[0].summary
    assert "LINE-ABS-01(RUNNING)" in result.items[0].summary
    assert "LINE-PP-02(MAINTENANCE)" in result.items[0].summary
    assert result.items[0].data["lineCount"] == 6
    assert result.items[0].data["lineCodes"] == [
        "LINE-ABS-01",
        "LINE-ABS-02",
        "LINE-PE-01",
        "LINE-PE-02",
        "LINE-PP-01",
        "LINE-PP-02",
    ]


def test_evidence_service_does_not_add_composition_summary_for_bottleneck_question() -> None:
    rdb_evidence_service = FakeLineRdbEvidenceService()
    request = _build_request().model_copy(
        update={"question": "전체 라인 병목 현황 알려줘"}
    )
    service = EvidenceService(
        Settings(
            evidence_lookup_enabled=True,
            rdb_evidence_enabled=True,
        ),
        rdb_evidence_service=rdb_evidence_service,
    )

    result = anyio.run(service.get_evidence, request, ChatIntent.LINE_BOTTLENECK)

    assert result.items[0].title == "LINE-ABS-01 RUNNING"


def test_evidence_service_summarizes_material_shortage_impacted_plans() -> None:
    rdb_evidence_service = FakeMaterialShortageImpactRdbEvidenceService()
    request = _build_request().model_copy(
        update={"question": "자재 부족으로 영향받는 생산계획을 알려줘"}
    )
    service = EvidenceService(
        Settings(
            evidence_lookup_enabled=True,
            rdb_evidence_enabled=True,
        ),
        rdb_evidence_service=rdb_evidence_service,
    )

    result = anyio.run(service.get_evidence, request, ChatIntent.MATERIAL_SHORTAGE)

    assert len(result.items) == 1
    summary_item = result.items[0]
    assert summary_item.type == "PLAN"
    assert summary_item.title == "자재 부족 영향 생산계획"
    assert "영향받는 생산계획은 총 3건" in summary_item.summary
    assert "계획 425 / ORD-202605-028 / MAT-HDPE / 부족 1250KG / LINE-PE-02" in (
        summary_item.summary
    )
    assert "부족 자재: MAT-HDPE 2개, MAT-PP-BASE 1개" in summary_item.summary
    assert "자재 상태: PARTIAL_RESERVED 1개, SHORTAGE 2개" in summary_item.summary
    assert "생산계획 ID:" not in summary_item.summary
    assert summary_item.url == "/production-plans?mode=read"
    assert summary_item.data["affectedPlanCount"] == 3
    assert summary_item.data["affectedPlanIds"] == [425, 423, 420]


def test_evidence_service_summarizes_overall_urgent_order_impact() -> None:
    rdb_evidence_service = FakeUrgentOrderImpactRdbEvidenceService()
    request = _build_request().model_copy(
        update={"question": "긴급 주문이 전체 생산계획에 미치는 영향을 알려줘"}
    )
    service = EvidenceService(
        Settings(
            evidence_lookup_enabled=True,
            rdb_evidence_enabled=True,
        ),
        rdb_evidence_service=rdb_evidence_service,
    )

    result = anyio.run(service.get_evidence, request, ChatIntent.URGENT_ORDER_IMPACT)

    assert len(result.items) == 1
    summary_item = result.items[0]
    assert summary_item.type == "PLAN"
    assert summary_item.title == "긴급 주문 전체 생산계획 영향"
    assert "영향 대상은 총 3건" in summary_item.summary
    assert "ORD-202605-020" in summary_item.summary
    assert "변경 후 지연 예상: 2건" in summary_item.summary
    assert "총 지연 감소: 7시간" in summary_item.summary
    assert "추천 등급: A 2개, B 1개" in summary_item.summary
    assert "simulation_detail_id" not in summary_item.summary
    assert summary_item.url == "/production-plans?mode=read"
    assert summary_item.data["orderCount"] == 3
    assert summary_item.data["delayedOrderCount"] == 2
    assert summary_item.data["totalDelayReductionHr"] == 7.0


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
                _build_request().model_copy(
                    update={"question": "자재 재고 부족한 항목 알려줘"}
                ),
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
