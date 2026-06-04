from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from typing import Any

import anyio

from app.core.config import Settings
from app.features.chat.rdb_evidence_providers import (
    CatalogRdbEvidenceProvider,
    build_default_rdb_evidence_providers,
)
from app.features.chat.rdb_evidence_view_catalog import get_rdb_evidence_view_definition
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatIntent,
    ChatUserContext,
    EvidenceLookupFilters,
)


def _build_request(role: str = "MANUFACTURING_MANAGER") -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=10,
        messageId=24,
        user=ChatUserContext(
            userId=1,
            role=role,
            companyName="S-MAP",
            status="ACTIVE",
        ),
        question="RM-AL-001 자재 부족 현황 알려줘",
        requestedAt=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
    )


class FakeRdbEvidenceViewClient:
    def __init__(self, rows: list[Mapping[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple] = []

    async def fetch_rows(
        self,
        definition,
        filters: EvidenceLookupFilters,
        limit: int,
    ) -> list[Mapping[str, Any]]:
        self.calls.append((definition, filters, limit))
        return self.rows


def test_catalog_rdb_evidence_provider_converts_material_view_rows_to_evidence() -> None:
    definition = get_rdb_evidence_view_definition(ChatIntent.MATERIAL_SHORTAGE)
    assert definition is not None
    row = {
        "plan_material_id": 7001,
        "plan_id": 1001,
        "order_id": 501,
        "order_no": "ORD-202605-001",
        "product_id": 301,
        "product_code": "PROD-A001",
        "product_name": "배터리 모듈",
        "line_id": 101,
        "line_code": "LINE-A01",
        "line_name": "배터리 모듈 생산 Line",
        "material_id": 11,
        "material_code": "RM-AL-001",
        "material_name": "알루미늄 원자재",
        "material_type": "원자재",
        "required_quantity": Decimal("150.0000"),
        "reserved_quantity": Decimal("90.0000"),
        "consumed_quantity": Decimal("0.0000"),
        "shortage_quantity": Decimal("60.0000"),
        "unit": "KG",
        "material_plan_status": "SHORTAGE",
        "current_quantity": Decimal("120.0000"),
        "available_quantity": Decimal("30.0000"),
        "inventory_reserved_quantity": Decimal("90.0000"),
        "safety_stock_quantity": Decimal("50.0000"),
        "expected_inbound_at": datetime.fromisoformat("2026-05-20T09:00:00+09:00"),
        "expected_inbound_quantity": Decimal("80.0000"),
        "inventory_status": "LOW",
        "planned_start_at": datetime.fromisoformat("2026-05-13T09:00:00+09:00"),
        "planned_end_at": datetime.fromisoformat("2026-05-13T18:00:00+09:00"),
        "plan_status": "SCHEDULED",
    }
    view_client = FakeRdbEvidenceViewClient([row])
    provider = CatalogRdbEvidenceProvider(definition, view_client)
    filters = EvidenceLookupFilters(
        limit=5,
        targetType="MATERIAL",
        targetCode="RM-AL-001",
    )

    items = anyio.run(provider.get_evidence, _build_request(), filters)

    assert len(items) == 1
    item = items[0]
    assert item.type == "MATERIAL"
    assert item.title == "RM-AL-001 알루미늄 원자재 SHORTAGE"
    assert "생산계획 ID: 1001" in item.summary
    assert "부족 수량: 60" in item.summary
    assert "입고 예정 시각: 2026.05.20 09:00" in item.summary
    assert "2026-05-20T09:00:00+09:00" not in item.summary
    assert item.url == "/materials/inventory/11?mode=read"
    assert item.source == "chat_material_shortage_evidence_view"
    assert item.reference_id == 7001
    assert item.data["planMaterialId"] == 7001
    assert item.data["materialCode"] == "RM-AL-001"
    assert item.data["shortageQuantity"] == 60.0
    assert item.data["expectedInboundAt"] == "2026-05-20T09:00:00+09:00"
    assert item.allowed_roles == [
        "OPERATOR",
        "EXECUTIVE",
        "MANUFACTURING_MANAGER",
    ]
    assert view_client.calls == [(definition, filters, 5)]


def test_catalog_rdb_evidence_provider_formats_line_units_and_rates() -> None:
    definition = get_rdb_evidence_view_definition(ChatIntent.LINE_BOTTLENECK)
    assert definition is not None
    row = {
        "line_status_id": 1,
        "line_id": 101,
        "line_code": "LINE-ABS-01",
        "line_name": "ABS 주 생산 Line",
        "operation_status": "RUNNING",
        "throughput_rate": Decimal("0.8700"),
        "current_yield_rate": Decimal("0.9340"),
        "waiting_quantity": 3200,
        "waiting_time_hr": Decimal("2.5000"),
        "utilization_rate": Decimal("0.8800"),
        "progress_rate": Decimal("0.5160"),
        "processed_quantity": 8256,
        "defect_quantity": 545,
        "recorded_at": datetime.fromisoformat("2026-06-01T00:00:00+00:00"),
        "abnormal_machine_count": 0,
    }
    view_client = FakeRdbEvidenceViewClient([row])
    provider = CatalogRdbEvidenceProvider(definition, view_client)

    items = anyio.run(
        provider.get_evidence,
        _build_request(role="MANUFACTURING_MANAGER"),
        EvidenceLookupFilters(limit=5, targetType="LINE", targetCode="LINE-ABS-01"),
    )

    assert len(items) == 1
    assert "처리량: 87%" in items[0].summary
    assert "현재 수율: 93.4%" in items[0].summary
    assert "대기 시간: 2.5시간" in items[0].summary
    assert "가동률: 88%" in items[0].summary
    assert "진행률: 51.6%" in items[0].summary
    assert "기록 시각: 2026.06.01 00:00" in items[0].summary
    assert "2026-06-01T00:00:00+00:00" not in items[0].summary


def test_catalog_rdb_evidence_provider_builds_read_only_url_for_non_material_source() -> None:
    definition = get_rdb_evidence_view_definition(ChatIntent.DELIVERY_RISK)
    assert definition is not None
    row = {
        "prediction_id": 41001,
        "order_no": "SM-2026-05-FR-004",
        "risk_level": "CRITICAL",
        "main_cause_type": "MATERIAL_SHORTAGE",
        "delay_probability": Decimal("0.7820"),
        "predicted_at": datetime.fromisoformat("2026-05-23T10:05:00+09:00"),
    }
    view_client = FakeRdbEvidenceViewClient([row])
    provider = CatalogRdbEvidenceProvider(definition, view_client)

    items = anyio.run(
        provider.get_evidence,
        _build_request(role="EXECUTIVE"),
        EvidenceLookupFilters(limit=5),
    )

    assert len(items) == 1
    assert items[0].url == "/predictions/41001?mode=read"


def test_catalog_rdb_evidence_provider_blocks_role_outside_view_definition() -> None:
    definition = get_rdb_evidence_view_definition(ChatIntent.URGENT_ORDER_IMPACT)
    assert definition is not None
    view_client = FakeRdbEvidenceViewClient([])
    provider = CatalogRdbEvidenceProvider(definition, view_client)

    items = anyio.run(
        provider.get_evidence,
        _build_request(role="ADMIN"),
        EvidenceLookupFilters(limit=5),
    )

    assert items == []
    assert view_client.calls == []


def test_build_default_rdb_evidence_providers_covers_catalog_definitions() -> None:
    providers = build_default_rdb_evidence_providers(Settings())

    assert {provider.intent for provider in providers} == {
        ChatIntent.DELIVERY_RISK,
        ChatIntent.MATERIAL_SHORTAGE,
        ChatIntent.PRODUCTION_PLAN,
        ChatIntent.URGENT_ORDER_IMPACT,
        ChatIntent.WORK_PRIORITY,
        ChatIntent.LINE_BOTTLENECK,
        ChatIntent.REPORT_LOOKUP,
    }
