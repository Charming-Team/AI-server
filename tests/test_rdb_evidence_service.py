from datetime import UTC, datetime

import anyio

from app.core.config import Settings
from app.features.chat.rdb_evidence_service import RdbEvidenceService
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatIntent,
    ChatUserContext,
    EvidenceItem,
    EvidenceLookupFilters,
)


def _build_request(question: str = "RM-AL-001 자재 부족 현황 알려줘") -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=10,
        messageId=24,
        user=ChatUserContext(
            userId=1,
            role="MANUFACTURING_MANAGER",
            companyName="S-MAP",
            status="ACTIVE",
        ),
        question=question,
        requestedAt=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
    )


class StubRdbEvidenceProvider:
    def __init__(self, intent: ChatIntent = ChatIntent.MATERIAL_SHORTAGE) -> None:
        self.intent = intent
        self.request: ChatAnswerRequest | None = None
        self.filters: EvidenceLookupFilters | None = None

    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        filters: EvidenceLookupFilters,
    ) -> list[EvidenceItem]:
        self.request = request
        self.filters = filters
        return [
            EvidenceItem(
                type="MATERIAL",
                title="RM-AL-001 알루미늄 원자재 재고 부족",
                summary="생산계획 1001에서 RM-AL-001 부족 수량은 60KG입니다.",
                url="/materials/inventory/11?mode=read",
                source="chat_material_shortage_evidence_view",
                referenceId=7001,
                data={
                    "materialCode": "RM-AL-001",
                    "shortageQuantity": 60,
                },
                allowedRoles=["MANUFACTURING_MANAGER", "EXECUTIVE"],
            )
        ]


def test_rdb_evidence_service_returns_empty_result_when_disabled() -> None:
    provider = StubRdbEvidenceProvider()
    service = RdbEvidenceService(
        Settings(rdb_evidence_enabled=False),
        providers=[provider],
    )
    request = _build_request()

    result = anyio.run(service.get_evidence, request, ChatIntent.MATERIAL_SHORTAGE)

    assert result.intent == ChatIntent.MATERIAL_SHORTAGE
    assert result.basis_time == request.requested_at
    assert result.items == []
    assert provider.request is None


def test_rdb_evidence_service_routes_to_matching_provider_with_filters() -> None:
    provider = StubRdbEvidenceProvider()
    basis_time = datetime(2026, 5, 12, 1, 35, tzinfo=UTC)
    service = RdbEvidenceService(
        Settings(rdb_evidence_enabled=True),
        providers=[provider],
        clock=lambda: basis_time,
    )
    request = _build_request()

    result = anyio.run(service.get_evidence, request, ChatIntent.MATERIAL_SHORTAGE)

    assert result.intent == ChatIntent.MATERIAL_SHORTAGE
    assert result.basis_time == basis_time
    assert len(result.items) == 1
    assert provider.request == request
    assert provider.filters == EvidenceLookupFilters(
        limit=5,
        targetType="MATERIAL",
        targetCode="RM-AL-001",
    )
    assert result.items[0].source == "chat_material_shortage_evidence_view"
    assert result.items[0].data["shortageQuantity"] == 60


def test_rdb_evidence_service_returns_empty_result_for_unsupported_intent() -> None:
    provider = StubRdbEvidenceProvider()
    service = RdbEvidenceService(
        Settings(rdb_evidence_enabled=True),
        providers=[provider],
    )
    request = _build_request()

    result = anyio.run(service.get_evidence, request, ChatIntent.LINE_BOTTLENECK)

    assert result.intent == ChatIntent.LINE_BOTTLENECK
    assert result.items == []
    assert provider.request is None


def test_rdb_evidence_service_caps_filter_limit_by_settings() -> None:
    provider = StubRdbEvidenceProvider()
    service = RdbEvidenceService(
        Settings(
            rdb_evidence_enabled=True,
            rdb_evidence_max_limit=3,
        ),
        providers=[provider],
    )
    request = _build_request()

    anyio.run(service.get_evidence, request, ChatIntent.MATERIAL_SHORTAGE)

    assert provider.filters is not None
    assert provider.filters.limit == 3


def test_rdb_evidence_service_passes_date_filters_from_question() -> None:
    provider = StubRdbEvidenceProvider()
    service = RdbEvidenceService(
        Settings(rdb_evidence_enabled=True),
        providers=[provider],
    )
    request = _build_request("이번 주 RM-AL-001 자재 부족 현황 알려줘")

    anyio.run(service.get_evidence, request, ChatIntent.MATERIAL_SHORTAGE)

    assert provider.filters is not None
    assert provider.filters.from_date == "2026-05-11"
    assert provider.filters.to_date == "2026-05-17"
    assert provider.filters.target_type == "MATERIAL"
    assert provider.filters.target_code == "RM-AL-001"


def test_rdb_evidence_service_skips_report_metadata_for_company_info_question() -> None:
    provider = StubRdbEvidenceProvider(intent=ChatIntent.REPORT_LOOKUP)
    service = RdbEvidenceService(
        Settings(rdb_evidence_enabled=True),
        providers=[provider],
    )
    request = _build_request("S-Map 회사 개요 알려줘")

    result = anyio.run(service.get_evidence, request, ChatIntent.REPORT_LOOKUP)

    assert result.intent == ChatIntent.REPORT_LOOKUP
    assert result.items == []
    assert provider.request is None


def test_rdb_evidence_service_uses_report_metadata_for_report_question() -> None:
    provider = StubRdbEvidenceProvider(intent=ChatIntent.REPORT_LOOKUP)
    service = RdbEvidenceService(
        Settings(rdb_evidence_enabled=True),
        providers=[provider],
    )
    request = _build_request("최근 보고서 목록 알려줘")

    result = anyio.run(service.get_evidence, request, ChatIntent.REPORT_LOOKUP)

    assert len(result.items) == 1
    assert provider.request == request
