from datetime import datetime

import anyio

from app.core.config import Settings
from app.schemas.chat import ChatAnswerRequest, ChatIntent, ChatUserContext
from app.services.evidence_service import EvidenceService


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
    request = _build_request()

    payload = service._build_payload(request, ChatIntent.MATERIAL_SHORTAGE)

    assert service._headers == {"X-Internal-Token": "internal-token"}
    assert payload["sessionId"] == 10
    assert payload["messageId"] == 24
    assert payload["intent"] == "MATERIAL_SHORTAGE"
    assert payload["user"]["userId"] == 1
    assert payload["user"]["role"] == "MANUFACTURING_MANAGER"
    assert payload["filters"]["limit"] == 5
