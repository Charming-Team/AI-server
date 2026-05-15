from datetime import datetime

from app.features.chat.evidence_access_policy import EvidenceAccessPolicy
from app.features.chat.schemas import ChatIntent, EvidenceItem, EvidenceResult


def _build_evidence_result() -> EvidenceResult:
    return EvidenceResult(
        intent=ChatIntent.DELIVERY_RISK,
        basisTime=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
        items=[
            EvidenceItem(
                type="ORDER",
                title="ORD-202605-001 납기 위험",
                summary="납기 지연 위험 등급은 WARNING입니다.",
                source="ai_prediction_results",
                data={
                    "riskLevel": "WARNING",
                    "contractAmount": 12000000,
                    "latePenaltyAmount": 500000,
                    "recommendedAction": "생산 순서 조정",
                    "nested": {
                        "costChangeAmount": 300000,
                        "lineCode": "LINE-A01",
                    },
                    "notes": [
                        "운영 확인 필요",
                        "계약 금액 확인 필요",
                    ],
                },
            ),
            EvidenceItem(
                type="ORDER",
                title="계약 금액 영향",
                summary="납기 지연 시 패널티 영향이 있습니다.",
                source="customer_orders",
                data={"orderNo": "ORD-202605-001"},
            ),
        ],
    )


def test_operator_evidence_removes_financial_items_and_data() -> None:
    policy = EvidenceAccessPolicy()

    result = policy.sanitize("OPERATOR", _build_evidence_result())

    assert len(result.items) == 1
    item_data = result.items[0].data
    assert item_data["riskLevel"] == "WARNING"
    assert item_data["recommendedAction"] == "생산 순서 조정"
    assert item_data["nested"] == {"lineCode": "LINE-A01"}
    assert item_data["notes"] == ["운영 확인 필요", "[권한 제한]"]
    assert "contractAmount" not in item_data
    assert "latePenaltyAmount" not in item_data


def test_executive_evidence_keeps_financial_data() -> None:
    policy = EvidenceAccessPolicy()

    result = policy.sanitize("EXECUTIVE", _build_evidence_result())

    assert len(result.items) == 2
    item_data = result.items[0].data
    assert item_data["contractAmount"] == 12000000
    assert item_data["latePenaltyAmount"] == 500000
    assert item_data["nested"]["costChangeAmount"] == 300000
