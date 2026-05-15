from datetime import datetime

from app.features.chat.evidence_access_policy import EvidenceAccessPolicy
from app.features.chat.schemas import (
    ChatIntent,
    ChatUserContext,
    EvidenceItem,
    EvidenceResult,
)


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


def test_evidence_access_policy_filters_allowed_roles_and_ignores_company_name() -> None:
    policy = EvidenceAccessPolicy()
    user = ChatUserContext(
        userId=1,
        role="EXECUTIVE",
        companyName="S-MAP",
        status="ACTIVE",
    )
    evidence_result = EvidenceResult(
        intent=ChatIntent.DELIVERY_RISK,
        basisTime=datetime.fromisoformat("2026-05-12T10:30:00+09:00"),
        items=[
            EvidenceItem(
                type="ORDER",
                title="허용된 주문",
                summary="현재 사용자 범위에 포함됩니다.",
                source="ai_prediction_results",
                allowedRoles=["EXECUTIVE"],
                companyName="S-MAP",
            ),
            EvidenceItem(
                type="ORDER",
                title="다른 Role 주문",
                summary="사용자 역할 범위 밖입니다.",
                source="ai_prediction_results",
                allowedRoles=["MANUFACTURING_MANAGER"],
                companyName="S-MAP",
            ),
            EvidenceItem(
                type="ORDER",
                title="다른 회사 주문",
                summary="회사명은 로그용이므로 접근 제어 조건으로 쓰지 않습니다.",
                source="ai_prediction_results",
                allowedRoles=["EXECUTIVE"],
                companyName="OTHER",
            ),
        ],
    )

    result = policy.sanitize(user, evidence_result)

    assert [item.title for item in result.items] == ["허용된 주문", "다른 회사 주문"]
