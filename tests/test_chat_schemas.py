from app.features.chat.schemas import ChatUserContext, EvidenceItem, EvidenceLookupUser


def test_chat_user_context_normalizes_role_and_status() -> None:
    user = ChatUserContext(
        userId=1,
        role=" operator ",
        companyName="S-MAP",
        status=" active ",
    )

    assert user.role == "OPERATOR"
    assert user.status == "ACTIVE"
    assert user.company_name == "S-MAP"


def test_evidence_lookup_user_normalizes_role() -> None:
    user = EvidenceLookupUser(
        userId=1,
        role=" manufacturing_manager ",
        companyName=" S-MAP ",
    )

    assert user.role == "MANUFACTURING_MANAGER"
    assert user.company_name == "S-MAP"


def test_evidence_item_normalizes_access_metadata() -> None:
    item = EvidenceItem(
        type="ORDER",
        title="ORD-202605-001 납기 위험",
        summary="납기 지연 위험 등급은 WARNING입니다.",
        source="ai_prediction_results",
        allowedRoles=[" operator ", "OPERATOR", "executive"],
    )

    assert item.allowed_roles == ["OPERATOR", "EXECUTIVE"]
