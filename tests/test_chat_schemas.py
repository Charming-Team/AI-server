from app.features.chat.schemas import ChatUserContext, EvidenceLookupUser


def test_chat_user_context_normalizes_role_and_status() -> None:
    user = ChatUserContext(
        userId=1,
        role=" operator ",
        companyName="S-MAP",
        status=" active ",
    )

    assert user.role == "OPERATOR"
    assert user.status == "ACTIVE"


def test_evidence_lookup_user_normalizes_role() -> None:
    user = EvidenceLookupUser(
        userId=1,
        role=" manufacturing_manager ",
        companyName="S-MAP",
    )

    assert user.role == "MANUFACTURING_MANAGER"
