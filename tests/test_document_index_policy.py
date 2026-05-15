import pytest

from app.features.chat.document_index_policy import DocumentIndexPolicy
from app.features.chat.document_payload import InternalDocumentInput
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode


def _build_document(
    *,
    document_type: str = "REPORT",
    allowed_roles: list[str] | None = None,
    company_name: str | None = "S-MAP",
    intent_tags: list[str] | None = None,
) -> InternalDocumentInput:
    return InternalDocumentInput(
        documentId="report-202605",
        documentType=document_type,
        title="2026년 5월 생산 리스크 보고서",
        content="자재 부족과 라인 병목이 주요 리스크입니다.",
        allowedRoles=(
            ["EXECUTIVE", "MANUFACTURING_MANAGER"]
            if allowed_roles is None
            else allowed_roles
        ),
        companyName=company_name,
        intentTags=["REPORT_LOOKUP"] if intent_tags is None else intent_tags,
    )


def test_document_index_policy_allows_report_and_company_info_documents() -> None:
    policy = DocumentIndexPolicy()

    policy.validate(_build_document(document_type="REPORT"))
    policy.validate(_build_document(document_type="COMPANY_INFO"))


def test_document_index_policy_rejects_unsupported_document_type() -> None:
    policy = DocumentIndexPolicy()

    with pytest.raises(ChatServiceError) as exc_info:
        policy.validate(_build_document(document_type="PROCESS"))

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == ChatErrorCode.CHAT_DOCUMENT_001
    assert exc_info.value.message == "문서 유형은 REPORT 또는 COMPANY_INFO만 허용됩니다."


def test_document_index_policy_requires_company_name() -> None:
    policy = DocumentIndexPolicy()

    with pytest.raises(ChatServiceError) as exc_info:
        policy.validate(_build_document(company_name=None))

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == ChatErrorCode.CHAT_DOCUMENT_002
    assert exc_info.value.message == "문서 인덱싱에는 회사명이 필요합니다."


def test_document_index_policy_rejects_empty_roles() -> None:
    policy = DocumentIndexPolicy()

    with pytest.raises(ChatServiceError) as exc_info:
        policy.validate(_build_document(allowed_roles=[]))

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == ChatErrorCode.CHAT_DOCUMENT_002
    assert exc_info.value.message == "문서 접근 가능 역할이 필요합니다."


def test_document_index_policy_rejects_admin_role() -> None:
    policy = DocumentIndexPolicy()

    with pytest.raises(ChatServiceError) as exc_info:
        policy.validate(_build_document(allowed_roles=["ADMIN"]))

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == ChatErrorCode.CHAT_DOCUMENT_002
    assert exc_info.value.message == "문서 접근 가능 역할이 올바르지 않습니다."


def test_document_index_policy_rejects_unknown_intent_tag() -> None:
    policy = DocumentIndexPolicy()

    with pytest.raises(ChatServiceError) as exc_info:
        policy.validate(_build_document(intent_tags=["UNKNOWN"]))

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == ChatErrorCode.CHAT_DOCUMENT_003
    assert exc_info.value.message == "문서 의도 태그가 올바르지 않습니다."
