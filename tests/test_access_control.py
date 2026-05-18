from app.features.chat.access_control import (
    ADMIN_ROLE,
    BUSINESS_ROLES,
    COMPANY_INFO_INDEXER_ROLES,
    EXECUTIVE_ROLE,
    MANUFACTURING_MANAGER_ROLE,
    OPERATOR_ROLE,
    ROLE_INTENT_MATRIX,
)
from app.features.chat.schemas import ChatIntent


def test_access_control_business_roles_exclude_admin() -> None:
    assert BUSINESS_ROLES == {
        OPERATOR_ROLE,
        EXECUTIVE_ROLE,
        MANUFACTURING_MANAGER_ROLE,
    }
    assert ADMIN_ROLE not in BUSINESS_ROLES


def test_access_control_company_info_indexer_roles() -> None:
    assert COMPANY_INFO_INDEXER_ROLES == {
        ADMIN_ROLE,
        MANUFACTURING_MANAGER_ROLE,
    }


def test_access_control_executive_and_manager_can_access_report_lookup() -> None:
    assert ChatIntent.REPORT_LOOKUP in ROLE_INTENT_MATRIX[EXECUTIVE_ROLE]
    assert ChatIntent.REPORT_LOOKUP in ROLE_INTENT_MATRIX[MANUFACTURING_MANAGER_ROLE]
    assert ChatIntent.REPORT_LOOKUP not in ROLE_INTENT_MATRIX[OPERATOR_ROLE]
