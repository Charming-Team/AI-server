from app.features.chat.access_control import (
    ADMIN_ROLE,
    BUSINESS_ROLES,
    COMPANY_INFO_INDEXER_ROLES,
    EXECUTIVE_ROLE,
    MANUFACTURING_MANAGER_ROLE,
    OPERATOR_RESTRICTED_TERMS,
    OPERATOR_ROLE,
    ROLE_INTENT_MATRIX,
)
from app.features.chat.document_index_policy import DocumentIndexPolicy
from app.features.chat.recommendation_service import RecommendationService
from app.features.chat.role_access_policy import RoleAccessPolicy
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


def test_access_control_role_matrix_covers_every_business_role() -> None:
    assert set(ROLE_INTENT_MATRIX) == set(BUSINESS_ROLES)


def test_access_control_policy_classes_use_shared_role_sets() -> None:
    assert RoleAccessPolicy.allowed_business_roles == BUSINESS_ROLES
    assert DocumentIndexPolicy.allowed_roles == BUSINESS_ROLES
    assert DocumentIndexPolicy.company_info_indexer_roles == COMPANY_INFO_INDEXER_ROLES


def test_access_control_recommendation_rules_follow_role_intent_matrix() -> None:
    for rule in RecommendationService._rules:
        assert set(rule.allowed_roles) <= set(BUSINESS_ROLES)
        for role in rule.allowed_roles:
            assert rule.intent in ROLE_INTENT_MATRIX[role]


def test_access_control_operator_restricted_terms_cover_financial_terms() -> None:
    assert "계약 금액" in OPERATOR_RESTRICTED_TERMS
    assert "패널티" in OPERATOR_RESTRICTED_TERMS
    assert "cost" in OPERATOR_RESTRICTED_TERMS
