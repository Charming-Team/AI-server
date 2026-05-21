import pytest

from app.features.chat.access_control import (
    EXECUTIVE_ROLE,
    MANUFACTURING_MANAGER_ROLE,
    OPERATOR_ROLE,
    ROLE_INTENT_MATRIX,
)
from app.features.chat.rdb_evidence_view_catalog import (
    RDB_EVIDENCE_VIEW_CATALOG,
    RDB_EVIDENCE_VIEW_DEFINITIONS,
    get_allowed_rdb_evidence_intents,
    get_rdb_evidence_view_definition,
)
from app.features.chat.schemas import ChatIntent


def test_rdb_evidence_view_catalog_covers_business_intents() -> None:
    expected_intents = {
        ChatIntent.DELIVERY_RISK,
        ChatIntent.MATERIAL_SHORTAGE,
        ChatIntent.PRODUCTION_PLAN,
        ChatIntent.URGENT_ORDER_IMPACT,
        ChatIntent.WORK_PRIORITY,
        ChatIntent.LINE_BOTTLENECK,
        ChatIntent.REPORT_LOOKUP,
    }

    assert set(RDB_EVIDENCE_VIEW_CATALOG) == expected_intents


def test_rdb_evidence_view_definitions_use_chat_read_only_views() -> None:
    for definition in RDB_EVIDENCE_VIEW_DEFINITIONS:
        assert definition.schema_name == "chat_evidence"
        assert definition.view_name.startswith("chat_")
        assert definition.view_name.endswith("_view")
        assert definition.source == definition.view_name
        assert definition.reference_id_column in definition.data_columns
        assert definition.title_columns
        assert definition.summary_columns
        assert definition.data_columns
        assert definition.allowed_roles


def test_rdb_evidence_view_definitions_follow_role_intent_matrix() -> None:
    for definition in RDB_EVIDENCE_VIEW_DEFINITIONS:
        for role in definition.allowed_roles:
            assert definition.intent in ROLE_INTENT_MATRIX[role]


def test_rdb_evidence_view_definitions_do_not_select_restricted_columns() -> None:
    for definition in RDB_EVIDENCE_VIEW_DEFINITIONS:
        selected_columns = (
            set(definition.title_columns)
            | set(definition.summary_columns)
            | set(definition.data_columns)
        )

        assert selected_columns.isdisjoint(definition.restricted_columns)


def test_rdb_evidence_view_definitions_define_selected_date_filter_columns() -> None:
    for definition in RDB_EVIDENCE_VIEW_DEFINITIONS:
        selected_columns = (
            set(definition.title_columns)
            | set(definition.summary_columns)
            | set(definition.data_columns)
        )

        assert definition.date_filter_columns
        assert set(definition.date_filter_columns).issubset(selected_columns)


@pytest.mark.parametrize(
    ("role", "expected_intents"),
    [
        (
            OPERATOR_ROLE,
            {
                ChatIntent.DELIVERY_RISK,
                ChatIntent.MATERIAL_SHORTAGE,
                ChatIntent.PRODUCTION_PLAN,
                ChatIntent.URGENT_ORDER_IMPACT,
                ChatIntent.WORK_PRIORITY,
                ChatIntent.LINE_BOTTLENECK,
                ChatIntent.REPORT_LOOKUP,
            },
        ),
        (
            EXECUTIVE_ROLE,
            {
                ChatIntent.DELIVERY_RISK,
                ChatIntent.MATERIAL_SHORTAGE,
                ChatIntent.PRODUCTION_PLAN,
                ChatIntent.URGENT_ORDER_IMPACT,
                ChatIntent.WORK_PRIORITY,
                ChatIntent.LINE_BOTTLENECK,
                ChatIntent.REPORT_LOOKUP,
            },
        ),
        (
            MANUFACTURING_MANAGER_ROLE,
            {
                ChatIntent.DELIVERY_RISK,
                ChatIntent.MATERIAL_SHORTAGE,
                ChatIntent.PRODUCTION_PLAN,
                ChatIntent.URGENT_ORDER_IMPACT,
                ChatIntent.WORK_PRIORITY,
                ChatIntent.LINE_BOTTLENECK,
                ChatIntent.REPORT_LOOKUP,
            },
        ),
    ],
)
def test_get_allowed_rdb_evidence_intents_returns_role_scoped_intents(
    role: str,
    expected_intents: set[ChatIntent],
) -> None:
    assert get_allowed_rdb_evidence_intents(role) == expected_intents


def test_get_rdb_evidence_view_definition_returns_none_for_unknown_intent() -> None:
    assert get_rdb_evidence_view_definition(ChatIntent.UNKNOWN) is None
