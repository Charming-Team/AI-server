from app.features.chat.rdb_evidence_view_catalog import get_rdb_evidence_view_definition
from app.features.chat.rdb_evidence_view_client import build_rdb_evidence_select_sql
from app.features.chat.schemas import ChatIntent, EvidenceLookupFilters


def test_build_rdb_evidence_select_sql_uses_catalog_view_and_columns() -> None:
    definition = get_rdb_evidence_view_definition(ChatIntent.MATERIAL_SHORTAGE)
    assert definition is not None

    sql, params = build_rdb_evidence_select_sql(
        definition,
        EvidenceLookupFilters(
            limit=5,
            targetType="MATERIAL",
            targetCode="RM-AL-001",
        ),
        5,
    )

    assert 'from "chat_material_shortage_evidence_view"' in sql
    assert '"plan_material_id"' in sql
    assert '"material_code"' in sql
    assert '"shortage_quantity"' in sql
    assert "production_plan_materials" not in sql
    assert "where (" in sql
    assert 'upper("material_code"::text) = $1' in sql
    assert "limit $2" in sql
    assert params == ["RM-AL-001", 5]


def test_build_rdb_evidence_select_sql_skips_target_filter_without_target_code() -> None:
    definition = get_rdb_evidence_view_definition(ChatIntent.LINE_BOTTLENECK)
    assert definition is not None

    sql, params = build_rdb_evidence_select_sql(
        definition,
        EvidenceLookupFilters(limit=3),
        3,
    )

    assert 'from "chat_line_bottleneck_evidence_view"' in sql
    assert "where (" not in sql
    assert "order by" in sql
    assert "limit $1" in sql
    assert params == [3]
