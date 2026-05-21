import sys
from datetime import date
from types import SimpleNamespace

import anyio
import pytest

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.rdb_evidence_view_catalog import get_rdb_evidence_view_definition
from app.features.chat.rdb_evidence_view_client import (
    AsyncpgRdbEvidenceViewClient,
    build_rdb_evidence_select_sql,
)
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

    assert 'from "chat_evidence"."chat_material_shortage_evidence_view"' in sql
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

    assert 'from "chat_evidence"."chat_line_bottleneck_evidence_view"' in sql
    assert "where (" not in sql
    assert "order by" in sql
    assert "limit $1" in sql
    assert params == [3]


def test_build_rdb_evidence_select_sql_uses_date_filters() -> None:
    definition = get_rdb_evidence_view_definition(ChatIntent.PRODUCTION_PLAN)
    assert definition is not None

    sql, params = build_rdb_evidence_select_sql(
        definition,
        EvidenceLookupFilters(
            limit=5,
            fromDate="2026-05-12",
            toDate="2026-05-18",
            targetType="LINE",
            targetCode="LINE-A01",
        ),
        5,
    )

    assert 'from "chat_evidence"."chat_production_plan_evidence_view"' in sql
    assert 'upper("line_code"::text) = $1' in sql
    assert '"planned_start_at"::date >= $2::date' in sql
    assert '"planned_start_at"::date <= $3::date' in sql
    assert "where (" in sql
    assert ") and (" in sql
    assert "limit $4" in sql
    assert params == ["LINE-A01", date(2026, 5, 12), date(2026, 5, 18), 5]


def test_build_rdb_evidence_select_sql_ignores_dates_without_catalog_date_columns() -> None:
    definition = get_rdb_evidence_view_definition(ChatIntent.REPORT_LOOKUP)
    assert definition is not None
    definition = definition.__class__(
        **{
            **definition.__dict__,
            "date_filter_columns": (),
        }
    )

    sql, params = build_rdb_evidence_select_sql(
        definition,
        EvidenceLookupFilters(
            limit=2,
            fromDate="2026-05-12",
            toDate="2026-05-18",
        ),
        2,
    )

    assert "::date" not in sql
    assert "where" not in sql
    assert "limit $1" in sql
    assert params == [2]


def test_asyncpg_rdb_evidence_view_client_wraps_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = get_rdb_evidence_view_definition(ChatIntent.MATERIAL_SHORTAGE)
    assert definition is not None

    async def fail_connect(**kwargs):
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(connect=fail_connect))
    client = AsyncpgRdbEvidenceViewClient(
        Settings(rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap")
    )

    async def run() -> None:
        await client.fetch_rows(definition, EvidenceLookupFilters(limit=5), 5)

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code.value == "CHAT_EVIDENCE_004"
    assert exc_info.value.message == "RDB Evidence View 조회에 실패했습니다."


def test_asyncpg_rdb_evidence_view_client_closes_connection_after_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = get_rdb_evidence_view_definition(ChatIntent.MATERIAL_SHORTAGE)
    assert definition is not None
    closed = False

    class FakeConnection:
        async def fetch(self, sql, *params, timeout):
            raise RuntimeError("query failed")

        async def close(self):
            nonlocal closed
            closed = True

    async def connect(**kwargs):
        return FakeConnection()

    monkeypatch.setitem(sys.modules, "asyncpg", SimpleNamespace(connect=connect))
    client = AsyncpgRdbEvidenceViewClient(
        Settings(rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap")
    )

    async def run() -> None:
        await client.fetch_rows(definition, EvidenceLookupFilters(limit=5), 5)

    with pytest.raises(ChatExternalServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_004"
    assert closed is True
