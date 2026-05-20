from pathlib import Path

from app.features.chat.rdb_evidence_view_catalog import RDB_EVIDENCE_VIEW_DEFINITIONS

POSTGRES_SCRIPT_DIR = Path("scripts/postgres")
CREATE_VIEWS_SQL = POSTGRES_SCRIPT_DIR / "001_create_chat_evidence_views.sql"
GRANT_READONLY_SQL = POSTGRES_SCRIPT_DIR / "002_grant_chat_evidence_readonly.sql"
VERIFY_READONLY_SQL = POSTGRES_SCRIPT_DIR / "003_verify_chat_evidence_readonly.sql"


def _read_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8").lower()


def test_chat_evidence_view_script_matches_rdb_catalog() -> None:
    sql = _read_sql(CREATE_VIEWS_SQL)

    assert "create schema if not exists chat_evidence" in sql
    for definition in RDB_EVIDENCE_VIEW_DEFINITIONS:
        view_name = f"{definition.schema_name}.{definition.view_name}".lower()
        assert f"create or replace view {view_name}" in sql


def test_chat_evidence_views_do_not_expose_sensitive_or_write_only_columns() -> None:
    sql = _read_sql(CREATE_VIEWS_SQL)
    blocked_terms = {
        "password",
        "refresh_tokens",
        " token",
        "contract_amount",
        "late_penalty_amount",
        "cost_change_amount",
    }

    for blocked_term in blocked_terms:
        assert blocked_term not in sql


def test_chat_evidence_views_do_not_use_select_star() -> None:
    sql = _read_sql(CREATE_VIEWS_SQL)

    assert "select *" not in sql


def test_readonly_role_script_grants_only_view_read_privileges() -> None:
    sql = _read_sql(GRANT_READONLY_SQL)

    assert "create role smap_chat_reader login" in sql
    assert "grant usage on schema chat_evidence to smap_chat_reader" in sql
    assert "grant select on all tables in schema chat_evidence to smap_chat_reader" in sql
    assert "revoke all privileges on all tables in schema public from smap_chat_reader" in sql
    assert (
        "revoke all privileges on all tables in schema chat_evidence "
        "from smap_chat_reader"
    ) in sql
    assert "default_transaction_read_only = on" in sql
    assert "password must be managed by external secret" in sql

    assert "grant insert" not in sql
    assert "grant update" not in sql
    assert "grant delete" not in sql
    assert "grant truncate" not in sql


def test_readonly_verification_script_checks_select_and_write_privileges() -> None:
    sql = _read_sql(VERIFY_READONLY_SQL)

    assert "has_table_privilege" in sql
    assert "chat_evidence_view_privilege" in sql
    assert "public_base_table_unexpected_privilege" in sql
    assert "'select'" in sql
    assert "'insert'" in sql
    assert "'update'" in sql
    assert "'delete'" in sql
    assert "'truncate'" in sql
    assert "information_schema.role_table_grants" in sql
