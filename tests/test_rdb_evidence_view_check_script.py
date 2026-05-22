from argparse import Namespace
from io import StringIO
from typing import Any

import anyio
import pytest

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.rdb_evidence_view_catalog import RDB_EVIDENCE_VIEW_DEFINITIONS
from scripts import check_rdb_evidence_views


def _build_args(**overrides: Any) -> Namespace:
    values = {
        "dsn": "postgresql://reader:secret@postgres.local:5432/smap",
        "env_file": None,
        "json": False,
        "validate_only": False,
        "skip_privilege_check": False,
    }
    values.update(overrides)
    return Namespace(**values)


class FakeConnection:
    def __init__(
        self,
        public_privilege_rows: list[dict[str, Any]] | None = None,
        fail_view_name: str | None = None,
        invalid_view_privilege: bool = False,
    ) -> None:
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.closed = False
        self.public_privilege_rows = public_privilege_rows or []
        self.fail_view_name = fail_view_name
        self.invalid_view_privilege = invalid_view_privilege

    async def fetchrow(self, sql: str, *args: Any, **kwargs: Any) -> dict[str, str]:
        self.fetch_calls.append((sql, args))
        return {
            "current_user": "smap_chat_reader",
            "current_database": "smap",
        }

    async def fetch(self, sql: str, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((sql, args))
        if self.fail_view_name and self.fail_view_name in sql:
            raise RuntimeError("view lookup failed")
        if "information_schema.views" in sql:
            return [
                {
                    "table_schema": definition.schema_name,
                    "table_name": definition.view_name,
                    "can_select": not self.invalid_view_privilege,
                    "can_insert": self.invalid_view_privilege,
                    "can_update": False,
                    "can_delete": False,
                    "can_truncate": False,
                }
                for definition in RDB_EVIDENCE_VIEW_DEFINITIONS
            ]
        if "public_table_permissions" in sql:
            return self.public_privilege_rows
        return [{"probe": 1}]

    async def close(self) -> None:
        self.closed = True


def test_check_rdb_evidence_views_script_builds_settings_from_cli() -> None:
    settings = check_rdb_evidence_views.build_settings(
        _build_args(dsn="postgresql://reader:secret@localhost:15432/smap")
    )

    assert settings.rdb_evidence_enabled is True
    assert settings.rdb_evidence_dsn == "postgresql://reader:secret@localhost:15432/smap"


def test_check_rdb_evidence_views_script_validate_only_result() -> None:
    settings = Settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
    )

    result = check_rdb_evidence_views.build_validate_only_result(settings)

    assert result["checkStatus"] == "VALIDATED"
    assert result["dsnConfigured"] is True
    assert result["networkChecked"] is False
    assert result["privilegeChecked"] is False
    assert result["viewCount"] == len(RDB_EVIDENCE_VIEW_DEFINITIONS)
    assert result["views"][0]["schema"] == "chat_evidence"


def test_check_rdb_evidence_views_script_probes_catalog_views_and_privileges() -> None:
    connection = FakeConnection()

    async def connect(settings: Settings) -> FakeConnection:
        assert settings.rdb_evidence_dsn is not None
        return connection

    async def run() -> dict[str, Any]:
        return await check_rdb_evidence_views.check_rdb_evidence_views(
            Settings(
                rdb_evidence_enabled=True,
                rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
            ),
            connection_factory=connect,
        )

    result = anyio.run(run)

    assert result["checkStatus"] == "PASS"
    assert result["database"] == "smap"
    assert result["user"] == "smap_chat_reader"
    assert result["viewCount"] == len(RDB_EVIDENCE_VIEW_DEFINITIONS)
    assert result["privilegeChecked"] is True
    assert result["invalidViewPrivileges"] == []
    assert result["unexpectedPublicPrivileges"] == []
    assert connection.closed is True
    assert any(
        '"chat_evidence"."chat_material_shortage_evidence_view"' in call[0]
        for call in connection.fetch_calls
    )


def test_check_rdb_evidence_views_script_fails_on_public_table_privilege() -> None:
    connection = FakeConnection(
        public_privilege_rows=[
            {
                "table_schema": "public",
                "table_name": "customer_orders",
                "can_select": True,
                "can_insert": False,
                "can_update": False,
                "can_delete": False,
                "can_truncate": False,
            }
        ]
    )

    async def connect(settings: Settings) -> FakeConnection:
        return connection

    async def run() -> None:
        await check_rdb_evidence_views.check_rdb_evidence_views(
            Settings(
                rdb_evidence_enabled=True,
                rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
            ),
            connection_factory=connect,
        )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_004"
    assert "public 원본 테이블 권한" in exc_info.value.message
    assert connection.closed is True


def test_check_rdb_evidence_views_script_fails_on_invalid_view_privilege() -> None:
    connection = FakeConnection(invalid_view_privilege=True)

    async def connect(settings: Settings) -> FakeConnection:
        return connection

    async def run() -> None:
        await check_rdb_evidence_views.check_rdb_evidence_views(
            Settings(
                rdb_evidence_enabled=True,
                rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
            ),
            connection_factory=connect,
        )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_004"
    assert "View 권한" in exc_info.value.message
    assert connection.closed is True


def test_check_rdb_evidence_views_script_fails_when_view_probe_fails() -> None:
    connection = FakeConnection(fail_view_name="chat_delivery_risk_evidence_view")

    async def connect(settings: Settings) -> FakeConnection:
        return connection

    async def run() -> None:
        await check_rdb_evidence_views.check_rdb_evidence_views(
            Settings(
                rdb_evidence_enabled=True,
                rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
            ),
            connection_factory=connect,
        )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_004"
    assert "chat_evidence.chat_delivery_risk_evidence_view" in exc_info.value.message
    assert connection.closed is True


def test_check_rdb_evidence_views_script_main_validate_only_does_not_expose_dsn() -> None:
    stdout = StringIO()

    exit_code = check_rdb_evidence_views.main(
        [
            "--dsn",
            "postgresql://reader:secret@postgres.local:5432/smap",
            "--validate-only",
        ],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "status=VALIDATED" in output
    assert "dsnConfigured=True" in output
    assert "secret" not in output


def test_check_rdb_evidence_views_script_main_returns_one_without_dsn() -> None:
    stderr = StringIO()

    exit_code = check_rdb_evidence_views.main(
        ["--validate-only"],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "RDB Evidence View 점검 실패" in stderr.getvalue()
    assert "code=CHAT_EVIDENCE_004" in stderr.getvalue()
