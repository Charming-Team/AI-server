import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TextIO

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError, ChatServiceError
from app.features.chat.rdb_evidence_view_catalog import RDB_EVIDENCE_VIEW_DEFINITIONS
from app.features.chat.rdb_evidence_view_client import build_rdb_evidence_select_sql
from app.features.chat.schemas import ChatErrorCode, EvidenceLookupFilters

ConnectionFactory = Callable[[Settings], Awaitable[Any]]

VIEW_PRIVILEGE_SQL = """
select
    table_schema,
    table_name,
    has_table_privilege(
        current_user,
        format('%I.%I', table_schema, table_name),
        'SELECT'
    ) as can_select,
    has_table_privilege(
        current_user,
        format('%I.%I', table_schema, table_name),
        'INSERT'
    ) as can_insert,
    has_table_privilege(
        current_user,
        format('%I.%I', table_schema, table_name),
        'UPDATE'
    ) as can_update,
    has_table_privilege(
        current_user,
        format('%I.%I', table_schema, table_name),
        'DELETE'
    ) as can_delete,
    has_table_privilege(
        current_user,
        format('%I.%I', table_schema, table_name),
        'TRUNCATE'
    ) as can_truncate
from information_schema.views
where table_schema = $1
  and table_name = any($2::text[])
order by table_name
"""

PUBLIC_BASE_TABLE_PRIVILEGE_SQL = """
with public_tables as (
    select
        table_schema,
        table_name,
        format('%I.%I', table_schema, table_name) as qualified_name
    from information_schema.tables
    where table_schema = 'public'
      and table_type = 'BASE TABLE'
),
public_table_permissions as (
    select
        table_schema,
        table_name,
        has_table_privilege(current_user, qualified_name, 'SELECT') as can_select,
        has_table_privilege(current_user, qualified_name, 'INSERT') as can_insert,
        has_table_privilege(current_user, qualified_name, 'UPDATE') as can_update,
        has_table_privilege(current_user, qualified_name, 'DELETE') as can_delete,
        has_table_privilege(current_user, qualified_name, 'TRUNCATE') as can_truncate
    from public_tables
)
select
    table_schema,
    table_name,
    can_select,
    can_insert,
    can_update,
    can_delete,
    can_truncate
from public_table_permissions
where can_select
   or can_insert
   or can_update
   or can_delete
   or can_truncate
order by table_name
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PostgreSQL 챗봇 RDB Evidence View 연결과 read-only 권한을 점검합니다."
    )
    parser.add_argument(
        "--dsn",
        help="RDB Evidence PostgreSQL DSN. 생략하면 Settings의 RDB_EVIDENCE_DSN을 사용합니다.",
    )
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. CLI 인자가 있으면 해당 값이 우선합니다.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="PostgreSQL 네트워크 연결 없이 설정과 View catalog만 검증합니다.",
    )
    parser.add_argument(
        "--skip-privilege-check",
        action="store_true",
        help="View SELECT 점검만 수행하고 DB role 권한 점검은 건너뜁니다.",
    )
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    values: dict[str, Any] = {"rdb_evidence_enabled": True}
    if args.dsn:
        values["rdb_evidence_dsn"] = args.dsn

    if args.env_file:
        return Settings(_env_file=args.env_file, **values)
    return Settings(**values)


def build_validate_only_result(settings: Settings) -> dict[str, Any]:
    validate_rdb_evidence_settings(settings)
    return {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "dsnConfigured": bool(settings.rdb_evidence_dsn),
        "networkChecked": False,
        "privilegeChecked": False,
        "viewCount": len(RDB_EVIDENCE_VIEW_DEFINITIONS),
        "views": [
            _format_view_definition(definition)
            for definition in RDB_EVIDENCE_VIEW_DEFINITIONS
        ],
    }


async def check_rdb_evidence_views(
    settings: Settings,
    connection_factory: ConnectionFactory | None = None,
    check_privileges: bool = True,
) -> dict[str, Any]:
    validate_rdb_evidence_settings(settings)
    connection = await _connect(settings, connection_factory)
    try:
        current_context = await connection.fetchrow(
            "select current_user as current_user, current_database() as current_database",
            timeout=settings.rdb_evidence_timeout_seconds,
        )
        view_results = await _probe_evidence_views(connection, settings)
        privilege_result = (
            await _check_readonly_privileges(connection, settings)
            if check_privileges
            else {
                "views": [],
                "invalidViewPrivileges": [],
                "unexpectedPublicPrivileges": [],
            }
        )
    finally:
        await connection.close()

    invalid_view_privileges = privilege_result["invalidViewPrivileges"]
    unexpected_privileges = privilege_result["unexpectedPublicPrivileges"]
    if invalid_view_privileges:
        raise ChatExternalServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_EVIDENCE_004,
            message="RDB Evidence View 권한이 read-only 계약과 다릅니다.",
        )
    if unexpected_privileges:
        raise ChatExternalServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_EVIDENCE_004,
            message="RDB Evidence 계정에 public 원본 테이블 권한이 남아 있습니다.",
        )

    return {
        "checkStatus": "PASS",
        "mode": "NETWORK",
        "database": _mapping_get(current_context, "current_database"),
        "user": _mapping_get(current_context, "current_user"),
        "networkChecked": True,
        "privilegeChecked": check_privileges,
        "viewCount": len(view_results),
        "views": view_results,
        "viewPrivileges": privilege_result["views"],
        "invalidViewPrivileges": invalid_view_privileges,
        "unexpectedPublicPrivileges": unexpected_privileges,
    }


def validate_rdb_evidence_settings(settings: Settings) -> None:
    if not settings.rdb_evidence_enabled:
        raise ChatExternalServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_EVIDENCE_004,
            message="RDB Evidence 조회가 비활성화되어 있습니다.",
        )
    if not settings.rdb_evidence_dsn:
        raise ChatExternalServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_EVIDENCE_004,
            message="RDB Evidence DSN이 설정되지 않았습니다.",
        )


async def _connect(
    settings: Settings,
    connection_factory: ConnectionFactory | None,
) -> Any:
    if connection_factory is not None:
        return await connection_factory(settings)

    try:
        import asyncpg
    except ImportError as exc:
        raise ChatExternalServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_EVIDENCE_004,
            message="RDB Evidence 조회 드라이버가 설치되지 않았습니다.",
        ) from exc

    return await asyncpg.connect(
        dsn=settings.rdb_evidence_dsn,
        timeout=settings.rdb_evidence_timeout_seconds,
    )


async def _probe_evidence_views(connection: Any, settings: Settings) -> list[dict[str, Any]]:
    view_results: list[dict[str, Any]] = []
    filters = EvidenceLookupFilters(limit=1)
    for definition in RDB_EVIDENCE_VIEW_DEFINITIONS:
        sql, params = build_rdb_evidence_select_sql(definition, filters, 1)
        try:
            rows = await connection.fetch(
                sql,
                *params,
                timeout=settings.rdb_evidence_timeout_seconds,
            )
        except Exception as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_EVIDENCE_004,
                message=(
                    "RDB Evidence View 조회에 실패했습니다. "
                    f"view={definition.schema_name}.{definition.view_name}"
                ),
            ) from exc

        view_results.append(
            {
                **_format_view_definition(definition),
                "selectable": True,
                "sampleRowCount": len(rows),
            }
        )
    return view_results


async def _check_readonly_privileges(
    connection: Any,
    settings: Settings,
) -> dict[str, list[dict[str, Any]]]:
    schema_name = RDB_EVIDENCE_VIEW_DEFINITIONS[0].schema_name
    view_names = [definition.view_name for definition in RDB_EVIDENCE_VIEW_DEFINITIONS]
    view_privileges = await connection.fetch(
        VIEW_PRIVILEGE_SQL,
        schema_name,
        view_names,
        timeout=settings.rdb_evidence_timeout_seconds,
    )
    public_privileges = await connection.fetch(
        PUBLIC_BASE_TABLE_PRIVILEGE_SQL,
        timeout=settings.rdb_evidence_timeout_seconds,
    )

    return {
        "views": [_normalize_privilege_row(row) for row in view_privileges],
        "invalidViewPrivileges": [
            row
            for row in [_normalize_privilege_row(row) for row in view_privileges]
            if _has_invalid_view_privilege(row)
        ],
        "unexpectedPublicPrivileges": [
            _normalize_privilege_row(row) for row in public_privileges
        ],
    }


def _format_view_definition(definition) -> dict[str, str]:
    return {
        "intent": definition.intent.value,
        "schema": definition.schema_name,
        "view": definition.view_name,
        "sourceType": definition.source_type,
    }


def _normalize_privilege_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": _mapping_get(row, "table_schema"),
        "table": _mapping_get(row, "table_name"),
        "canSelect": bool(_mapping_get(row, "can_select")),
        "canInsert": bool(_mapping_get(row, "can_insert")),
        "canUpdate": bool(_mapping_get(row, "can_update")),
        "canDelete": bool(_mapping_get(row, "can_delete")),
        "canTruncate": bool(_mapping_get(row, "can_truncate")),
    }


def _has_invalid_view_privilege(row: Mapping[str, Any]) -> bool:
    return (
        not row["canSelect"]
        or row["canInsert"]
        or row["canUpdate"]
        or row["canDelete"]
        or row["canTruncate"]
    )


def _mapping_get(row: Mapping[str, Any] | None, key: str) -> Any:
    if row is None:
        return None
    return row[key]


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"mode={result['mode']}",
        f"dsnConfigured={result.get('dsnConfigured', True)}",
        f"networkChecked={result['networkChecked']}",
        f"privilegeChecked={result['privilegeChecked']}",
        f"viewCount={result['viewCount']}",
    ]
    if result["mode"] == "NETWORK":
        lines.extend(
            [
                f"database={result['database']}",
                f"user={result['user']}",
                "unexpectedPublicPrivilegeCount="
                f"{len(result['unexpectedPublicPrivileges'])}",
            ]
        )
    return "\n".join(lines)


def format_json_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


def main(
    argv: list[str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    args = build_parser().parse_args(argv)

    try:
        settings = build_settings(args)
        if args.validate_only:
            result = build_validate_only_result(settings)
        else:
            result = asyncio.run(
                check_rdb_evidence_views(
                    settings,
                    check_privileges=not args.skip_privilege_check,
                )
            )
    except ChatServiceError as exc:
        print(f"RDB Evidence View 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        return 1
    except Exception as exc:
        print(f"RDB Evidence View 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
