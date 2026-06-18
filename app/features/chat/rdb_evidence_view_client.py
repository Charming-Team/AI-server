import re
from collections.abc import Mapping
from datetime import date
from typing import Any, Protocol

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.rdb_evidence_view_catalog import RdbEvidenceViewDefinition
from app.features.chat.schemas import ChatErrorCode, EvidenceLookupFilters


class RdbEvidenceViewClient(Protocol):
    async def fetch_rows(
        self,
        definition: RdbEvidenceViewDefinition,
        filters: EvidenceLookupFilters,
        limit: int,
    ) -> list[Mapping[str, Any]]:
        """사전에 정의된 read-only Evidence View에서 행을 조회한다."""


class AsyncpgRdbEvidenceViewClient:
    _identifier_pattern = re.compile(r"^[a-z][a-z0-9_]*$")

    def __init__(self, settings: Settings) -> None:
        self.dsn = settings.rdb_evidence_dsn
        self.timeout_seconds = settings.rdb_evidence_timeout_seconds

    async def fetch_rows(
        self,
        definition: RdbEvidenceViewDefinition,
        filters: EvidenceLookupFilters,
        limit: int,
    ) -> list[Mapping[str, Any]]:
        if not self.dsn:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_EVIDENCE_004,
                message="RDB Evidence DSN이 설정되지 않았습니다.",
            )

        try:
            import asyncpg
        except ImportError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_EVIDENCE_004,
                message="RDB Evidence 조회 드라이버가 설치되지 않았습니다.",
            ) from exc

        sql, params = build_rdb_evidence_select_sql(definition, filters, limit)
        connection = None
        try:
            connection = await asyncpg.connect(
                dsn=self.dsn,
                timeout=self.timeout_seconds,
            )
            rows = await connection.fetch(sql, *params, timeout=self.timeout_seconds)
            return [dict(row) for row in rows]
        except Exception as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_EVIDENCE_004,
                message="RDB Evidence View 조회에 실패했습니다.",
            ) from exc
        finally:
            if connection is not None:
                try:
                    await connection.close()
                except Exception:
                    pass


def build_rdb_evidence_select_sql(
    definition: RdbEvidenceViewDefinition,
    filters: EvidenceLookupFilters,
    limit: int,
) -> tuple[str, list[Any]]:
    columns = _selected_columns(definition)
    quoted_columns = ", ".join(_quote_identifier(column) for column in columns)
    sql_parts = [
        f"select {quoted_columns}",
        (
            f"from {_quote_identifier(definition.schema_name)}."
            f"{_quote_identifier(definition.view_name)}"
        ),
    ]
    params: list[Any] = []
    where_clauses: list[str] = []

    if filters.target_code and definition.target_code_columns:
        params.append(filters.target_code.upper())
        target_conditions = [
            f"upper({_quote_identifier(column)}::text) = ${len(params)}"
            for column in definition.target_code_columns
        ]
        where_clauses.append(f"({' or '.join(target_conditions)})")

    if filters.from_date and definition.date_filter_columns:
        params.append(date.fromisoformat(filters.from_date))
        from_filter_columns = (
            definition.date_filter_end_columns or definition.date_filter_columns
        )
        from_conditions = [
            f"{_quote_identifier(column)}::date >= ${len(params)}::date"
            for column in from_filter_columns
        ]
        where_clauses.append(f"({' or '.join(from_conditions)})")

    if filters.to_date and definition.date_filter_columns:
        params.append(date.fromisoformat(filters.to_date))
        to_conditions = [
            f"{_quote_identifier(column)}::date <= ${len(params)}::date"
            for column in definition.date_filter_columns
        ]
        where_clauses.append(f"({' or '.join(to_conditions)})")

    if where_clauses:
        sql_parts.append(f"where {' and '.join(where_clauses)}")

    if definition.default_order_columns:
        order_clause = ", ".join(
            f"{_quote_identifier(column)} desc nulls last"
            for column in definition.default_order_columns
        )
        sql_parts.append(f"order by {order_clause}")

    params.append(max(1, limit))
    sql_parts.append(f"limit ${len(params)}")
    return "\n".join(sql_parts), params


def _selected_columns(definition: RdbEvidenceViewDefinition) -> tuple[str, ...]:
    ordered_columns = (
        definition.reference_id_column,
        *definition.title_columns,
        *definition.summary_columns,
        *definition.data_columns,
    )
    selected_columns: list[str] = []
    seen_columns: set[str] = set()
    for column in ordered_columns:
        if column in seen_columns:
            continue
        seen_columns.add(column)
        selected_columns.append(column)
    return tuple(selected_columns)


def _quote_identifier(identifier: str) -> str:
    if not AsyncpgRdbEvidenceViewClient._identifier_pattern.fullmatch(identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier}")
    return f'"{identifier}"'
