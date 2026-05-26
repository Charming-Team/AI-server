from typing import Any

from sqlalchemy import text

from app.core.database import engine


class ReportQueryRepository:
    def fetch_reports(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        report_type: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        where_conditions = []
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }

        if report_type:
            where_conditions.append("report_type = CAST(:report_type AS report_type_enum)")
            params["report_type"] = report_type

        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)

        list_query = text(
            f"""
            SELECT
                report_id,
                report_type::TEXT AS report_type,
                report_title,
                author_id,
                target_start_date,
                target_end_date,
                related_simulation_id,
                created_at,
                updated_at
            FROM reports
            {where_clause}
            ORDER BY created_at DESC, report_id DESC
            LIMIT :limit
            OFFSET :offset
            """
        )

        count_query = text(
            f"""
            SELECT COUNT(*) AS total_count
            FROM reports
            {where_clause}
            """
        )

        with engine.connect() as conn:
            rows = conn.execute(list_query, params).mappings().all()
            total_count = conn.execute(count_query, params).scalar_one()

        return [dict(row) for row in rows], int(total_count)

    def fetch_report_detail(
        self,
        report_id: int,
    ) -> dict[str, Any] | None:
        query = text(
            """
            SELECT
                report_id,
                report_type::TEXT AS report_type,
                report_title,
                author_id,
                target_start_date,
                target_end_date,
                included_items,
                report_content,
                report_evidence,
                related_simulation_id,
                created_at,
                updated_at
            FROM reports
            WHERE report_id = :report_id
            """
        )

        with engine.connect() as conn:
            row = conn.execute(
                query,
                {"report_id": report_id},
            ).mappings().first()

        if row is None:
            return None

        return dict(row)