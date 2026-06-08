from sqlalchemy import text

from app.core.database import engine
from app.features.business_report.schemas.source import BusinessReportSource


class PostgresBusinessReportRepository:
    def get_report_by_id(
        self,
        report_id: int,
    ) -> BusinessReportSource | None:
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
            FROM public.reports
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
        return BusinessReportSource.model_validate(dict(row))
