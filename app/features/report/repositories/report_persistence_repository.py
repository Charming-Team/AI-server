import json
from datetime import date
from typing import Any

from sqlalchemy import text

from app.core.database import engine


class ReportPersistenceRepository:
    def save_report(
        self,
        *,
        report_type: str,
        report_title: str,
        author_id: int | None,
        target_start_date: date,
        target_end_date: date,
        markdown: str,
        sections: dict[str, Any],
        evidence: list[dict[str, Any]],
        related_simulation_id: int | None = None,
    ) -> int:
        query = text(
            """
            INSERT INTO reports (
                report_type,
                report_title,
                author_id,
                target_start_date,
                target_end_date,
                included_items,
                report_content,
                report_evidence,
                related_simulation_id
            )
            VALUES (
                CAST(:report_type AS report_type_enum),
                :report_title,
                :author_id,
                :target_start_date,
                :target_end_date,
                CAST(:included_items AS jsonb),
                :report_content,
                CAST(:report_evidence AS jsonb),
                :related_simulation_id
            )
            RETURNING report_id
            """
        )

        params = {
            "report_type": report_type,
            "report_title": report_title,
            "author_id": author_id,
            "target_start_date": target_start_date,
            "target_end_date": target_end_date,
            "included_items": json.dumps(sections, ensure_ascii=False),
            "report_content": markdown,
            "report_evidence": json.dumps(evidence, ensure_ascii=False),
            "related_simulation_id": related_simulation_id,
        }

        with engine.begin() as conn:
            report_id = conn.execute(query, params).scalar_one()

        return int(report_id)

    def mark_job_success(
        self,
        *,
        job_id: int,
        report_id: int,
    ) -> None:
        query = text(
            """
            UPDATE report_jobs
            SET
                report_id = :report_id,
                job_status = CAST('SUCCESS' AS job_status_enum),
                finished_at = NOW(),
                updated_at = NOW(),
                error_message = NULL
            WHERE job_id = :job_id
            """
        )

        with engine.begin() as conn:
            result = conn.execute(
                query,
                {
                    "job_id": job_id,
                    "report_id": report_id,
                },
            )

            if result.rowcount == 0:
                self._insert_success_job(
                    conn=conn,
                    job_id=job_id,
                    report_id=report_id,
                )

    def mark_job_failed(
        self,
        *,
        job_id: int,
        requested_by: int,
        request_payload: dict[str, Any],
        error_message: str,
    ) -> None:
        update_query = text(
            """
            UPDATE report_jobs
            SET
                job_status = CAST('FAILED' AS job_status_enum),
                error_message = :error_message,
                finished_at = NOW(),
                updated_at = NOW()
            WHERE job_id = :job_id
            """
        )

        with engine.begin() as conn:
            result = conn.execute(
                update_query,
                {
                    "job_id": job_id,
                    "error_message": error_message,
                },
            )

            if result.rowcount == 0:
                insert_query = text(
                    """
                    INSERT INTO report_jobs (
                        job_id,
                        requested_by,
                        job_status,
                        request_payload,
                        error_message,
                        retry_count,
                        started_at,
                        finished_at
                    )
                    VALUES (
                        :job_id,
                        :requested_by,
                        CAST('FAILED' AS job_status_enum),
                        CAST(:request_payload AS jsonb),
                        :error_message,
                        0,
                        NOW(),
                        NOW()
                    )
                    """
                )

                conn.execute(
                    insert_query,
                    {
                        "job_id": job_id,
                        "requested_by": requested_by,
                        "request_payload": json.dumps(
                            request_payload,
                            ensure_ascii=False,
                        ),
                        "error_message": error_message,
                    },
                )

    def _insert_success_job(
        self,
        *,
        conn,
        job_id: int,
        report_id: int,
    ) -> None:
        report_query = text(
            """
            SELECT
                author_id,
                report_type,
                target_start_date,
                target_end_date
            FROM reports
            WHERE report_id = :report_id
            """
        )

        report = conn.execute(
            report_query,
            {"report_id": report_id},
        ).mappings().first()

        if report is None:
            return

        insert_query = text(
            """
            INSERT INTO report_jobs (
                job_id,
                report_id,
                requested_by,
                job_status,
                request_payload,
                retry_count,
                started_at,
                finished_at
            )
            VALUES (
                :job_id,
                :report_id,
                :requested_by,
                CAST('SUCCESS' AS job_status_enum),
                CAST(:request_payload AS jsonb),
                0,
                NOW(),
                NOW()
            )
            """
        )

        request_payload = {
            "reportType": str(report["report_type"]),
            "period": {
                "startDate": report["target_start_date"].isoformat(),
                "endDate": report["target_end_date"].isoformat(),
            },
        }

        conn.execute(
            insert_query,
            {
                "job_id": job_id,
                "report_id": report_id,
                "requested_by": report["author_id"],
                "request_payload": json.dumps(request_payload, ensure_ascii=False),
            },
        )