from app.features.report.repositories.report_query_repository import (
    ReportQueryRepository,
)
from app.features.report.schemas.response import (
    ReportDetailResponse,
    ReportListItemResponse,
    ReportListResponse,
)


class ReportQueryService:
    def __init__(self) -> None:
        self.report_query_repository = ReportQueryRepository()

    def get_reports(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        report_type: str | None = None,
    ) -> ReportListResponse:
        rows, total_count = self.report_query_repository.fetch_reports(
            limit=limit,
            offset=offset,
            report_type=report_type,
        )

        reports = [
            ReportListItemResponse(
                reportId=row["report_id"],
                reportType=row["report_type"],
                reportTitle=row["report_title"],
                authorId=row["author_id"],
                targetStartDate=row["target_start_date"],
                targetEndDate=row["target_end_date"],
                relatedSimulationId=row["related_simulation_id"],
                createdAt=row["created_at"],
                updatedAt=row["updated_at"],
            )
            for row in rows
        ]

        return ReportListResponse(
            reports=reports,
            totalCount=total_count,
        )

    def get_report_detail(
        self,
        report_id: int,
    ) -> ReportDetailResponse:
        row = self.report_query_repository.fetch_report_detail(report_id)

        if row is None:
            raise ValueError(f"보고서를 찾을 수 없습니다. report_id={report_id}")

        return ReportDetailResponse(
            reportId=row["report_id"],
            reportType=row["report_type"],
            reportTitle=row["report_title"],
            authorId=row["author_id"],
            targetStartDate=row["target_start_date"],
            targetEndDate=row["target_end_date"],
            includedItems=row["included_items"],
            reportContent=row["report_content"],
            reportEvidence=row["report_evidence"],
            relatedSimulationId=row["related_simulation_id"],
            createdAt=row["created_at"],
            updatedAt=row["updated_at"],
        )