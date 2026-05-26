from app.features.report.repositories.postgres_report_repository import (
    PostgresReportRepository,
)
from app.features.report.schemas.state import ReportAgentState


class RdbDataCollectionAgent:
    def __init__(self) -> None:
        self.repository = PostgresReportRepository()

    def run(self, state: ReportAgentState) -> ReportAgentState:
        start_date = state.request.period.start_date
        end_date = state.request.period.end_date

        raw_data = self.repository.fetch_report_source_data(
            start_date=start_date,
            end_date=end_date,
        )

        state.raw_data = raw_data
        return state