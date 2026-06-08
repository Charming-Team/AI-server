import json
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from app.features.business_report.repositories.postgres_business_report_repository import (
    PostgresBusinessReportRepository,
)
from app.features.business_report.schemas.request import BusinessReportGenerateRequest
from app.features.business_report.schemas.response import BusinessReportGenerateResponse
from app.features.business_report.schemas.source import BusinessReportSource
from app.features.business_report.transformers.llm_business_report_transformer import (
    LlmBusinessReportTransformer,
)


class BusinessReportGenerationService:
    def __init__(
        self,
        repository: PostgresBusinessReportRepository | None = None,
        llm_business_report_transformer: LlmBusinessReportTransformer | None = None,
    ) -> None:
        self.repository = repository or PostgresBusinessReportRepository()
        self.llm_business_report_transformer = (
            llm_business_report_transformer or LlmBusinessReportTransformer()
        )

    async def generate_business_report(
        self,
        request: BusinessReportGenerateRequest,
    ) -> BusinessReportGenerateResponse:
        source = self.repository.get_report_by_id(request.report_id)
        if source is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "BUSINESS_REPORT_404",
                    "message": f"보고서를 찾을 수 없습니다: report_id={request.report_id}",
                },
            )

        self._validate_source(source)

        try:
            transformed_text = self.llm_business_report_transformer.run(source)
        except RuntimeError as error:
            message = str(error)
            status_code = 503 if "설정" in message or "client" in message else 502
            error_code = (
                "BUSINESS_REPORT_LLM_001" if status_code == 503 else "BUSINESS_REPORT_LLM_002"
            )
            raise HTTPException(
                status_code=status_code,
                detail={
                    "code": error_code,
                    "message": message,
                },
            ) from error

        current_timestamp = datetime.now(UTC)
        transformed_payload = self._build_response_payload(
            source=source,
            llm_payload=self._parse_llm_json(transformed_text),
            current_timestamp=current_timestamp,
        )
        return BusinessReportGenerateResponse.model_validate(transformed_payload)

    def _validate_source(self, source: BusinessReportSource) -> None:
        if source.report_type not in {"ON_DEMAND", "MONTHLY"}:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "BUSINESS_REPORT_400",
                    "message": f"지원하지 않는 report_type입니다: {source.report_type}",
                },
            )
        if not isinstance(source.report_content, dict):
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "BUSINESS_REPORT_500",
                    "message": "report_content 형식이 올바르지 않습니다.",
                },
            )

    def _parse_llm_json(
        self,
        answer: str,
    ) -> dict[str, Any]:
        normalized_answer = answer.strip()
        if normalized_answer.startswith("```"):
            normalized_answer = self._strip_code_fence(normalized_answer)

        try:
            payload = json.loads(normalized_answer)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "BUSINESS_REPORT_LLM_002",
                    "message": "LLM 응답 JSON 파싱에 실패했습니다.",
                },
            ) from exc

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "BUSINESS_REPORT_LLM_002",
                    "message": "LLM 응답이 JSON 객체가 아닙니다.",
                },
            )
        return payload

    def _build_response_payload(
        self,
        *,
        source: BusinessReportSource,
        llm_payload: dict[str, Any],
        current_timestamp: datetime,
    ) -> dict[str, Any]:
        timestamp = current_timestamp.isoformat().replace("+00:00", "Z")
        report_content = llm_payload.get("report_content")

        if not isinstance(report_content, dict):
            report_content = source.report_content

        return {
            "report_id": source.report_id,
            "report_type": self._build_business_report_type(source.report_type),
            "report_title": self._build_business_report_title(source),
            "author_id": source.author_id,
            "target_start_date": source.target_start_date.isoformat(),
            "target_end_date": source.target_end_date.isoformat(),
            "report_content": report_content,
            "report_evidence": source.report_evidence,
            "related_simulation_id": source.related_simulation_id,
            "created_at": timestamp,
            "updated_at": timestamp,
        }

    def _build_business_report_type(
        self,
        source_report_type: str,
    ) -> str:
        report_type_map = {
            "ON_DEMAND": "ON_DEMAND_BUSINESS",
            "MONTHLY": "MONTHLY_BUSINESS",
        }

        return report_type_map[source_report_type]

    def _build_business_report_title(
        self,
        source: BusinessReportSource,
    ) -> str:
        if source.report_type == "MONTHLY":
            period = source.target_start_date.strftime("%Y년 %m월")
            return f"[{period}] 생산계획 월간 비즈니스 보고서"

        return (
            f"[{source.target_start_date.isoformat()} ~ "
            f"{source.target_end_date.isoformat()}] 생산계획 리스크 비즈니스 보고서"
        )

    def _strip_code_fence(self, answer: str) -> str:
        stripped = answer.strip()
        if stripped.startswith("```json"):
            stripped = stripped[len("```json") :]
        elif stripped.startswith("```"):
            stripped = stripped[len("```") :]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        return stripped.strip()
