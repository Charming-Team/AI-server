import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException

from app.features.business_report.repositories.postgres_business_report_repository import (
    PostgresBusinessReportRepository,
)
from app.features.business_report.schemas.request import (
    BusinessReportGenerateRequest,
    BusinessReportSourceRequest,
)
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
        source = self._build_source_from_request(request.source_report)
        if source is None:
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
            transformed_text = await asyncio.to_thread(
                self.llm_business_report_transformer.run,
                source,
            )
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

    def _build_source_from_request(
        self,
        source_report: BusinessReportSourceRequest | None,
    ) -> BusinessReportSource | None:
        if source_report is None:
            return None

        current_timestamp = datetime.now(UTC)
        return BusinessReportSource(
            report_id=source_report.report_id,
            report_type=source_report.report_type,
            report_title=source_report.report_title,
            author_id=source_report.author_id,
            target_start_date=source_report.target_start_date,
            target_end_date=source_report.target_end_date,
            included_items=source_report.sections,
            report_content=self._build_source_report_content(source_report),
            report_evidence=source_report.report_evidence,
            related_simulation_id=source_report.related_simulation_id,
            created_at=current_timestamp,
            updated_at=current_timestamp,
        )

    def _build_source_report_content(
        self,
        source_report: BusinessReportSourceRequest,
    ) -> dict[str, Any]:
        report_content = source_report.report_content
        content = dict(report_content) if isinstance(report_content, dict) else {}

        if source_report.markdown is not None:
            content["markdown"] = source_report.markdown

        if source_report.sections is not None:
            content["sections"] = source_report.sections

        return content

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
        report_content = self._build_business_report_content(
            source=source,
            llm_payload=llm_payload,
        )

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

    def _build_business_report_content(
        self,
        *,
        source: BusinessReportSource,
        llm_payload: dict[str, Any],
    ) -> dict[str, Any]:
        llm_report_content = llm_payload.get("report_content")
        source_report_content = (
            source.report_content if isinstance(source.report_content, dict) else {}
        )

        report_content = dict(source_report_content)
        if isinstance(llm_report_content, dict):
            report_content.update(llm_report_content)

        markdown = report_content.get("markdown")
        if not isinstance(markdown, str) or not markdown.strip():
            markdown = self._build_fallback_markdown(source)

        sections = self._extract_business_sections(report_content, source)

        return {
            "markdown": markdown,
            "sections": self._ensure_frontend_sections(sections),
        }

    def _extract_business_sections(
        self,
        report_content: dict[str, Any],
        source: BusinessReportSource,
    ) -> dict[str, Any]:
        content_sections = report_content.get("sections")
        if isinstance(content_sections, dict):
            return content_sections

        if isinstance(source.included_items, dict):
            return source.included_items

        return {}

    def _ensure_frontend_sections(
        self,
        sections: dict[str, Any],
    ) -> dict[str, Any]:
        frontend_sections = dict(sections)
        frontend_sections["summaryRows"] = self._ensure_rows(
            frontend_sections.get("summaryRows"),
            self._build_unknown_summary_rows(),
        )
        frontend_sections["lineRows"] = self._ensure_rows(
            frontend_sections.get("lineRows"),
            [
                {
                    "line": "확인 필요",
                    "utilization": "-",
                    "completed": "-",
                    "defectRate": "-",
                    "note": "확인 필요",
                }
            ],
        )
        frontend_sections["equipmentRows"] = self._ensure_rows(
            frontend_sections.get("equipmentRows"),
            [
                {
                    "name": "확인 필요",
                    "utilization": "확인 필요",
                    "downTime": "확인 필요",
                    "status": "확인 필요",
                }
            ],
        )
        frontend_sections["analysis"] = self._ensure_analysis(
            frontend_sections.get("analysis")
        )
        return frontend_sections

    def _ensure_rows(
        self,
        rows: Any,
        fallback_rows: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        if isinstance(rows, list) and rows:
            return rows

        return fallback_rows

    def _ensure_analysis(
        self,
        analysis: Any,
    ) -> dict[str, Any]:
        if not isinstance(analysis, dict):
            return self._build_fallback_analysis()

        overview = analysis.get("overview")
        analysis_sections = analysis.get("sections")
        recommendation = analysis.get("recommendation")

        return {
            "overview": overview if overview else "분석 내용이 없습니다.",
            "sections": analysis_sections
            if isinstance(analysis_sections, list) and analysis_sections
            else [{"title": "종합 분석", "items": ["분석 내용이 없습니다."]}],
            "recommendation": recommendation if recommendation else "생성 필요",
        }

    def _build_unknown_summary_rows(self) -> list[dict[str, str]]:
        labels = [
            "보고서 기간",
            "보고서 유형",
            "총 주문 수",
            "총 생산계획 수",
            "총 생산 계획 수량",
            "총 생산 완료 수량",
            "생산 계획 대비 실적",
            "라인 가동률",
            "불량 수량",
            "불량률",
            "납기 위험 주문 수",
            "자재 위험 품목 수",
            "비정상 설비 상태 수",
        ]
        return [{"label": label, "value": "확인 필요", "change": "-"} for label in labels]

    def _build_fallback_analysis(self) -> dict[str, Any]:
        return {
            "overview": "분석 내용이 없습니다.",
            "sections": [{"title": "종합 분석", "items": ["분석 내용이 없습니다."]}],
            "recommendation": "생성 필요",
        }

    def _build_fallback_markdown(
        self,
        source: BusinessReportSource,
    ) -> str:
        return (
            f"# {self._build_business_report_title(source)}\n\n"
            "선택한 일반 보고서를 기반으로 경영진용 보고서를 생성했습니다."
        )

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
