from app.features.report.schemas.request import ReportGenerateRequest
from app.features.report.schemas.response import (
    EvidenceType,
    ReportEvidence,
    ReportGenerateResponse,
    ReportStatus,
    ReportValidationResult,
)


class ReportGenerationService:
    def generate_report(self, request: ReportGenerateRequest) -> ReportGenerateResponse:
        period_text = f"{request.period.start_date} ~ {request.period.end_date}"

        title = self._build_title(request)
        markdown = self._build_dummy_markdown(title, period_text)

        sections = {
            "summary": {
                "period": period_text,
                "totalPlannedQuantity": 0,
                "totalCompletedQuantity": 0,
                "achievementRate": 0,
                "delayRiskOrderCount": 0,
                "materialRiskCount": 0,
            },
            "linePerformance": [],
            "materialRisk": [],
            "riskAnalysis": [],
            "recommendation": {
                "priority": "현재는 더미 보고서 응답입니다. 이후 RDB 데이터 수집 Agent와 연결됩니다."
            },
        }

        evidence = [
            ReportEvidence(
                type=EvidenceType.AGENT,
                source="report_generation_service",
                description="더미 보고서 생성 응답",
            )
        ]

        validation = ReportValidationResult(
            requiredSectionIncluded=True,
            groundednessPassed=True,
            missingFields=[],
        )

        return ReportGenerateResponse(
            reportJobId=request.report_job_id,
            status=ReportStatus.COMPLETED,
            title=title,
            reportType=request.report_type.value,
            markdown=markdown,
            sections=sections,
            evidence=evidence,
            validation=validation,
            errorMessage=None,
        )

    def _build_title(self, request: ReportGenerateRequest) -> str:
        if request.report_type.value == "MONTHLY":
            return f"{request.period.start_date.strftime('%Y년 %m월')} 생산 운영 보고서"

        return f"{request.period.start_date} ~ {request.period.end_date} 수시 생산 운영 보고서"

    def _build_dummy_markdown(self, title: str, period_text: str) -> str:
        return f"""# {title}

## 1. 주요 요약

- 보고서 기간: {period_text}
- 총 생산 계획 수량: 데이터 수집 전
- 총 생산 완료 수량: 데이터 수집 전
- 계획 대비 실적률: 데이터 수집 전
- 납기 위험 주문 수: 데이터 수집 전
- 자재 부족 품목 수: 데이터 수집 전

## 2. 생산 실적 분석

현재 단계에서는 FastAPI Report Agent 기본 연결을 확인하기 위한 더미 보고서입니다.

## 3. 라인별 성과

라인별 성과 데이터는 이후 RDB 데이터 수집 Agent 연결 후 제공됩니다.

## 4. 자재 및 재고 리스크

자재 부족 위험 데이터는 이후 자재/재고 조회 Agent 연결 후 제공됩니다.

## 5. 리스크 분석

납기 위험 및 생산 지연 리스크는 이후 리스크 분석 데이터와 연결됩니다.

## 6. 종합 의견 및 제안

현재는 API 계약과 기본 응답 구조를 검증하는 단계입니다.
"""