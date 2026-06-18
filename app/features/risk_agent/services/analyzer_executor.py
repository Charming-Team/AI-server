from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from app.features.risk_agent.schemas.common import AnalyzerName
from app.features.risk_agent.schemas.evidence import RiskAgentEvidence
from app.features.risk_agent.schemas.state import AnalyzerFinding


class RiskAnalyzer(Protocol):
    name: AnalyzerName

    def analyze(
        self,
        context: RiskAgentEvidence,
    ) -> list[AnalyzerFinding]:
        ...


@dataclass(frozen=True)
class AnalyzerBatchResult:
    findings: list[AnalyzerFinding]
    failed_analyzers: list[AnalyzerName]


class AnalyzerExecutor:
    def __init__(
        self,
        analyzers: list[RiskAnalyzer],
    ) -> None:
        self.analyzers = tuple(analyzers)

    async def run(
        self,
        context: RiskAgentEvidence,
    ) -> AnalyzerBatchResult:
        tasks = [
            asyncio.to_thread(analyzer.analyze, context)
            for analyzer in self.analyzers
        ]

        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        findings: list[AnalyzerFinding] = []
        failed_analyzers: list[AnalyzerName] = []

        for analyzer, result in zip(
            self.analyzers,
            results,
            strict=True,
        ):
            if isinstance(result, BaseException):
                failed_analyzers.append(analyzer.name)

                findings.append(
                    AnalyzerFinding(
                        analyzer=analyzer.name,
                        detected=False,
                        summary="Analyzer 실행에 실패했습니다.",
                        reasoning="해당 영역의 분석 결과는 최종 판단에서 제외됩니다.",
                        error_message=str(result),
                    )
                )
                continue

            findings.extend(result)

        return AnalyzerBatchResult(
            findings=findings,
            failed_analyzers=failed_analyzers,
        )