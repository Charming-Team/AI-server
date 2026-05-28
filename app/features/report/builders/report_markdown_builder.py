from typing import Any


class ReportMarkdownBuilder:
    def build(
        self,
        title: str,
        period_text: str,
        sections: dict[str, Any],
    ) -> str:
        executive = sections["executiveSummary"]
        issue_history = sections["issueHistory"]
        delay_response = sections["delayResponseAnalysis"]
        economic = sections["economicAnalysis"]
        conclusion = sections["conclusion"]
        appendix = sections["appendix"]

        risk_issue_lines = self._build_top_risk_order_lines(
            issue_history.get("riskIssues", [])
        )
        machine_issue_lines = self._build_top_machine_status_lines(
            issue_history.get("machineIssues", [])
        )
        line_issue_lines = self._build_top_line_status_lines(
            issue_history.get("lineIssues", [])
        )
        material_impact_lines = self._build_top_material_shortage_lines(
            delay_response.get("materialImpact", [])
        )
        economic_analysis_lines = self._build_economic_analysis_lines(economic)
        source_lines = self._build_source_lines(appendix.get("sources", []))
        key_finding_lines = self._build_key_finding_lines(
            executive.get("keyFindings", [])
        )
        priority_action_lines = self._build_key_finding_lines(
            conclusion.get("priorityActions", [])
        )

        return f"""# {title}

## 1. Executive Summary

- 보고서 기간: {period_text}
- 총 주문 수: {executive["totalOrderCount"]}건
- 총 생산계획 수: {executive["totalPlanCount"]}건
- 납기 위험 주문 수: {executive["delayRiskOrderCount"]}건
- CRITICAL 위험 수: {executive["criticalRiskCount"]}건
- WARNING 위험 수: {executive["warningRiskCount"]}건
- 자재 위험 품목 수: {executive["materialRiskCount"]}건
- 평균 지연 확률: {executive["avgDelayProbability"]}%
- 평균 예상 지연일: {executive["avgPredictedDelayDays"]}일

### 핵심 요약

{key_finding_lines}

## 2. 선택한 날짜 사이의 이슈 이력

### 2-1. 지연 예측 이슈

{risk_issue_lines}

### 2-2. 머신 이슈

{machine_issue_lines}

### 2-3. 라인 이슈

{line_issue_lines}

## 3. 지연 대응 Solution 분석

### 3-1. 주문 관점

{risk_issue_lines}

### 3-2. 생산계획 관점

{line_issue_lines}

### 3-3. 자재 관점

{material_impact_lines}

## 4. 경제성 분석

{economic_analysis_lines}

## 5. 결론

{priority_action_lines}

{conclusion["finalComment"]}

## 6. Appendix / 출처

{source_lines}
"""

    def _build_economic_analysis_lines(
        self,
        economic_analysis: dict[str, Any],
    ) -> str:
        simulation_results = economic_analysis.get("simulationResults", [])
        best_scenario = economic_analysis.get("bestScenario")
        comment = economic_analysis.get("comment")

        lines: list[str] = []

        if comment:
            lines.append(comment)
            lines.append("")

        if not simulation_results:
            lines.append("- 선택 기간 내 조회 가능한 시뮬레이션 결과가 없습니다.")
            return "\n".join(lines)

        if best_scenario:
            lines.append("### 4-1. 추천 대응안")
            lines.append("")
            lines.append(f"- 대응안: {best_scenario.get('simulationName')}")
            lines.append(f"- 유형: {best_scenario.get('simulationType')}")
            lines.append(f"- 추천 등급: {best_scenario.get('recommendationGrade')}")
            lines.append(f"- 지연 감소 시간: {best_scenario.get('delayReductionHr')}시간")
            lines.append(f"- 적용 후 총 지연 시간: {best_scenario.get('afterTotalDelayHr')}시간")
            lines.append(
                f"- 비용 변화 금액: {self._format_number(best_scenario.get('costChangeAmount'))}원"
            )
            lines.append(f"- 선정 사유: {best_scenario.get('reason')}")
            lines.append("")

        lines.append("### 4-2. 대응안별 적용 전후 비교")
        lines.append("")
        lines.append(
            "| 대응안 | 유형 | 적용 전 지연(hr) | 적용 후 지연(hr) | "
            "지연 감소(hr) | 비용 변화 | 추천 등급 |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---|")

        for item in simulation_results:
            lines.append(
                "| {name} | {type} | {before} | {after} | {reduction} | {cost} | {grade} |".format(
                    name=item.get("simulationName", "-"),
                    type=item.get("simulationType", "-"),
                    before=item.get("beforeTotalDelayHr", "-"),
                    after=item.get("afterTotalDelayHr", "-"),
                    reduction=item.get("delayReductionHr", "-"),
                    cost=self._format_number(item.get("costChangeAmount")),
                    grade=item.get("recommendationGrade", "-"),
                )
            )

        lines.append("")
        lines.append("### 4-3. 주요 일정 변경 내역")
        lines.append("")

        detail_count = 0

        for item in simulation_results:
            details = item.get("details", [])
            if not details:
                continue

            lines.append(f"#### {item.get('simulationName')}")
            lines.append("")
            lines.append(
                "| 주문 ID | 계획 ID | 변경 전 라인 | 변경 후 라인 | "
                "변경 후 지연 여부 | 변경 사유 |"
            )
            lines.append("|---:|---:|---:|---:|---|---|")

            for detail in details[:5]:
                detail_count += 1
                lines.append(
                    (
                        "| {order_id} | {plan_id} | {before_line} | {after_line} | "
                        "{after_delayed} | {reason} |"
                    ).format(
                        order_id=detail.get("orderId", "-"),
                        plan_id=detail.get("planId", "-"),
                        before_line=detail.get("beforeLineId", "-"),
                        after_line=detail.get("afterLineId", "-"),
                        after_delayed="지연" if detail.get("afterIsDelayed") else "정상",
                        reason=detail.get("changeReason", "-"),
                    )
                )

            lines.append("")

        if detail_count == 0:
            lines.append("- 조회 가능한 상세 일정 변경 내역이 없습니다.")

        return "\n".join(lines)

    def _build_top_risk_order_lines(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "- 조회된 주요 납기 위험 주문이 없습니다."

        lines = []

        for index, row in enumerate(rows, start=1):
            cause = row.get("cause_detail") or row.get("analysis_summary") or "원인 확인 필요"
            action = row.get("recommended_action") or "추천 조치 확인 필요"
            evidence = (
                f"prediction_id={row.get('prediction_id')}, "
                f"predicted_at={row.get('predicted_at')}"
            )

            lines.append(
                f"- {index}. 위험 대상: 주문 {row.get('order_id')} / "
                f"고객사: {row.get('customer_name')} / "
                f"제품: {row.get('product_name')} / "
                f"위험도: {row.get('risk_level')} / "
                f"예상 지연 시간: {row.get('predicted_delay_days')}일 / "
                f"주요 원인: {cause} / "
                f"추천 조치: {action} / "
                f"근거 데이터: {evidence}"
            )

        return "\n".join(lines)

    def _build_top_material_shortage_lines(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "- 조회된 주요 자재 부족 계획이 없습니다."

        lines = []

        for index, row in enumerate(rows, start=1):
            lines.append(
                f"- {index}. 계획 {row.get('plan_id')} / "
                f"자재: {row.get('material_name')} / "
                f"필요 수량: {row.get('required_quantity')} / "
                f"예약 수량: {row.get('reserved_quantity')} / "
                f"부족 수량: {row.get('shortage_quantity')} / "
                f"상태: {row.get('material_plan_status')}"
            )

        return "\n".join(lines)

    def _build_top_line_status_lines(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "- 조회된 주요 라인 상태가 없습니다."

        lines = []

        for index, row in enumerate(rows, start=1):
            lines.append(
                f"- {index}. {row.get('line_code')} {row.get('line_name')} / "
                f"상태: {row.get('operation_status')} / "
                f"가동률: {row.get('utilization_rate')} / "
                f"진행률: {row.get('progress_rate')} / "
                f"대기 시간: {row.get('waiting_time_hr')}시간"
            )

        return "\n".join(lines)

    def _build_top_machine_status_lines(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "- 조회된 주요 설비 상태가 없습니다."

        lines = []

        for index, row in enumerate(rows, start=1):
            lines.append(
                f"- {index}. {row.get('machine_code')} {row.get('machine_name')} / "
                f"라인: {row.get('line_code')} / "
                f"상태: {row.get('operation_status')} / "
                f"처리 수량: {row.get('processed_quantity')} / "
                f"불량 수량: {row.get('defect_quantity')} / "
                f"비고: {row.get('status_note')}"
            )

        return "\n".join(lines)

    def _build_key_finding_lines(self, findings: list[str]) -> str:
        if not findings:
            return "- 주요 요약 내용이 없습니다."

        return "\n".join(f"- {finding}" for finding in findings)

    def _build_source_lines(self, sources: list[str]) -> str:
        if not sources:
            return "- 참조한 데이터 출처가 없습니다."

        return "\n".join(f"- {source}" for source in sources)

    def _format_number(self, value: Any) -> str:
        if value is None:
            return "-"

        try:
            return f"{float(value):,.0f}"
        except (TypeError, ValueError):
            return str(value)
