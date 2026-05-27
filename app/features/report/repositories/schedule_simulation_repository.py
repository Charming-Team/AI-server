# 경제성 분석 전용 시뮬레이션 레포지토리
# 역할: 경제성 분석에 필요한 시뮬레이션 데이터 조회
# schedule_simulation_results 조회, schedule_simulation_details 조회
# economicAnalysis 생성, bestScenario 계산

from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from sqlalchemy import bindparam, text

from app.core.database import engine
from app.features.report.repositories.report_repository_utils import (
    to_float,
    to_int,
    to_iso,
)


class ScheduleSimulationRepository:
    def fetch_economic_analysis(
        self,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        simulation_results = self._fetch_simulation_results(start_date, end_date)

        if not simulation_results:
            return {
                "simulationResults": [],
                "bestScenario": None,
                "comment": "선택 기간 내 조회 가능한 시뮬레이션 결과가 없습니다.",
            }

        simulation_ids = [
            item["simulationId"]
            for item in simulation_results
            if item.get("simulationId") is not None
        ]

        details = self._fetch_simulation_details(simulation_ids)

        details_by_simulation_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for detail in details:
            details_by_simulation_id[detail["simulationId"]].append(detail)

        for item in simulation_results:
            item["details"] = details_by_simulation_id.get(item["simulationId"], [])

        best_scenario = self._select_best_simulation_scenario(simulation_results)

        return {
            "simulationResults": simulation_results,
            "bestScenario": best_scenario,
            "comment": (
                "선택 기간 내 시뮬레이션 결과를 기준으로 적용 전후 지연 시간, "
                "비용 변화, 라인 가동률을 비교했습니다."
            ),
        }

    def _fetch_simulation_results(
        self,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT
                simulation_id,
                simulation_group_id,
                simulation_name,
                prediction_id,
                simulation_type,
                action_result,
                before_total_delay_hr,
                after_total_delay_hr,
                delay_reduction_hr,
                before_avg_line_utilization_rate,
                after_avg_line_utilization_rate,
                before_total_production_quantity,
                after_total_production_quantity,
                before_bottleneck_line_id,
                after_bottleneck_line_id,
                cost_change_amount,
                recommendation_grade,
                applied_by,
                applied_at,
                created_at
            FROM schedule_simulation_results
            WHERE created_at >= :start_date
              AND created_at < :end_date_exclusive
            ORDER BY created_at DESC, simulation_id DESC
            """
        )

        params = {
            "start_date": start_date,
            "end_date_exclusive": end_date + timedelta(days=1),
        }

        with engine.connect() as conn:
            rows = conn.execute(query, params).mappings().all()

        return [
            {
                "simulationId": row["simulation_id"],
                "simulationGroupId": row["simulation_group_id"],
                "simulationName": row["simulation_name"],
                "predictionId": row["prediction_id"],
                "simulationType": row["simulation_type"],
                "actionResult": row["action_result"],
                "beforeTotalDelayHr": to_float(row["before_total_delay_hr"]),
                "afterTotalDelayHr": to_float(row["after_total_delay_hr"]),
                "delayReductionHr": to_float(row["delay_reduction_hr"]),
                "beforeAvgLineUtilizationRate": to_float(row["before_avg_line_utilization_rate"]),
                "afterAvgLineUtilizationRate": to_float(row["after_avg_line_utilization_rate"]),
                "beforeTotalProductionQuantity": to_int(row["before_total_production_quantity"]),
                "afterTotalProductionQuantity": to_int(row["after_total_production_quantity"]),
                "beforeBottleneckLineId": row["before_bottleneck_line_id"],
                "afterBottleneckLineId": row["after_bottleneck_line_id"],
                "costChangeAmount": to_float(row["cost_change_amount"]),
                "recommendationGrade": row["recommendation_grade"],
                "appliedBy": row["applied_by"],
                "appliedAt": to_iso(row["applied_at"]),
                "createdAt": to_iso(row["created_at"]),
                "details": [],
            }
            for row in rows
        ]

    def _fetch_simulation_details(
        self,
        simulation_ids: list[int],
    ) -> list[dict[str, Any]]:
        if not simulation_ids:
            return []

        query = (
            text(
                """
                SELECT
                    simulation_detail_id,
                    simulation_id,
                    order_id,
                    plan_id,
                    before_line_id,
                    after_line_id,
                    before_sequence,
                    after_sequence,
                    before_start_at,
                    before_end_at,
                    after_start_at,
                    after_end_at,
                    expected_completion_date,
                    after_is_delayed,
                    before_quantity,
                    after_quantity,
                    change_reason,
                    created_at
                FROM schedule_simulation_details
                WHERE simulation_id IN :simulation_ids
                ORDER BY simulation_id, simulation_detail_id
                """
            )
            .bindparams(bindparam("simulation_ids", expanding=True))
        )

        with engine.connect() as conn:
            rows = conn.execute(
                query,
                {"simulation_ids": simulation_ids},
            ).mappings().all()

        return [
            {
                "simulationDetailId": row["simulation_detail_id"],
                "simulationId": row["simulation_id"],
                "orderId": row["order_id"],
                "planId": row["plan_id"],
                "beforeLineId": row["before_line_id"],
                "afterLineId": row["after_line_id"],
                "beforeSequence": row["before_sequence"],
                "afterSequence": row["after_sequence"],
                "beforeStartAt": to_iso(row["before_start_at"]),
                "beforeEndAt": to_iso(row["before_end_at"]),
                "afterStartAt": to_iso(row["after_start_at"]),
                "afterEndAt": to_iso(row["after_end_at"]),
                "expectedCompletionDate": to_iso(row["expected_completion_date"]),
                "afterIsDelayed": row["after_is_delayed"],
                "beforeQuantity": to_int(row["before_quantity"]),
                "afterQuantity": to_int(row["after_quantity"]),
                "changeReason": row["change_reason"],
                "createdAt": to_iso(row["created_at"]),
            }
            for row in rows
        ]

    def _select_best_simulation_scenario(
        self,
        simulation_results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not simulation_results:
            return None

        grade_priority = {
            "HIGH": 1,
            "MEDIUM": 2,
            "LOW": 3,
        }

        best = sorted(
            simulation_results,
            key=lambda item: (
                grade_priority.get(item.get("recommendationGrade"), 99),
                -(item.get("delayReductionHr") or 0),
                item.get("costChangeAmount")
                if item.get("costChangeAmount") is not None
                else float("inf"),
                item.get("afterTotalDelayHr")
                if item.get("afterTotalDelayHr") is not None
                else float("inf"),
            ),
        )[0]

        return {
            "simulationId": best.get("simulationId"),
            "simulationGroupId": best.get("simulationGroupId"),
            "simulationName": best.get("simulationName"),
            "simulationType": best.get("simulationType"),
            "recommendationGrade": best.get("recommendationGrade"),
            "delayReductionHr": best.get("delayReductionHr"),
            "costChangeAmount": best.get("costChangeAmount"),
            "afterTotalDelayHr": best.get("afterTotalDelayHr"),
            "reason": self._build_best_scenario_reason(best),
        }

    def _build_best_scenario_reason(
        self,
        best: dict[str, Any],
    ) -> str:
        simulation_name = best.get("simulationName") or "선택된 대응안"
        grade = best.get("recommendationGrade") or "미정"
        delay_reduction_hr = best.get("delayReductionHr")
        cost_change_amount = best.get("costChangeAmount")
        after_total_delay_hr = best.get("afterTotalDelayHr")

        parts = [f"{simulation_name}은 추천 등급이 {grade}입니다."]

        if delay_reduction_hr is not None:
            parts.append(
                f"총 지연 시간을 {delay_reduction_hr}시간 감소시키는 것으로 분석되었습니다."
            )

        if after_total_delay_hr is not None:
            parts.append(f"적용 후 총 지연 시간은 {after_total_delay_hr}시간입니다.")

        if cost_change_amount is not None:
            parts.append(f"비용 변화 금액은 {cost_change_amount:,.0f}원입니다.")

        return " ".join(parts)
