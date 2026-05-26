from datetime import date
from typing import Any

from sqlalchemy import text

from app.core.database import engine


class PostgresReportRepository:
    def fetch_report_source_data(
        self,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        return {
            "order_summary": self._fetch_order_summary(start_date, end_date),
            "production_plan_summary": self._fetch_production_plan_summary(start_date, end_date),
            "production_result_summary": self._fetch_production_result_summary(start_date, end_date),
            "material_summary": self._fetch_material_summary(),
            "plan_material_summary": self._fetch_plan_material_summary(start_date, end_date),
            "risk_summary": self._fetch_risk_summary(start_date, end_date),
            "line_summary": self._fetch_line_summary(start_date, end_date),
            "machine_summary": self._fetch_machine_summary(start_date, end_date),

            "top_risk_orders": self._fetch_top_risk_orders(start_date, end_date),
            "top_material_shortages": self._fetch_top_material_shortages(start_date, end_date),
            "top_line_statuses": self._fetch_top_line_statuses(start_date, end_date),
            "top_machine_statuses": self._fetch_top_machine_statuses(start_date, end_date),
        }

    def _fetch_order_summary(
        self,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        query = text(
            """
            SELECT
                COUNT(*) AS total_order_count,
                COALESCE(SUM(order_quantity), 0) AS total_order_quantity,
                COUNT(*) FILTER (
                    WHERE due_date BETWEEN :start_date AND :end_date
                ) AS due_order_count,
                COUNT(*) FILTER (
                    WHERE order_status::TEXT ILIKE '%DELAY%'
                ) AS delayed_order_count
            FROM customer_orders
            WHERE order_date <= :end_date
              AND due_date >= :start_date
            """
        )

        with engine.connect() as conn:
            row = conn.execute(
                query,
                {"start_date": start_date, "end_date": end_date},
            ).mappings().first()

        return dict(row or {})

    def _fetch_production_plan_summary(
        self,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        query = text(
            """
            SELECT
                COUNT(*) AS total_plan_count,
                COALESCE(SUM(planned_quantity), 0) AS total_planned_quantity,
                COALESCE(SUM(estimated_duration_hr), 0) AS total_estimated_duration_hr,
                COUNT(DISTINCT line_id) AS active_line_count,
                COUNT(*) FILTER (
                    WHERE plan_status::TEXT ILIKE '%DELAY%'
                ) AS delayed_plan_count
            FROM production_plans
            WHERE planned_start_at::DATE <= :end_date
              AND planned_end_at::DATE >= :start_date
            """
        )

        with engine.connect() as conn:
            row = conn.execute(
                query,
                {"start_date": start_date, "end_date": end_date},
            ).mappings().first()

        return dict(row or {})

    def _fetch_production_result_summary(
        self,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        query = text(
            """
            SELECT
                COUNT(pr.*) AS total_result_count,
                COALESCE(SUM(pp.planned_quantity), 0) AS result_planned_quantity,
                COALESCE(SUM(pr.actual_quantity), 0) AS total_actual_quantity,
                COALESCE(SUM(pr.defect_quantity), 0) AS total_defect_quantity,
                COALESCE(AVG(pr.yield_rate), 0) AS avg_yield_rate,
                COALESCE(
                    SUM(
                        CASE
                            WHEN pr.actual_end_at IS NOT NULL
                             AND pp.planned_end_at IS NOT NULL
                             AND pr.actual_end_at > pp.planned_end_at
                            THEN EXTRACT(EPOCH FROM (pr.actual_end_at - pp.planned_end_at)) / 3600
                            ELSE 0
                        END
                    ),
                    0
                ) AS total_actual_delay_hr,
                COUNT(*) FILTER (
                    WHERE pr.actual_end_at IS NOT NULL
                      AND pp.planned_end_at IS NOT NULL
                      AND pr.actual_end_at > pp.planned_end_at
                ) AS delayed_result_count
            FROM production_results pr
            JOIN production_plans pp
              ON pr.plan_id = pp.plan_id
            WHERE pp.planned_start_at::DATE <= :end_date
              AND pp.planned_end_at::DATE >= :start_date
            """
        )

        with engine.connect() as conn:
            row = conn.execute(
                query,
                {"start_date": start_date, "end_date": end_date},
            ).mappings().first()

        return dict(row or {})

    def _fetch_material_summary(self) -> dict[str, Any]:
        query = text(
            """
            SELECT
                COUNT(*) AS total_material_count,
                COUNT(*) FILTER (
                    WHERE available_quantity < safety_stock_quantity
                ) AS safety_stock_shortage_count,
                COUNT(*) FILTER (
                    WHERE inventory_status::TEXT ILIKE '%SHORT%'
                       OR inventory_status::TEXT ILIKE '%RISK%'
                ) AS risk_material_count,
                COALESCE(SUM(current_quantity), 0) AS total_current_quantity,
                COALESCE(SUM(available_quantity), 0) AS total_available_quantity,
                COALESCE(SUM(reserved_quantity), 0) AS total_reserved_quantity
            FROM material_inventories
            """
        )

        with engine.connect() as conn:
            row = conn.execute(query).mappings().first()

        return dict(row or {})

    def _fetch_plan_material_summary(
        self,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        query = text(
            """
            SELECT
                COUNT(*) AS total_plan_material_count,
                COALESCE(SUM(ppm.required_quantity), 0) AS total_required_quantity,
                COALESCE(SUM(ppm.reserved_quantity), 0) AS total_reserved_quantity,
                COALESCE(SUM(ppm.shortage_quantity), 0) AS total_shortage_quantity,
                COUNT(*) FILTER (
                    WHERE ppm.shortage_quantity > 0
                ) AS shortage_plan_material_count
            FROM production_plan_materials ppm
            JOIN production_plans pp
              ON ppm.plan_id = pp.plan_id
            WHERE pp.planned_start_at::DATE <= :end_date
              AND pp.planned_end_at::DATE >= :start_date
            """
        )

        with engine.connect() as conn:
            row = conn.execute(
                query,
                {"start_date": start_date, "end_date": end_date},
            ).mappings().first()

        return dict(row or {})

    def _fetch_risk_summary(
        self,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        query = text(
            """
            SELECT
                COUNT(*) AS total_prediction_count,
                COUNT(*) FILTER (
                    WHERE risk_level::TEXT IN ('WARNING', 'CRITICAL')
                ) AS delay_risk_order_count,
                COUNT(*) FILTER (
                    WHERE risk_level::TEXT = 'CRITICAL'
                ) AS critical_risk_count,
                COUNT(*) FILTER (
                    WHERE risk_level::TEXT = 'WARNING'
                ) AS warning_risk_count,
                COALESCE(AVG(delay_probability), 0) AS avg_delay_probability,
                COALESCE(AVG(predicted_delay_days), 0) AS avg_predicted_delay_days
            FROM ai_prediction_results
            WHERE predicted_at::DATE BETWEEN :start_date AND :end_date
            """
        )

        with engine.connect() as conn:
            row = conn.execute(
                query,
                {"start_date": start_date, "end_date": end_date},
            ).mappings().first()

        return dict(row or {})

    def _fetch_line_summary(
        self,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        query = text(
            """
            SELECT
                COUNT(*) AS total_line_status_count,
                COUNT(DISTINCT line_id) AS observed_line_count,
                COALESCE(AVG(utilization_rate), 0) AS avg_line_utilization_rate,
                COALESCE(AVG(progress_rate), 0) AS avg_line_progress_rate,
                COALESCE(AVG(waiting_time_hr), 0) AS avg_waiting_time_hr,
                COALESCE(SUM(processed_quantity), 0) AS total_line_processed_quantity,
                COALESCE(SUM(defect_quantity), 0) AS total_line_defect_quantity,
                COUNT(*) FILTER (
                    WHERE operation_status::TEXT NOT IN ('RUNNING', 'OPERATING', 'ACTIVE')
                ) AS non_running_line_status_count
            FROM line_status
            WHERE recorded_at::DATE BETWEEN :start_date AND :end_date
            """
        )

        with engine.connect() as conn:
            row = conn.execute(
                query,
                {"start_date": start_date, "end_date": end_date},
            ).mappings().first()

        return dict(row or {})

    def _fetch_machine_summary(
        self,
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:
        query = text(
            """
            SELECT
                COUNT(*) AS total_machine_status_count,
                COUNT(DISTINCT machine_id) AS observed_machine_count,
                COALESCE(SUM(processed_quantity), 0) AS total_machine_processed_quantity,
                COALESCE(SUM(defect_quantity), 0) AS total_machine_defect_quantity,
                COUNT(*) FILTER (
                    WHERE operation_status::TEXT NOT IN ('RUNNING', 'OPERATING', 'ACTIVE')
                ) AS abnormal_machine_status_count
            FROM machine_statuses
            WHERE recorded_at::DATE BETWEEN :start_date AND :end_date
            """
        )

        with engine.connect() as conn:
            row = conn.execute(
                query,
                {"start_date": start_date, "end_date": end_date},
            ).mappings().first()

        return dict(row or {})
    

    # 납기 지연 위험이 높은 상위 5개 주문 조회
    def _fetch_top_risk_orders(
        self,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT
                apr.prediction_id,
                apr.order_id,
                co.customer_name,
                p.product_name,
                co.order_quantity,
                co.due_date,
                apr.delay_probability,
                apr.predicted_delay_days,
                apr.risk_level::TEXT AS risk_level,
                apr.analysis_summary,
                apr.recommended_action,
                apr.predicted_at
            FROM ai_prediction_results apr
            JOIN customer_orders co
              ON apr.order_id = co.order_id
            JOIN products p
              ON co.product_id = p.product_id
            WHERE apr.predicted_at::DATE BETWEEN :start_date AND :end_date
              AND apr.risk_level::TEXT IN ('WARNING', 'CRITICAL')
            ORDER BY
                CASE
                    WHEN apr.risk_level::TEXT = 'CRITICAL' THEN 1
                    WHEN apr.risk_level::TEXT = 'WARNING' THEN 2
                    ELSE 3
                END,
                apr.delay_probability DESC,
                apr.predicted_delay_days DESC
            LIMIT 5
            """
        )

        with engine.connect() as conn:
            rows = conn.execute(
                query,
                {"start_date": start_date, "end_date": end_date},
            ).mappings().all()

        return [dict(row) for row in rows]
    
    # 자재 부족 위험이 높은 상위 5개 자재 조회
    def _fetch_top_material_shortages(
        self,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT
                ppm.plan_material_id,
                ppm.plan_id,
                pp.order_id,
                m.material_id,
                m.material_name,
                ppm.required_quantity,
                ppm.reserved_quantity,
                ppm.shortage_quantity,
                ppm.material_plan_status::TEXT AS material_plan_status,
                pp.planned_start_at,
                pp.planned_end_at
            FROM production_plan_materials ppm
            JOIN production_plans pp
              ON ppm.plan_id = pp.plan_id
            JOIN materials m
              ON ppm.material_id = m.material_id
            WHERE pp.planned_start_at::DATE <= :end_date
              AND pp.planned_end_at::DATE >= :start_date
              AND ppm.shortage_quantity > 0
            ORDER BY ppm.shortage_quantity DESC
            LIMIT 5
            """
        )

        with engine.connect() as conn:
            rows = conn.execute(
                query,
                {"start_date": start_date, "end_date": end_date},
            ).mappings().all()

        return [dict(row) for row in rows]
    
    # 비정상 상태가 자주 관측된 상위 5개 라인 조회
    def _fetch_top_line_statuses(
        self,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT
                ls.line_status_id,
                ls.line_id,
                pl.line_code,
                pl.line_name,
                ls.operation_status::TEXT AS operation_status,
                ls.utilization_rate,
                ls.progress_rate,
                ls.waiting_time_hr,
                ls.processed_quantity,
                ls.defect_quantity,
                ls.recorded_at
            FROM line_status ls
            JOIN production_lines pl
              ON ls.line_id = pl.line_id
            WHERE ls.recorded_at::DATE BETWEEN :start_date AND :end_date
            ORDER BY
                CASE
                    WHEN ls.operation_status::TEXT IN ('DOWN', 'ERROR', 'MAINTENANCE', 'STOPPED') THEN 1
                    ELSE 2
                END,
                ls.waiting_time_hr DESC,
                ls.utilization_rate DESC
            LIMIT 5
            """
        )

        with engine.connect() as conn:
            rows = conn.execute(
                query,
                {"start_date": start_date, "end_date": end_date},
            ).mappings().all()

        return [dict(row) for row in rows]
    
    # 비정상 상태가 자주 관측된 상위 5개 설비 조회
    def _fetch_top_machine_statuses(
        self,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT
                ms.machine_status_id,
                ms.machine_id,
                pm.machine_code,
                pm.machine_name,
                pm.line_id,
                pl.line_code,
                ms.operation_status::TEXT AS operation_status,
                ms.processed_quantity,
                ms.defect_quantity,
                ms.status_note,
                ms.recorded_at
            FROM machine_statuses ms
            JOIN production_machines pm
              ON ms.machine_id = pm.machine_id
            JOIN production_lines pl
              ON pm.line_id = pl.line_id
            WHERE ms.recorded_at::DATE BETWEEN :start_date AND :end_date
            ORDER BY
                CASE
                    WHEN ms.operation_status::TEXT IN ('ERROR', 'DOWN', 'MAINTENANCE', 'STOPPED') THEN 1
                    ELSE 2
                END,
                ms.defect_quantity DESC,
                ms.processed_quantity DESC
            LIMIT 5
            """
        )

        with engine.connect() as conn:
            rows = conn.execute(
                query,
                {"start_date": start_date, "end_date": end_date},
            ).mappings().all()

        return [dict(row) for row in rows]