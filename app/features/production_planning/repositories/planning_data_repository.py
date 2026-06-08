"""
Planning Data Repository

Loads all production planning input data from the ai_planning schema views
using the read-only smap_planning PostgreSQL role.

All queries target ai_planning.v_* views only — no direct access to public base tables.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.core.database import planning_engine
from app.features.production_planning.exceptions import PlanningDataAccessError
from app.features.production_planning.schemas import (
    BomItemInput,
    ChangeoverRuleInput,
    ExistingScheduleInput,
    MaterialInput,
    OrderInput,
    ProductInput,
    ProductionLineInput,
    ProductLineCapabilityInput,
)


def _due_date_to_datetime(value: _dt.date | _dt.datetime) -> _dt.datetime:
    """DB의 date 타입 납기일을 당일 23:59:59 UTC datetime으로 변환한다."""
    if isinstance(value, _dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=_dt.UTC)
    return _dt.datetime.combine(value, _dt.time(23, 59, 59), tzinfo=_dt.UTC)


@dataclass
class PlanningInputBundle:
    """
    Aggregated planning inputs loaded from all ai_planning views.

    Pass this bundle to build_planning_request() to construct a
    ProductionPlanningRequest for the solver.
    """

    orders: list[OrderInput]
    products: list[ProductInput]
    production_lines: list[ProductionLineInput]
    product_line_capabilities: list[ProductLineCapabilityInput]
    existing_schedules: list[ExistingScheduleInput]
    changeover_rules: list[ChangeoverRuleInput]
    material_inventories: list[MaterialInput]
    bom_items: list[BomItemInput]
    line_statuses: list[dict[str, Any]]
    machine_statuses: list[dict[str, Any]]


class PlanningDataRepository:
    """
    Read-only repository for ai_planning schema views.

    Each public method loads one category of planning input data and maps
    it to the production planning schema types. All queries run inside a
    single shared connection provided by load_planning_input_bundle().
    """

    def load_planning_input_bundle(
        self,
        planning_start: _dt.datetime | None = None,
        planning_end: _dt.datetime | None = None,
        excluded_order_statuses: list[str] | None = None,
        use_existing_plans_as_orders: bool = False,
        excluded_rescheduling_plan_statuses: list[str] | None = None,
    ) -> PlanningInputBundle:
        """
        Parameters:
            - planning_start: Optional inclusive planning window start for DB-side filters.
            - planning_end: Optional exclusive planning window end for DB-side filters.
            - excluded_order_statuses: Optional order status labels to exclude in the DB query.
            - use_existing_plans_as_orders: Whether future production plans become the
              rescheduling targets.
            - excluded_rescheduling_plan_statuses: Plan statuses excluded from rescheduling.

        Methodology:
            - Open a single connection and run all view queries sequentially.
            - Map every result set to the corresponding planning input type.
            - Apply planning window filters where rows represent time-bounded planning inputs.

        Output:
            - PlanningInputBundle containing all data needed to build a ProductionPlanningRequest.
        """
        try:
            with planning_engine.connect() as conn:
                orders = (
                    self.get_reschedulable_orders_from_existing_plans(
                        conn,
                        planning_start,
                        excluded_rescheduling_plan_statuses,
                    )
                    if use_existing_plans_as_orders
                    else self.get_orders_for_planning(
                        conn,
                        planning_start,
                        planning_end,
                        excluded_order_statuses,
                    )
                )
                return PlanningInputBundle(
                    orders=orders,
                    products=self.get_products_for_planning(conn),
                    production_lines=self.get_production_lines_for_planning(conn),
                    product_line_capabilities=self.get_product_line_capabilities_for_planning(conn),
                    existing_schedules=self.get_existing_schedules_for_planning(
                        conn,
                        planning_start,
                        planning_end,
                        excluded_rescheduling_plan_statuses
                        if use_existing_plans_as_orders
                        else None,
                    ),
                    changeover_rules=self.get_changeover_rules_for_planning(conn),
                    material_inventories=self.get_material_inventories_for_planning(
                        conn,
                        planning_end,
                    ),
                    bom_items=self.get_bom_items_for_planning(conn),
                    line_statuses=self.get_latest_line_statuses_for_planning(conn),
                    machine_statuses=self.get_latest_machine_statuses_for_planning(conn),
                )
        except PlanningDataAccessError:
            raise
        except Exception as exc:
            raise PlanningDataAccessError(
                f"Failed to load planning input bundle: {exc}"
            ) from exc

    def get_reschedulable_orders_from_existing_plans(
        self,
        conn: Connection,
        planning_start: _dt.datetime | None,
        excluded_plan_statuses: list[str] | None = None,
    ) -> list[OrderInput]:
        """
        Parameters:
            - conn: Active SQLAlchemy connection to the planning database.
            - planning_start: Inclusive lower bound for existing plan start times.
            - excluded_plan_statuses: Plan statuses that must not be rescheduled.

        Methodology:
            - Read existing production plans that start at or after planning_start.
            - Exclude completed and in-progress plans from the rescheduling target set.
            - Join order attributes from the planning order view first, then the historical
              order view because completed source orders can still have future open plans.
            - Convert each plan row into a unique OrderInput so CP-SAT can rearrange plan-level
              work. If one source order appears in multiple plan rows, split quantity and
              monetary values across those plan targets.

        Output:
            - List of OrderInput objects representing reschedulable production plan rows.
        """
        if planning_start is None:
            raise PlanningDataAccessError(
                "planning_start is required when existing plans are used as planning targets.",
                view="ai_planning.v_existing_schedules_for_planning",
            )

        params: dict[str, Any] = {"planning_start": planning_start}
        excluded_status_clause = _build_status_exclusion_clause(
            "plan_status",
            excluded_plan_statuses or ["COMPLETED", "IN_PROGRESS"],
            params,
            key_prefix="excluded_plan_status",
        )
        where_clauses = ["start_time >= :planning_start"]
        if excluded_status_clause:
            where_clauses.append(excluded_status_clause)

        query = text(
            f"""
            WITH target_plan AS (
                SELECT
                    schedule_id,
                    order_id AS source_order_id,
                    product_id AS plan_product_id,
                    line_id AS original_line_id,
                    start_time,
                    end_time,
                    plan_status,
                    COUNT(*) OVER (PARTITION BY order_id) AS split_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY order_id
                        ORDER BY start_time, schedule_id
                    ) AS split_index
                FROM ai_planning.v_existing_schedules_for_planning
                WHERE {" AND ".join(where_clauses)}
            ),
            all_orders AS (
                SELECT
                    1 AS source_priority,
                    order_id,
                    product_id,
                    order_quantity,
                    due_date,
                    contract_amount,
                    late_penalty_amount,
                    order_status
                FROM ai_planning.v_orders_for_planning
                UNION ALL
                SELECT
                    2 AS source_priority,
                    order_id,
                    product_id,
                    order_quantity,
                    due_date,
                    contract_amount,
                    late_penalty_amount,
                    order_status
                FROM ai_planning.v_orders_history_for_sampling
            ),
            order_data AS (
                SELECT DISTINCT ON (order_id)
                    order_id,
                    product_id,
                    order_quantity,
                    due_date,
                    contract_amount,
                    late_penalty_amount,
                    order_status
                FROM all_orders
                ORDER BY order_id, source_priority
            )
            SELECT
                target_plan.schedule_id,
                target_plan.source_order_id,
                target_plan.plan_product_id,
                target_plan.original_line_id,
                target_plan.start_time,
                target_plan.end_time,
                target_plan.plan_status,
                target_plan.split_count,
                target_plan.split_index,
                order_data.order_id AS joined_order_id,
                order_data.product_id AS order_product_id,
                order_data.order_quantity,
                order_data.due_date,
                order_data.contract_amount,
                order_data.late_penalty_amount,
                order_data.order_status
            FROM target_plan
            LEFT JOIN order_data
              ON order_data.order_id = target_plan.source_order_id
            ORDER BY target_plan.start_time, target_plan.schedule_id
            """
        )

        try:
            rows = conn.execute(query, params).mappings().all()
        except Exception as exc:
            raise PlanningDataAccessError(
                f"Failed to query reschedulable existing plans: {exc}",
                view="ai_planning.v_existing_schedules_for_planning",
            ) from exc

        missing_order_ids = [
            str(row["source_order_id"])
            for row in rows
            if row["joined_order_id"] is None
        ]
        if missing_order_ids:
            raise PlanningDataAccessError(
                "Order data is missing for reschedulable plans: "
                + ", ".join(sorted(missing_order_ids)),
                view="ai_planning.v_existing_schedules_for_planning",
            )
        if not rows:
            raise PlanningDataAccessError(
                "No reschedulable existing production plans were found.",
                view="ai_planning.v_existing_schedules_for_planning",
            )

        return [_reschedulable_plan_row_to_order(row) for row in rows]

    def get_orders_for_planning(
        self,
        conn: Connection,
        planning_start: _dt.datetime | None = None,
        planning_end: _dt.datetime | None = None,
        excluded_order_statuses: list[str] | None = None,
    ) -> list[OrderInput]:
        """
        Parameters:
            - conn: Active SQLAlchemy connection to the planning database.
            - planning_start: Optional inclusive planning date lower bound.
            - planning_end: Optional inclusive planning date upper bound.
            - excluded_order_statuses: Optional order status labels to remove in SQL.

        Methodology:
            - Query ai_planning.v_orders_for_planning.
            - Exclude completed-style statuses while leaving final policy filtering to
              SolverConfig during preprocessing.
            - When a planning end is supplied, include unfinished overdue orders by filtering
              only due_date <= planning_end.
            - Map contract_amount → order_amount, late_penalty_amount as-is.

        Output:
            - List of OrderInput objects ready for the planning solver.
        """
        where_clauses = ["1 = 1"]
        params: dict[str, Any] = {}
        statuses_to_exclude = excluded_order_statuses or ["COMPLETED"]
        status_clause = _build_status_exclusion_clause(
            "order_status",
            statuses_to_exclude,
            params,
        )
        if status_clause:
            where_clauses.append(status_clause)
        if planning_end is not None:
            where_clauses.append("due_date <= CAST(:planning_end AS date)")
            params["planning_end"] = planning_end

        query = text(
            f"""
            SELECT
                order_id,
                product_id,
                order_quantity,
                due_date,
                contract_amount,
                late_penalty_amount,
                order_status
            FROM ai_planning.v_orders_for_planning
            WHERE {" AND ".join(where_clauses)}
            ORDER BY order_id
            """
        )

        try:
            rows = conn.execute(query, params).mappings().all()
        except Exception as exc:
            raise PlanningDataAccessError(
                f"Failed to query v_orders_for_planning: {exc}",
                view="ai_planning.v_orders_for_planning",
            ) from exc

        if not rows:
            raise PlanningDataAccessError(
                "v_orders_for_planning returned no plannable orders.",
                view="ai_planning.v_orders_for_planning",
            )

        return [
            OrderInput(
                order_id=str(row["order_id"]),
                product_id=str(row["product_id"]),
                order_quantity=int(row["order_quantity"]),
                due_date=_due_date_to_datetime(row["due_date"]),
                order_amount=int(row["contract_amount"] or 0),
                late_penalty_amount=int(row["late_penalty_amount"] or 0),
                status=str(row["order_status"]),
            )
            for row in rows
        ]

    def get_products_for_planning(self, conn: Connection) -> list[ProductInput]:
        """
        Parameters:
            - conn: Active SQLAlchemy connection.

        Methodology:
            - Query ai_planning.v_products_for_planning.
            - Map product_id and product_name; other fields are informational.

        Output:
            - List of ProductInput objects.
        """
        query = text(
            """
            SELECT
                product_id,
                product_name,
                product_category,
                unit,
                average_yield_rate,
                min_production_quantity
            FROM ai_planning.v_products_for_planning
            ORDER BY product_id
            """
        )

        try:
            rows = conn.execute(query).mappings().all()
        except Exception as exc:
            raise PlanningDataAccessError(
                f"Failed to query v_products_for_planning: {exc}",
                view="ai_planning.v_products_for_planning",
            ) from exc

        if not rows:
            raise PlanningDataAccessError(
                "v_products_for_planning returned no products.",
                view="ai_planning.v_products_for_planning",
            )

        return [
            ProductInput(
                product_id=str(row["product_id"]),
                product_name=str(row["product_name"]),
                grade=str(row["product_category"]) if row["product_category"] else None,
                unit=str(row["unit"]) if row["unit"] else None,
                average_yield_rate=(
                    Decimal(str(row["average_yield_rate"]))
                    if row["average_yield_rate"] is not None
                    else None
                ),
                min_production_quantity=(
                    int(row["min_production_quantity"])
                    if row["min_production_quantity"] is not None
                    else None
                ),
            )
            for row in rows
        ]

    def get_production_lines_for_planning(self, conn: Connection) -> list[ProductionLineInput]:
        """
        Parameters:
            - conn: Active SQLAlchemy connection.

        Methodology:
            - Query ai_planning.v_production_lines_for_planning.
            - Include only active lines (is_active = true).
            - available_from and available_to are left as None so that preprocessing
              defaults them to planning_start and planning_end.

        Output:
            - List of ProductionLineInput objects.
        """
        query = text(
            """
            SELECT
                line_id,
                line_name,
                is_active,
                max_capacity_per_day,
                capacity_unit,
                supports_changeover
            FROM ai_planning.v_production_lines_for_planning
            WHERE is_active = true
            ORDER BY line_id
            """
        )

        try:
            rows = conn.execute(query).mappings().all()
        except Exception as exc:
            raise PlanningDataAccessError(
                f"Failed to query v_production_lines_for_planning: {exc}",
                view="ai_planning.v_production_lines_for_planning",
            ) from exc

        if not rows:
            raise PlanningDataAccessError(
                "v_production_lines_for_planning returned no active production lines.",
                view="ai_planning.v_production_lines_for_planning",
            )

        return [
            ProductionLineInput(
                line_id=str(row["line_id"]),
                line_name=str(row["line_name"]),
                is_active=bool(row["is_active"]),
                max_capacity_per_day=(
                    int(row["max_capacity_per_day"])
                    if row["max_capacity_per_day"] is not None
                    else None
                ),
                capacity_unit=str(row["capacity_unit"]) if row["capacity_unit"] else None,
                supports_changeover=bool(row["supports_changeover"]),
            )
            for row in rows
        ]

    def get_product_line_capabilities_for_planning(
        self,
        conn: Connection,
    ) -> list[ProductLineCapabilityInput]:
        """
        Parameters:
            - conn: Active SQLAlchemy connection.

        Methodology:
            - Query ai_planning.v_product_line_capabilities_for_planning.
            - Convert standard_production_time_hr → process_time_per_unit_minutes using ceil.
            - Convert standard_yield_rate → yield_rate_scaled (×10000) as integer.
            - fixed_setup_minutes is NOT sourced from this view; changeover time
              comes from v_changeover_rules_for_planning instead.

        Output:
            - List of ProductLineCapabilityInput objects.
        """
        query = text(
            """
            SELECT
                product_id,
                line_id,
                standard_production_time_hr,
                capacity_per_day,
                standard_yield_rate,
                priority_rank
            FROM ai_planning.v_product_line_capabilities_for_planning
            ORDER BY product_id, line_id
            """
        )

        try:
            rows = conn.execute(query).mappings().all()
        except Exception as exc:
            raise PlanningDataAccessError(
                f"Failed to query v_product_line_capabilities_for_planning: {exc}",
                view="ai_planning.v_product_line_capabilities_for_planning",
            ) from exc

        return [
            ProductLineCapabilityInput(
                product_id=str(row["product_id"]),
                line_id=str(row["line_id"]),
                process_time_per_unit_minutes=_to_process_time_minutes(
                    row["standard_production_time_hr"]
                ),
                capacity_per_day=(
                    int(row["capacity_per_day"])
                    if row["capacity_per_day"] is not None
                    else None
                ),
                yield_rate_scaled=_to_yield_rate_scaled(row["standard_yield_rate"]),
                priority_rank=(
                    int(row["priority_rank"]) if row["priority_rank"] is not None else None
                ),
            )
            for row in rows
        ]

    def get_existing_schedules_for_planning(
        self,
        conn: Connection,
        planning_start: _dt.datetime | None = None,
        planning_end: _dt.datetime | None = None,
        excluded_rescheduling_plan_statuses: list[str] | None = None,
    ) -> list[ExistingScheduleInput]:
        """
        Parameters:
            - conn: Active SQLAlchemy connection.
            - planning_start: Optional inclusive planning window start.
            - planning_end: Optional exclusive planning window end.
            - excluded_rescheduling_plan_statuses: When supplied, plans starting at or after
              planning_start and not in these statuses are rescheduling targets, not blockers.

        Methodology:
            - Query ai_planning.v_existing_schedules_for_planning.
            - Include only SCHEDULED, IN_PROGRESS, and DELAYED plans as fixed blockers.
            - When a planning window is supplied, keep only schedules that overlap it.
            - These intervals are added to CP-SAT NoOverlap constraints so new orders
              cannot be placed on top of them.

        Output:
            - List of ExistingScheduleInput objects with is_locked=True.
        """
        where_clauses = ["plan_status IN ('SCHEDULED', 'IN_PROGRESS', 'DELAYED')"]
        params: dict[str, Any] = {}
        if planning_start is not None:
            where_clauses.append("end_time > :planning_start")
            params["planning_start"] = planning_start
        if planning_end is not None:
            where_clauses.append("start_time < :planning_end")
            params["planning_end"] = planning_end
        if planning_start is not None and excluded_rescheduling_plan_statuses is not None:
            rescheduling_params: dict[str, Any] = {}
            rescheduling_status_clause = _build_status_exclusion_clause(
                "plan_status",
                excluded_rescheduling_plan_statuses,
                rescheduling_params,
                key_prefix="reschedule_excluded_status",
            )
            if rescheduling_status_clause:
                params.update(rescheduling_params)
                where_clauses.append(
                    "NOT (start_time >= :planning_start "
                    f"AND {rescheduling_status_clause})"
                )

        query = text(
            f"""
            SELECT
                schedule_id,
                line_id,
                product_id,
                order_id,
                start_time,
                end_time,
                plan_status
            FROM ai_planning.v_existing_schedules_for_planning
            WHERE {" AND ".join(where_clauses)}
            ORDER BY line_id, start_time
            """
        )

        try:
            rows = conn.execute(query, params).mappings().all()
        except Exception as exc:
            raise PlanningDataAccessError(
                f"Failed to query v_existing_schedules_for_planning: {exc}",
                view="ai_planning.v_existing_schedules_for_planning",
            ) from exc

        return [
            ExistingScheduleInput(
                schedule_id=str(row["schedule_id"]),
                line_id=str(row["line_id"]),
                product_id=str(row["product_id"]),
                order_id=str(row["order_id"]) if row["order_id"] else None,
                start_time=row["start_time"],
                end_time=row["end_time"],
                is_locked=True,
                plan_status=str(row["plan_status"]) if row["plan_status"] else None,
            )
            for row in rows
        ]

    def get_changeover_rules_for_planning(
        self,
        conn: Connection,
    ) -> list[ChangeoverRuleInput]:
        """
        Parameters:
            - conn: Active SQLAlchemy connection.

        Methodology:
            - Query ai_planning.v_changeover_rules_for_planning.
            - Convert total_changeover_time_hr → changeover_minutes using ceil.
            - changeover_cost defaults to 0 when null.

        Output:
            - List of ChangeoverRuleInput objects.
        """
        query = text(
            """
            SELECT
                line_id,
                from_product_id,
                to_product_id,
                cleaning_time_hr,
                stabilization_time_hr,
                total_changeover_time_hr,
                changeover_cost,
                changeover_difficulty
            FROM ai_planning.v_changeover_rules_for_planning
            ORDER BY from_product_id, to_product_id
            """
        )

        try:
            rows = conn.execute(query).mappings().all()
        except Exception as exc:
            raise PlanningDataAccessError(
                f"Failed to query v_changeover_rules_for_planning: {exc}",
                view="ai_planning.v_changeover_rules_for_planning",
            ) from exc

        return [
            ChangeoverRuleInput(
                from_product_id=str(row["from_product_id"]),
                to_product_id=str(row["to_product_id"]),
                line_id=str(row["line_id"]) if row["line_id"] else None,
                cleaning_time_hr=_to_decimal(row["cleaning_time_hr"]),
                stabilization_time_hr=_to_decimal(row["stabilization_time_hr"]),
                total_changeover_time_hr=_to_decimal(row["total_changeover_time_hr"]),
                changeover_cost=(
                    int(row["changeover_cost"]) if row["changeover_cost"] is not None else 0
                ),
                changeover_difficulty=(
                    str(row["changeover_difficulty"])
                    if row["changeover_difficulty"]
                    else None
                ),
            )
            for row in rows
        ]

    def get_material_inventories_for_planning(
        self,
        conn: Connection,
        planning_end: _dt.datetime | None = None,
    ) -> list[MaterialInput]:
        """
        Parameters:
            - conn: Active SQLAlchemy connection.
            - planning_end: Optional planning window end used to ignore later inbound stock.

        Methodology:
            - Query ai_planning.v_material_inventories_for_planning.
            - Use available_now_quantity as the planning-safe inventory level.
              (available_now_quantity already excludes reserved and safety stock.)
            - Keep expected inbound only when it can arrive during the planning window.

        Output:
            - List of MaterialInput objects.
        """
        if planning_end is None:
            query = text(
                """
                SELECT
                    material_id,
                    available_now_quantity,
                    expected_inbound_quantity,
                    expected_inbound_at
                FROM ai_planning.v_material_inventories_for_planning
                ORDER BY material_id
                """
            )
            params = {}
        else:
            query = text(
                """
                SELECT
                    material_id,
                    available_now_quantity,
                    CASE
                        WHEN expected_inbound_at <= :planning_end
                        THEN expected_inbound_quantity
                        ELSE NULL
                    END AS expected_inbound_quantity,
                    CASE
                        WHEN expected_inbound_at <= :planning_end
                        THEN expected_inbound_at
                        ELSE NULL
                    END AS expected_inbound_at
                FROM ai_planning.v_material_inventories_for_planning
                ORDER BY material_id
                """
            )
            params = {"planning_end": planning_end}

        try:
            rows = conn.execute(query, params).mappings().all()
        except Exception as exc:
            raise PlanningDataAccessError(
                f"Failed to query v_material_inventories_for_planning: {exc}",
                view="ai_planning.v_material_inventories_for_planning",
            ) from exc

        return [
            MaterialInput(
                material_id=str(row["material_id"]),
                material_name=str(row["material_id"]),
                available_quantity=_to_decimal(row["available_now_quantity"]) or Decimal("0"),
                expected_inbound_quantity=_to_decimal(row["expected_inbound_quantity"]),
                expected_inbound_at=row["expected_inbound_at"],
            )
            for row in rows
        ]

    def get_bom_items_for_planning(self, conn: Connection) -> list[BomItemInput]:
        """
        Parameters:
            - conn: Active SQLAlchemy connection.

        Methodology:
            - Query ai_planning.v_bom_items_for_planning.
            - Apply loss_rate to required_quantity_per_unit in Python.
            - Round to nearest integer for CP-SAT integer constraints.

        Output:
            - List of BomItemInput objects with loss rate applied.
        """
        query = text(
            """
            SELECT
                product_id,
                material_id,
                required_quantity_per_unit,
                unit,
                loss_rate
            FROM ai_planning.v_bom_items_for_planning
            ORDER BY product_id, material_id
            """
        )

        try:
            rows = conn.execute(query).mappings().all()
        except Exception as exc:
            raise PlanningDataAccessError(
                f"Failed to query v_bom_items_for_planning: {exc}",
                view="ai_planning.v_bom_items_for_planning",
            ) from exc

        return [
            BomItemInput(
                product_id=str(row["product_id"]),
                material_id=str(row["material_id"]),
                required_quantity_per_unit=_to_decimal(row["required_quantity_per_unit"])
                or Decimal("0"),
                unit=str(row["unit"]) if row["unit"] else None,
                loss_rate=_to_decimal(row["loss_rate"]) or Decimal("0"),
            )
            for row in rows
        ]

    def get_latest_line_statuses_for_planning(
        self,
        conn: Connection,
    ) -> list[dict[str, Any]]:
        """
        Parameters:
            - conn: Active SQLAlchemy connection.

        Methodology:
            - Query ai_planning.v_line_status_latest_for_planning.
            - Returns raw dicts for use by Simulation Node and auxiliary Planning Node logic.

        Output:
            - List of line status dicts keyed by column name.
        """
        query = text(
            """
            SELECT
                line_status_id,
                line_id,
                product_id,
                plan_id,
                recorded_at,
                operation_status,
                throughput_rate,
                current_yield_rate,
                waiting_quantity,
                waiting_time_hr,
                processed_quantity,
                defect_quantity,
                utilization_rate,
                progress_rate
            FROM ai_planning.v_line_status_latest_for_planning
            ORDER BY line_id
            """
        )

        try:
            rows = conn.execute(query).mappings().all()
        except Exception as exc:
            raise PlanningDataAccessError(
                f"Failed to query v_line_status_latest_for_planning: {exc}",
                view="ai_planning.v_line_status_latest_for_planning",
            ) from exc

        return [dict(row) for row in rows]

    def get_latest_machine_statuses_for_planning(
        self,
        conn: Connection,
    ) -> list[dict[str, Any]]:
        """
        Parameters:
            - conn: Active SQLAlchemy connection.

        Methodology:
            - Query ai_planning.v_machine_status_latest_for_planning.
            - Returns raw dicts for use by Simulation Node and auxiliary Planning Node logic.

        Output:
            - List of machine status dicts keyed by column name.
        """
        query = text(
            """
            SELECT
                machine_status_id,
                machine_id,
                line_id,
                plan_id,
                product_id,
                recorded_at,
                operation_status,
                processed_quantity,
                defect_quantity
            FROM ai_planning.v_machine_status_latest_for_planning
            ORDER BY machine_id
            """
        )

        try:
            rows = conn.execute(query).mappings().all()
        except Exception as exc:
            raise PlanningDataAccessError(
                f"Failed to query v_machine_status_latest_for_planning: {exc}",
                view="ai_planning.v_machine_status_latest_for_planning",
            ) from exc

        return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------

def _to_process_time_minutes(standard_production_time_hr: Any) -> float | None:
    """Convert DB standard_production_time_hr from hours per ton to minutes per kg."""
    if standard_production_time_hr is None:
        return None
    return float(standard_production_time_hr) * 60 / 1000


def _to_yield_rate_scaled(standard_yield_rate: Any) -> int | None:
    """Scale yield rate (0.0–1.0) to integer ×10000 for CP-SAT integer constraints."""
    if standard_yield_rate is None:
        return None
    return round(float(standard_yield_rate) * 10_000)


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _build_status_exclusion_clause(
    column_name: str,
    statuses: list[str],
    params: dict[str, Any],
    key_prefix: str = "excluded_order_status",
) -> str | None:
    normalized_statuses = [
        status.strip().upper()
        for status in statuses
        if status is not None and status.strip()
    ]
    if not normalized_statuses:
        return None

    placeholders = []
    for index, status in enumerate(normalized_statuses):
        key = f"{key_prefix}_{index}"
        params[key] = status
        placeholders.append(f":{key}")
    return f"UPPER(TRIM({column_name}::text)) NOT IN ({', '.join(placeholders)})"


def _reschedulable_plan_row_to_order(row) -> OrderInput:
    source_quantity = int(row["order_quantity"])
    split_count = int(row["split_count"] or 1)
    split_index = int(row["split_index"] or 1)
    plan_quantity = _split_integer_value(source_quantity, split_count, split_index)
    amount = _split_money_value(row["contract_amount"], source_quantity, plan_quantity)
    late_penalty = _split_money_value(row["late_penalty_amount"], source_quantity, plan_quantity)

    return OrderInput(
        order_id=f"PLAN-{row['schedule_id']}",
        product_id=str(row["plan_product_id"]),
        customer_id=str(row["source_order_id"]) if row["source_order_id"] is not None else None,
        order_quantity=plan_quantity,
        due_date=_due_date_to_datetime(row["due_date"]),
        order_amount=amount,
        late_penalty_amount=late_penalty,
        status=str(row["plan_status"]) if row["plan_status"] else None,
    )


def _split_integer_value(total: int, split_count: int, split_index: int) -> int:
    if split_count <= 1:
        return total
    base = total // split_count
    remainder = total % split_count
    return base + (1 if split_index <= remainder else 0)


def _split_money_value(value: Any, source_quantity: int, plan_quantity: int) -> int:
    if value is None or source_quantity <= 0:
        return 0
    return round(Decimal(str(value)) * Decimal(plan_quantity) / Decimal(source_quantity))


def _apply_loss_rate(required_quantity_per_unit: Any, loss_rate: Any) -> Decimal:
    """Return quantity_per_unit × (1 + loss_rate) as Decimal without rounding."""
    base = Decimal(str(required_quantity_per_unit or 0))
    rate = Decimal(str(loss_rate or 0))
    return base * (1 + rate)
