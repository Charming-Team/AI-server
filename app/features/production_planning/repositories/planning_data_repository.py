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
from math import ceil
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

    def load_planning_input_bundle(self) -> PlanningInputBundle:
        """
        Parameters:
            - None.

        Methodology:
            - Open a single connection and run all view queries sequentially.
            - Map every result set to the corresponding planning input type.

        Output:
            - PlanningInputBundle containing all data needed to build a ProductionPlanningRequest.
        """
        try:
            with planning_engine.connect() as conn:
                return PlanningInputBundle(
                    orders=self.get_orders_for_planning(conn),
                    products=self.get_products_for_planning(conn),
                    production_lines=self.get_production_lines_for_planning(conn),
                    product_line_capabilities=self.get_product_line_capabilities_for_planning(conn),
                    existing_schedules=self.get_existing_schedules_for_planning(conn),
                    changeover_rules=self.get_changeover_rules_for_planning(conn),
                    material_inventories=self.get_material_inventories_for_planning(conn),
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

    def get_orders_for_planning(self, conn: Connection) -> list[OrderInput]:
        """
        Parameters:
            - conn: Active SQLAlchemy connection to the planning database.

        Methodology:
            - Query ai_planning.v_orders_for_planning.
            - Include only WAITING, IN_PROGRESS, and DELAYED orders.
            - Map contract_amount → order_amount, late_penalty_amount as-is.

        Output:
            - List of OrderInput objects ready for the planning solver.
        """
        query = text(
            """
            SELECT
                order_id,
                product_id,
                order_quantity,
                due_date,
                contract_amount,
                late_penalty_amount,
                order_status
            FROM ai_planning.v_orders_for_planning
            WHERE order_status IN ('WAITING', 'IN_PROGRESS', 'DELAYED')
            ORDER BY order_id
            """
        )

        try:
            rows = conn.execute(query).mappings().all()
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
    ) -> list[ExistingScheduleInput]:
        """
        Parameters:
            - conn: Active SQLAlchemy connection.

        Methodology:
            - Query ai_planning.v_existing_schedules_for_planning.
            - Include only SCHEDULED, IN_PROGRESS, and DELAYED plans as fixed blockers.
            - These intervals are added to CP-SAT NoOverlap constraints so new orders
              cannot be placed on top of them.

        Output:
            - List of ExistingScheduleInput objects with is_locked=True.
        """
        query = text(
            """
            SELECT
                schedule_id,
                line_id,
                product_id,
                order_id,
                start_time,
                end_time,
                plan_status
            FROM ai_planning.v_existing_schedules_for_planning
            WHERE plan_status IN ('SCHEDULED', 'IN_PROGRESS', 'DELAYED')
            ORDER BY line_id, start_time
            """
        )

        try:
            rows = conn.execute(query).mappings().all()
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
    ) -> list[MaterialInput]:
        """
        Parameters:
            - conn: Active SQLAlchemy connection.

        Methodology:
            - Query ai_planning.v_material_inventories_for_planning.
            - Use available_now_quantity as the planning-safe inventory level.
              (available_now_quantity already excludes reserved and safety stock.)
            - expected_inbound_quantity is mapped to confirmed_inbound_quantity
              for potential inbound-aware material constraints.

        Output:
            - List of MaterialInput objects.
        """
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

        try:
            rows = conn.execute(query).mappings().all()
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

def _to_process_time_minutes(standard_production_time_hr: Any) -> int | None:
    """Convert DB standard_production_time_hr into whole CP-SAT minutes."""
    if standard_production_time_hr is None:
        return None
    return ceil(float(standard_production_time_hr) * 60)


def _to_yield_rate_scaled(standard_yield_rate: Any) -> int | None:
    """Scale yield rate (0.0–1.0) to integer ×10000 for CP-SAT integer constraints."""
    if standard_yield_rate is None:
        return None
    return round(float(standard_yield_rate) * 10_000)


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _apply_loss_rate(required_quantity_per_unit: Any, loss_rate: Any) -> Decimal:
    """Return quantity_per_unit × (1 + loss_rate) as Decimal without rounding."""
    base = Decimal(str(required_quantity_per_unit or 0))
    rate = Decimal(str(loss_rate or 0))
    return base * (1 + rate)
