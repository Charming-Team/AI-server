"""
Simulation Data Repository

Loads Monte Carlo sampling inputs from ai_planning history views through the
read-only planning database connection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.core.database import planning_engine
from app.features.production_planning.exceptions import PlanningDataAccessError

logger = logging.getLogger(__name__)

DEFAULT_SAMPLING_LOOKBACK_DAYS = 365


@dataclass
class SimulationInputBundle:
    production_results: list[dict[str, Any]]
    production_result_causes: list[dict[str, Any]]
    changeover_sequences: list[dict[str, Any]]
    orders_history: list[dict[str, Any]]
    production_plans_history: list[dict[str, Any]]
    ai_prediction_results: list[dict[str, Any]]
    ai_prediction_causes: list[dict[str, Any]]
    simulation_results_history: list[dict[str, Any]]
    simulation_details_history: list[dict[str, Any]]
    machine_status_history: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SimulationDataRepository:
    """
    Read-only repository for ai_planning simulation sampling views.

    Primary historical views fail fast because they define the empirical simulation model.
    Auxiliary views degrade to empty lists so Monte Carlo can still use primary or fallback
    distributions.
    """

    def load_simulation_input_bundle(
        self,
        planning_start: datetime | None = None,
        planning_end: datetime | None = None,
    ) -> SimulationInputBundle:
        """
        Parameters:
            - planning_start: Optional plan start; sampling history must end before this time.
            - planning_end: Optional plan end retained for interface symmetry.

        Methodology:
            - Open a single connection and query all required sampling views sequentially.
            - Use historical production rows before planning_start to avoid future data leakage.
            - Primary view failures raise PlanningDataAccessError.
            - Auxiliary view failures return empty lists with logged warnings.
            - Machine status history is loaded for diagnostics only, not breakdown probability.

        Output:
            - SimulationInputBundle with raw rows for empirical distribution building.
        """
        warnings: list[str] = []
        sampling_start = _calculate_sampling_start(planning_start)
        try:
            with planning_engine.connect() as conn:
                return SimulationInputBundle(
                    production_results=self._get_production_results(
                        conn,
                        sampling_start,
                        planning_start,
                    ),
                    production_result_causes=self._get_production_result_causes(
                        conn,
                        sampling_start,
                        planning_start,
                    ),
                    changeover_sequences=self._get_auxiliary_rows(
                        conn,
                        "ai_planning.v_observed_changeover_sequence_history_for_sampling",
                        warnings,
                    ),
                    orders_history=self._get_auxiliary_rows(
                        conn,
                        "ai_planning.v_orders_history_for_sampling",
                        warnings,
                    ),
                    production_plans_history=self._get_auxiliary_rows(
                        conn,
                        "ai_planning.v_production_plans_history_for_sampling",
                        warnings,
                    ),
                    ai_prediction_results=self._get_auxiliary_rows(
                        conn,
                        "ai_planning.v_ai_prediction_results_history_for_sampling",
                        warnings,
                    ),
                    ai_prediction_causes=self._get_auxiliary_rows(
                        conn,
                        "ai_planning.v_ai_prediction_causes_history_for_sampling",
                        warnings,
                    ),
                    simulation_results_history=self._get_auxiliary_rows(
                        conn,
                        "ai_planning.v_schedule_simulation_results_history_for_sampling",
                        warnings,
                    ),
                    simulation_details_history=self._get_auxiliary_rows(
                        conn,
                        "ai_planning.v_schedule_simulation_details_history_for_sampling",
                        warnings,
                    ),
                    machine_status_history=self._get_auxiliary_rows(
                        conn,
                        "ai_planning.v_machine_status_history_for_sampling",
                        warnings,
                    ),
                    warnings=warnings,
                )
        except PlanningDataAccessError:
            raise
        except Exception as exc:
            raise PlanningDataAccessError(
                f"Failed to load simulation input bundle: {exc}"
            ) from exc

    def _get_production_results(
        self,
        conn: Connection,
        sampling_start: datetime | None,
        sampling_end: datetime | None,
    ) -> list[dict[str, Any]]:
        where_clauses = ["1 = 1"]
        params: dict[str, Any] = {}
        if sampling_start is not None:
            where_clauses.append(
                "COALESCE(actual_end_at, planned_end_at) >= :sampling_start"
            )
            params["sampling_start"] = sampling_start
        if sampling_end is not None:
            where_clauses.append(
                "COALESCE(actual_end_at, planned_end_at) < :sampling_end"
            )
            params["sampling_end"] = sampling_end

        query = text(
            f"""
            SELECT
                result_id,
                plan_id,
                order_id,
                product_id,
                line_id,
                NULL::text AS product_category,
                planned_start_at,
                actual_start_at,
                planned_end_at,
                actual_end_at,
                planned_quantity,
                actual_duration_hr,
                actual_setup_time_hr,
                actual_delay_hr,
                is_delayed,
                actual_delay_reason,
                yield_rate,
                actual_quantity,
                defect_quantity,
                result_status
            FROM ai_planning.v_production_results_history_for_sampling
            WHERE {" AND ".join(where_clauses)}
            ORDER BY product_id, line_id
            """
        )
        return self._execute_primary_query(
            conn,
            query,
            "ai_planning.v_production_results_history_for_sampling",
            params,
        )

    def _get_production_result_causes(
        self,
        conn: Connection,
        sampling_start: datetime | None,
        sampling_end: datetime | None,
    ) -> list[dict[str, Any]]:
        where_clauses = ["1 = 1"]
        params: dict[str, Any] = {}
        if sampling_start is not None:
            where_clauses.append("created_at >= :sampling_start")
            params["sampling_start"] = sampling_start
        if sampling_end is not None:
            where_clauses.append("created_at < :sampling_end")
            params["sampling_end"] = sampling_end

        query = text(
            f"""
            SELECT
                NULL::text AS product_id,
                NULL::text AS line_id,
                NULL::text AS product_category,
                cause_type,
                created_at
            FROM ai_planning.v_production_result_causes_history_for_sampling
            WHERE {" AND ".join(where_clauses)}
            ORDER BY cause_type, created_at
            """
        )
        return self._execute_primary_query(
            conn,
            query,
            "ai_planning.v_production_result_causes_history_for_sampling",
            params,
        )

    def _execute_primary_query(
        self,
        conn: Connection,
        query,
        view: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            return [dict(row) for row in conn.execute(query, params or {}).mappings().all()]
        except Exception as exc:
            raise PlanningDataAccessError(f"Failed to query {view}: {exc}", view=view) from exc

    def _get_auxiliary_rows(
        self,
        conn: Connection,
        view: str,
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        try:
            rows = conn.execute(text(f"SELECT * FROM {view}")).mappings().all()
            return [dict(row) for row in rows]
        except Exception as exc:
            message = f"Auxiliary sampling view skipped: {view} ({exc})"
            warnings.append(message)
            logger.warning(
                "production_planning.simulation.auxiliary_view_skipped",
                extra={"view": view},
            )
            return []


def _calculate_sampling_start(planning_start: datetime | None) -> datetime | None:
    if planning_start is None:
        return None
    return planning_start - timedelta(days=DEFAULT_SAMPLING_LOOKBACK_DAYS)
