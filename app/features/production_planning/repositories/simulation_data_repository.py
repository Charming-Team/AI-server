"""
Simulation Data Repository

Loads Monte Carlo sampling inputs from ai_planning history views through the
read-only planning database connection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.core.database import planning_engine
from app.features.production_planning.exceptions import PlanningDataAccessError

logger = logging.getLogger(__name__)


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

    def load_simulation_input_bundle(self) -> SimulationInputBundle:
        """
        Parameters:
            - None.

        Methodology:
            - Open a single connection and query all required sampling views sequentially.
            - Primary view failures raise PlanningDataAccessError.
            - Auxiliary view failures return empty lists with logged warnings.
            - Machine status history is loaded only for diagnostics and is not used as a
              Monte Carlo sampling source.

        Output:
            - SimulationInputBundle with raw rows for empirical distribution building.
        """
        warnings: list[str] = []
        try:
            with planning_engine.connect() as conn:
                return SimulationInputBundle(
                    production_results=self._get_production_results(conn),
                    production_result_causes=self._get_production_result_causes(conn),
                    changeover_sequences=self._get_changeover_sequences(conn),
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

    def _get_production_results(self, conn: Connection) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT
                product_id,
                line_id,
                NULL::text AS product_category,
                actual_duration_hr,
                actual_setup_time_hr,
                actual_delay_hr,
                is_delayed,
                yield_rate,
                actual_quantity,
                defect_quantity
            FROM ai_planning.v_production_results_history_for_sampling
            ORDER BY product_id, line_id
            """
        )
        return self._execute_primary_query(
            conn,
            query,
            "ai_planning.v_production_results_history_for_sampling",
        )

    def _get_production_result_causes(self, conn: Connection) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT product_id, line_id, cause_type
            FROM ai_planning.v_production_result_causes_history_for_sampling
            ORDER BY product_id, line_id
            """
        )
        return self._execute_primary_query(
            conn,
            query,
            "ai_planning.v_production_result_causes_history_for_sampling",
        )

    def _get_changeover_sequences(self, conn: Connection) -> list[dict[str, Any]]:
        query = text(
            """
            SELECT
                line_id,
                from_product_id,
                to_product_id,
                observed_gap_minutes
            FROM ai_planning.v_observed_changeover_sequence_history_for_sampling
            ORDER BY from_product_id, to_product_id, line_id
            """
        )
        return self._execute_primary_query(
            conn,
            query,
            "ai_planning.v_observed_changeover_sequence_history_for_sampling",
        )

    def _execute_primary_query(
        self,
        conn: Connection,
        query,
        view: str,
    ) -> list[dict[str, Any]]:
        try:
            return [dict(row) for row in conn.execute(query).mappings().all()]
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
