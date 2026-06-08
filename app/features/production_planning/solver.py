import logging

from ortools.sat.python import cp_model

from app.features.production_planning.config import SolverConfig
from app.features.production_planning.cpsat_model_builder import CpsatModelBundle
from app.features.production_planning.exceptions import SolverExecutionError
from app.features.production_planning.schemas import RawSolverResult

logger = logging.getLogger(__name__)


class _LoggingCallback(cp_model.CpSolverSolutionCallback):
    def on_solution_callback(self) -> None:
        logger.info(
            "solution found | objective=%.2f | wall_time=%.2fs | branches=%d",
            self.ObjectiveValue(),
            self.WallTime(),
            self.NumBranches(),
        )


def solve_model(
    bundle: CpsatModelBundle,
    variant_code: str,
    config: SolverConfig,
) -> RawSolverResult:
    """
    Parameters:
        - bundle: CP-SAT model bundle with objective already applied.
        - variant_code: Plan variant code being solved (for example, "DUE_DATE_OPTIMAL").
        - config: Solver execution configuration.

    Methodology:
        - Configure CpSolver with time limit, worker count, and logging flag.
        - Solve the model and collect solver metadata without throwing on infeasible status.

    Output:
        - RawSolverResult containing status, objective value, timing, search stats, and solver.
    """
    try:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = config.time_limit_seconds
        solver.parameters.num_search_workers = config.num_search_workers
        solver.parameters.log_search_progress = config.log_solver
        if config.log_solver:
            solver.parameters.log_search_progress = True
        callback = _LoggingCallback() if config.log_solver else None
        status_code = solver.Solve(bundle.model, callback)
        status = solver.StatusName(status_code)
        has_solution = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        return RawSolverResult(
            plan_variant_code=variant_code,
            status=status,
            objective_value=solver.ObjectiveValue() if has_solution else None,
            wall_time_seconds=solver.WallTime(),
            conflicts=solver.NumConflicts(),
            branches=solver.NumBranches(),
            response_stats=solver.ResponseStats(),
            solver=solver,
        )
    except (RuntimeError, ValueError) as exc:
        raise SolverExecutionError(f"Failed to solve {variant_code} model: {exc}") from exc
