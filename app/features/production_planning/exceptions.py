from __future__ import annotations

from typing import Any


class ProductionPlanningErrorMixin:
    """Base behavior for production planning exceptions that become API errors."""

    default_error_code = "PRODUCTION_PLANNING_ERROR"

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        view: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.default_error_code
        self.details = details or {}
        self.view = view

    def to_response_error(self) -> dict[str, Any]:
        """
        Parameters:
            - None.

        Methodology:
            - Convert the exception into a stable response payload.
            - Include optional context only when the caller supplied it.

        Output:
            - Dictionary with error_code, message, and optional view/details fields.
        """
        payload: dict[str, Any] = {
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.view:
            payload["view"] = self.view
        if self.details:
            payload["details"] = self.details
        return payload

    def __str__(self) -> str:
        if self.view:
            return f"[{self.view}] {self.message}"
        return self.message


class PlanningValidationError(ProductionPlanningErrorMixin, ValueError):
    default_error_code = "PLANNING_VALIDATION_ERROR"


class PlanningInfeasibleError(ProductionPlanningErrorMixin, RuntimeError):
    default_error_code = "PLANNING_INFEASIBLE"


class SolverExecutionError(ProductionPlanningErrorMixin, RuntimeError):
    default_error_code = "SOLVER_EXECUTION_ERROR"


class SolutionExtractionError(ProductionPlanningErrorMixin, RuntimeError):
    default_error_code = "SOLUTION_EXTRACTION_ERROR"


class PlanningDataAccessError(ProductionPlanningErrorMixin, RuntimeError):
    """
    Raised when the planning data repository cannot load data from the database.

    Attributes:
        message: Human-readable error description.
        error_code: Stable error code for response serialization.
        details: Optional structured data for callers that need context.
        view: Name of the ai_planning view that caused the error, if applicable.
    """
    default_error_code = "PLANNING_DATA_ACCESS_ERROR"

    def __init__(
        self,
        message: str,
        view: str | None = None,
        *,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            details=details,
            view=view,
        )
