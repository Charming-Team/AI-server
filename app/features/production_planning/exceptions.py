from __future__ import annotations

from typing import Any


class ProductionPlanningErrorMixin:
    """Base behavior for production planning exceptions that become API errors."""

    default_error_code = "PRODUCTION_PLANNING_ERROR"
    response_status = "500 INTERNAL_SERVER_ERROR"

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
            - Convert the exception into the compact API error format used by clients.
            - Keep diagnostic details on the exception object for logs instead of exposing
              them through the response body.

        Output:
            - Dictionary with status and message fields.
        """
        return {
            "status": self.response_status,
            "message": self.message,
        }

    def __str__(self) -> str:
        if self.view:
            return f"[{self.view}] {self.message}"
        return self.message


class PlanningValidationError(ProductionPlanningErrorMixin, ValueError):
    default_error_code = "PLANNING_VALIDATION_ERROR"
    response_status = "400 BAD_REQUEST"


class PlanningInfeasibleError(ProductionPlanningErrorMixin, RuntimeError):
    default_error_code = "PLANNING_INFEASIBLE"
    response_status = "409 CONFLICT"


class SolverExecutionError(ProductionPlanningErrorMixin, RuntimeError):
    default_error_code = "SOLVER_EXECUTION_ERROR"
    response_status = "408 REQUEST_TIMEOUT"


class SolutionExtractionError(ProductionPlanningErrorMixin, RuntimeError):
    default_error_code = "SOLUTION_EXTRACTION_ERROR"
    response_status = "500 INTERNAL_SERVER_ERROR"


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
    response_status = "503 SERVICE_UNAVAILABLE"

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
