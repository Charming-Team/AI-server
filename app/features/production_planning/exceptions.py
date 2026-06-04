class PlanningValidationError(ValueError):
    pass


class PlanningInfeasibleError(RuntimeError):
    pass


class SolverExecutionError(RuntimeError):
    pass


class SolutionExtractionError(RuntimeError):
    pass


class PlanningDataAccessError(RuntimeError):
    """
    Raised when the planning data repository cannot load data from the database.

    Attributes:
        message: Human-readable error description.
        view: Name of the ai_planning view that caused the error, if applicable.
    """

    def __init__(self, message: str, view: str | None = None) -> None:
        super().__init__(message)
        self.view = view

    def __str__(self) -> str:
        if self.view:
            return f"[{self.view}] {super().__str__()}"
        return super().__str__()
