class PlanningValidationError(ValueError):
    pass


class PlanningInfeasibleError(RuntimeError):
    pass


class SolverExecutionError(RuntimeError):
    pass


class SolutionExtractionError(RuntimeError):
    pass
