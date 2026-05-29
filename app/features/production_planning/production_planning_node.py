from app.features.production_planning.langgraph_workflow import run_production_planning_graph
from app.features.production_planning.schemas import (
    ProductionPlanningRequest,
    ProductionPlanningResult,
)


def generate_production_plans(
    request: ProductionPlanningRequest,
) -> ProductionPlanningResult:
    """
    Parameters:
        - request: Production planning request containing orders, resources, constraints, and
          solver configuration.

    Methodology:
        - Delegate orchestration to the LangGraph production planning workflow.
        - The graph connects validation, preprocessing, model-building, solving, extraction,
          comparison, and finalization nodes.

    Output:
        - ProductionPlanningResult containing due-date plan, amount plan, comparison summary,
          and combined solver metadata.
    """
    return run_production_planning_graph(request)
