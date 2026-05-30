from datetime import datetime

from app.features.production_planning.config import SolverConfig
from app.features.production_planning.langgraph_workflow import run_production_planning_graph
from app.features.production_planning.repositories.planning_data_repository import (
    PlanningDataRepository,
)
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
        - ProductionPlanningResult with 6 plan variants and comparison summary.
    """
    return run_production_planning_graph(request)


def generate_production_plans_from_db(
    planning_start: datetime,
    planning_end: datetime,
    solver_config: SolverConfig | None = None,
) -> ProductionPlanningResult:
    """
    Parameters:
        - planning_start: Inclusive start of the planning window (timezone-aware datetime).
        - planning_end: Exclusive end of the planning window (timezone-aware datetime).
        - solver_config: Optional solver configuration; defaults to SolverConfig() when omitted.

    Methodology:
        - Load all planning input data from ai_planning schema views via PlanningDataRepository.
        - Build a ProductionPlanningRequest from the loaded bundle and the supplied time window.
        - Delegate to generate_production_plans() to run the full 6-variant optimization.

    Output:
        - ProductionPlanningResult with 6 plan variants and comparison summary.
    """
    repository = PlanningDataRepository()
    bundle = repository.load_planning_input_bundle()

    request = ProductionPlanningRequest(
        planning_start=planning_start,
        planning_end=planning_end,
        orders=bundle.orders,
        products=bundle.products,
        production_lines=bundle.production_lines,
        product_line_capabilities=bundle.product_line_capabilities,
        existing_schedules=bundle.existing_schedules,
        changeover_rules=bundle.changeover_rules,
        materials=bundle.material_inventories,
        bom_items=bundle.bom_items,
        solver_config=solver_config or SolverConfig(),
    )

    return generate_production_plans(request)
