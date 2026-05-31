from datetime import UTC, datetime, timedelta

from app.features.production_planning.config import SolverConfig
from app.features.production_planning.langgraph_workflow import (
    build_production_planning_graph,
    run_production_planning_graph,
)
from app.features.production_planning.production_planning_node import generate_production_plans
from app.features.production_planning.schemas import (
    OrderInput,
    ProductInput,
    ProductionLineInput,
    ProductionPlanningRequest,
    ProductLineCapabilityInput,
)


def _request() -> ProductionPlanningRequest:
    start = datetime(2026, 5, 22, 9, tzinfo=UTC)
    return ProductionPlanningRequest(
        planning_start=start,
        planning_end=start + timedelta(hours=3),
        orders=[
            OrderInput(
                order_id="O-1",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=1),
                order_amount=100_000,
            ),
            OrderInput(
                order_id="O-2",
                product_id="P-1",
                quantity=1,
                due_date=start + timedelta(hours=2),
                order_amount=50_000,
            ),
        ],
        products=[
            ProductInput(
                product_id="P-1",
                product_name="Product",
                default_process_time_minutes=30,
            )
        ],
        production_lines=[
            ProductionLineInput(line_id="L-1", line_name="Line 1", is_active=True),
        ],
        product_line_capabilities=[
            ProductLineCapabilityInput(product_id="P-1", line_id="L-1"),
        ],
        solver_config=SolverConfig(time_limit_seconds=5, num_search_workers=1),
    )


def test_production_planning_graph_connects_required_nodes() -> None:
    workflow = build_production_planning_graph()
    graph = workflow.get_graph()
    node_names = set(graph.nodes)
    edge_pairs = {(edge.source, edge.target) for edge in graph.edges}

    assert {
        "validate_request",
        "normalize_request",
        "generate_plan_variants",
        "load_simulation_input_from_db",
        "build_sampling_distributions",
        "simulate_plan_candidates",
        "compare_plan_variants",
        "compare_simulation_results",
        "recommend_final_plans",
        "finalize_result",
    }.issubset(node_names)
    assert ("__start__", "validate_request") in edge_pairs
    assert ("validate_request", "normalize_request") in edge_pairs
    assert ("generate_plan_variants", "load_simulation_input_from_db") in edge_pairs
    assert ("load_simulation_input_from_db", "build_sampling_distributions") in edge_pairs
    assert ("build_sampling_distributions", "simulate_plan_candidates") in edge_pairs
    assert ("simulate_plan_candidates", "compare_plan_variants") in edge_pairs
    assert ("compare_simulation_results", "recommend_final_plans") in edge_pairs
    assert ("finalize_result", "__end__") in edge_pairs


def test_generate_production_plans_runs_through_langgraph_workflow() -> None:
    direct_result = run_production_planning_graph(_request())
    node_result = generate_production_plans(_request())

    assert len(direct_result.plan_results) == 6
    assert direct_result.recommended_plan_variant_code is not None
    assert len(direct_result.simulation_results) == 6
    assert direct_result.recommended_due_date_plan is not None
    assert direct_result.recommended_cost_plan is not None
    assert len(node_result.plan_results) == 6
    assert all(plan.status in {"OPTIMAL", "FEASIBLE"} for plan in node_result.plan_results)
