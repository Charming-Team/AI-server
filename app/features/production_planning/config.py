from typing import Literal

from pydantic import BaseModel, Field


class DueDateOptimizationConfig(BaseModel):
    # Formula: weight_k > weight_(k+1) * max_orders * max_horizon_minutes.
    unscheduled_order_weight: int = 100_000_000_000_000
    delayed_order_count_weight: int = 100_000_000_000
    total_tardiness_weight: int = 10_000
    max_tardiness_weight: int = 1_000
    material_shortage_weight: int = 10_000
    due_date_completion_priority_weight: int = 100
    total_changeover_minutes_weight: int = 10
    line_priority_weight: int = 1
    makespan_weight: int = 1


class CostOptimizationConfig(BaseModel):
    unscheduled_contract_amount_weight: int = 1_000_000_000
    late_penalty_amount_weight: int = 100_000_000
    delayed_order_count_weight: int = 10_000_000_000
    total_cleaning_cost_weight: int = 1_000_000
    line_change_cost_weight: int = 1_000_000
    different_product_sequence_count_weight: int = 1_000_000
    total_tardiness_minutes_weight: int = 10_000
    makespan_minutes_weight: int = 1
    scheduled_amount_reward_weight: int = 1_000_000
    amount_weighted_tardiness_weight: int = 10_000
    unscheduled_amount_penalty_weight: int = 100_000
    high_amount_delay_penalty_weight: int = 1_000
    late_penalty_weight: int = 1_000_000_000
    changeover_cost_weight: int = 1_000
    line_priority_weight: int = 1_000
    makespan_weight: int = 10
    due_date_safety_weight: int = 1
    amount_scale_unit: int = 1_000
    high_amount_threshold_policy: Literal[
        "TOP_20_PERCENT",
        "ABOVE_AVERAGE",
        "CUSTOM_THRESHOLD",
    ] = "TOP_20_PERCENT"
    custom_high_amount_threshold: int | None = None


class SimulationConfig(BaseModel):
    enabled: bool = True
    num_iterations: int = Field(default=100, ge=1)
    random_seed: int = 20260531
    min_samples: int = Field(default=10, ge=1)
    duration_variation_ratio: float = Field(default=0.10, ge=0.0, le=1.0)
    changeover_variation_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    delay_probability: float = Field(default=0.05, ge=0.0, le=1.0)
    min_delay_minutes: int = Field(default=30, ge=0)
    max_delay_minutes: int = Field(default=240, ge=0)
    yield_rate_mean: float = Field(default=0.97, ge=0.0, le=1.0)
    yield_rate_stddev: float = Field(default=0.02, ge=0.0, le=1.0)
    defect_rate_mean: float = Field(default=0.01, ge=0.0, le=1.0)
    defect_rate_stddev: float = Field(default=0.005, ge=0.0, le=1.0)
    material_shortage_delay_minutes: int = Field(default=24 * 60, ge=0)
    setup_delay_probability: float = Field(default=0.03, ge=0.0, le=1.0)
    min_setup_delay_minutes: int = Field(default=10, ge=0)
    max_setup_delay_minutes: int = Field(default=120, ge=0)
    machine_breakdown_probability: float = Field(default=0.02, ge=0.0, le=1.0)
    min_machine_repair_minutes: int = Field(default=60, ge=0)
    max_machine_repair_minutes: int = Field(default=480, ge=0)
    line_change_delay_probability: float = Field(default=0.01, ge=0.0, le=1.0)
    min_line_change_delay_minutes: int = Field(default=10, ge=0)
    max_line_change_delay_minutes: int = Field(default=180, ge=0)
    material_inbound_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    min_material_inbound_delay_minutes: int = Field(default=0, ge=0)
    max_material_inbound_delay_minutes: int = Field(default=7 * 24 * 60, ge=0)
    event_timeline_limit: int = Field(default=50, ge=0, le=500)


class SolverConfig(BaseModel):
    time_limit_seconds: int = Field(default=30, ge=1)
    num_search_workers: int = Field(default=8, ge=1)
    default_changeover_minutes: int = Field(default=0, ge=0)
    default_changeover_cost: int = Field(default=0, ge=0)
    default_line_priority_rank: int = Field(default=100, ge=1)
    interchangeable_line_groups: list[list[str]] = Field(default_factory=list)
    due_date_policy: Literal["SOFT", "HARD"] = "SOFT"
    allow_unscheduled_orders: bool = True
    unscheduled_penalty: int = Field(default=1_000_000, ge=0)
    tardiness_penalty_per_minute: int = Field(default=1_000, ge=0)
    makespan_weight: int = Field(default=1, ge=0)
    changeover_weight: int = Field(default=1, ge=0)
    use_existing_schedule_locks: bool = True
    use_material_constraints: bool = True
    log_solver: bool = False
    excluded_order_statuses: list[str] = Field(
        default_factory=lambda: ["CANCELLED", "COMPLETED", "IN_PROGRESS"]
    )
    locked_plan_statuses: list[str] = Field(
        default_factory=lambda: ["SCHEDULED", "IN_PROGRESS", "DELAYED"]
    )
    excluded_rescheduling_plan_statuses: list[str] = Field(
        default_factory=lambda: ["COMPLETED", "IN_PROGRESS"]
    )
    due_date_optimization: DueDateOptimizationConfig = Field(
        default_factory=DueDateOptimizationConfig
    )
    amount_optimization: CostOptimizationConfig = Field(
        default_factory=CostOptimizationConfig
    )


AmountOptimizationConfig = CostOptimizationConfig
