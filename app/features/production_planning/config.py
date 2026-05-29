from typing import Literal

from pydantic import BaseModel, Field


class DueDateOptimizationConfig(BaseModel):
    # Formula: weight_k > weight_(k+1) * max_orders * max_horizon_minutes.
    unscheduled_order_weight: int = 100_000_000_000_000
    delayed_order_count_weight: int = 100_000_000_000
    total_tardiness_weight: int = 10_000
    max_tardiness_weight: int = 1_000
    due_date_completion_priority_weight: int = 100
    total_changeover_minutes_weight: int = 10
    line_priority_weight: int = 1
    makespan_weight: int = 1


class CostOptimizationConfig(BaseModel):
    scheduled_amount_reward_weight: int = 1_000_000
    amount_weighted_tardiness_weight: int = 10_000
    unscheduled_amount_penalty_weight: int = 100_000
    high_amount_delay_penalty_weight: int = 1_000
    changeover_cost_weight: int = 10
    line_priority_weight: int = 1
    makespan_weight: int = 1
    due_date_safety_weight: int = 1_000
    amount_scale_unit: int = 1_000
    high_amount_threshold_policy: Literal[
        "TOP_20_PERCENT",
        "ABOVE_AVERAGE",
        "CUSTOM_THRESHOLD",
    ] = "TOP_20_PERCENT"
    custom_high_amount_threshold: int | None = None


class SolverConfig(BaseModel):
    time_limit_seconds: int = Field(default=30, ge=1)
    num_search_workers: int = Field(default=8, ge=1)
    default_changeover_minutes: int = Field(default=0, ge=0)
    default_changeover_cost: int = Field(default=0, ge=0)
    default_line_priority_rank: int = Field(default=100, ge=1)
    interchangeable_line_groups: list[list[str]] = Field(
        default_factory=lambda: [["1", "2"], ["3", "4"], ["5", "6"]]
    )
    due_date_policy: Literal["SOFT", "HARD"] = "SOFT"
    allow_unscheduled_orders: bool = True
    unscheduled_penalty: int = Field(default=1_000_000, ge=0)
    tardiness_penalty_per_minute: int = Field(default=1_000, ge=0)
    makespan_weight: int = Field(default=1, ge=0)
    changeover_weight: int = Field(default=1, ge=0)
    use_existing_schedule_locks: bool = True
    log_solver: bool = False
    excluded_order_statuses: list[str] = Field(
        default_factory=lambda: ["CANCELLED", "COMPLETED"]
    )
    locked_plan_statuses: list[str] = Field(
        default_factory=lambda: ["SCHEDULED", "IN_PROGRESS", "DELAYED"]
    )
    due_date_optimization: DueDateOptimizationConfig = Field(
        default_factory=DueDateOptimizationConfig
    )
    amount_optimization: CostOptimizationConfig = Field(
        default_factory=CostOptimizationConfig
    )


AmountOptimizationConfig = CostOptimizationConfig
