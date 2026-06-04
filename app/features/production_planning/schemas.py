from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.features.production_planning.config import SolverConfig

PlanFamily = Literal["DUE_DATE_OPTIMAL", "COST_OPTIMAL"]

PlanVariantCode = Literal[
    "DUE_DATE_MIN_DELAY_COUNT",
    "DUE_DATE_MIN_TOTAL_TARDINESS",
    "DUE_DATE_BALANCED",
    "COST_MIN_TOTAL_COST",
    "COST_MIN_CHANGEOVER_COST",
    "COST_BALANCED",
]

PLAN_VARIANTS: list[str] = [
    "DUE_DATE_MIN_DELAY_COUNT",
    "DUE_DATE_MIN_TOTAL_TARDINESS",
    "DUE_DATE_BALANCED",
    "COST_MIN_TOTAL_COST",
    "COST_MIN_CHANGEOVER_COST",
    "COST_BALANCED",
]

PLAN_VARIANT_FAMILIES: dict[str, str] = {
    "DUE_DATE_MIN_DELAY_COUNT": "DUE_DATE_OPTIMAL",
    "DUE_DATE_MIN_TOTAL_TARDINESS": "DUE_DATE_OPTIMAL",
    "DUE_DATE_BALANCED": "DUE_DATE_OPTIMAL",
    "COST_MIN_TOTAL_COST": "COST_OPTIMAL",
    "COST_MIN_CHANGEOVER_COST": "COST_OPTIMAL",
    "COST_BALANCED": "COST_OPTIMAL",
}

PLAN_VARIANT_NAMES: dict[str, str] = {
    "DUE_DATE_MIN_DELAY_COUNT": "Due-Date: Minimize Delay Count",
    "DUE_DATE_MIN_TOTAL_TARDINESS": "Due-Date: Minimize Total Tardiness",
    "DUE_DATE_BALANCED": "Due-Date: Balanced Optimization",
    "COST_MIN_TOTAL_COST": "Cost: Minimize Total Cost",
    "COST_MIN_CHANGEOVER_COST": "Cost: Minimize Changeover Cost",
    "COST_BALANCED": "Cost: Balanced Optimization",
}


class OrderInput(BaseModel):
    order_id: str
    product_id: str
    customer_id: str | None = None
    order_quantity: int | None = None
    quantity: int | None = None
    due_date: datetime
    order_amount: int
    late_penalty_amount: int = 0
    priority: int | None = None
    status: str | None = None
    is_locked: bool = False

    @model_validator(mode="after")
    def sync_order_quantity_fields(self) -> OrderInput:
        if self.order_quantity is None and self.quantity is None:
            raise ValueError("order_quantity or quantity is required.")
        if self.order_quantity is None:
            self.order_quantity = self.quantity
        if self.quantity is None:
            self.quantity = self.order_quantity
        return self


class ProductInput(BaseModel):
    product_id: str
    product_name: str
    grade: str | None = None
    default_process_time_minutes: float | None = None
    unit: str | None = None
    average_yield_rate: Decimal | None = None
    min_production_quantity: int | None = None


class ProductionLineInput(BaseModel):
    line_id: str
    line_name: str
    is_active: bool
    max_capacity_per_day: int | None = None
    capacity_unit: str | None = None
    supports_changeover: bool = True
    available_from: datetime | None = None
    available_to: datetime | None = None


class ProductLineCapabilityInput(BaseModel):
    product_id: str
    line_id: str
    process_time_per_unit_minutes: float | None = None
    fixed_setup_minutes: int | None = None
    yield_rate_scaled: int | None = None
    standard_yield_rate: Decimal | None = None
    capacity_per_day: int | None = None
    priority_rank: int | None = None

    @model_validator(mode="after")
    def sync_yield_rate_fields(self) -> ProductLineCapabilityInput:
        if self.yield_rate_scaled is None and self.standard_yield_rate is not None:
            self.yield_rate_scaled = int(
                (self.standard_yield_rate * Decimal("10000")).to_integral_value()
            )
        return self


class ExistingScheduleInput(BaseModel):
    schedule_id: str
    line_id: str
    product_id: str
    order_id: str | None = None
    start_time: datetime
    end_time: datetime
    is_locked: bool = True
    plan_status: str | None = None


class MaterialInput(BaseModel):
    material_id: str
    material_name: str
    material_unit: str | None = None
    available_quantity: Decimal
    safety_stock_quantity: Decimal = Decimal("0")
    expected_inbound_quantity: Decimal | None = None
    expected_inbound_at: datetime | None = None
    confirmed_inbound_quantity: Decimal | None = None
    confirmed_inbound_time: datetime | None = None


class BomItemInput(BaseModel):
    product_id: str
    material_id: str
    required_quantity_per_unit: Decimal
    unit: str | None = None
    loss_rate: Decimal = Decimal("0")


class ChangeoverRuleInput(BaseModel):
    from_product_id: str
    to_product_id: str
    line_id: str | None = None
    cleaning_time_hr: Decimal | None = None
    stabilization_time_hr: Decimal | None = None
    total_changeover_time_hr: Decimal | None = None
    changeover_minutes: int | None = None
    changeover_cost: int | None = None
    changeover_difficulty: str | None = None


class ProductionPlanningRequest(BaseModel):
    planning_start: datetime
    planning_end: datetime
    orders: list[OrderInput]
    products: list[ProductInput]
    production_lines: list[ProductionLineInput]
    product_line_capabilities: list[ProductLineCapabilityInput]
    existing_schedules: list[ExistingScheduleInput] = Field(default_factory=list)
    materials: list[MaterialInput] = Field(default_factory=list)
    bom_items: list[BomItemInput] = Field(default_factory=list)
    changeover_rules: list[ChangeoverRuleInput] = Field(default_factory=list)
    solver_config: SolverConfig = Field(default_factory=SolverConfig)


class ScheduleItem(BaseModel):
    order_id: str
    product_id: str
    line_id: str
    start_time: datetime
    end_time: datetime
    quantity: int
    order_quantity: int
    planned_production_quantity: int
    order_amount: int
    tardiness_minutes: int
    status: str


class PlanMetrics(BaseModel):
    scheduled_count: int
    unscheduled_count: int
    delayed_order_count: int
    on_time_order_count: int
    total_tardiness_minutes: int
    max_tardiness_minutes: int
    total_scheduled_amount: int
    on_time_amount: int
    delayed_amount: int
    unscheduled_amount: int
    makespan_minutes: int
    total_changeover_minutes: int
    total_late_penalty_amount: int = 0
    total_changeover_cost: int = 0
    estimated_total_cost: int = 0


class PlanResult(BaseModel):
    plan_family: str
    plan_variant_code: str
    plan_variant_name: str
    status: str
    schedule_items: list[ScheduleItem]
    unscheduled_orders: list[str]
    metrics: PlanMetrics
    warnings: list[str] = Field(default_factory=list)
    solver_metadata: dict[str, Any] = Field(default_factory=dict)


class PlanRankingItem(BaseModel):
    rank: int
    plan_variant_code: str
    plan_family: str
    plan_variant_name: str
    status: str
    delayed_order_count: int
    total_tardiness_minutes: int
    total_late_penalty_amount: int
    total_changeover_cost: int
    estimated_total_cost: int
    unscheduled_count: int
    recommendation_note: str


class PlanComparisonSummary(BaseModel):
    recommended_plan_variant_code: str
    recommendation_reason: str
    plan_rankings: list[PlanRankingItem]


class ProductionPlanningResult(BaseModel):
    request_id: str | None = None
    status: str = "COMPLETED"
    plan_results: list[PlanResult]
    recommended_plan_variant_code: str | None = None
    comparison_summary: PlanComparisonSummary
    warnings: list[str] = Field(default_factory=list)
    solver_metadata: dict[str, Any]


@dataclass(frozen=True)
class NormalizedOrder:
    order: OrderInput
    due_date_offset: int
    priority: int
    normalized_amount: int


@dataclass(frozen=True)
class NormalizedLine:
    line: ProductionLineInput
    available_from_offset: int
    available_to_offset: int


@dataclass(frozen=True)
class ProcessingCandidate:
    order_id: str
    product_id: str
    line_id: str
    duration_minutes: int
    priority_rank: int
    planned_production_quantity: int


@dataclass(frozen=True)
class ExistingScheduleBlock:
    schedule: ExistingScheduleInput
    start_offset: int
    end_offset: int
    duration_minutes: int


@dataclass(frozen=True)
class AmountReferenceData:
    order_amount_by_order_id: dict[str, int]
    normalized_amount_by_order_id: dict[str, int]
    priority_by_order_id: dict[str, int]
    high_amount_order_ids: set[str]
    customer_priority_by_order_id: dict[str, int]
    margin_amount_by_order_id: dict[str, int]
    warnings: list[str]


@dataclass(frozen=True)
class AmountObjectiveTerms:
    total_scheduled_amount: Any
    amount_weighted_tardiness: Any
    unscheduled_amount_penalty: Any
    high_amount_delay_penalty: Any
    total_changeover_cost: Any
    total_line_priority_penalty: Any
    makespan: Any
    total_tardiness: Any


@dataclass
class NormalizedPlanningData:
    planning_start: datetime
    planning_end: datetime
    horizon_minutes: int
    orders: dict[str, NormalizedOrder]
    products: dict[str, ProductInput]
    lines: dict[str, NormalizedLine]
    capabilities: dict[tuple[str, str], ProductLineCapabilityInput]
    candidates_by_order_id: dict[str, list[ProcessingCandidate]]
    candidates_by_line_id: dict[str, list[ProcessingCandidate]]
    existing_schedule_blocks_by_line_id: dict[str, list[ExistingScheduleBlock]]
    materials: dict[str, MaterialInput]
    bom_items_by_product_id: dict[str, list[BomItemInput]]
    changeover_rules: dict[tuple[str, str, str | None], ChangeoverRuleInput]
    warnings: list[str] = field(default_factory=list)


@dataclass
class RawSolverResult:
    plan_variant_code: str
    status: str
    objective_value: float | None
    wall_time_seconds: float
    conflicts: int
    branches: int
    response_stats: str
    solver: Any
