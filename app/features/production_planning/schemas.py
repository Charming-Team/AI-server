from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.features.production_planning.config import SimulationConfig, SolverConfig

PlanFamily = Literal["DUE_DATE_OPTIMAL", "AMOUNT_OPTIMAL"]

PlanVariantCode = Literal[
    "DUE_DATE_OPTIMAL",
    "AMOUNT_OPTIMAL",
]

PLAN_VARIANTS: list[str] = [
    "DUE_DATE_OPTIMAL",
    "AMOUNT_OPTIMAL",
]

PLAN_VARIANT_FAMILIES: dict[str, str] = {
    "DUE_DATE_OPTIMAL": "DUE_DATE_OPTIMAL",
    "AMOUNT_OPTIMAL": "AMOUNT_OPTIMAL",
}

PLAN_VARIANT_NAMES: dict[str, str] = {
    "DUE_DATE_OPTIMAL": "Due-Date Optimal",
    "AMOUNT_OPTIMAL": "Amount Optimal",
}

DB_BIGINT_MIN = -(2**63)
DB_BIGINT_MAX = 2**63 - 1
DbBigInt = Annotated[
    int,
    Field(ge=DB_BIGINT_MIN, le=DB_BIGINT_MAX, strict=True),
]
MoneyNumeric = Annotated[
    Decimal,
    Field(ge=Decimal("0"), max_digits=15, decimal_places=2),
]


def parse_db_datetime(value):
    """
    Parameters:
        - value: Datetime value from Python, ISO JSON, or DB-style text.

    Methodology:
        - Accept PostgreSQL-style strings such as "2026-04-14 09:00:00.000 +0900".
        - Leave non-string values unchanged so Pydantic can handle native datetimes.

    Output:
        - Datetime-compatible value for Pydantic validation.
    """
    if not isinstance(value, str):
        return value
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass
    normalized = value.replace("Z", "+00:00")
    if len(normalized) >= 6 and normalized[-6] == " " and normalized[-5] in {"+", "-"}:
        normalized = f"{normalized[:-6]}{normalized[-5:]}"
    return datetime.fromisoformat(normalized)


class LockedPlanInput(BaseModel):
    line_id: str
    planned_start_at: datetime
    planned_end_at: datetime

    @field_validator("planned_start_at", "planned_end_at", mode="before")
    @classmethod
    def parse_locked_datetimes(cls, value):
        return parse_db_datetime(value)

    @field_validator("line_id", mode="before")
    @classmethod
    def normalize_line_id(cls, value) -> str:
        return str(value)


class LockedPlanPatchInput(BaseModel):
    line_id: DbBigInt
    planned_start_at: datetime
    planned_end_at: datetime

    @field_validator("planned_start_at", "planned_end_at", mode="before")
    @classmethod
    def parse_locked_datetimes(cls, value):
        return parse_db_datetime(value)


class OrderInput(BaseModel):
    order_id: str
    order_no: str | None = None
    product_id: str
    customer_id: str | None = None
    order_quantity: int | None = None
    quantity: int | None = None
    due_date: datetime
    contract_amount: int | None = None
    order_amount: int | None = None
    late_penalty_amount: int = 0
    priority: int | None = None
    order_status: str | None = None
    status: str | None = None
    is_locked: bool = False
    locked_plan: LockedPlanInput | None = None

    @field_validator("order_id", "product_id", mode="before")
    @classmethod
    def normalize_ids(cls, value) -> str:
        return str(value)

    @model_validator(mode="after")
    def sync_compatible_fields(self) -> OrderInput:
        if self.order_quantity is None and self.quantity is None:
            raise ValueError("order_quantity or quantity is required.")
        if self.order_quantity is None:
            self.order_quantity = self.quantity
        if self.quantity is None:
            self.quantity = self.order_quantity
        if self.order_amount is None and self.contract_amount is None:
            raise ValueError("order_amount or contract_amount is required.")
        if self.order_amount is None:
            self.order_amount = self.contract_amount
        if self.contract_amount is None:
            self.contract_amount = self.order_amount
        if self.status is None:
            self.status = self.order_status
        if self.order_status is None:
            self.order_status = self.status
        return self


class PlanningOrderPatchInput(BaseModel):
    order_id: DbBigInt
    order_no: str | None = None
    product_id: DbBigInt
    order_quantity: int
    due_date: datetime
    contract_amount: MoneyNumeric
    late_penalty_amount: MoneyNumeric = Decimal("0.00")
    order_status: str | None = None
    locked_plan: LockedPlanPatchInput | None = None

    @field_validator("due_date", mode="before")
    @classmethod
    def parse_due_date(cls, value):
        return parse_db_datetime(value)


class ProductionPlanningAdjustmentRequest(BaseModel):
    planning_start: datetime
    planning_end: datetime
    edit_orders: list[PlanningOrderPatchInput] = Field(default_factory=list)
    add_orders: list[PlanningOrderPatchInput] = Field(default_factory=list)

    @field_validator("planning_start", "planning_end", mode="before")
    @classmethod
    def parse_planning_window(cls, value):
        return parse_db_datetime(value)


class OperatorInput(BaseModel):
    operator_id: DbBigInt
    operator_name: str | None = None
    is_active: bool = True


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
    operators: list[OperatorInput] = Field(default_factory=list)
    solver_config: SolverConfig = Field(default_factory=SolverConfig)
    simulation_config: SimulationConfig = Field(default_factory=SimulationConfig)


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
    total_material_shortage_quantity: int = 0
    total_material_shortage_penalty_amount: int = 0
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


class PlanSimulationResult(BaseModel):
    plan_variant_code: str
    plan_family: str
    iterations: int
    delay_probability: float
    expected_delayed_order_count: float
    expected_tardiness_minutes: float
    p50_tardiness_minutes: float
    p90_tardiness_minutes: float
    p95_tardiness_minutes: float
    expected_late_penalty_amount: float
    p95_late_penalty_amount: float
    expected_changeover_cost: float
    expected_material_shortage_penalty_amount: float
    expected_total_risk_cost: float
    material_shortage_probability: float
    expected_material_shortage_count: float
    expected_material_shortage_quantity: float
    expected_yield_rate: float | None
    expected_defect_quantity: float | None
    top_delay_causes: list[dict[str, Any]]
    order_delay_probabilities: dict[str, float] = Field(default_factory=dict)
    order_duration_estimates: list[dict[str, Any]] = Field(default_factory=list)
    sampling_summary: dict[str, Any] = Field(default_factory=dict)
    event_summary: list[dict[str, Any]] = Field(default_factory=list)
    event_timeline: list[dict[str, Any]] = Field(default_factory=list)
    scenario_summary: dict[str, Any]


class SimulationRankingItem(BaseModel):
    rank: int
    plan_variant_code: str
    plan_family: str
    delay_probability: float
    expected_delayed_order_count: float
    expected_total_risk_cost: float
    material_shortage_probability: float
    recommendation_note: str


class SimulationComparisonSummary(BaseModel):
    due_date_recommended_plan_variant_code: str
    cost_recommended_plan_variant_code: str
    due_date_rankings: list[SimulationRankingItem]
    cost_rankings: list[SimulationRankingItem]


class FinalPlanRecommendation(BaseModel):
    plan_variant_code: str
    plan_family: str
    recommendation_reason: str
    simulation_metrics: PlanSimulationResult
    schedule_items: list[ScheduleItem]
    risk_summary: dict[str, Any]


class AdjustedPlanRow(BaseModel):
    order_id: str
    product_id: str
    line_id: str
    operator_id: int | None
    planned_start_at: datetime
    planned_end_at: datetime
    estimated_duration_hr: float
    planned_quantity: int
    plan_sequence: int
    plan_status: str
    updated_at: datetime


class AdjustedPlanCandidate(BaseModel):
    plan_variant_code: str
    plan_variant_name: str
    status: str
    plans: list[AdjustedPlanRow]


class ProductionPlanningResult(BaseModel):
    request_id: str | None = None
    status: str = "COMPLETED"
    adjusted_plan_candidates: list[AdjustedPlanCandidate] = Field(default_factory=list)
    dashboard_response: dict[str, Any] | None = None
    plan_results: list[PlanResult]
    recommended_plan_variant_code: str | None = None
    comparison_summary: PlanComparisonSummary
    simulation_results: list[PlanSimulationResult] = Field(default_factory=list)
    simulation_comparison_summary: SimulationComparisonSummary | None = None
    recommended_due_date_plan: FinalPlanRecommendation | None = None
    recommended_cost_plan: FinalPlanRecommendation | None = None
    warnings: list[str] = Field(default_factory=list)
    solver_metadata: dict[str, Any]

    def to_adjusted_plan_response(self) -> dict[str, Any]:
        """
        Parameters:
            - None.

        Methodology:
            - Serialize only the production_plans-compatible adjusted plan candidates.
            - Leave internal solver, comparison, and simulation fields out of the API payload.

        Output:
            - Dictionary shaped as {"adjusted_plan_candidates": [...]}.
        """
        return {
            "adjusted_plan_candidates": [
                _serialize_adjusted_plan_candidate(candidate)
                for candidate in self.adjusted_plan_candidates
            ]
        }


def _serialize_adjusted_plan_candidate(
    candidate: AdjustedPlanCandidate,
) -> dict[str, Any]:
    payload = candidate.model_dump(mode="json")
    payload["plans"] = [
        _serialize_adjusted_plan_row(row)
        for row in candidate.plans
    ]
    return payload


def _serialize_adjusted_plan_row(row: AdjustedPlanRow) -> dict[str, Any]:
    payload = row.model_dump(mode="json")
    for field_name in ("order_id", "product_id", "line_id", "operator_id"):
        payload[field_name] = _public_numeric_id(payload.get(field_name))
    return payload


def _public_numeric_id(value: Any) -> Any:
    if value is None:
        return None
    text = str(value)
    if text.startswith("PLAN-") and text.removeprefix("PLAN-").isdigit():
        return int(text.removeprefix("PLAN-"))
    if text.isdigit():
        return int(text)
    return value


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
    unscheduled_contract_amount: Any
    late_penalty_amount: Any
    total_cleaning_cost: Any
    line_change_cost: Any
    different_product_sequence_count: Any
    total_tardiness_minutes: Any
    makespan_minutes: Any


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
