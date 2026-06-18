from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Generic, TypeVar

from pydantic import Field

from app.features.risk_agent.schemas.common import CamelModel, RiskLevel


class MaterialEvidence(CamelModel):
    material_id: int
    material_code: str | None = None
    material_name: str | None = None
    material_type: str | None = None
    unit: str | None = None

    required_quantity: Decimal | None = None
    reserved_quantity: Decimal | None = None
    consumed_quantity: Decimal | None = None
    shortage_quantity: Decimal | None = None
    material_plan_status: str | None = None

    current_inventory_quantity: Decimal | None = None
    available_inventory_quantity: Decimal | None = None
    inventory_reserved_quantity: Decimal | None = None
    safety_stock_quantity: Decimal | None = None
    expected_inbound_at: datetime | None = None
    expected_inbound_quantity: Decimal | None = None
    inventory_status: str | None = None


class MachineEvidence(CamelModel):
    machine_id: int
    machine_code: str | None = None
    machine_name: str | None = None
    machine_type: str | None = None
    machine_role: str | None = None
    machine_order: int | None = None

    operation_status: str | None = None
    recorded_at: datetime | None = None
    processed_quantity: int | None = None
    defect_quantity: int | None = None
    status_note: str | None = None


class LineQueueOrderEvidence(CamelModel):
    order_id: int
    order_no: str | None = None
    plan_id: int
    plan_sequence: int | None = None
    planned_start_at: datetime | None = None
    planned_end_at: datetime | None = None
    planned_quantity: int | None = None
    completed_quantity: int | None = None
    remaining_quantity: int | None = None
    due_date: date | None = None
    plan_status: str | None = None


class RiskAgentEvidence(CamelModel):
    prediction_id: int
    order_id: int
    order_no: str
    customer_name: str | None = None

    product_id: int
    product_code: str | None = None
    product_name: str | None = None

    order_quantity: int
    completed_quantity: int
    remaining_quantity: int
    progress_rate: Decimal

    order_date: date | None = None
    due_date: date
    days_until_due: int | None = None

    contract_amount: Decimal | None = None
    late_penalty_amount: Decimal | None = None

    risk_level: RiskLevel
    delay_probability: Decimal
    predicted_delay_days: Decimal | None = None
    predicted_at: datetime
    ml_cause_detail_json: str | None = None

    plan_id: int | None = None
    plan_status: str | None = None
    planned_start_at: datetime | None = None
    planned_end_at: datetime | None = None
    planned_quantity: int | None = None
    estimated_duration_hr: Decimal | None = None
    plan_sequence: int | None = None

    line_id: int | None = None
    line_code: str | None = None
    line_name: str | None = None
    line_max_capacity_per_day: int | None = None
    line_load_ratio: Decimal | None = None

    line_operation_status: str | None = None
    line_throughput_rate: Decimal | None = None
    line_yield_rate: Decimal | None = None
    line_waiting_quantity: int | None = None
    line_waiting_time_hr: Decimal | None = None
    line_utilization_rate: Decimal | None = None

    actual_yield_rate: Decimal | None = None
    defect_quantity: int | None = None

    materials: list[MaterialEvidence] = Field(default_factory=list)
    machines: list[MachineEvidence] = Field(default_factory=list)
    competing_orders: list[LineQueueOrderEvidence] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)

    evidence_captured_at: datetime


T = TypeVar("T")


class SpringEnvelope(CamelModel, Generic[T]):
    success: bool
    code: str | None = None
    message: str | None = None
    data: T | None = None