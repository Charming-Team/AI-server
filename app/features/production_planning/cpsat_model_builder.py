from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from ortools.sat.python import cp_model

from app.features.production_planning.config import SolverConfig
from app.features.production_planning.exceptions import PlanningValidationError
from app.features.production_planning.preprocessing import (
    calculate_required_material_quantity_scaled,
    get_changeover_cost,
    get_changeover_minutes,
    get_inbound_material_quantity_scaled,
    get_inbound_material_time,
    get_initial_available_material_quantity_scaled,
    to_minute_offset,
    to_minute_offset_ceil,
)
from app.features.production_planning.schemas import NormalizedPlanningData, ProcessingCandidate


@dataclass
class CpsatModelBundle:
    model: cp_model.CpModel
    order_vars: dict[str, dict[str, Any]]
    line_vars: dict[str, dict[str, Any]]
    interval_vars: dict[tuple[str, str], cp_model.IntervalVar]
    tardiness_vars: dict[str, cp_model.IntVar]
    unscheduled_vars: dict[str, cp_model.IntVar]
    changeover_vars: dict[str, list[dict[str, Any]]]
    metric_vars: dict[str, Any]
    data: NormalizedPlanningData
    warnings: list[str] = field(default_factory=list)


def build_base_cpsat_model(
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> CpsatModelBundle:
    """
    Parameters:
        - data: Normalized planning data containing integer offsets and candidate assignments.
        - config: Solver configuration used for hard-constraint defaults.

    Methodology:
        - Build the common CP-SAT model shared by due-date and amount objectives.
        - Create optional intervals per order-line candidate, no-overlap constraints per line,
          locked existing schedule intervals, material constraints, and pairwise sequencing.

    Output:
        - CpsatModelBundle containing the model, decision variables, metric references, and data.
    """
    model = cp_model.CpModel()
    bundle = CpsatModelBundle(
        model=model,
        order_vars={},
        line_vars={line_id: {"intervals": []} for line_id in data.lines},
        interval_vars={},
        tardiness_vars={},
        unscheduled_vars={},
        changeover_vars={"sequence": []},
        metric_vars={},
        data=data,
        warnings=list(data.warnings),
    )

    _build_order_assignment_variables(bundle, data, config)
    _add_locked_order_constraints(bundle, data)
    _add_line_no_overlap_constraints(bundle, data, config)
    _add_line_availability_constraints(bundle, data)
    _add_daily_capacity_constraints(bundle, data)
    _add_no_changeover_line_constraints(bundle, data)
    if config.use_material_constraints:
        _add_material_constraints(bundle, data)
    _add_changeover_constraints(bundle, data, config)
    bundle.metric_vars["optional_interval_count"] = len(bundle.interval_vars)
    bundle.metric_vars["changeover_constraint_count"] = len(bundle.changeover_vars["sequence"])
    return bundle


def _build_order_assignment_variables(
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> None:
    for order_id, normalized_order in data.orders.items():
        scheduled = bundle.model.NewBoolVar(f"scheduled_{order_id}")
        unscheduled = bundle.model.NewBoolVar(f"unscheduled_{order_id}")
        completion = bundle.model.NewIntVar(0, data.horizon_minutes, f"completion_{order_id}")
        tardiness = bundle.model.NewIntVar(0, data.horizon_minutes, f"tardiness_{order_id}")
        due_date_met = bundle.model.NewBoolVar(f"due_date_met_{order_id}")
        assigned_vars: dict[str, cp_model.IntVar] = {}
        start_vars: dict[str, cp_model.IntVar] = {}
        end_vars: dict[str, cp_model.IntVar] = {}

        for candidate in data.candidates_by_order_id.get(order_id, []):
            assigned = bundle.model.NewBoolVar(f"assigned_{order_id}_{candidate.line_id}")
            start = bundle.model.NewIntVar(
                0,
                data.horizon_minutes,
                f"start_{order_id}_{candidate.line_id}",
            )
            end = bundle.model.NewIntVar(
                0,
                data.horizon_minutes,
                f"end_{order_id}_{candidate.line_id}",
            )
            interval = bundle.model.NewOptionalIntervalVar(
                start,
                candidate.duration_minutes,
                end,
                assigned,
                f"interval_{order_id}_{candidate.line_id}",
            )
            assigned_vars[candidate.line_id] = assigned
            start_vars[candidate.line_id] = start
            end_vars[candidate.line_id] = end
            bundle.interval_vars[(order_id, candidate.line_id)] = interval
            bundle.line_vars[candidate.line_id]["intervals"].append(interval)
            bundle.model.Add(completion == end).OnlyEnforceIf(assigned)

        bundle.model.Add(sum(assigned_vars.values()) == scheduled)
        if config.allow_unscheduled_orders:
            bundle.model.Add(scheduled + unscheduled == 1)
            bundle.model.Add(completion == 0).OnlyEnforceIf(unscheduled)
        else:
            bundle.model.Add(scheduled == 1)
            bundle.model.Add(unscheduled == 0)

        due_offset = max(0, min(data.horizon_minutes, normalized_order.due_date_offset))
        bundle.model.Add(tardiness >= completion - due_offset)
        bundle.model.Add(tardiness >= 0)
        bundle.model.Add(tardiness == 0).OnlyEnforceIf(unscheduled)
        bundle.model.Add(due_date_met <= scheduled)
        bundle.model.Add(completion <= due_offset).OnlyEnforceIf(due_date_met)
        bundle.model.Add(completion >= due_offset + 1).OnlyEnforceIf(
            [scheduled, due_date_met.Not()]
        )
        if config.due_date_policy == "HARD" and not normalized_order.order.is_locked:
            # Locked edits are fixed user decisions; their lateness is measured, not repaired.
            bundle.model.Add(completion <= normalized_order.due_date_offset).OnlyEnforceIf(
                scheduled
            )

        bundle.order_vars[order_id] = {
            "scheduled": scheduled,
            "unscheduled": unscheduled,
            "completion": completion,
            "due_date_met": due_date_met,
            "assigned": assigned_vars,
            "start": start_vars,
            "end": end_vars,
        }
        bundle.tardiness_vars[order_id] = tardiness
        bundle.unscheduled_vars[order_id] = unscheduled


def _add_locked_order_constraints(
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> None:
    for order_id, normalized_order in data.orders.items():
        order = normalized_order.order
        if not order.is_locked:
            continue
        if order.locked_plan is None:
            raise PlanningValidationError(f"Locked order {order_id} requires locked_plan.")
        locked_line_id = order.locked_plan.line_id
        order_vars = bundle.order_vars[order_id]
        if locked_line_id not in order_vars["assigned"]:
            raise PlanningValidationError(
                f"Locked order {order_id} has no assignment variable on line {locked_line_id}."
            )
        locked_start = to_minute_offset(order.locked_plan.planned_start_at, data.planning_start)
        locked_end = to_minute_offset_ceil(order.locked_plan.planned_end_at, data.planning_start)

        bundle.model.Add(order_vars["scheduled"] == 1)
        bundle.model.Add(order_vars["unscheduled"] == 0)
        bundle.model.Add(order_vars["completion"] == locked_end)
        for line_id, assigned in order_vars["assigned"].items():
            if line_id == locked_line_id:
                bundle.model.Add(assigned == 1)
                bundle.model.Add(order_vars["start"][line_id] == locked_start)
                bundle.model.Add(order_vars["end"][line_id] == locked_end)
            else:
                bundle.model.Add(assigned == 0)


def _add_line_no_overlap_constraints(
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> None:
    for line_id, line_vars in bundle.line_vars.items():
        intervals = list(line_vars["intervals"])
        if config.use_existing_schedule_locks:
            for block in data.existing_schedule_blocks_by_line_id.get(line_id, []):
                intervals.append(
                    bundle.model.NewIntervalVar(
                        block.start_offset,
                        block.duration_minutes,
                        block.end_offset,
                        f"locked_{block.schedule.schedule_id}",
                    )
                )
        if intervals:
            bundle.model.AddNoOverlap(intervals)


def _add_line_availability_constraints(
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> None:
    for order_var in bundle.order_vars.values():
        for line_id, assigned in order_var["assigned"].items():
            line = data.lines[line_id]
            bundle.model.Add(
                order_var["start"][line_id] >= line.available_from_offset
            ).OnlyEnforceIf(assigned)
            bundle.model.Add(order_var["end"][line_id] <= line.available_to_offset).OnlyEnforceIf(
                assigned
            )


def _add_material_constraints(
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> None:
    for material_id, material in data.materials.items():
        required_terms = []
        for order_id, normalized_order in data.orders.items():
            quantity_required = _calculate_order_material_requirement_scaled(
                normalized_order.order,
                data,
                material_id,
            )
            if quantity_required:
                required_terms.append(quantity_required * bundle.order_vars[order_id]["scheduled"])

        if not required_terms:
            continue

        initial_available = get_initial_available_material_quantity_scaled(material)
        inbound_quantity = get_inbound_material_quantity_scaled(material)
        inbound_time = get_inbound_material_time(material)
        if inbound_time is None:
            if initial_available == 0:
                continue
            bundle.model.Add(sum(required_terms) <= initial_available)
            continue

        inbound_offset = to_minute_offset(inbound_time, data.planning_start)
        if inbound_offset <= 0:
            total = initial_available + inbound_quantity
            if total == 0:
                continue
            bundle.model.Add(sum(required_terms) <= total)
            continue
        if inbound_offset > data.horizon_minutes:
            if initial_available == 0:
                continue
            bundle.model.Add(sum(required_terms) <= initial_available)
            continue

        before_inbound_terms = []
        for order_id, normalized_order in data.orders.items():
            quantity_required = _calculate_order_material_requirement_scaled(
                normalized_order.order,
                data,
                material_id,
            )
            if not quantity_required:
                continue
            for line_id, assigned in bundle.order_vars[order_id]["assigned"].items():
                starts_before = bundle.model.NewBoolVar(
                    f"material_{material_id}_{order_id}_{line_id}_before_inbound"
                )
                bundle.model.Add(starts_before <= assigned)
                bundle.model.Add(
                    bundle.order_vars[order_id]["start"][line_id] <= inbound_offset - 1
                ).OnlyEnforceIf(starts_before)
                bundle.model.Add(
                    bundle.order_vars[order_id]["start"][line_id] >= inbound_offset
                ).OnlyEnforceIf([assigned, starts_before.Not()])
                before_inbound_terms.append(quantity_required * starts_before)

        bundle.model.Add(sum(before_inbound_terms) <= initial_available)
        bundle.model.Add(sum(required_terms) <= initial_available + inbound_quantity)


def _calculate_order_material_requirement_scaled(order, data, material_id: str) -> int:
    quantity_required = 0
    product = data.products[order.product_id]
    for bom_item in data.bom_items_by_product_id.get(order.product_id, []):
        if bom_item.material_id == material_id:
            quantity_required += calculate_required_material_quantity_scaled(
                order,
                product,
                bom_item,
            )
    return quantity_required


def _add_daily_capacity_constraints(
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> None:
    bucket_count = ceil_div(data.horizon_minutes, 24 * 60)
    for line_id, candidates in data.candidates_by_line_id.items():
        line = data.lines[line_id].line
        for bucket_index in range(bucket_count):
            line_terms = []
            for candidate in candidates:
                if not _fits_completion_bucket_capacity(
                    bundle,
                    "line",
                    candidate,
                    line.max_capacity_per_day,
                ):
                    continue
                in_bucket = _build_completion_bucket_var(bundle, candidate, bucket_index)
                line_terms.append(candidate.planned_production_quantity * in_bucket)
            if line.max_capacity_per_day is not None and line_terms:
                bundle.model.Add(sum(line_terms) <= line.max_capacity_per_day)

        for bucket_index in range(bucket_count):
            candidates_by_product: dict[str, list[ProcessingCandidate]] = {}
            for candidate in candidates:
                candidates_by_product.setdefault(candidate.product_id, []).append(candidate)
            for product_id, product_candidates in candidates_by_product.items():
                capability = data.capabilities[(product_id, line_id)]
                if capability.capacity_per_day is None:
                    continue
                terms = []
                for candidate in product_candidates:
                    if not _fits_completion_bucket_capacity(
                        bundle,
                        "product-line",
                        candidate,
                        capability.capacity_per_day,
                    ):
                        continue
                    terms.append(
                        candidate.planned_production_quantity
                        * _build_completion_bucket_var(bundle, candidate, bucket_index)
                    )
                if not terms:
                    continue
                bundle.model.Add(sum(terms) <= capability.capacity_per_day)


def _fits_completion_bucket_capacity(
    bundle: CpsatModelBundle,
    scope: str,
    candidate: ProcessingCandidate,
    capacity: int | None,
) -> bool:
    if capacity is None or candidate.planned_production_quantity <= capacity:
        return True

    skipped = bundle.metric_vars.setdefault("oversized_capacity_candidates", set())
    key = (scope, candidate.order_id, candidate.line_id)
    if key not in skipped:
        skipped.add(key)
        bundle.warnings.append(
            f"Skipped {scope} completion-bucket capacity for order {candidate.order_id} "
            f"on line {candidate.line_id}: planned quantity "
            f"{candidate.planned_production_quantity} exceeds daily capacity {capacity}."
        )
    return False


def _build_completion_bucket_var(
    bundle: CpsatModelBundle,
    candidate: ProcessingCandidate,
    bucket_index: int,
) -> cp_model.IntVar:
    metric_key = "completion_bucket_vars"
    bucket_vars = bundle.metric_vars.setdefault(metric_key, {})
    key = (candidate.order_id, candidate.line_id, bucket_index)
    if key in bucket_vars:
        return bucket_vars[key]

    assigned = bundle.order_vars[candidate.order_id]["assigned"][candidate.line_id]
    end = bundle.order_vars[candidate.order_id]["end"][candidate.line_id]
    bucket_start = bucket_index * 24 * 60
    bucket_next_start = min((bucket_index + 1) * 24 * 60, bundle.data.horizon_minutes)
    if bucket_next_start == bundle.data.horizon_minutes:
        bucket_end = bucket_next_start
    else:
        bucket_end = bucket_next_start - 1
    in_bucket = bundle.model.NewBoolVar(
        f"completion_bucket_{candidate.order_id}_{candidate.line_id}_{bucket_index}"
    )
    bundle.model.Add(in_bucket <= assigned)
    bundle.model.Add(end >= bucket_start).OnlyEnforceIf(in_bucket)
    bundle.model.Add(end <= bucket_end).OnlyEnforceIf(in_bucket)
    bucket_vars[key] = in_bucket

    candidate_bucket_keys = [
        existing_key
        for existing_key in bucket_vars
        if existing_key[0] == candidate.order_id and existing_key[1] == candidate.line_id
    ]
    if len(candidate_bucket_keys) == ceil_div(bundle.data.horizon_minutes, 24 * 60):
        bundle.model.Add(
            sum(bucket_vars[item_key] for item_key in candidate_bucket_keys) == assigned
        )
    return in_bucket


def _add_no_changeover_line_constraints(
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
) -> None:
    for line_id, line in data.lines.items():
        if line.line.supports_changeover:
            continue
        candidates = data.candidates_by_line_id.get(line_id, [])
        for left_candidate, right_candidate in combinations(candidates, 2):
            if left_candidate.product_id == right_candidate.product_id:
                continue
            bundle.model.Add(
                bundle.order_vars[left_candidate.order_id]["assigned"][line_id]
                + bundle.order_vars[right_candidate.order_id]["assigned"][line_id]
                <= 1
            )


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _add_changeover_constraints(
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    config: SolverConfig,
) -> None:
    # TODO: Replace pairwise sequencing with circuit-based sequencing if performance becomes poor.
    for line_id, candidates in data.candidates_by_line_id.items():
        candidates_by_order = {candidate.order_id: candidate for candidate in candidates}
        for left_order_id, right_order_id in combinations(sorted(candidates_by_order), 2):
            _add_pairwise_changeover_constraint(
                bundle,
                data,
                config,
                line_id,
                candidates_by_order[left_order_id],
                candidates_by_order[right_order_id],
            )


def _add_pairwise_changeover_constraint(
    bundle: CpsatModelBundle,
    data: NormalizedPlanningData,
    config: SolverConfig,
    line_id: str,
    left_candidate: ProcessingCandidate,
    right_candidate: ProcessingCandidate,
) -> None:
    left_assigned = bundle.order_vars[left_candidate.order_id]["assigned"][line_id]
    right_assigned = bundle.order_vars[right_candidate.order_id]["assigned"][line_id]
    both_assigned = bundle.model.NewBoolVar(
        f"both_{left_candidate.order_id}_{right_candidate.order_id}_{line_id}"
    )
    left_before_right = bundle.model.NewBoolVar(
        f"before_{left_candidate.order_id}_{right_candidate.order_id}_{line_id}"
    )
    right_before_left = bundle.model.NewBoolVar(
        f"before_{right_candidate.order_id}_{left_candidate.order_id}_{line_id}"
    )

    bundle.model.Add(both_assigned <= left_assigned)
    bundle.model.Add(both_assigned <= right_assigned)
    bundle.model.Add(both_assigned >= left_assigned + right_assigned - 1)
    bundle.model.Add(left_before_right + right_before_left == both_assigned)

    left_to_right_minutes = get_changeover_minutes(
        left_candidate.product_id,
        right_candidate.product_id,
        line_id,
        data.changeover_rules,
        config.default_changeover_minutes,
    )
    right_to_left_minutes = get_changeover_minutes(
        right_candidate.product_id,
        left_candidate.product_id,
        line_id,
        data.changeover_rules,
        config.default_changeover_minutes,
    )
    bundle.model.Add(
        bundle.order_vars[right_candidate.order_id]["start"][line_id]
        >= bundle.order_vars[left_candidate.order_id]["end"][line_id] + left_to_right_minutes
    ).OnlyEnforceIf(left_before_right)
    bundle.model.Add(
        bundle.order_vars[left_candidate.order_id]["start"][line_id]
        >= bundle.order_vars[right_candidate.order_id]["end"][line_id] + right_to_left_minutes
    ).OnlyEnforceIf(right_before_left)

    bundle.changeover_vars["sequence"].append(
        {
            "line_id": line_id,
            "from_order_id": left_candidate.order_id,
            "to_order_id": right_candidate.order_id,
            "before_var": left_before_right,
            "minutes": left_to_right_minutes,
            "cost": get_changeover_cost(
                left_candidate.product_id,
                right_candidate.product_id,
                line_id,
                data.changeover_rules,
                config.default_changeover_cost,
            ),
        }
    )
    bundle.changeover_vars["sequence"].append(
        {
            "line_id": line_id,
            "from_order_id": right_candidate.order_id,
            "to_order_id": left_candidate.order_id,
            "before_var": right_before_left,
            "minutes": right_to_left_minutes,
            "cost": get_changeover_cost(
                right_candidate.product_id,
                left_candidate.product_id,
                line_id,
                data.changeover_rules,
                config.default_changeover_cost,
            ),
        }
    )
