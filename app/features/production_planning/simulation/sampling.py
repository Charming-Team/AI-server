from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from random import Random
from typing import Any

from app.features.production_planning.config import SimulationConfig
from app.features.production_planning.repositories.simulation_data_repository import (
    SimulationInputBundle,
)
from app.features.production_planning.schemas import NormalizedPlanningData, PlanResult


@dataclass(frozen=True)
class SamplingDistributions:
    duration_variation_ratio: float
    changeover_variation_ratio: float
    delay_probability: float
    min_delay_minutes: int
    max_delay_minutes: int
    yield_rate_mean: float
    yield_rate_stddev: float
    defect_rate_mean: float
    defect_rate_stddev: float
    duration_params_by_context: dict[tuple[str, str], tuple[float, float]] = field(
        default_factory=dict
    )
    yield_params_by_context: dict[tuple[str, str], tuple[float, float]] = field(
        default_factory=dict
    )
    defect_params_by_context: dict[tuple[str, str], tuple[float, float]] = field(
        default_factory=dict
    )
    changeover_params_by_context: dict[tuple[str, str, str], tuple[float, float]] = field(
        default_factory=dict
    )
    delay_params_by_context: dict[tuple[str, str], tuple[float, float, float]] = field(
        default_factory=dict
    )
    cause_weights_by_context: dict[tuple[str, str], dict[str, float]] = field(
        default_factory=dict
    )
    warnings: list[str] = field(default_factory=list)


def build_sampling_distributions(
    plan_results: list[PlanResult],
    data: NormalizedPlanningData,
    config: SimulationConfig,
) -> SamplingDistributions:
    """
    Parameters:
        - plan_results: Six CP-SAT candidate plans that will be simulated.
        - data: Normalized planning data used for fallback grouping context.
        - config: Simulation configuration containing fallback distribution parameters.

    Methodology:
        - Build an explicit fallback distribution bundle when historical sampling rows are
          unavailable in the in-memory planning request path.
        - Keep the object deterministic and side-effect free so a DB-backed sampling repository
          can replace the fallback values later.

    Output:
        - SamplingDistributions used by the DES runner.
    """
    warnings = []
    if len(plan_results) != 6:
        warnings.append(f"Expected 6 plan candidates, received {len(plan_results)}.")
    if not data.lines:
        warnings.append("No active line state is available for simulation.")
    warnings.append(
        "Historical sampling views are not loaded in this path; "
        "using configured fallback distributions."
    )
    return SamplingDistributions(
        duration_variation_ratio=config.duration_variation_ratio,
        changeover_variation_ratio=config.changeover_variation_ratio,
        delay_probability=config.delay_probability,
        min_delay_minutes=config.min_delay_minutes,
        max_delay_minutes=config.max_delay_minutes,
        yield_rate_mean=config.yield_rate_mean,
        yield_rate_stddev=config.yield_rate_stddev,
        defect_rate_mean=config.defect_rate_mean,
        defect_rate_stddev=config.defect_rate_stddev,
        warnings=warnings,
    )


def build_empirical_sampling_distributions(
    raw_data: SimulationInputBundle,
    data: NormalizedPlanningData,
    config: SimulationConfig,
) -> SamplingDistributions:
    """
    Parameters:
        - raw_data: Historical rows from simulation sampling views.
        - data: Normalized planning data used to enumerate current product-line contexts.
        - config: Simulation configuration containing fallback values and min sample count.

    Methodology:
        - Fit empirical parameters for every current product-line context using the configured
          fallback hierarchy.
        - Populate exact context keys even when the data came from broader product, line,
          category, or global history.
        - Preserve global fallback scalar parameters for sparse or missing contexts.

    Output:
        - SamplingDistributions with empirical context dictionaries and fallback values.
    """
    warnings = list(raw_data.warnings)
    contexts = _current_product_line_contexts(data)
    duration_params = {}
    yield_params = {}
    defect_params = {}
    delay_params = {}
    cause_weights = {}

    for product_id, line_id, product_category in contexts:
        rows = _collect_with_fallback(
            raw_data.production_results,
            product_id,
            line_id,
            product_category,
            config.min_samples,
        )
        if rows:
            context_key = (product_id, line_id)
            duration_params[context_key] = _fit_duration_params(rows)
            yield_params[context_key] = _fit_yield_params(
                rows,
                config.yield_rate_mean,
                config.yield_rate_stddev,
            )
            defect_params[context_key] = _fit_defect_params(
                rows,
                config.defect_rate_mean,
                config.defect_rate_stddev,
            )
            delay_params[context_key] = _fit_delay_params(
                rows,
                config.min_delay_minutes,
                config.max_delay_minutes,
            )
        else:
            warnings.append(
                f"Insufficient production history for product={product_id}, line={line_id}; "
                "using fallback simulation scalars."
            )

        cause_rows = _collect_with_fallback(
            raw_data.production_result_causes,
            product_id,
            line_id,
            product_category,
            config.min_samples,
        )
        if cause_rows:
            cause_weights[(product_id, line_id)] = _build_cause_weights(cause_rows)

    changeover_params = _build_changeover_params_by_context(
        raw_data.changeover_sequences,
        data,
        config.min_samples,
        warnings,
    )
    if raw_data.machine_status_history:
        warnings.append(
            "v_machine_status_history_for_sampling was loaded for diagnostics only and "
            "excluded from Monte Carlo sampling."
        )

    return SamplingDistributions(
        duration_variation_ratio=config.duration_variation_ratio,
        changeover_variation_ratio=config.changeover_variation_ratio,
        delay_probability=config.delay_probability,
        min_delay_minutes=config.min_delay_minutes,
        max_delay_minutes=config.max_delay_minutes,
        yield_rate_mean=config.yield_rate_mean,
        yield_rate_stddev=config.yield_rate_stddev,
        defect_rate_mean=config.defect_rate_mean,
        defect_rate_stddev=config.defect_rate_stddev,
        duration_params_by_context=duration_params,
        yield_params_by_context=yield_params,
        defect_params_by_context=defect_params,
        changeover_params_by_context=changeover_params,
        delay_params_by_context=delay_params,
        cause_weights_by_context=cause_weights,
        warnings=warnings,
    )


def sample_duration_minutes(
    planned_duration_minutes: int,
    distributions: SamplingDistributions,
    rng: Random,
    *,
    product_id: str | None = None,
    line_id: str | None = None,
) -> int:
    """
    Parameters:
        - planned_duration_minutes: Deterministic CP-SAT duration for a schedule item.
        - distributions: Distribution parameters used by the simulation.
        - rng: Iteration-local random generator.
        - product_id: Optional product context for empirical lookup.
        - line_id: Optional line context for empirical lookup.

    Methodology:
        - Use fitted empirical duration ratios for the product-line context when available.
        - Fall back to symmetric uniform variation around the planned duration.

    Output:
        - Sampled production duration in whole minutes.
    """
    if product_id is not None and line_id is not None:
        params = distributions.duration_params_by_context.get((product_id, line_id))
        if params is not None:
            mean_ratio, stddev_ratio = params
            ratio = max(0.1, rng.gauss(mean_ratio, stddev_ratio))
            return max(1, round(planned_duration_minutes * ratio))
    spread = distributions.duration_variation_ratio
    multiplier = rng.uniform(1.0 - spread, 1.0 + spread)
    return max(1, round(planned_duration_minutes * multiplier))


def sample_changeover_minutes(
    standard_changeover_minutes: int,
    distributions: SamplingDistributions,
    rng: Random,
    *,
    from_product_id: str | None = None,
    to_product_id: str | None = None,
    line_id: str | None = None,
) -> int:
    """
    Parameters:
        - standard_changeover_minutes: Changeover minutes from planning rules.
        - distributions: Distribution parameters used by the simulation.
        - rng: Iteration-local random generator.
        - from_product_id: Optional previous product context.
        - to_product_id: Optional next product context.
        - line_id: Optional line context.

    Methodology:
        - Use fitted empirical changeover ratios for the transition context when available.
        - Return zero for same-product transitions or missing standard changeover.

    Output:
        - Sampled changeover gap in minutes.
    """
    if standard_changeover_minutes <= 0:
        return 0
    if from_product_id is not None and to_product_id is not None and line_id is not None:
        params = distributions.changeover_params_by_context.get(
            (from_product_id, to_product_id, line_id)
        )
        if params is not None:
            mean_ratio, stddev_ratio = params
            ratio = max(0.0, rng.gauss(mean_ratio, stddev_ratio))
            return max(0, round(standard_changeover_minutes * ratio))
    spread = distributions.changeover_variation_ratio
    return max(0, round(standard_changeover_minutes * rng.uniform(1.0 - spread, 1.0 + spread)))


def sample_operational_delay_minutes(
    distributions: SamplingDistributions,
    rng: Random,
    *,
    product_id: str | None = None,
    line_id: str | None = None,
) -> int:
    """
    Parameters:
        - distributions: Distribution parameters used by the simulation.
        - rng: Iteration-local random generator.
        - product_id: Optional product context for empirical lookup.
        - line_id: Optional line context for empirical lookup.

    Methodology:
        - Draw delay probability and delay duration from empirical context parameters when
          available.
        - Fall back to configured Bernoulli and bounded uniform delay policy.

    Output:
        - Sampled delay minutes; zero when no delay event occurs.
    """
    probability = distributions.delay_probability
    min_minutes = distributions.min_delay_minutes
    max_minutes = distributions.max_delay_minutes
    if product_id is not None and line_id is not None:
        params = distributions.delay_params_by_context.get((product_id, line_id))
        if params is not None:
            probability, min_minutes, max_minutes = params
    if rng.random() > probability:
        return 0
    lower = round(min_minutes)
    upper = round(max_minutes)
    if upper <= lower:
        return max(0, lower)
    return rng.randint(lower, upper)


def sample_yield_rate(
    distributions: SamplingDistributions,
    rng: Random,
    *,
    product_id: str | None = None,
    line_id: str | None = None,
) -> float:
    """
    Parameters:
        - distributions: Distribution parameters used by the simulation.
        - rng: Iteration-local random generator.
        - product_id: Optional product context for empirical lookup.
        - line_id: Optional line context for empirical lookup.

    Methodology:
        - Draw a bounded normal yield sample from empirical context parameters when available.

    Output:
        - Yield rate in the inclusive range [0, 1].
    """
    mean = distributions.yield_rate_mean
    stddev = distributions.yield_rate_stddev
    if product_id is not None and line_id is not None:
        params = distributions.yield_params_by_context.get((product_id, line_id))
        if params is not None:
            mean, stddev = params
    return min(
        1.0,
        max(0.0, rng.gauss(mean, stddev)),
    )


def sample_defect_rate(
    distributions: SamplingDistributions,
    rng: Random,
    *,
    product_id: str | None = None,
    line_id: str | None = None,
) -> float:
    """
    Parameters:
        - distributions: Distribution parameters used by the simulation.
        - rng: Iteration-local random generator.
        - product_id: Optional product context for empirical lookup.
        - line_id: Optional line context for empirical lookup.

    Methodology:
        - Draw a bounded normal defect-rate sample from empirical context parameters when
          available.

    Output:
        - Defect rate in the inclusive range [0, 1].
    """
    mean = distributions.defect_rate_mean
    stddev = distributions.defect_rate_stddev
    if product_id is not None and line_id is not None:
        params = distributions.defect_params_by_context.get((product_id, line_id))
        if params is not None:
            mean, stddev = params
    return min(
        1.0,
        max(0.0, rng.gauss(mean, stddev)),
    )


def sample_cause(
    distributions: SamplingDistributions,
    rng: Random,
    *,
    product_id: str | None = None,
    line_id: str | None = None,
) -> str:
    """
    Parameters:
        - distributions: Sampling distributions containing cause weights.
        - rng: Iteration-local random generator.
        - product_id: Optional product context for empirical lookup.
        - line_id: Optional line context for empirical lookup.

    Methodology:
        - Draw a weighted random cause for the product-line context when available.
        - Use OPERATIONAL_DELAY as the stable fallback cause.

    Output:
        - Delay cause string.
    """
    if product_id is None or line_id is None:
        return "OPERATIONAL_DELAY"
    weights = distributions.cause_weights_by_context.get((product_id, line_id))
    if not weights:
        return "OPERATIONAL_DELAY"
    causes = list(weights)
    probabilities = [weights[cause] for cause in causes]
    return rng.choices(causes, weights=probabilities, k=1)[0]


def _current_product_line_contexts(
    data: NormalizedPlanningData,
) -> list[tuple[str, str, str | None]]:
    contexts = set()
    for candidate in data.candidates_by_order_id.values():
        for item in candidate:
            product = data.products.get(item.product_id)
            contexts.add((item.product_id, item.line_id, product.grade if product else None))
    return sorted(contexts)


def _collect_with_fallback(
    rows: list[dict[str, Any]],
    product_id: str,
    line_id: str,
    product_category: str | None,
    min_samples: int,
) -> list[dict[str, Any]]:
    filters = [
        lambda row: _matches(row, "product_id", product_id) and _matches(row, "line_id", line_id),
        lambda row: _matches(row, "product_id", product_id),
        lambda row: _matches(row, "line_id", line_id),
        lambda row: product_category is not None
        and _matches(row, "product_category", product_category),
        lambda row: True,
    ]
    for row_filter in filters:
        subset = [row for row in rows if row_filter(row)]
        if len(subset) >= min_samples:
            return subset
    return []


def _build_changeover_params_by_context(
    rows: list[dict[str, Any]],
    data: NormalizedPlanningData,
    min_samples: int,
    warnings: list[str],
) -> dict[tuple[str, str, str], tuple[float, float]]:
    params = {}
    transition_contexts = set()
    for candidates in data.candidates_by_line_id.values():
        products = sorted({candidate.product_id for candidate in candidates})
        for from_product_id in products:
            for to_product_id in products:
                if from_product_id != to_product_id:
                    transition_contexts.add((from_product_id, to_product_id, candidates[0].line_id))

    for from_product_id, to_product_id, line_id in sorted(transition_contexts):
        subset = _collect_changeover_with_fallback(
            rows,
            from_product_id,
            to_product_id,
            line_id,
            min_samples,
        )
        if subset:
            params[(from_product_id, to_product_id, line_id)] = _fit_changeover_ratio_params(subset)
        else:
            warnings.append(
                f"Insufficient changeover history for {from_product_id}->{to_product_id} "
                f"on line={line_id}; using fallback changeover variation."
            )
    return params


def _collect_changeover_with_fallback(
    rows: list[dict[str, Any]],
    from_product_id: str,
    to_product_id: str,
    line_id: str,
    min_samples: int,
) -> list[dict[str, Any]]:
    filters = [
        lambda row: _matches(row, "from_product_id", from_product_id)
        and _matches(row, "to_product_id", to_product_id)
        and _matches(row, "line_id", line_id),
        lambda row: _matches(row, "from_product_id", from_product_id)
        and _matches(row, "to_product_id", to_product_id),
        lambda row: _matches(row, "line_id", line_id),
        lambda row: True,
    ]
    for row_filter in filters:
        subset = [row for row in rows if row_filter(row)]
        if len(subset) >= min_samples:
            return subset
    return []


def _fit_duration_params(rows: list[dict[str, Any]]) -> tuple[float, float]:
    values = [_positive_float(row.get("actual_duration_hr")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return (1.0, 0.0)
    mean_value = _mean(values)
    return (1.0, _safe_stddev(values) / mean_value if mean_value else 0.0)


def _fit_yield_params(
    rows: list[dict[str, Any]],
    fallback_mean: float,
    fallback_stddev: float,
) -> tuple[float, float]:
    values = [_bounded_float(row.get("yield_rate"), 0.0, 1.0) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return (fallback_mean, fallback_stddev)
    return (_mean(values), _safe_stddev(values))


def _fit_defect_params(
    rows: list[dict[str, Any]],
    fallback_mean: float,
    fallback_stddev: float,
) -> tuple[float, float]:
    values = []
    for row in rows:
        actual_quantity = _positive_float(row.get("actual_quantity"))
        defect_quantity = _nonnegative_float(row.get("defect_quantity"))
        if actual_quantity and defect_quantity is not None:
            values.append(min(1.0, defect_quantity / actual_quantity))
    if not values:
        return (fallback_mean, fallback_stddev)
    return (_mean(values), _safe_stddev(values))


def _fit_delay_params(
    rows: list[dict[str, Any]],
    fallback_min_minutes: int,
    fallback_max_minutes: int,
) -> tuple[float, float, float]:
    delayed_rows = [row for row in rows if bool(row.get("is_delayed"))]
    probability = len(delayed_rows) / len(rows) if rows else 0.0
    delay_minutes = [
        value * 60
        for value in (_nonnegative_float(row.get("actual_delay_hr")) for row in delayed_rows)
        if value is not None
    ]
    if not delay_minutes:
        return (probability, float(fallback_min_minutes), float(fallback_max_minutes))
    return (probability, min(delay_minutes), max(delay_minutes))


def _fit_changeover_ratio_params(rows: list[dict[str, Any]]) -> tuple[float, float]:
    values = [_positive_float(row.get("observed_gap_minutes")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return (1.0, 0.0)
    mean_value = _mean(values)
    return (1.0, _safe_stddev(values) / mean_value if mean_value else 0.0)


def _build_cause_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for row in rows:
        cause_type = row.get("cause_type")
        if cause_type:
            cause = str(cause_type)
            counts[cause] = counts.get(cause, 0) + 1
    total = sum(counts.values())
    if not total:
        return {}
    return {cause: count / total for cause, count in counts.items()}


def _matches(row: dict[str, Any], key: str, expected: str) -> bool:
    value = row.get(key)
    return value is not None and str(value) == str(expected)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return sqrt(variance)


def _positive_float(value: Any) -> float | None:
    parsed = _nonnegative_float(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _nonnegative_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed < 0:
        return None
    return parsed


def _bounded_float(value: Any, lower: float, upper: float) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed < lower or parsed > upper:
        return None
    return parsed
