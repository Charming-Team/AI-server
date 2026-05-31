from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from app.features.production_planning.config import SimulationConfig
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


def sample_duration_minutes(
    planned_duration_minutes: int,
    distributions: SamplingDistributions,
    rng: Random,
) -> int:
    """
    Parameters:
        - planned_duration_minutes: Deterministic CP-SAT duration for a schedule item.
        - distributions: Distribution parameters used by the simulation.
        - rng: Iteration-local random generator.

    Methodology:
        - Apply symmetric uniform variation around the planned duration.
        - Clamp to at least one minute so the DES clock always advances.

    Output:
        - Sampled production duration in whole minutes.
    """
    spread = distributions.duration_variation_ratio
    multiplier = rng.uniform(1.0 - spread, 1.0 + spread)
    return max(1, round(planned_duration_minutes * multiplier))


def sample_changeover_minutes(
    standard_changeover_minutes: int,
    distributions: SamplingDistributions,
    rng: Random,
) -> int:
    """
    Parameters:
        - standard_changeover_minutes: Changeover minutes from planning rules.
        - distributions: Distribution parameters used by the simulation.
        - rng: Iteration-local random generator.

    Methodology:
        - Apply uniform uncertainty around the standard changeover time.
        - Return zero for same-product transitions or missing standard changeover.

    Output:
        - Sampled changeover gap in minutes.
    """
    if standard_changeover_minutes <= 0:
        return 0
    spread = distributions.changeover_variation_ratio
    return max(0, round(standard_changeover_minutes * rng.uniform(1.0 - spread, 1.0 + spread)))


def sample_operational_delay_minutes(
    distributions: SamplingDistributions,
    rng: Random,
) -> int:
    """
    Parameters:
        - distributions: Distribution parameters used by the simulation.
        - rng: Iteration-local random generator.

    Methodology:
        - Draw a Bernoulli delay event and then a bounded uniform delay duration.
        - This represents non-material operational uncertainty until historical cause samples
          are connected.

    Output:
        - Sampled delay minutes; zero when no delay event occurs.
    """
    if rng.random() > distributions.delay_probability:
        return 0
    return rng.randint(distributions.min_delay_minutes, distributions.max_delay_minutes)


def sample_yield_rate(distributions: SamplingDistributions, rng: Random) -> float:
    """
    Parameters:
        - distributions: Distribution parameters used by the simulation.
        - rng: Iteration-local random generator.

    Methodology:
        - Draw a bounded normal yield sample around the configured fallback mean.

    Output:
        - Yield rate in the inclusive range [0, 1].
    """
    return min(
        1.0,
        max(0.0, rng.gauss(distributions.yield_rate_mean, distributions.yield_rate_stddev)),
    )


def sample_defect_rate(distributions: SamplingDistributions, rng: Random) -> float:
    """
    Parameters:
        - distributions: Distribution parameters used by the simulation.
        - rng: Iteration-local random generator.

    Methodology:
        - Draw a bounded normal defect-rate sample around the configured fallback mean.

    Output:
        - Defect rate in the inclusive range [0, 1].
    """
    return min(
        1.0,
        max(0.0, rng.gauss(distributions.defect_rate_mean, distributions.defect_rate_stddev)),
    )
