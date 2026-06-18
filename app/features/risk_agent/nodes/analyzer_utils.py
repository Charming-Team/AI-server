from __future__ import annotations

import json
import math
from decimal import Decimal
from typing import Any, Iterable


def as_float(
    value: Decimal | int | float | str | None,
    default: float = 0.0,
) -> float:
    if value is None:
        return default

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default

    return converted if math.isfinite(converted) else default


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def unique_texts(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def parse_ml_factors(raw_json: str | None) -> list[dict[str, Any]]:
    if not raw_json:
        return []

    try:
        payload = json.loads(raw_json)
    except (TypeError, json.JSONDecodeError):
        return []

    if not isinstance(payload, dict):
        return []

    factors = payload.get("risk_increase_factors")

    if not isinstance(factors, list) or not factors:
        factors = payload.get("top_factors")

    if not isinstance(factors, list):
        return []

    return [
        factor
        for factor in factors
        if isinstance(factor, dict)
    ]


def ml_cause_boost(
    raw_json: str | None,
    accepted_tags: set[str],
) -> float:
    relevant_impacts: list[float] = []

    for factor in parse_ml_factors(raw_json):
        cause_tag = str(factor.get("cause_tag") or "").upper()
        direction = str(factor.get("direction") or "").lower()

        if cause_tag not in accepted_tags:
            continue

        if direction == "decrease":
            continue

        impact = as_float(
            factor.get("abs_impact"),
            abs(as_float(factor.get("impact"))),
        )
        relevant_impacts.append(impact)

    if not relevant_impacts:
        return 0.0

    normalized_impact = clamp(max(relevant_impacts) / 2.0)

    return clamp(0.05 + normalized_impact * 0.15)


def ml_factor_evidence(
    raw_json: str | None,
    accepted_tags: set[str],
) -> list[str]:
    evidence: list[str] = []

    for factor in parse_ml_factors(raw_json):
        cause_tag = str(factor.get("cause_tag") or "").upper()
        direction = str(factor.get("direction") or "").lower()

        if cause_tag not in accepted_tags or direction == "decrease":
            continue

        name = (
            factor.get("feature_name_ko")
            or factor.get("feature")
            or cause_tag
        )
        value = factor.get("feature_value")
        impact = factor.get("impact")

        evidence.append(
            f"ML SHAP: {name}={value}, impact={impact}"
        )

    return evidence