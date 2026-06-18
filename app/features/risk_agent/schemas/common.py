from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


def to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )


class RiskLevel(StrEnum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class DelayCauseType(StrEnum):
    MATERIAL_SHORTAGE = "MATERIAL_SHORTAGE"
    MATERIAL_DELAY = "MATERIAL_DELAY"
    LOW_YIELD = "LOW_YIELD"
    MACHINE_ABNORMAL = "MACHINE_ABNORMAL"
    LINE_ABNORMAL = "LINE_ABNORMAL"


class AnalyzerName(StrEnum):
    MATERIAL = "MATERIAL"
    YIELD = "YIELD"
    MACHINE = "MACHINE"
    LINE_PROCESS = "LINE_PROCESS"
    DUE_IMPACT = "DUE_IMPACT"


class WorkflowStatus(StrEnum):
    CREATED = "CREATED"
    CONTEXT_LOADED = "CONTEXT_LOADED"
    ANALYZING = "ANALYZING"
    ANALYZED = "ANALYZED"
    RANKED = "RANKED"
    GENERATED = "GENERATED"
    VALIDATED = "VALIDATED"
    PERSISTED = "PERSISTED"
    SKIPPED_SAFE = "SKIPPED_SAFE"
    FAILED = "FAILED"


class ConfidenceLevel(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"