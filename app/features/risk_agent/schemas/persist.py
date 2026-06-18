from __future__ import annotations

from pydantic import Field, field_validator

from app.features.risk_agent.schemas.common import (
    CamelModel,
    DelayCauseType,
)


class RiskAgentPersistRequest(CamelModel):
    workflow_run_id: str = Field(
        min_length=1,
        max_length=200,
    )
    prediction_id: int = Field(gt=0)
    order_id: int = Field(gt=0)

    analysis_summary: str = Field(
        min_length=30,
        max_length=1500,
    )
    recommended_action: str = Field(
        min_length=20,
        max_length=1500,
    )

    cause_types: list[DelayCauseType] = Field(
        min_length=1,
        max_length=3,
    )

    @field_validator("cause_types")
    @classmethod
    def validate_unique_cause_types(
        cls,
        value: list[DelayCauseType],
    ) -> list[DelayCauseType]:
        if len(set(value)) != len(value):
            raise ValueError(
                "causeTypes에는 중복 원인을 넣을 수 없습니다."
            )

        return value