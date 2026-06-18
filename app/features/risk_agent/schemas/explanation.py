from __future__ import annotations

import re

from pydantic import Field, field_validator

from app.features.risk_agent.schemas.common import CamelModel


class RiskExplanationDraft(CamelModel):
    analysis_summary: str = Field(
        min_length=30,
        max_length=1500,
    )
    recommended_action: str = Field(
        min_length=20,
        max_length=1500,
    )

    @field_validator("analysis_summary", mode="before")
    @classmethod
    def normalize_analysis_summary(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("분석 요약은 문자열이어야 합니다.")

        normalized = " ".join(value.split())

        if not normalized:
            raise ValueError("분석 요약은 비어 있을 수 없습니다.")

        return normalized

    @field_validator("recommended_action", mode="before")
    @classmethod
    def normalize_recommended_action(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("권고 조치는 문자열이어야 합니다.")

        normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()

        normalized = re.sub(
            r"\s+(?=\d+\)\s*)",
            "\n",
            normalized,
        )

        lines = [
            " ".join(line.split())
            for line in normalized.split("\n")
            if line.strip()
        ]

        normalized = "\n".join(lines)

        if not normalized:
            raise ValueError("권고 조치는 비어 있을 수 없습니다.")

        return normalized