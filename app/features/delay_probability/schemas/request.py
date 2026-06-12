# app/features/delay_probability/schemas/request.py
"""
Request schemas for delay probability prediction API.

Spring Back-end -> AI-server FastAPI 요청 DTO입니다.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DelayProbabilityPredictRequest(BaseModel):
    """
    주문 1건의 지연 확률 예측 요청입니다.

    예상 요청 예시:
    {
      "orderId": 314,
      "topN": 5
    }
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "orderId": 314,
                "topN": 5,
            }
        },
    )

    order_id: int = Field(
        alias="orderId",
        ge=1,
        description="지연 확률 예측 대상 주문 ID",
    )

    plan_id: int | None = Field(
        default=None,
        alias="planId",
        ge=1,
        description=(
            "특정 생산계획 기준으로 예측할 때 사용하는 plan ID. "
            "주문 기준 최신/대표 계획을 사용할 경우 null."
        ),
    )

    top_n: int = Field(
        default=5,
        alias="topN",
        ge=1,
        le=20,
        description="응답에 포함할 SHAP 상위 요인 개수",
    )


__all__ = [
    "DelayProbabilityPredictRequest",
]