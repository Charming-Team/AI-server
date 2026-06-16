# app/features/delay_probability/schemas/response.py
"""
Response schemas for delay probability prediction API.

DB 저장 매핑:
- delay_probability -> ai_prediction_results.delay_probability
- risk_level -> ai_prediction_results.risk_level
- model_name -> ai_prediction_results.model_name
- model_version -> ai_prediction_results.model_version
- predicted_at -> ai_prediction_results.predicted_at
- cause_detail -> ai_prediction_results.cause_detail JSONB
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

RiskLevel = Literal["SAFE", "CAUTION", "WARNING", "CRITICAL"]
ImpactDirection = Literal["increase", "decrease"]
FeatureValue = str | int | float | bool | None


class ShapFactor(BaseModel):
    """
    SHAP 기반 지연 위험 영향 요인입니다.

    direction:
    - increase: 지연 위험 증가 방향
    - decrease: 지연 위험 감소 방향
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    feature: str = Field(
        min_length=1,
        description="모델 원본 feature 이름",
        examples=["capacity_load_ratio"],
    )

    feature_name_ko: str = Field(
        validation_alias=AliasChoices("feature_name_ko", "featureNameKo"),
        serialization_alias="feature_name_ko",
        min_length=1,
        description="feature 한글 표시명",
        examples=["라인 생산능력 대비 주문 부하"],
    )

    cause_tag: str = Field(
        validation_alias=AliasChoices("cause_tag", "causeTag"),
        serialization_alias="cause_tag",
        min_length=1,
        description="feature 기반 원인 태그",
        examples=["LINE_LOAD"],
    )

    feature_value: FeatureValue = Field(
        validation_alias=AliasChoices("feature_value", "featureValue"),
        serialization_alias="feature_value",
        description="예측에 사용된 해당 feature 값",
        examples=[0.6542056074766355],
    )

    impact: float = Field(
        description="SHAP impact 값. 양수면 지연 위험 증가, 음수면 감소",
        examples=[-0.3560287654399872],
    )

    abs_impact: float = Field(
        validation_alias=AliasChoices("abs_impact", "absImpact"),
        serialization_alias="abs_impact",
        ge=0,
        description="SHAP impact 절댓값",
        examples=[0.3560287654399872],
    )

    direction: ImpactDirection = Field(
        description="지연 위험 영향 방향",
        examples=["decrease"],
    )


class CauseDetail(BaseModel):
    """
    ai_prediction_results.cause_detail JSONB에 저장할 ML SHAP 분석 payload입니다.
    Risk Agent의 Risk Explanation LLM Node에서도 이 값을 재사용합니다.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    raw_delay_probability: float = Field(
        validation_alias=AliasChoices("raw_delay_probability", "rawDelayProbability"),
        serialization_alias="raw_delay_probability",
        ge=0,
        le=1,
        description="calibration 적용 전 raw 지연 확률",
        examples=[0.012722019106149673],
    )

    calibrated_delay_probability: float = Field(
        validation_alias=AliasChoices(
            "calibrated_delay_probability",
            "calibratedDelayProbability",
        ),
        serialization_alias="calibrated_delay_probability",
        ge=0,
        le=1,
        description="calibration 적용 후 지연 확률",
        examples=[0.014733265154063702],
    )

    probability_output: str = Field(
        validation_alias=AliasChoices("probability_output", "probabilityOutput"),
        serialization_alias="probability_output",
        min_length=1,
        description="확률 출력 방식",
        examples=["calibrated_sigmoid"],
    )

    top_factors: list[ShapFactor] = Field(
        validation_alias=AliasChoices("top_factors", "topFactors"),
        serialization_alias="top_factors",
        default_factory=list,
        description="절댓값 기준 상위 SHAP 요인",
    )

    risk_increase_factors: list[ShapFactor] = Field(
        validation_alias=AliasChoices("risk_increase_factors", "riskIncreaseFactors"),
        serialization_alias="risk_increase_factors",
        default_factory=list,
        description="지연 위험 증가 방향 SHAP 요인",
    )

    risk_decrease_factors: list[ShapFactor] = Field(
        validation_alias=AliasChoices("risk_decrease_factors", "riskDecreaseFactors"),
        serialization_alias="risk_decrease_factors",
        default_factory=list,
        description="지연 위험 감소 방향 SHAP 요인",
    )


class DelayProbabilityPredictResponse(BaseModel):
    """
    지연 확률 예측 API 응답 DTO입니다.

    현재 모델 반환 구조와 동일하게 snake_case로 직렬화합니다.
    Spring에서는 @JsonProperty 또는 snake_case naming strategy로 수신하면 됩니다.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        json_schema_extra={
            "example": {
                "order_id": 314,
                "product_id": 10,
                "plan_id": None,
                "line_id": 5,
                "raw_delay_probability": 0.012722019106149673,
                "delay_probability": 0.014733265154063702,
                "risk_level": "SAFE",
                "model_name": "xgboost_delay_probability",
                "model_version": "v1.0.0",
                "probability_output": "calibrated_sigmoid",
                "predicted_at": "2026-06-11T09:22:30.889312+00:00",
                "top_factors": [
                    {
                        "feature": "due_margin_to_duration_ratio_capped",
                        "feature_name_ko": "생산 소요시간 대비 납기 여유 비율",
                        "cause_tag": "DUE_MARGIN_RISK",
                        "feature_value": 3.0,
                        "impact": -1.798938512802124,
                        "abs_impact": 1.798938512802124,
                        "direction": "decrease",
                    }
                ],
                "risk_increase_factors": [],
                "risk_decrease_factors": [
                    {
                        "feature": "due_margin_to_duration_ratio_capped",
                        "feature_name_ko": "생산 소요시간 대비 납기 여유 비율",
                        "cause_tag": "DUE_MARGIN_RISK",
                        "feature_value": 3.0,
                        "impact": -1.798938512802124,
                        "abs_impact": 1.798938512802124,
                        "direction": "decrease",
                    }
                ],
                "cause_detail": {
                    "raw_delay_probability": 0.012722019106149673,
                    "calibrated_delay_probability": 0.014733265154063702,
                    "probability_output": "calibrated_sigmoid",
                    "top_factors": [
                        {
                            "feature": "due_margin_to_duration_ratio_capped",
                            "feature_name_ko": "생산 소요시간 대비 납기 여유 비율",
                            "cause_tag": "DUE_MARGIN_RISK",
                            "feature_value": 3.0,
                            "impact": -1.798938512802124,
                            "abs_impact": 1.798938512802124,
                            "direction": "decrease",
                        }
                    ],
                    "risk_increase_factors": [],
                    "risk_decrease_factors": [
                        {
                            "feature": "due_margin_to_duration_ratio_capped",
                            "feature_name_ko": "생산 소요시간 대비 납기 여유 비율",
                            "cause_tag": "DUE_MARGIN_RISK",
                            "feature_value": 3.0,
                            "impact": -1.798938512802124,
                            "abs_impact": 1.798938512802124,
                            "direction": "decrease",
                        }
                    ],
                },
            }
        },
    )

    order_id: int = Field(
        validation_alias=AliasChoices("order_id", "orderId"),
        serialization_alias="order_id",
        ge=1,
        description="주문 ID",
    )

    product_id: int = Field(
        validation_alias=AliasChoices("product_id", "productId"),
        serialization_alias="product_id",
        ge=1,
        description="제품 ID",
    )

    plan_id: int | None = Field(
        validation_alias=AliasChoices("plan_id", "planId"),
        serialization_alias="plan_id",
        default=None,
        ge=1,
        description="생산계획 ID. 없을 경우 null",
    )

    line_id: int | None = Field(
        validation_alias=AliasChoices("line_id", "lineId"),
        serialization_alias="line_id",
        default=None,
        ge=1,
        description="라인 ID. 없을 경우 null",
    )

    raw_delay_probability: float = Field(
        validation_alias=AliasChoices("raw_delay_probability", "rawDelayProbability"),
        serialization_alias="raw_delay_probability",
        ge=0,
        le=1,
        description="calibration 적용 전 raw 지연 확률",
    )

    delay_probability: float = Field(
        validation_alias=AliasChoices("delay_probability", "delayProbability"),
        serialization_alias="delay_probability",
        ge=0,
        le=1,
        description="calibration 적용 후 지연 확률. DB 저장 대상",
    )

    risk_level: RiskLevel = Field(
        validation_alias=AliasChoices("risk_level", "riskLevel"),
        serialization_alias="risk_level",
        description="지연 위험 등급",
    )

    model_name: str = Field(
        validation_alias=AliasChoices("model_name", "modelName"),
        serialization_alias="model_name",
        min_length=1,
        description="모델명",
    )

    model_version: str = Field(
        validation_alias=AliasChoices("model_version", "modelVersion"),
        serialization_alias="model_version",
        min_length=1,
        description="모델 버전",
    )

    probability_output: str = Field(
        validation_alias=AliasChoices("probability_output", "probabilityOutput"),
        serialization_alias="probability_output",
        min_length=1,
        description="확률 출력 방식",
    )

    predicted_at: datetime = Field(
        validation_alias=AliasChoices("predicted_at", "predictedAt"),
        serialization_alias="predicted_at",
        description="ML 지연 확률 예측 시각",
    )

    top_factors: list[ShapFactor] = Field(
        validation_alias=AliasChoices("top_factors", "topFactors"),
        serialization_alias="top_factors",
        default_factory=list,
        description="절댓값 기준 상위 SHAP 요인",
    )

    risk_increase_factors: list[ShapFactor] = Field(
        validation_alias=AliasChoices("risk_increase_factors", "riskIncreaseFactors"),
        serialization_alias="risk_increase_factors",
        default_factory=list,
        description="지연 위험 증가 방향 SHAP 요인",
    )

    risk_decrease_factors: list[ShapFactor] = Field(
        validation_alias=AliasChoices("risk_decrease_factors", "riskDecreaseFactors"),
        serialization_alias="risk_decrease_factors",
        default_factory=list,
        description="지연 위험 감소 방향 SHAP 요인",
    )

    cause_detail: CauseDetail = Field(
        validation_alias=AliasChoices("cause_detail", "causeDetail"),
        serialization_alias="cause_detail",
        description="ai_prediction_results.cause_detail JSONB 저장 대상",
    )

    @classmethod
    def from_prediction_result(
        cls,
        prediction_result: dict[str, Any],
    ) -> DelayProbabilityPredictResponse:
        """
        artifact.predict_one(...) 또는 inference_utils.predict_delay_probability_one(...) 결과를
        응답 DTO로 변환합니다.
        """

        return cls.model_validate(prediction_result)


__all__ = [
    "RiskLevel",
    "ImpactDirection",
    "FeatureValue",
    "ShapFactor",
    "CauseDetail",
    "DelayProbabilityPredictResponse",
]
