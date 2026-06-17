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
        serialization_alias="featureNameKo",
        min_length=1,
        description="feature 한글 표시명",
        examples=["라인 생산능력 대비 주문 부하"],
    )

    cause_tag: str = Field(
        validation_alias=AliasChoices("cause_tag", "causeTag"),
        serialization_alias="causeTag",
        min_length=1,
        description="feature 기반 원인 태그",
        examples=["LINE_LOAD"],
    )

    feature_value: FeatureValue = Field(
        validation_alias=AliasChoices("feature_value", "featureValue"),
        serialization_alias="featureValue",
        description="예측에 사용된 해당 feature 값",
        examples=[0.6542056074766355],
    )

    impact: float = Field(
        description="SHAP impact 값. 양수면 지연 위험 증가, 음수면 감소",
        examples=[-0.3560287654399872],
    )

    abs_impact: float = Field(
        validation_alias=AliasChoices("abs_impact", "absImpact"),
        serialization_alias="absImpact",
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
        serialization_alias="rawDelayProbability",
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
        serialization_alias="calibratedDelayProbability",
        ge=0,
        le=1,
        description="calibration 적용 후 지연 확률",
        examples=[0.014733265154063702],
    )

    probability_output: str = Field(
        validation_alias=AliasChoices("probability_output", "probabilityOutput"),
        serialization_alias="probabilityOutput",
        min_length=1,
        description="확률 출력 방식",
        examples=["calibrated_sigmoid"],
    )

    top_factors: list[ShapFactor] = Field(
        validation_alias=AliasChoices("top_factors", "topFactors"),
        serialization_alias="topFactors",
        default_factory=list,
        description="절댓값 기준 상위 SHAP 요인",
    )

    risk_increase_factors: list[ShapFactor] = Field(
        validation_alias=AliasChoices("risk_increase_factors", "riskIncreaseFactors"),
        serialization_alias="riskIncreaseFactors",
        default_factory=list,
        description="지연 위험 증가 방향 SHAP 요인",
    )

    risk_decrease_factors: list[ShapFactor] = Field(
        validation_alias=AliasChoices("risk_decrease_factors", "riskDecreaseFactors"),
        serialization_alias="riskDecreaseFactors",
        default_factory=list,
        description="지연 위험 감소 방향 SHAP 요인",
    )


class DelayProbabilityPredictResponse(BaseModel):
    """
    지연 확률 예측 API 응답 DTO입니다.

    내부 모델 반환 구조는 snake_case로 받아들이고, 외부 응답은 camelCase로 직렬화합니다.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        json_schema_extra={
            "example": {
                "orderId": 314,
                "productId": 10,
                "planId": None,
                "lineId": 5,
                "rawDelayProbability": 0.012722019106149673,
                "delayProbability": 0.014733265154063702,
                "riskLevel": "SAFE",
                "modelName": "xgboost_delay_probability",
                "modelVersion": "v1.0.0",
                "probabilityOutput": "calibrated_sigmoid",
                "predictedAt": "2026-06-11T09:22:30.889312+00:00",
                "topFactors": [
                    {
                        "feature": "due_margin_to_duration_ratio_capped",
                        "featureNameKo": "생산 소요시간 대비 납기 여유 비율",
                        "causeTag": "DUE_MARGIN_RISK",
                        "featureValue": 3.0,
                        "impact": -1.798938512802124,
                        "absImpact": 1.798938512802124,
                        "direction": "decrease",
                    }
                ],
                "riskIncreaseFactors": [],
                "riskDecreaseFactors": [
                    {
                        "feature": "due_margin_to_duration_ratio_capped",
                        "featureNameKo": "생산 소요시간 대비 납기 여유 비율",
                        "causeTag": "DUE_MARGIN_RISK",
                        "featureValue": 3.0,
                        "impact": -1.798938512802124,
                        "absImpact": 1.798938512802124,
                        "direction": "decrease",
                    }
                ],
                "causeDetail": {
                    "rawDelayProbability": 0.012722019106149673,
                    "calibratedDelayProbability": 0.014733265154063702,
                    "probabilityOutput": "calibrated_sigmoid",
                    "topFactors": [
                        {
                            "feature": "due_margin_to_duration_ratio_capped",
                            "featureNameKo": "생산 소요시간 대비 납기 여유 비율",
                            "causeTag": "DUE_MARGIN_RISK",
                            "featureValue": 3.0,
                            "impact": -1.798938512802124,
                            "absImpact": 1.798938512802124,
                            "direction": "decrease",
                        }
                    ],
                    "riskIncreaseFactors": [],
                    "riskDecreaseFactors": [
                        {
                            "feature": "due_margin_to_duration_ratio_capped",
                            "featureNameKo": "생산 소요시간 대비 납기 여유 비율",
                            "causeTag": "DUE_MARGIN_RISK",
                            "featureValue": 3.0,
                            "impact": -1.798938512802124,
                            "absImpact": 1.798938512802124,
                            "direction": "decrease",
                        }
                    ],
                },
            }
        },
    )

    order_id: int = Field(
        validation_alias=AliasChoices("order_id", "orderId"),
        serialization_alias="orderId",
        ge=1,
        description="주문 ID",
    )

    product_id: int = Field(
        validation_alias=AliasChoices("product_id", "productId"),
        serialization_alias="productId",
        ge=1,
        description="제품 ID",
    )

    plan_id: int | None = Field(
        validation_alias=AliasChoices("plan_id", "planId"),
        serialization_alias="planId",
        default=None,
        ge=1,
        description="생산계획 ID. 없을 경우 null",
    )

    line_id: int | None = Field(
        validation_alias=AliasChoices("line_id", "lineId"),
        serialization_alias="lineId",
        default=None,
        ge=1,
        description="라인 ID. 없을 경우 null",
    )

    raw_delay_probability: float = Field(
        validation_alias=AliasChoices("raw_delay_probability", "rawDelayProbability"),
        serialization_alias="rawDelayProbability",
        ge=0,
        le=1,
        description="calibration 적용 전 raw 지연 확률",
    )

    delay_probability: float = Field(
        validation_alias=AliasChoices("delay_probability", "delayProbability"),
        serialization_alias="delayProbability",
        ge=0,
        le=1,
        description="calibration 적용 후 지연 확률. DB 저장 대상",
    )

    risk_level: RiskLevel = Field(
        validation_alias=AliasChoices("risk_level", "riskLevel"),
        serialization_alias="riskLevel",
        description="지연 위험 등급",
    )

    model_name: str = Field(
        validation_alias=AliasChoices("model_name", "modelName"),
        serialization_alias="modelName",
        min_length=1,
        description="모델명",
    )

    model_version: str = Field(
        validation_alias=AliasChoices("model_version", "modelVersion"),
        serialization_alias="modelVersion",
        min_length=1,
        description="모델 버전",
    )

    probability_output: str = Field(
        validation_alias=AliasChoices("probability_output", "probabilityOutput"),
        serialization_alias="probabilityOutput",
        min_length=1,
        description="확률 출력 방식",
    )

    predicted_at: datetime = Field(
        validation_alias=AliasChoices("predicted_at", "predictedAt"),
        serialization_alias="predictedAt",
        description="ML 지연 확률 예측 시각",
    )

    top_factors: list[ShapFactor] = Field(
        validation_alias=AliasChoices("top_factors", "topFactors"),
        serialization_alias="topFactors",
        default_factory=list,
        description="절댓값 기준 상위 SHAP 요인",
    )

    risk_increase_factors: list[ShapFactor] = Field(
        validation_alias=AliasChoices("risk_increase_factors", "riskIncreaseFactors"),
        serialization_alias="riskIncreaseFactors",
        default_factory=list,
        description="지연 위험 증가 방향 SHAP 요인",
    )

    risk_decrease_factors: list[ShapFactor] = Field(
        validation_alias=AliasChoices("risk_decrease_factors", "riskDecreaseFactors"),
        serialization_alias="riskDecreaseFactors",
        default_factory=list,
        description="지연 위험 감소 방향 SHAP 요인",
    )

    cause_detail: CauseDetail = Field(
        validation_alias=AliasChoices("cause_detail", "causeDetail"),
        serialization_alias="causeDetail",
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
