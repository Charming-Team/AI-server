# app/features/delay_probability/inference_utils.py
"""
Delay probability inference utilities.

artifact_io.py에서 로드한 모델/전처리기/metadata를 전달받아
단건 지연 확률 예측, 위험 등급 산정, SHAP factor 생성만 수행합니다.

역할:
- source row -> selected feature frame 생성
- raw delay probability 산출
- calibrated delay probability 산출
- SAFE / CAUTION / WARNING / CRITICAL 산정
- XGBoost pred_contribs 기반 SHAP contribution 계산
- encoded feature를 원본 feature 단위로 그룹화
- top_factors, risk_increase_factors, risk_decrease_factors 생성
- cause_detail 생성
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import prepare_selected_X
from .preprocess import prepare_single_inference_source_row

DEFAULT_RISK_THRESHOLDS = {
    "SAFE_MAX": 0.10,
    "CAUTION_MAX": 0.40,
    "WARNING_MAX": 0.70,
    "CRITICAL_MAX": 1.00,
}


def _get_threshold(
    thresholds: Mapping[str, Any] | None,
    upper_key: str,
    lower_key: str,
    camel_key: str,
    default_value: float,
) -> float:
    if not thresholds:
        return default_value

    for key in (upper_key, lower_key, camel_key):
        value = thresholds.get(key)
        if value is not None:
            return float(value)

    return default_value


def to_risk_level(
    probability: float,
    thresholds: Mapping[str, Any] | None = None,
) -> str:
    """
    calibrated delay probability를 risk level로 변환합니다.

    기준:
    - SAFE: p <= SAFE_MAX
    - CAUTION: p <= CAUTION_MAX
    - WARNING: p <= WARNING_MAX
    - CRITICAL: p > WARNING_MAX
    """

    if probability is None:
        raise ValueError("probability must not be None")

    probability = float(probability)

    if np.isnan(probability):
        raise ValueError("probability must not be NaN")

    safe_max = _get_threshold(
        thresholds,
        "SAFE_MAX",
        "safe_max",
        "safeMax",
        DEFAULT_RISK_THRESHOLDS["SAFE_MAX"],
    )
    caution_max = _get_threshold(
        thresholds,
        "CAUTION_MAX",
        "caution_max",
        "cautionMax",
        DEFAULT_RISK_THRESHOLDS["CAUTION_MAX"],
    )
    warning_max = _get_threshold(
        thresholds,
        "WARNING_MAX",
        "warning_max",
        "warningMax",
        DEFAULT_RISK_THRESHOLDS["WARNING_MAX"],
    )

    if probability <= safe_max:
        return "SAFE"

    if probability <= caution_max:
        return "CAUTION"

    if probability <= warning_max:
        return "WARNING"

    return "CRITICAL"


def normalize_feature_name(feature_name: str) -> str:
    """
    OneHotEncoder/ColumnTransformer 결과 feature명을 원본 모델 feature명으로 정규화합니다.

    예:
    - cat__product_code_ABS001 -> product_code
    - cat__planned_quantity_gap_bin_NO_GAP -> planned_quantity_gap_bin
    - num__capacity_load_ratio -> capacity_load_ratio
    """

    name = str(feature_name).replace("cat__", "").replace("num__", "")

    if name.startswith("product_code_"):
        return "product_code"

    if name.startswith("primary_line_id_"):
        return "primary_line_id"

    if name.startswith("planned_quantity_gap_bin_"):
        return "planned_quantity_gap_bin"

    if name.startswith("duration_to_leadtime_bin_"):
        return "duration_to_leadtime_bin"

    return name


def to_json_safe_value(value: Any) -> Any:
    """
    pandas/numpy 값을 FastAPI JSON 응답에 안전한 값으로 변환합니다.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    return str(value)


def get_original_feature_value(row: pd.Series, feature: str) -> Any:
    """
    SHAP factor에 표시할 원본 feature 값을 반환합니다.

    feature가 X_one에 없으면 None을 반환합니다.
    """

    if feature not in row.index:
        return None

    return to_json_safe_value(row[feature])


def get_optional_int(row: pd.Series, *column_names: str) -> int | None:
    """
    여러 후보 컬럼 중 존재하는 첫 값을 int로 반환합니다.

    예:
    - line_id가 있으면 line_id 사용
    - 없으면 primary_line_id 사용
    """

    for column_name in column_names:
        if column_name not in row.index:
            continue

        value = row[column_name]

        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            continue

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return None


def get_grouped_shap_for_one_row(
    shap_values_1d: np.ndarray,
    encoded_feature_names: list[str] | np.ndarray,
) -> pd.DataFrame:
    """
    encoded feature 단위 SHAP 값을 원본 feature 단위로 합산합니다.

    OneHotEncoder 때문에 encoded feature가 여러 개로 나뉘는 경우,
    product_code, primary_line_id, planned_quantity_gap_bin 등으로 다시 묶습니다.
    """

    if len(shap_values_1d) != len(encoded_feature_names):
        raise ValueError(
            "SHAP value 개수와 encoded feature name 개수가 일치하지 않습니다. "
            f"shap_values={len(shap_values_1d)}, "
            f"encoded_feature_names={len(encoded_feature_names)}"
        )

    temp = pd.DataFrame(
        {
            "encoded_feature": list(map(str, encoded_feature_names)),
            "normalized_feature": [normalize_feature_name(name) for name in encoded_feature_names],
            "shap_value": np.asarray(shap_values_1d, dtype=float),
        }
    )

    grouped = temp.groupby("normalized_feature", as_index=False).agg(
        shap_value=("shap_value", "sum")
    )

    grouped["abs_shap_value"] = grouped["shap_value"].abs()

    return grouped.sort_values("abs_shap_value", ascending=False)


def _get_booster(xgb_model: Any) -> xgb.Booster:
    """
    sklearn XGBClassifier 또는 xgboost.Booster에서 Booster를 추출합니다.
    """

    if isinstance(xgb_model, xgb.Booster):
        return xgb_model

    if hasattr(xgb_model, "get_booster"):
        return xgb_model.get_booster()

    raise TypeError(
        "xgb_model은 xgboost.Booster 또는 get_booster()를 가진 XGBClassifier여야 합니다. "
        f"입력 타입: {type(xgb_model)!r}"
    )


def _ensure_encoded_feature_names(
    encoded_feature_names: list[str] | np.ndarray,
    transformed_feature_count: int,
) -> list[str]:
    names = list(map(str, encoded_feature_names))

    if len(names) != transformed_feature_count:
        raise ValueError(
            "encoded_feature_names 개수가 전처리 결과 feature 개수와 일치하지 않습니다. "
            f"encoded_feature_names={len(names)}, "
            f"transformed_feature_count={transformed_feature_count}"
        )

    return names


def predict_raw_probability(
    xgb_pipeline: Any,
    X_one: pd.DataFrame,
) -> float:
    """
    calibration 적용 전 raw delay probability를 반환합니다.
    """

    if not hasattr(xgb_pipeline, "predict_proba"):
        raise TypeError("xgb_pipeline must have predict_proba().")

    return float(xgb_pipeline.predict_proba(X_one)[:, 1][0])


def predict_calibrated_probability(
    calibrated_model: Any,
    X_one: pd.DataFrame,
) -> float:
    """
    calibration 적용 후 delay probability를 반환합니다.
    """

    if not hasattr(calibrated_model, "predict_proba"):
        raise TypeError("calibrated_model must have predict_proba().")

    return float(calibrated_model.predict_proba(X_one)[:, 1][0])


def compute_grouped_shap_values(
    *,
    preprocessor: Any,
    xgb_model: Any,
    X_one: pd.DataFrame,
    encoded_feature_names: list[str] | np.ndarray,
) -> pd.DataFrame:
    """
    단건 X에 대해 XGBoost pred_contribs 기반 SHAP contribution을 계산하고,
    원본 feature 단위로 그룹화합니다.

    마지막 bias/base value 컬럼은 제외합니다.
    """

    if not hasattr(preprocessor, "transform"):
        raise TypeError("preprocessor must have transform().")

    X_transformed = preprocessor.transform(X_one)

    transformed_feature_count = X_transformed.shape[1]
    encoded_names = _ensure_encoded_feature_names(
        encoded_feature_names,
        transformed_feature_count,
    )

    dmat = xgb.DMatrix(
        X_transformed,
        feature_names=encoded_names,
    )

    booster = _get_booster(xgb_model)

    shap_contribs = booster.predict(
        dmat,
        pred_contribs=True,
    )

    if shap_contribs.ndim != 2 or shap_contribs.shape[0] != 1:
        raise ValueError(
            f"단건 추론의 SHAP contribution shape이 올바르지 않습니다. shape={shap_contribs.shape}"
        )

    # 마지막 컬럼은 bias/base value입니다.
    shap_values = shap_contribs[:, :-1]

    return get_grouped_shap_for_one_row(
        shap_values[0],
        encoded_names,
    )


def build_shap_factors(
    *,
    grouped_shap: pd.DataFrame,
    X_one: pd.DataFrame,
    feature_name_map: Mapping[str, str] | None,
    feature_cause_map: Mapping[str, str] | None,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    grouped SHAP DataFrame을 API 응답용 factor list로 변환합니다.
    """

    top_n = max(1, int(top_n))
    feature_name_map = feature_name_map or {}
    feature_cause_map = feature_cause_map or {}

    top_factors: list[dict[str, Any]] = []
    feature_row = X_one.iloc[0]

    for _, factor in grouped_shap.head(top_n).iterrows():
        feature = str(factor["normalized_feature"])
        impact = float(factor["shap_value"])
        direction = "increase" if impact > 0 else "decrease"

        top_factors.append(
            {
                "feature": feature,
                "feature_name_ko": feature_name_map.get(feature, feature),
                "cause_tag": feature_cause_map.get(feature, "ETC"),
                "feature_value": get_original_feature_value(feature_row, feature),
                "impact": impact,
                "abs_impact": float(abs(impact)),
                "direction": direction,
            }
        )

    return top_factors


def split_risk_factors(
    top_factors: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    top_factors를 위험 증가/감소 요인으로 분리합니다.
    기존 모델 응답 계약과 동일하게 top_factors 안에서만 분리합니다.
    """

    risk_increase_factors = [
        factor for factor in top_factors if factor.get("direction") == "increase"
    ]

    risk_decrease_factors = [
        factor for factor in top_factors if factor.get("direction") == "decrease"
    ]

    return risk_increase_factors, risk_decrease_factors


def build_cause_detail(
    *,
    raw_probability: float,
    calibrated_probability: float,
    probability_output: str,
    top_factors: list[dict[str, Any]],
    risk_increase_factors: list[dict[str, Any]],
    risk_decrease_factors: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    ai_prediction_results.cause_detail JSONB에 저장할 payload를 생성합니다.
    """

    return {
        "raw_delay_probability": raw_probability,
        "calibrated_delay_probability": calibrated_probability,
        "probability_output": probability_output,
        "top_factors": top_factors,
        "risk_increase_factors": risk_increase_factors,
        "risk_decrease_factors": risk_decrease_factors,
    }


def predict_delay_probability_one(
    source_row: dict[str, Any] | pd.Series | pd.DataFrame,
    *,
    xgb_pipeline: Any,
    calibrated_model: Any,
    preprocessor: Any,
    xgb_model: Any,
    encoded_feature_names: list[str] | np.ndarray,
    risk_thresholds: Mapping[str, Any] | None,
    feature_name_map: Mapping[str, str] | None,
    feature_cause_map: Mapping[str, str] | None,
    metadata: Mapping[str, Any] | None,
    top_n: int = 5,
) -> dict[str, Any]:
    """
    주문 1건에 대해 지연 확률 예측 결과를 반환합니다.

    입력:
    - source_row:
      inference 전용 DB view에서 조회한 단건 row
    - xgb_pipeline:
      raw probability 산출용 pipeline
    - calibrated_model:
      calibrated probability 산출용 모델
    - preprocessor:
      SHAP 계산을 위한 fitted preprocessor
    - xgb_model:
      SHAP pred_contribs 계산을 위한 XGBClassifier 또는 Booster
    - encoded_feature_names:
      preprocessor transform 이후 feature names
    - risk_thresholds:
      SAFE/CAUTION/WARNING/CRITICAL 임계값
    - feature_name_map:
      feature -> 한글명 mapping
    - feature_cause_map:
      feature -> cause_tag mapping
    - metadata:
      model_name, model_version, probability_output 등

    출력:
    - 현재 모델 반환 구조와 동일한 dict
    """

    metadata = metadata or {}

    raw_df = prepare_single_inference_source_row(source_row)
    source = raw_df.iloc[0]

    order_id = get_optional_int(source, "order_id")
    product_id = get_optional_int(source, "product_id")
    plan_id = get_optional_int(source, "plan_id")
    line_id = get_optional_int(source, "line_id", "primary_line_id")

    X_one = prepare_selected_X(raw_df)

    raw_probability = predict_raw_probability(
        xgb_pipeline,
        X_one,
    )

    calibrated_probability = predict_calibrated_probability(
        calibrated_model,
        X_one,
    )

    risk_level = to_risk_level(
        calibrated_probability,
        risk_thresholds,
    )

    grouped_shap = compute_grouped_shap_values(
        preprocessor=preprocessor,
        xgb_model=xgb_model,
        X_one=X_one,
        encoded_feature_names=encoded_feature_names,
    )

    top_factors = build_shap_factors(
        grouped_shap=grouped_shap,
        X_one=X_one,
        feature_name_map=feature_name_map,
        feature_cause_map=feature_cause_map,
        top_n=top_n,
    )

    risk_increase_factors, risk_decrease_factors = split_risk_factors(
        top_factors,
    )

    probability_output = str(metadata.get("probability_output", "calibrated_sigmoid"))

    predicted_at = datetime.now(UTC).isoformat()

    cause_detail = build_cause_detail(
        raw_probability=raw_probability,
        calibrated_probability=calibrated_probability,
        probability_output=probability_output,
        top_factors=top_factors,
        risk_increase_factors=risk_increase_factors,
        risk_decrease_factors=risk_decrease_factors,
    )

    return {
        "order_id": order_id,
        "product_id": product_id,
        "plan_id": plan_id,
        "line_id": line_id,
        "raw_delay_probability": raw_probability,
        "delay_probability": calibrated_probability,
        "risk_level": risk_level,
        "model_name": str(metadata.get("model_name", "xgboost_delay_probability")),
        "model_version": str(metadata.get("model_version", "v1.0.0")),
        "probability_output": probability_output,
        "predicted_at": predicted_at,
        "top_factors": top_factors,
        "risk_increase_factors": risk_increase_factors,
        "risk_decrease_factors": risk_decrease_factors,
        "cause_detail": cause_detail,
    }
