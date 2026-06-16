# app/features/delay_probability/features.py
"""
Delay probability model feature definitions and inference-time feature builder.
FastAPI 운영 추론에서 사용하는 feature 생성 전용 모듈입니다.

- DB 조회는 하지 않습니다.
- 학습/검증 split 함수는 포함하지 않습니다.
- inference 전용 DB view는 REQUIRED_SOURCE_COLS_INFERENCE 컬럼을 제공해야 합니다.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# 운영 추론에서는 사용하지 않지만, artifact metadata나 일부 테스트 코드와의
# 호환성을 위해 상수만 유지합니다.
TARGET_COL = "target_delay"


SELECTED_CATEGORICAL_COLS = [
    "product_code",
    "primary_line_id",
    "planned_quantity_gap_bin",
    "duration_to_leadtime_bin",
]

BASE_NUMERIC_COLS = [
    "order_quantity",
    "order_month",
    "order_dayofweek",
    "due_month",
    "due_dayofweek",
    "order_to_plan_start_days",
    "is_multi_plan",
    "is_multi_line",
    "avg_estimated_duration_hr",
    "min_production_quantity",
    "avg_standard_production_time_hr",
    "avg_capacity_per_day",
    "avg_standard_yield_rate",
    "distinct_material_count",
    "ever_had_material_shortage",
    "inbound_delay_days",
    "material_ready_before_start",
]

DERIVED_NUMERIC_COLS = [
    "capacity_load_ratio",
    "due_margin_to_duration_ratio_capped",
]

SELECTED_NUMERIC_COLS = BASE_NUMERIC_COLS + DERIVED_NUMERIC_COLS
SELECTED_FEATURE_COLS = SELECTED_CATEGORICAL_COLS + SELECTED_NUMERIC_COLS

DERIVED_SOURCE_COLS = [
    "planned_quantity_ratio",
    "order_lead_time_days",
    "due_margin_hr",
    "total_estimated_duration_hr",
]

REQUIRED_SOURCE_COLS_INFERENCE = (
    ["product_code", "primary_line_id"] + BASE_NUMERIC_COLS + DERIVED_SOURCE_COLS
)

# 운영 추론에서는 사용하지 않습니다.
# 다만 validate_required_source_columns(..., require_target=True)를 호출하는
# 테스트나 과거 코드가 있어도 NameError가 나지 않도록 유지합니다.
REQUIRED_SOURCE_COLS_TRAINING = REQUIRED_SOURCE_COLS_INFERENCE + [TARGET_COL]

BINARY_COLS = [
    "is_multi_plan",
    "is_multi_line",
    "ever_had_material_shortage",
    "material_ready_before_start",
]


def _unique(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def _to_binary(series: pd.Series) -> pd.Series:
    """
    bool/numeric/string 형태의 이진 컬럼을 0/1 int로 정규화합니다.
    """

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 1).astype(int)

    return (
        series.astype("string")
        .str.lower()
        .str.strip()
        .map(
            {
                "true": 1,
                "t": 1,
                "1": 1,
                "yes": 1,
                "y": 1,
                "false": 0,
                "f": 0,
                "0": 0,
                "no": 0,
                "n": 0,
            }
        )
        .fillna(0)
        .astype(int)
    )


def validate_required_source_columns(
    df: pd.DataFrame,
    require_target: bool = False,
) -> None:
    """
    inference source DataFrame이 모델 feature 생성에 필요한 원천 컬럼을 갖고 있는지 검증합니다.

    require_target=True는 운영 FastAPI에서는 사용하지 않습니다.
    과거 학습 코드 호환용으로만 남겨둔 옵션입니다.
    """

    required_cols = (
        REQUIRED_SOURCE_COLS_TRAINING if require_target else REQUIRED_SOURCE_COLS_INFERENCE
    )

    missing_cols = [col for col in _unique(required_cols) if col not in df.columns]

    if missing_cols:
        raise ValueError(
            "지연 확률 예측 inference view에서 누락된 원천 컬럼이 있습니다: "
            f"{missing_cols}. "
            "DB inference 전용 view의 SELECT 컬럼과 "
            "app/features/delay_probability/features.py의 "
            "REQUIRED_SOURCE_COLS_INFERENCE를 맞춰주세요."
        )


def _make_planned_quantity_gap_bin(gap_ratio: pd.Series) -> pd.Series:
    gap = (
        pd.to_numeric(gap_ratio, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0)
    )

    bins = [-np.inf, 0, 0.01, 0.05, 0.10, np.inf]
    labels = [
        "NO_GAP",
        "GAP_0_1PCT",
        "GAP_1_5PCT",
        "GAP_5_10PCT",
        "GAP_OVER_10PCT",
    ]

    return pd.cut(
        gap,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype("string")


def add_selected_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델 학습 시 사용한 파생변수를 inference 시점에 동일하게 생성합니다.
    """

    df = df.copy()
    eps = 1e-6

    required_cols = [
        "planned_quantity_ratio",
        "order_quantity",
        "avg_capacity_per_day",
        "total_estimated_duration_hr",
        "order_lead_time_days",
        "due_margin_hr",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"파생변수 생성에 필요한 컬럼이 없습니다: {missing_cols}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    planned_quantity_gap_ratio = (1 - df["planned_quantity_ratio"]).clip(lower=0)
    df["planned_quantity_gap_bin"] = _make_planned_quantity_gap_bin(planned_quantity_gap_ratio)

    df["capacity_load_ratio"] = np.where(
        df["avg_capacity_per_day"].abs() > eps,
        df["order_quantity"] / df["avg_capacity_per_day"],
        0,
    )

    duration_ratio = np.where(
        (df["order_lead_time_days"] * 24).abs() > eps,
        df["total_estimated_duration_hr"] / (df["order_lead_time_days"] * 24),
        0,
    )
    duration_ratio = (
        pd.Series(duration_ratio, index=df.index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0, upper=2)
    )

    df["duration_to_leadtime_bin"] = (
        pd.cut(
            duration_ratio,
            bins=[-0.001, 0.10, 0.25, 0.50, 1.00, 2.00],
            labels=["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"],
            include_lowest=True,
        )
        .astype("string")
        .fillna("UNKNOWN")
    )

    due_margin_to_duration_ratio = np.where(
        df["total_estimated_duration_hr"].abs() > eps,
        df["due_margin_hr"] / df["total_estimated_duration_hr"],
        0,
    )

    df["due_margin_to_duration_ratio_capped"] = (
        pd.Series(due_margin_to_duration_ratio, index=df.index)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(-3, 3)
    )

    for col in DERIVED_NUMERIC_COLS:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        )

    for col in ["planned_quantity_gap_bin", "duration_to_leadtime_bin"]:
        df[col] = df[col].astype("string").fillna("UNKNOWN")

    return df


def validate_selected_features(df: pd.DataFrame) -> None:
    missing_cols = [col for col in SELECTED_FEATURE_COLS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"최종 모델 feature 중 누락된 컬럼이 있습니다: {missing_cols}")


def _prepare_selected_X_internal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = _to_binary(df[col])

    df = add_selected_derived_features(df)
    validate_selected_features(df)

    for col in SELECTED_CATEGORICAL_COLS:
        df[col] = df[col].astype("string").fillna("UNKNOWN")

    for col in SELECTED_NUMERIC_COLS:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
        )

    return df[SELECTED_FEATURE_COLS].copy()


def prepare_selected_X(df: pd.DataFrame) -> pd.DataFrame:
    """
    FastAPI 서비스 추론용 X 생성 함수입니다.
    target_delay 없이 동작합니다.
    """

    df = df.copy()
    validate_required_source_columns(df, require_target=False)
    return _prepare_selected_X_internal(df)


def prepare_selected_xy(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    과거 테스트/검증 코드 호환용 함수입니다.
    FastAPI 운영 추론에서는 사용하지 않습니다.
    """

    df = df.copy()
    validate_required_source_columns(df, require_target=True)

    if target_col not in df.columns:
        raise ValueError(f"target 컬럼이 없습니다: {target_col}")

    df = df[df[target_col].notna()].copy()
    df[target_col] = df[target_col].astype(int)

    X = _prepare_selected_X_internal(df)
    y = df[target_col].astype(int).copy()

    return X, y


def get_required_inference_columns() -> list[str]:
    return _unique(REQUIRED_SOURCE_COLS_INFERENCE)


def get_selected_feature_columns() -> list[str]:
    return list(SELECTED_FEATURE_COLS)
