# app/features/delay_probability/preprocess.py
"""
Inference-time preprocessing utilities for delay probability prediction.

- 이 파일은 DB를 직접 조회하지 않습니다.
- DB 조회는 repositories/delay_probability_repository.py가 담당하고,
  이 파일은 repository가 반환한 dict/RowMapping/DataFrame을 모델 입력 source frame으로 정규화합니다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .features import (
    BASE_NUMERIC_COLS,
    BINARY_COLS,
    DERIVED_SOURCE_COLS,
    REQUIRED_SOURCE_COLS_INFERENCE,
    validate_required_source_columns,
)


REQUIRED_IDENTIFIER_COLS = [
    "order_id",
    "product_id",
]

OPTIONAL_IDENTIFIER_COLS = [
    "plan_id",
    "line_id",
]


COLUMN_ALIASES = {
    "orderid": "order_id",
    "order_id": "order_id",
    "productid": "product_id",
    "product_id": "product_id",
    "planid": "plan_id",
    "plan_id": "plan_id",
    "lineid": "line_id",
    "line_id": "line_id",
    "primarylineid": "primary_line_id",
    "primary_line_id": "primary_line_id",
}


def _to_snake_case(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[\s\-]+", "_", value)
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"__+", "_", value)
    return value.lower()


def _as_record(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)

    if isinstance(row, pd.Series):
        return row.to_dict()

    # SQLAlchemy Row는 _mapping을 갖는 경우가 많습니다.
    if hasattr(row, "_mapping"):
        return dict(row._mapping)

    raise TypeError(
        "inference row는 dict, Mapping, pandas.Series, SQLAlchemy RowMapping 형태여야 합니다. "
        f"입력 타입: {type(row)!r}"
    )


def to_inference_source_frame(
    rows: Mapping[str, Any] | pd.Series | pd.DataFrame | Sequence[Any],
) -> pd.DataFrame:
    """
    repository 결과를 pandas DataFrame으로 변환합니다.

    허용 입력:
    - dict
    - SQLAlchemy RowMapping
    - pandas.Series
    - pandas.DataFrame
    - dict/RowMapping list
    """

    if isinstance(rows, pd.DataFrame):
        return rows.copy()

    if isinstance(rows, (Mapping, pd.Series)) or hasattr(rows, "_mapping"):
        return pd.DataFrame([_as_record(rows)])

    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
        records = [_as_record(row) for row in rows]
        return pd.DataFrame(records)

    raise TypeError(
        "rows는 dict, RowMapping, pandas.Series, pandas.DataFrame, 또는 row sequence여야 합니다. "
        f"입력 타입: {type(rows)!r}"
    )


def normalize_inference_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    camelCase, PascalCase, 공백 포함 컬럼명을 snake_case로 정규화합니다.
    Spring/DB view 컬럼명이 snake_case라면 그대로 유지됩니다.
    """

    df = df.copy()

    rename_map: dict[Any, str] = {}
    for column in df.columns:
        normalized = _to_snake_case(str(column))
        normalized = COLUMN_ALIASES.get(normalized, normalized)
        rename_map[column] = normalized

    return df.rename(columns=rename_map)


def validate_identifier_columns(df: pd.DataFrame) -> None:
    missing_cols = [col for col in REQUIRED_IDENTIFIER_COLS if col not in df.columns]

    if missing_cols:
        raise ValueError(
            "지연 확률 예측 응답 저장에 필요한 식별자 컬럼이 inference view에 없습니다: "
            f"{missing_cols}. 최소 order_id, product_id는 필요합니다."
        )

    null_required = [
        col for col in REQUIRED_IDENTIFIER_COLS
        if df[col].isna().any()
    ]

    if null_required:
        raise ValueError(
            "지연 확률 예측 응답 저장에 필요한 식별자 컬럼에 NULL 값이 있습니다: "
            f"{null_required}."
        )


def coerce_inference_source_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델 입력 source 컬럼 타입을 느슨하게 정규화합니다.

    BINARY_COLS는 features.py의 _to_binary()에서 처리해야 하므로
    여기서 숫자로 강제 변환하지 않습니다.
    """

    df = df.copy()

    numeric_source_cols = [
        col for col in BASE_NUMERIC_COLS
        if col not in BINARY_COLS
    ] + DERIVED_SOURCE_COLS

    identifier_numeric_cols = [
        "order_id",
        "product_id",
        "plan_id",
        "line_id",
        "primary_line_id",
    ]

    for col in numeric_source_cols + identifier_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "product_code" in df.columns:
        df["product_code"] = df["product_code"].astype("string").fillna("UNKNOWN")

    return df


def validate_inference_source_columns(df: pd.DataFrame) -> None:
    validate_identifier_columns(df)
    validate_required_source_columns(df, require_target=False)


def prepare_inference_source_frame(
    rows: Mapping[str, Any] | pd.Series | pd.DataFrame | Sequence[Any],
    *,
    expect_single: bool = False,
) -> pd.DataFrame:
    """
    repository에서 조회한 inference row를 모델 추론 가능한 source DataFrame으로 정리합니다.

    이 함수는 아직 모델 feature DataFrame을 만들지 않습니다.
    최종 X 생성은 features.prepare_selected_X()가 담당합니다.
    """

    df = to_inference_source_frame(rows)
    df = normalize_inference_column_names(df)

    if df.empty:
        raise ValueError("지연 확률 예측에 사용할 inference row가 없습니다.")

    if expect_single and len(df) != 1:
        raise ValueError(
            "단건 지연 확률 예측에는 정확히 1개의 row가 필요합니다. "
            f"입력 row 수: {len(df)}"
        )

    validate_inference_source_columns(df)
    df = coerce_inference_source_types(df)

    return df.reset_index(drop=True)


def prepare_single_inference_source_row(
    row: Mapping[str, Any] | pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    return prepare_inference_source_frame(row, expect_single=True)


def get_required_inference_view_columns() -> list[str]:
    """
    inference 전용 DB view가 제공해야 하는 최소 컬럼 목록입니다.

    repository 작성 시 이 목록을 기준으로 SELECT 컬럼을 맞추면 됩니다.
    """

    result: list[str] = []

    for col in REQUIRED_IDENTIFIER_COLS + OPTIONAL_IDENTIFIER_COLS + REQUIRED_SOURCE_COLS_INFERENCE:
        if col not in result:
            result.append(col)

    return result