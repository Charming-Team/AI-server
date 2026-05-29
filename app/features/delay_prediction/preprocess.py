from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.features.delay_prediction.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


def validate_training_dataframe(df: pd.DataFrame) -> None:
    missing_columns = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_training_dataframe(df)
    cleaned_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    cleaned_df[TARGET_COLUMN] = pd.to_numeric(cleaned_df[TARGET_COLUMN], errors="coerce")
    cleaned_df = cleaned_df.dropna(axis=0).reset_index(drop=True)

    x = cleaned_df[FEATURE_COLUMNS].copy()
    y = cleaned_df[TARGET_COLUMN].copy()
    return x, y


def build_preprocessor(*, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_transformer = "passthrough"
    if scale_numeric:
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_steps = [("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("categorical", Pipeline(steps=categorical_steps), CATEGORICAL_COLUMNS),
        ]
    )
