from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.training.features import FEATURE_COLUMNS
from src.training.metrics import clip_predictions


def prepare_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required inference columns: {missing_columns}")
    return df[FEATURE_COLUMNS].copy()


def predict_delay_hours(model, df: pd.DataFrame):
    inference_frame = prepare_inference_frame(df)
    return clip_predictions(model.predict(inference_frame))


def save_model_artifacts(model, artifact_dir: Path, metadata: dict[str, object]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "model.joblib")
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
