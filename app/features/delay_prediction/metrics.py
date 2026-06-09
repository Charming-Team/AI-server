from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def clip_predictions(predictions) -> np.ndarray:
    return np.clip(np.asarray(predictions, dtype=float), a_min=0.0, a_max=None)


def calculate_regression_metrics(y_true, y_pred) -> dict[str, float]:
    clipped_predictions = clip_predictions(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, clipped_predictions)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, clipped_predictions))),
        "r2": float(r2_score(y_true, clipped_predictions)),
    }
