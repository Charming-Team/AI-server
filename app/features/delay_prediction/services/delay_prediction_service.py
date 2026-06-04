import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import HTTPException

from app.features.delay_prediction.inference_utils import predict_delay_hours
from app.features.delay_prediction.repository import DelayPredictionRepository
from app.features.delay_prediction.schemas.request import DelayPredictionRequest
from app.features.delay_prediction.schemas.response import DelayPredictionResponse


class DelayPredictionService:
    def __init__(
        self,
        artifact_dir: Path | None = None,
        repository: DelayPredictionRepository | None = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self._artifact_dir = artifact_dir or base_dir / "artifacts"
        self._repository = repository or DelayPredictionRepository()

    def predict_delay(self, request: DelayPredictionRequest) -> DelayPredictionResponse:
        model = joblib.load(self._artifact_dir / "model.joblib")
        metadata = self._load_metadata()
        inference_row = self._build_inference_row(request)
        inference_frame = pd.DataFrame([inference_row])
        predictions = predict_delay_hours(model, inference_frame)

        return DelayPredictionResponse(
            orderId=request.order_id,
            predictedDelayHours=float(predictions[0]),
            modelName=str(metadata.get("model_name", "unknown_model")),
        )

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self._artifact_dir / "metadata.json"
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _build_inference_row(
        self,
        request: DelayPredictionRequest,
    ) -> dict[str, Any]:
        payload = self._repository.fetch_inference_row(request.order_id)
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "DELAY_PREDICTION_404",
                    "message": f"예측 대상 주문을 찾을 수 없습니다: order_id={request.order_id}",
                },
            )

        return {key: payload[key] for key in payload if key != "order_id"}
