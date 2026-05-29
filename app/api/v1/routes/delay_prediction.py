from fastapi import APIRouter
from pydantic import BaseModel

from app.features.delay_prediction.schemas.request import DelayPredictionRequest
from app.features.delay_prediction.schemas.response import DelayPredictionResponse
from app.features.delay_prediction.services.delay_prediction_service import (
    DelayPredictionService,
)

router = APIRouter(prefix="/delay-prediction", tags=["delay-prediction"])


class DelayPredictionHealthResponse(BaseModel):
    status: str
    feature: str

delay_prediction_service = DelayPredictionService()


@router.post("/predict", response_model=DelayPredictionResponse)
def predict_delay(
    request: DelayPredictionRequest,
) -> DelayPredictionResponse:
    return delay_prediction_service.predict_delay(request)


@router.get("/health", response_model=DelayPredictionHealthResponse)
def delay_prediction_health() -> DelayPredictionHealthResponse:
    return DelayPredictionHealthResponse(
        status="ok",
        feature="delay-prediction",
    )
