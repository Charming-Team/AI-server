from fastapi import APIRouter

from app.features.delay_prediction.schemas.request import DelayPredictionRequest
from app.features.delay_prediction.schemas.response import DelayPredictionResponse
from app.features.delay_prediction.services.delay_prediction_service import (
    DelayPredictionService,
)
from app.schemas.base import ApiSchema

router = APIRouter(
    prefix="/delay-prediction",
    tags=["delay-prediction"],
)


class DelayPredictionHealthResponse(ApiSchema):
    status: str
    feature: str


delay_prediction_service = DelayPredictionService()


@router.post("/predict", response_model=DelayPredictionResponse)
def predict_delay(
    request: DelayPredictionRequest,
) -> DelayPredictionResponse:
    """Predict order delay hours from read-only inference features."""
    return delay_prediction_service.predict_delay(request)


@router.get("/health", response_model=DelayPredictionHealthResponse)
def get_delay_prediction_health() -> DelayPredictionHealthResponse:
    """Return route availability without touching DB or model dependencies."""
    return DelayPredictionHealthResponse(
        status="ok",
        feature="delay-prediction",
    )
