# app/api/v1/routes/delay_probability.py
"""
Delay probability prediction API routes.

Endpoint:
- POST /delay-probability/predict
최종 경로:
- POST /api/v1/delay-probability/predict
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.features.delay_probability.repositories.delay_probability_repository import (
    DelayProbabilityInferenceRowDuplicatedError,
    DelayProbabilityInferenceRowNotFoundError,
    DelayProbabilityRepositoryError,
)
from app.features.delay_probability.schemas.request import (
    DelayProbabilityPredictRequest,
)
from app.features.delay_probability.schemas.response import (
    DelayProbabilityPredictResponse,
)
from app.features.delay_probability.services.delay_probability_service import (
    get_delay_probability_prediction_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/delay-probability",
    tags=["delay-probability"],
)


@router.post(
    "/predict",
    response_model=DelayProbabilityPredictResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_200_OK,
    summary="주문별 지연 확률 예측",
    description=(
        "orderId 기준으로 inference 전용 view에서 주문 단위 feature row를 조회한 뒤, "
        "XGBoost 지연 확률 예측 모델을 실행하여 delayProbability, riskLevel, "
        "SHAP 기반 causeDetail을 반환합니다."
    ),
)
def predict_delay_probability(
    request: DelayProbabilityPredictRequest,
) -> DelayProbabilityPredictResponse:
    """
    주문 1건의 지연 확률 예측을 수행합니다.

    요청 예시:
    {
      "orderId": 314,
      "topN": 5
    }
    """

    service = get_delay_probability_prediction_service()

    try:
        return service.predict(request)

    except DelayProbabilityInferenceRowNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except DelayProbabilityInferenceRowDuplicatedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except DelayProbabilityRepositoryError as exc:
        logger.exception("Delay probability inference repository error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="지연 확률 예측용 inference 데이터 조회 중 오류가 발생했습니다.",
        ) from exc

    except FileNotFoundError as exc:
        logger.exception("Delay probability model artifact file not found")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="지연 확률 예측 모델 artifact 파일을 찾을 수 없습니다.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected delay probability prediction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="지연 확률 예측 처리 중 알 수 없는 오류가 발생했습니다.",
        ) from exc
