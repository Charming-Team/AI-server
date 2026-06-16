# app/features/delay_probability/services/delay_probability_service.py
"""
Service for delay probability prediction.

역할:
- request DTO 수신
- repository에서 order_id 기준 inference source row 조회
- artifact wrapper로 모델 추론 실행
- response DTO로 변환

중요:
- 이 기능은 plan별 예측이 아니라 주문별 지연 확률 예측입니다.
- plan_id가 request에 남아 있더라도 service에서는 사용하지 않습니다.
- plan_id가 전달되면 잘못된 사용으로 보고 오류를 발생시킵니다.
"""

from __future__ import annotations

from app.features.delay_probability.artifact_io import (
    DelayProbabilityArtifact,
    get_default_delay_probability_artifact,
)
from app.features.delay_probability.repositories.delay_probability_repository import (
    DelayProbabilityRepository,
    get_delay_probability_repository,
)
from app.features.delay_probability.schemas.request import (
    DelayProbabilityPredictRequest,
)
from app.features.delay_probability.schemas.response import (
    DelayProbabilityPredictResponse,
)


class DelayProbabilityPredictionService:
    def __init__(
        self,
        *,
        repository: DelayProbabilityRepository | None = None,
        artifact: DelayProbabilityArtifact | None = None,
    ) -> None:
        self.repository = repository or get_delay_probability_repository()
        self._artifact = artifact

    @property
    def artifact(self) -> DelayProbabilityArtifact:
        """
        모델 artifact는 최초 요청 시점에 lazy loading합니다.
        앱 import 시점에 모델을 바로 로드하지 않기 위함입니다.
        """

        if self._artifact is None:
            self._artifact = get_default_delay_probability_artifact()

        return self._artifact

    def predict(
        self,
        request: DelayProbabilityPredictRequest,
    ) -> DelayProbabilityPredictResponse:
        """
        주문 1건의 지연 확률 예측을 수행합니다.

        처리 흐름:
        1. order_id 기준으로 inference view row 1건 조회
        2. artifact.predict_one(...) 실행
        3. DelayProbabilityPredictResponse로 변환
        """

        plan_id = getattr(request, "plan_id", None)

        if plan_id is not None:
            raise ValueError(
                "지연 확률 예측은 plan별 예측이 아니라 order별 예측입니다. "
                "요청에서 planId를 제거하고 orderId만 전달하세요."
            )

        source_row = self.repository.find_inference_row_by_order_id(
            order_id=request.order_id,
        )

        prediction_result = self.artifact.predict_one(
            source_row,
            top_n=request.top_n,
        )

        return DelayProbabilityPredictResponse.from_prediction_result(prediction_result)


_default_service: DelayProbabilityPredictionService | None = None


def get_delay_probability_prediction_service() -> DelayProbabilityPredictionService:
    global _default_service

    if _default_service is None:
        _default_service = DelayProbabilityPredictionService()

    return _default_service
