from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.features.chat.recommendation_service import RecommendationService
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatAnswerResponse,
    ChatRecommendationRequest,
    ChatRecommendationResponse,
    ErrorResponse,
)
from app.features.chat.service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_chat_service(settings: SettingsDep) -> ChatService:
    return ChatService(settings)


def get_recommendation_service() -> RecommendationService:
    return RecommendationService()


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
RecommendationServiceDep = Annotated[
    RecommendationService,
    Depends(get_recommendation_service),
]


chat_error_responses = {
    400: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}


@router.post(
    "/answer",
    response_model=ChatAnswerResponse,
    responses=chat_error_responses,
)
async def create_chat_answer(
    request: ChatAnswerRequest,
    chat_service: ChatServiceDep,
) -> ChatAnswerResponse:
    return await chat_service.create_answer(request)


@router.post(
    "/recommendations",
    response_model=ChatRecommendationResponse,
    responses=chat_error_responses,
)
async def get_chat_recommendations(
    request: ChatRecommendationRequest,
    recommendation_service: RecommendationServiceDep,
) -> ChatRecommendationResponse:
    return recommendation_service.get_recommendations(request)
