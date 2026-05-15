from secrets import compare_digest
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

from app.core.config import Settings, get_settings
from app.features.chat.document_index_service import (
    DocumentIndexResult,
    DocumentIndexService,
)
from app.features.chat.document_payload import InternalDocumentInput
from app.features.chat.recommendation_service import RecommendationService
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatAnswerResponse,
    ChatErrorCode,
    ChatRecommendationRequest,
    ChatRecommendationResponse,
    ErrorResponse,
)
from app.features.chat.service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
InternalTokenHeader = Annotated[str | None, Header(alias="X-Internal-Token")]


def get_chat_service(settings: SettingsDep) -> ChatService:
    return ChatService(settings)


def get_recommendation_service() -> RecommendationService:
    return RecommendationService()


def get_document_index_service(settings: SettingsDep) -> DocumentIndexService:
    return DocumentIndexService(settings)


def verify_document_index_token(
    settings: SettingsDep,
    x_internal_token: InternalTokenHeader = None,
) -> None:
    expected_token = settings.document_index_internal_token
    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail={
                "code": ChatErrorCode.CHAT_SECURITY_003,
                "message": "문서 인덱싱 내부 토큰이 설정되지 않았습니다.",
            },
        )

    if not x_internal_token or not compare_digest(x_internal_token, expected_token):
        raise HTTPException(
            status_code=403,
            detail={
                "code": ChatErrorCode.CHAT_SECURITY_003,
                "message": "문서 인덱싱 권한이 없습니다.",
            },
        )


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
RecommendationServiceDep = Annotated[
    RecommendationService,
    Depends(get_recommendation_service),
]
DocumentIndexServiceDep = Annotated[
    DocumentIndexService,
    Depends(get_document_index_service),
]


chat_error_responses = {
    400: {"model": ErrorResponse},
    403: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
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


@router.post(
    "/internal/documents/index",
    response_model=DocumentIndexResult,
    dependencies=[Depends(verify_document_index_token)],
    responses=chat_error_responses,
)
async def index_internal_document(
    request: InternalDocumentInput,
    document_index_service: DocumentIndexServiceDep,
) -> DocumentIndexResult:
    return await document_index_service.index_document(request)
