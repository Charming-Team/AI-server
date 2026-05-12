from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.chat import ChatAnswerRequest, ChatAnswerResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_chat_service(settings: SettingsDep) -> ChatService:
    return ChatService(settings)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


@router.post("/answer", response_model=ChatAnswerResponse)
async def create_chat_answer(
    request: ChatAnswerRequest,
    chat_service: ChatServiceDep,
) -> ChatAnswerResponse:
    return await chat_service.create_answer(request)
