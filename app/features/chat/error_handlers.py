from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.schemas import ChatErrorCode, ErrorResponse


def register_chat_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ChatExternalServiceError, handle_external_service_error)
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(Exception, handle_unexpected_exception)


async def handle_external_service_error(
    request: Request,
    exc: ChatExternalServiceError,
) -> JSONResponse:
    return _error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
    )


async def handle_request_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return _error_response(
        status_code=422,
        code=ChatErrorCode.CHAT_REQUEST_001,
        message="요청 본문 형식이 올바르지 않습니다.",
    )


async def handle_http_exception(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    if isinstance(exc.detail, dict):
        code = exc.detail.get("code", ChatErrorCode.CHAT_REQUEST_002)
        message = exc.detail.get("message", "요청을 처리할 수 없습니다.")
    else:
        code = ChatErrorCode.CHAT_REQUEST_002
        message = str(exc.detail or "요청을 처리할 수 없습니다.")

    return _error_response(
        status_code=exc.status_code,
        code=code,
        message=message,
    )


async def handle_unexpected_exception(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return _error_response(
        status_code=500,
        code=ChatErrorCode.CHAT_SERVER_001,
        message="서버 처리 중 오류가 발생했습니다.",
    )


def _error_response(
    status_code: int,
    code: ChatErrorCode | str,
    message: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(code=code, message=message).model_dump(mode="json"),
    )
