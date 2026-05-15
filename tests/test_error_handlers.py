from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.features.chat.error_handlers import register_chat_error_handlers
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.schemas import ChatErrorCode
from app.main import app

client = TestClient(app)


def test_validation_error_returns_error_response() -> None:
    response = client.post(
        "/api/v1/chat/answer",
        json={
            "sessionId": 10,
            "messageId": 24,
            "user": {
                "userId": 1,
                "role": "MANUFACTURING_MANAGER",
                "companyName": "S-MAP",
                "status": "ACTIVE",
            },
            "requestedAt": "2026-05-12T10:30:00+09:00",
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "CHAT_REQUEST_001",
        "message": "요청 본문 형식이 올바르지 않습니다.",
    }


def test_http_exception_returns_error_response() -> None:
    response = client.get("/missing")

    assert response.status_code == 404
    assert response.json() == {
        "code": "CHAT_REQUEST_002",
        "message": "Not Found",
    }


def test_unexpected_exception_returns_error_response() -> None:
    test_app = FastAPI()
    register_chat_error_handlers(test_app)

    @test_app.get("/broken")
    async def broken() -> None:
        raise RuntimeError("boom")

    test_client = TestClient(test_app, raise_server_exceptions=False)

    response = test_client.get("/broken")

    assert response.status_code == 500
    assert response.json() == {
        "code": "CHAT_SERVER_001",
        "message": "서버 처리 중 오류가 발생했습니다.",
    }


def test_external_service_exception_returns_error_response() -> None:
    test_app = FastAPI()
    register_chat_error_handlers(test_app)

    @test_app.get("/qdrant-error")
    async def qdrant_error() -> None:
        raise ChatExternalServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_QDRANT_002,
            message="Qdrant 검색에 실패했습니다.",
        )

    test_client = TestClient(test_app, raise_server_exceptions=False)

    response = test_client.get("/qdrant-error")

    assert response.status_code == 503
    assert response.json() == {
        "code": "CHAT_QDRANT_002",
        "message": "Qdrant 검색에 실패했습니다.",
    }
