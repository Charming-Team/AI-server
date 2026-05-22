from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.features.chat.error_handlers import register_chat_error_handlers
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.router import get_chat_service
from app.features.chat.schemas import ChatErrorCode
from app.features.chat.service import ChatService
from app.main import app

client = TestClient(app)
CHAT_ANSWER_INTERNAL_TOKEN = "chat-answer-token"
CHAT_ANSWER_HEADERS = {"X-Internal-Token": CHAT_ANSWER_INTERNAL_TOKEN}
_MISSING_OVERRIDE = object()


def _post_chat_answer(*, json: dict):
    previous_override = app.dependency_overrides.get(get_settings, _MISSING_OVERRIDE)
    app.dependency_overrides[get_settings] = lambda: Settings(
        chat_answer_internal_token=CHAT_ANSWER_INTERNAL_TOKEN
    )
    try:
        return client.post(
            "/api/v1/chat/answer",
            headers=CHAT_ANSWER_HEADERS,
            json=json,
        )
    finally:
        if previous_override is _MISSING_OVERRIDE:
            app.dependency_overrides.pop(get_settings, None)
        else:
            app.dependency_overrides[get_settings] = previous_override


def _post_chat_answer_raw(*, content: str):
    previous_override = app.dependency_overrides.get(get_settings, _MISSING_OVERRIDE)
    app.dependency_overrides[get_settings] = lambda: Settings(
        chat_answer_internal_token=CHAT_ANSWER_INTERNAL_TOKEN
    )
    try:
        return client.post(
            "/api/v1/chat/answer",
            headers={
                **CHAT_ANSWER_HEADERS,
                "Content-Type": "application/json",
            },
            content=content,
        )
    finally:
        if previous_override is _MISSING_OVERRIDE:
            app.dependency_overrides.pop(get_settings, None)
        else:
            app.dependency_overrides[get_settings] = previous_override


def test_validation_error_returns_error_response() -> None:
    response = _post_chat_answer(
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


def test_malformed_json_returns_error_response() -> None:
    response = _post_chat_answer_raw(content='{"sessionId": 10,')

    assert response.status_code == 422
    assert response.json() == {
        "code": "CHAT_REQUEST_001",
        "message": "요청 본문 형식이 올바르지 않습니다.",
    }


def test_question_length_validation_returns_error_response() -> None:
    response = _post_chat_answer(
        json={
            "sessionId": 10,
            "messageId": 24,
            "user": {
                "userId": 1,
                "role": "MANUFACTURING_MANAGER",
                "companyName": "S-MAP",
                "status": "ACTIVE",
            },
            "question": "가" * 1001,
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


def test_chat_answer_returns_evidence_error_when_rdb_lookup_fails() -> None:
    class FailingEvidenceService:
        async def get_evidence(self, request, intent):
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_EVIDENCE_004,
                message="RDB Evidence View 조회에 실패했습니다.",
            )

    def build_chat_service() -> ChatService:
        service = ChatService(Settings())
        service.evidence_service = FailingEvidenceService()
        return service

    previous_settings_override = app.dependency_overrides.get(
        get_settings,
        _MISSING_OVERRIDE,
    )
    previous_service_override = app.dependency_overrides.get(
        get_chat_service,
        _MISSING_OVERRIDE,
    )
    app.dependency_overrides[get_settings] = lambda: Settings(
        chat_answer_internal_token=CHAT_ANSWER_INTERNAL_TOKEN
    )
    app.dependency_overrides[get_chat_service] = build_chat_service
    try:
        response = client.post(
            "/api/v1/chat/answer",
            headers=CHAT_ANSWER_HEADERS,
            json={
                "sessionId": 10,
                "messageId": 24,
                "user": {
                    "userId": 1,
                    "role": "MANUFACTURING_MANAGER",
                    "companyName": "S-MAP",
                    "status": "ACTIVE",
                },
                "question": "자재 부족 현황 알려줘",
                "requestedAt": "2026-05-12T10:30:00+09:00",
            },
        )
    finally:
        if previous_settings_override is _MISSING_OVERRIDE:
            app.dependency_overrides.pop(get_settings, None)
        else:
            app.dependency_overrides[get_settings] = previous_settings_override

        if previous_service_override is _MISSING_OVERRIDE:
            app.dependency_overrides.pop(get_chat_service, None)
        else:
            app.dependency_overrides[get_chat_service] = previous_service_override

    assert response.status_code == 503
    assert response.json() == {
        "code": "CHAT_EVIDENCE_004",
        "message": "RDB Evidence View 조회에 실패했습니다.",
    }
