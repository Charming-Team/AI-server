from argparse import Namespace
from io import StringIO

import anyio
import httpx

from app.core.config import Settings
from app.features.chat.schemas import ChatIntent
from scripts import check_evidence_lookup


def _build_args(**overrides):
    values = {
        "base_url": "http://spring.local",
        "path": "/internal/chat/evidence",
        "token": "internal-token",
        "env_file": None,
        "timeout_seconds": None,
        "intent": "MATERIAL_SHORTAGE",
        "question": "LINE-A01 자재 부족 알려줘",
        "role": "MANUFACTURING_MANAGER",
        "user_id": 1,
        "company_name": "S-MAP",
        "session_id": 10,
        "message_id": 24,
        "requested_at": "2026-05-12T10:30:00+09:00",
        "json": False,
        "validate_only": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_check_evidence_lookup_script_builds_settings_from_cli() -> None:
    settings = check_evidence_lookup.build_settings(
        _build_args(
            base_url="http://spring-api.local",
            path="internal/chat/evidence",
            token="secret-token",
            timeout_seconds=3.5,
        )
    )

    assert settings.evidence_lookup_enabled is True
    assert settings.evidence_lookup_base_url == "http://spring-api.local"
    assert settings.evidence_lookup_path == "internal/chat/evidence"
    assert settings.evidence_lookup_internal_token == "secret-token"
    assert settings.evidence_lookup_timeout_seconds == 3.5


def test_check_evidence_lookup_script_builds_request() -> None:
    request = check_evidence_lookup.build_request(
        _build_args(
            intent="LINE_BOTTLENECK",
            question="LINE-A01 병목 원인을 알려줘",
            role=" executive ",
            user_id=7,
        )
    )

    assert request.session_id == 10
    assert request.message_id == 24
    assert request.user.user_id == 7
    assert request.user.role == "EXECUTIVE"
    assert request.question == "LINE-A01 병목 원인을 알려줘"


def test_check_evidence_lookup_script_validate_only_result() -> None:
    settings = Settings(
        evidence_lookup_enabled=True,
        evidence_lookup_base_url="http://spring.local/",
        evidence_lookup_path="internal/chat/evidence",
        evidence_lookup_internal_token="internal-token",
    )
    request = check_evidence_lookup.build_request(
        _build_args(question="LINE-A01 병목 원인을 알려줘")
    )

    result = check_evidence_lookup.build_validate_only_result(
        settings,
        request,
        ChatIntent.LINE_BOTTLENECK,
    )

    assert result["checkStatus"] == "VALIDATED"
    assert result["url"] == "http://spring.local/internal/chat/evidence"
    assert result["tokenConfigured"] is True
    assert result["networkChecked"] is False
    assert result["payload"]["intent"] == "LINE_BOTTLENECK"
    assert result["payload"]["filters"]["targetType"] == "LINE"
    assert result["payload"]["filters"]["targetCode"] == "LINE-A01"


def test_check_evidence_lookup_script_calls_spring_contract() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["token"] = request.headers.get("X-Internal-Token")
        captured_request["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "intent": "MATERIAL_SHORTAGE",
                "basisTime": "2026-05-12T10:35:00+09:00",
                "items": [
                    {
                        "type": "MATERIAL",
                        "title": "MAT-001 재고 부족",
                        "summary": "가용 재고가 안전 재고보다 낮습니다.",
                        "source": "material_inventories",
                        "referenceId": 11,
                    }
                ],
            },
            request=request,
        )

    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_evidence_lookup.check_evidence_lookup(
                Settings(
                    evidence_lookup_enabled=True,
                    evidence_lookup_base_url="http://spring.local",
                    evidence_lookup_internal_token="internal-token",
                ),
                check_evidence_lookup.build_request(_build_args()),
                ChatIntent.MATERIAL_SHORTAGE,
                http_client=http_client,
            )

    result = anyio.run(run)

    assert captured_request["url"] == "http://spring.local/internal/chat/evidence"
    assert captured_request["token"] == "internal-token"
    assert '"intent":"MATERIAL_SHORTAGE"' in captured_request["body"]
    assert result == {
        "checkStatus": "PASS",
        "mode": "NETWORK",
        "url": "http://spring.local/internal/chat/evidence",
        "intent": "MATERIAL_SHORTAGE",
        "basisTime": "2026-05-12T10:35:00+09:00",
        "itemCount": 1,
        "sourceTypes": ["MATERIAL"],
        "networkChecked": True,
    }


def test_check_evidence_lookup_script_main_validate_only_does_not_expose_secret() -> None:
    stdout = StringIO()

    exit_code = check_evidence_lookup.main(
        [
            "--base-url",
            "http://spring.local",
            "--token",
            "secret-token",
            "--validate-only",
        ],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "status=VALIDATED" in output
    assert "tokenConfigured=True" in output
    assert "secret-token" not in output


def test_check_evidence_lookup_script_main_returns_one_without_token() -> None:
    stderr = StringIO()

    exit_code = check_evidence_lookup.main(
        ["--base-url", "http://spring.local", "--validate-only"],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "Spring Evidence 연결 점검 실패" in stderr.getvalue()
    assert "code=CHAT_SECURITY_003" in stderr.getvalue()
