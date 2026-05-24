from argparse import Namespace
from io import StringIO

import anyio
import httpx
import pytest

from scripts import check_chat_grounding


def _build_args(**overrides):
    values = {
        "env_file": None,
        "spring_base_url": "http://spring.local",
        "spring_path": "/internal/chat/evidence",
        "spring_token": "spring-token",
        "fastapi_base_url": "http://fastapi.local",
        "fastapi_path": "/api/v1/chat/answer",
        "fastapi_token": "answer-token",
        "timeout_seconds": 10.0,
        "intent": "MATERIAL_SHORTAGE",
        "question": "자재 부족 현황 알려줘",
        "role": "MANUFACTURING_MANAGER",
        "user_id": 1,
        "company_name": "S-MAP",
        "session_id": 10,
        "message_id": 24,
        "requested_at": "2026-05-12T10:30:00+09:00",
        "min_evidence_count": 1,
        "allow_non_rdb_evidence": False,
        "json": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _spring_evidence_response() -> dict:
    return {
        "success": True,
        "code": "COMMON200",
        "message": "요청 성공",
        "data": {
            "intent": "MATERIAL_SHORTAGE",
            "basisTime": "2026-05-12T10:35:00+09:00",
            "items": [
                {
                    "type": "MATERIAL",
                    "title": "RM-AL-001 알루미늄 원자재 재고 부족",
                    "summary": "생산계획 1001에서 부족 상태입니다.",
                    "url": "/materials/inventory/1?mode=read",
                    "source": "production_plan_materials",
                    "referenceId": 1,
                    "allowedRoles": [
                        "OPERATOR",
                        "EXECUTIVE",
                        "MANUFACTURING_MANAGER",
                    ],
                }
            ],
        },
    }


def _fastapi_answer_response() -> dict:
    return {
        "sessionId": 10,
        "messageId": 24,
        "intent": "MATERIAL_SHORTAGE",
        "answer": "근거는 조회됐지만 LLM 답변 생성 기능이 아직 활성화되지 않았습니다.",
        "basisTime": "2026-05-12T10:35:00+09:00",
        "urls": [
            {
                "label": "RM-AL-001 알루미늄 원자재 재고 부족",
                "url": "/materials/inventory/1?mode=read",
                "type": "MATERIAL",
            }
        ],
        "sources": [
            {
                "sourceType": "MATERIAL",
                "title": "RM-AL-001 알루미늄 원자재 재고 부족",
                "summary": "생산계획 1001에서 부족 상태입니다.",
                "url": "/materials/inventory/1?mode=read",
                "referenceId": 1,
                "source": "production_plan_materials",
                "basisTime": "2026-05-12T10:35:00+09:00",
                "sourceOrigin": "RDB",
            }
        ],
        "securityResult": {
            "status": "PASSED",
            "code": None,
            "reason": "보안 필터를 통과했고 내부 근거가 확인되었습니다.",
        },
        "modelResult": {
            "usedVectorSearch": False,
            "usedRdbEvidence": True,
            "usedLlmGeneration": False,
            "llmCacheHit": False,
            "rdbEvidenceCount": 1,
            "documentSourceCount": 0,
            "evidenceCount": 1,
            "vectorSearchSkippedReason": None,
            "llmGenerationSkippedReason": "LLM 답변 생성 기능이 비활성화되어 있습니다.",
        },
    }


def test_check_chat_grounding_script_builds_settings_from_cli() -> None:
    settings = check_chat_grounding.build_settings(
        _build_args(
            spring_base_url="http://spring-api.local",
            spring_path="internal/chat/evidence",
            spring_token="internal-token",
            timeout_seconds=3.5,
        )
    )
    evidence_settings = check_chat_grounding.build_evidence_settings(
        _build_args(spring_token="internal-token")
    )

    assert settings.evidence_lookup_base_url == "http://spring-api.local"
    assert settings.evidence_lookup_path == "internal/chat/evidence"
    assert settings.evidence_lookup_internal_token == "internal-token"
    assert settings.evidence_lookup_timeout_seconds == 3.5
    assert evidence_settings.evidence_lookup_enabled is True


def test_check_chat_grounding_script_calls_spring_and_fastapi_contracts() -> None:
    captured: dict = {}

    def spring_handler(request: httpx.Request) -> httpx.Response:
        captured["spring_url"] = str(request.url)
        captured["spring_token"] = request.headers.get("X-Internal-Token")
        captured["spring_body"] = request.read().decode()
        return httpx.Response(200, json=_spring_evidence_response(), request=request)

    def fastapi_handler(request: httpx.Request) -> httpx.Response:
        captured["fastapi_url"] = str(request.url)
        captured["fastapi_token"] = request.headers.get("X-Internal-Token")
        captured["fastapi_body"] = request.read().decode()
        return httpx.Response(200, json=_fastapi_answer_response(), request=request)

    async def run() -> dict:
        spring_transport = httpx.MockTransport(spring_handler)
        fastapi_transport = httpx.MockTransport(fastapi_handler)
        async with httpx.AsyncClient(transport=spring_transport) as spring_client:
            async with httpx.AsyncClient(transport=fastapi_transport) as fastapi_client:
                return await check_chat_grounding.check_chat_grounding(
                    _build_args(),
                    spring_http_client=spring_client,
                    fastapi_http_client=fastapi_client,
                )

    result = anyio.run(run)

    assert captured["spring_url"] == "http://spring.local/internal/chat/evidence"
    assert captured["spring_token"] == "spring-token"
    assert '"intent":"MATERIAL_SHORTAGE"' in captured["spring_body"]
    assert captured["fastapi_url"] == "http://fastapi.local/api/v1/chat/answer"
    assert captured["fastapi_token"] == "answer-token"
    assert '"question":"자재 부족 현황 알려줘"' in captured["fastapi_body"]
    assert result["checkStatus"] == "PASS"
    assert result["springEvidence"]["itemCount"] == 1
    assert result["fastapiAnswer"]["usedRdbEvidence"] is True
    assert result["fastapiAnswer"]["evidenceCount"] == 1


def test_check_chat_grounding_script_formats_text_result() -> None:
    result = {
        "checkStatus": "PASS",
        "intent": "MATERIAL_SHORTAGE",
        "minEvidenceCount": 1,
        "springEvidence": {
            "url": "http://spring.local/internal/chat/evidence",
            "itemCount": 1,
            "sourceTypes": ["MATERIAL"],
        },
        "fastapiAnswer": {
            "url": "http://fastapi.local/api/v1/chat/answer",
            "securityStatus": "PASSED",
            "evidenceCount": 1,
            "usedRdbEvidence": True,
            "sourceCount": 1,
            "urlCount": 1,
        },
    }

    output = check_chat_grounding.format_text_result(result)

    assert "status=PASS" in output
    assert "spring.itemCount=1" in output
    assert "fastapi.usedRdbEvidence=True" in output


def test_check_chat_grounding_script_main_does_not_expose_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_chat_grounding(args) -> dict:
        return {
            "checkStatus": "PASS",
            "intent": "MATERIAL_SHORTAGE",
            "minEvidenceCount": 1,
            "springEvidence": {
                "url": "http://spring.local/internal/chat/evidence",
                "itemCount": 1,
                "sourceTypes": ["MATERIAL"],
            },
            "fastapiAnswer": {
                "url": "http://fastapi.local/api/v1/chat/answer",
                "securityStatus": "PASSED",
                "evidenceCount": 1,
                "usedRdbEvidence": True,
                "sourceCount": 1,
                "urlCount": 1,
            },
        }

    monkeypatch.setattr(
        check_chat_grounding,
        "check_chat_grounding",
        fake_check_chat_grounding,
    )
    stdout = StringIO()

    exit_code = check_chat_grounding.main(
        [
            "--spring-token",
            "secret-spring-token",
            "--fastapi-token",
            "secret-answer-token",
        ],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "status=PASS" in output
    assert "secret-spring-token" not in output
    assert "secret-answer-token" not in output


def test_check_chat_grounding_script_main_returns_one_without_fastapi_token() -> None:
    stderr = StringIO()

    exit_code = check_chat_grounding.main(
        ["--spring-token", "spring-token"],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "챗봇 Grounding 점검 실패" in stderr.getvalue()
    assert "code=CHAT_SECURITY_003" in stderr.getvalue()
