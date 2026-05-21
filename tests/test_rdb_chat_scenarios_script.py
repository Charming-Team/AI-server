import json
from argparse import Namespace
from io import StringIO
from typing import Any

import anyio
import httpx
import pytest

from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatIntent
from scripts import check_rdb_chat_scenarios


def _build_args(**overrides: Any) -> Namespace:
    values = {
        "base_url": "http://fastapi.local",
        "path": "/api/v1/chat/answer",
        "token": "answer-token",
        "env_file": None,
        "timeout_seconds": 10.0,
        "role": "MANUFACTURING_MANAGER",
        "user_id": 1,
        "company_name": "S-MAP",
        "session_id": 10,
        "message_id": 24,
        "requested_at": "2026-05-12T10:30:00+09:00",
        "scenario": None,
        "scenario_group": None,
        "min_evidence_count": None,
        "json": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _answer_response(
    intent: ChatIntent,
    evidence_count: int = 1,
    security_status: str = "PASSED",
) -> dict:
    security_code = None
    if security_status == "BLOCKED_UNAUTHORIZED":
        security_code = "CHAT_SECURITY_004"
    if security_status == "INSUFFICIENT_EVIDENCE":
        security_code = "CHAT_EVIDENCE_001"

    return {
        "sessionId": 10,
        "messageId": 24,
        "intent": intent.value,
        "answer": (
            "근거 기반 답변입니다."
            if security_status == "PASSED"
            else "현재 역할 권한으로는 답변할 수 없는 요청입니다."
        ),
        "basisTime": "2026-05-12T10:35:00+09:00",
        "urls": (
            [
                {
                    "label": "근거 화면",
                    "url": "/plans/1?mode=read",
                    "type": "PLAN",
                }
            ]
            if evidence_count > 0
            else []
        ),
        "sources": (
            [
                {
                    "sourceType": "PLAN",
                    "title": "생산계획 근거",
                    "summary": "RDB View에서 조회한 근거입니다.",
                    "url": "/plans/1?mode=read",
                    "referenceId": 1,
                    "source": "chat_production_plan_evidence_view",
                    "basisTime": "2026-05-12T10:35:00+09:00",
                    "sourceOrigin": "RDB",
                }
            ]
            if evidence_count > 0
            else []
        ),
        "securityResult": {
            "status": security_status,
            "code": security_code,
            "reason": (
                "보안 필터를 통과했고 내부 근거가 확인되었습니다."
                if security_status == "PASSED"
                else "근거가 부족하거나 역할 권한으로 차단되었습니다."
            ),
        },
        "modelResult": {
            "usedVectorSearch": False,
            "usedRdbEvidence": evidence_count > 0,
            "usedLlmGeneration": False,
            "rdbEvidenceCount": evidence_count,
            "documentSourceCount": 0,
            "evidenceCount": evidence_count,
            "vectorSearchSkippedReason": None,
            "llmGenerationSkippedReason": "LLM 답변 생성 기능이 비활성화되어 있습니다.",
        },
    }


def test_select_scenarios_returns_all_by_default() -> None:
    scenarios = check_rdb_chat_scenarios.select_scenarios(None)

    assert len(scenarios) == 5
    assert {scenario.intent for scenario in scenarios} == {
        ChatIntent.MATERIAL_SHORTAGE,
        ChatIntent.DELIVERY_RISK,
        ChatIntent.PRODUCTION_PLAN,
        ChatIntent.LINE_BOTTLENECK,
        ChatIntent.WORK_PRIORITY,
    }


def test_select_scenarios_filters_by_requested_ids() -> None:
    scenarios = check_rdb_chat_scenarios.select_scenarios(
        ["material-shortage", "line-bottleneck"],
    )

    assert [scenario.scenario_id for scenario in scenarios] == [
        "material-shortage",
        "line-bottleneck",
    ]


def test_select_scenarios_supports_scenario_groups() -> None:
    scenarios = check_rdb_chat_scenarios.select_scenarios(
        None,
        ["access", "filtered"],
    )

    assert [scenario.scenario_id for scenario in scenarios] == [
        "operator-report-allowed",
        "operator-urgent-order-blocked",
        "operator-financial-blocked",
        "admin-chat-blocked",
        "material-shortage-this-week-target",
        "line-bottleneck-today-target",
        "production-plan-date-range",
    ]
    assert scenarios[0].role == "OPERATOR"
    assert scenarios[0].expected_security_statuses == (
        "INSUFFICIENT_EVIDENCE",
        "PASSED",
    )
    assert scenarios[0].expected_security_codes == ("CHAT_EVIDENCE_001", None)


def test_select_scenarios_finds_explicit_scenario_without_group() -> None:
    scenarios = check_rdb_chat_scenarios.select_scenarios(
        ["operator-report-allowed"],
    )

    assert [scenario.scenario_id for scenario in scenarios] == [
        "operator-report-allowed",
    ]


def test_check_rdb_chat_scenarios_calls_fastapi_for_each_scenario() -> None:
    captured_questions: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        captured_questions.append(body)
        for scenario in check_rdb_chat_scenarios.DEFAULT_RDB_CHAT_SCENARIOS:
            if scenario.question in body:
                return httpx.Response(
                    200,
                    json=_answer_response(scenario.intent),
                    request=request,
                )
        return httpx.Response(500, json={}, request=request)

    async def run() -> dict[str, Any]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_rdb_chat_scenarios.check_rdb_chat_scenarios(
                _build_args(),
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["checkStatus"] == "PASS"
    assert result["scenarioCount"] == 5
    assert len(captured_questions) == 5
    assert all(scenario["usedRdbEvidence"] is True for scenario in result["scenarios"])
    assert {scenario["expectedIntent"] for scenario in result["scenarios"]} == {
        ChatIntent.MATERIAL_SHORTAGE.value,
        ChatIntent.DELIVERY_RISK.value,
        ChatIntent.PRODUCTION_PLAN.value,
        ChatIntent.LINE_BOTTLENECK.value,
        ChatIntent.WORK_PRIORITY.value,
    }
    assert all(
        scenario["expectedSecurityStatuses"] == ["PASSED"]
        for scenario in result["scenarios"]
    )
    assert all(
        scenario["expectedSecurityCodes"] == [None]
        for scenario in result["scenarios"]
    )
    assert all(
        scenario["expectedSecurityResults"] == [{"status": "PASSED", "code": None}]
        for scenario in result["scenarios"]
    )


def test_check_rdb_chat_scenarios_verifies_access_control_group() -> None:
    captured_roles: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        payload = json.loads(body)
        for scenario in check_rdb_chat_scenarios.ACCESS_CONTROL_RDB_CHAT_SCENARIOS:
            if scenario.question in body:
                captured_roles.append(payload["user"]["role"])
                return httpx.Response(
                    200,
                    json=_answer_response(
                        scenario.intent,
                        evidence_count=0,
                        security_status=scenario.expected_security_statuses[0],
                    ),
                    request=request,
                )
        return httpx.Response(500, json={}, request=request)

    async def run() -> dict[str, Any]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_rdb_chat_scenarios.check_rdb_chat_scenarios(
                _build_args(scenario_group=["access"]),
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["checkStatus"] == "PASS"
    assert result["scenarioCount"] == 4
    assert captured_roles == ["OPERATOR", "OPERATOR", "OPERATOR", "ADMIN"]
    assert [scenario["securityStatus"] for scenario in result["scenarios"]] == [
        "INSUFFICIENT_EVIDENCE",
        "BLOCKED_UNAUTHORIZED",
        "BLOCKED_UNAUTHORIZED",
        "BLOCKED_UNAUTHORIZED",
    ]
    assert all(
        scenario["requireRdbEvidence"] is False
        for scenario in result["scenarios"]
    )


def test_check_rdb_chat_scenarios_fails_when_intent_is_different() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(ChatIntent.DELIVERY_RISK),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_rdb_chat_scenarios.check_rdb_chat_scenarios(
                _build_args(scenario=["material-shortage"]),
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_001"
    assert "expected=MATERIAL_SHORTAGE" in exc_info.value.message
    assert "actual=DELIVERY_RISK" in exc_info.value.message


def test_check_rdb_chat_scenarios_fails_when_security_status_is_different() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                ChatIntent.DELIVERY_RISK,
                evidence_count=0,
                security_status="PASSED",
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_rdb_chat_scenarios.check_rdb_chat_scenarios(
                _build_args(
                    scenario_group=["access"],
                    scenario=["operator-financial-blocked"],
                ),
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_SECURITY_001"
    assert "expected=BLOCKED_UNAUTHORIZED" in exc_info.value.message
    assert "actual=PASSED" in exc_info.value.message


def test_check_rdb_chat_scenarios_fails_when_security_code_is_different() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                **_answer_response(
                    ChatIntent.DELIVERY_RISK,
                    evidence_count=0,
                    security_status="BLOCKED_UNAUTHORIZED",
                ),
                "securityResult": {
                    "status": "BLOCKED_UNAUTHORIZED",
                    "code": "CHAT_SECURITY_001",
                    "reason": "다른 보안 코드입니다.",
                },
            },
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_rdb_chat_scenarios.check_rdb_chat_scenarios(
                _build_args(
                    scenario_group=["access"],
                    scenario=["operator-financial-blocked"],
                ),
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_SECURITY_001"
    assert "expected=CHAT_SECURITY_004" in exc_info.value.message
    assert "actual=CHAT_SECURITY_001" in exc_info.value.message


def test_check_rdb_chat_scenarios_formats_text_result() -> None:
    result = {
        "checkStatus": "PASS",
        "scenarioCount": 1,
        "scenarios": [
            {
                "scenarioId": "material-shortage",
                "role": "MANUFACTURING_MANAGER",
                "intent": "MATERIAL_SHORTAGE",
                "securityStatus": "PASSED",
                "securityCode": None,
                "expectedSecurityStatuses": ["PASSED"],
                "expectedSecurityCodes": [None],
                "expectedSecurityResults": [{"status": "PASSED", "code": None}],
                "requireRdbEvidence": True,
                "rdbEvidenceCount": 3,
                "sourceCount": 3,
                "urlCount": 2,
            }
        ],
    }

    output = check_rdb_chat_scenarios.format_text_result(result)

    assert "status=PASS" in output
    assert "scenarioCount=1" in output
    assert "scenario=material-shortage" in output
    assert "role=MANUFACTURING_MANAGER" in output
    assert "securityCode=None" in output
    assert "requireRdbEvidence=True" in output
    assert "rdbEvidenceCount=3" in output


def test_check_rdb_chat_scenarios_main_does_not_expose_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_rdb_chat_scenarios(args) -> dict:
        return {
            "checkStatus": "PASS",
            "scenarioCount": 1,
            "scenarios": [
                {
                    "scenarioId": "material-shortage",
                    "role": "MANUFACTURING_MANAGER",
                    "intent": "MATERIAL_SHORTAGE",
                    "securityStatus": "PASSED",
                    "securityCode": None,
                    "expectedSecurityStatuses": ["PASSED"],
                    "expectedSecurityCodes": [None],
                    "expectedSecurityResults": [{"status": "PASSED", "code": None}],
                    "requireRdbEvidence": True,
                    "rdbEvidenceCount": 1,
                    "sourceCount": 1,
                    "urlCount": 1,
                }
            ],
        }

    monkeypatch.setattr(
        check_rdb_chat_scenarios,
        "check_rdb_chat_scenarios",
        fake_check_rdb_chat_scenarios,
    )
    stdout = StringIO()

    exit_code = check_rdb_chat_scenarios.main(
        ["--token", "secret-answer-token"],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "status=PASS" in output
    assert "secret-answer-token" not in output


def test_check_rdb_chat_scenarios_main_returns_one_without_token() -> None:
    stderr = StringIO()

    exit_code = check_rdb_chat_scenarios.main(
        ["--base-url", "http://fastapi.local"],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "RDB 챗봇 시나리오 점검 실패" in stderr.getvalue()
    assert "code=CHAT_SECURITY_003" in stderr.getvalue()
