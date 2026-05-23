import json
from argparse import Namespace
from io import StringIO
from typing import Any

import anyio
import httpx
import pytest

from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatIntent
from scripts import check_rag_chat_scenarios


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
        "min_rdb_evidence_count": None,
        "min_document_source_count": None,
        "require_llm_generation": False,
        "markdown": False,
        "json": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _answer_response(
    intent: ChatIntent,
    evidence_count: int = 2,
    rdb_evidence_count: int = 1,
    document_source_count: int = 1,
    used_vector_search: bool = True,
    security_status: str = "PASSED",
    answer: str = (
        "핵심 답변: RDB 근거와 Qdrant 문서 근거를 함께 확인했습니다.\n\n"
        "확인 필요: 위 근거에 없는 내용은 추가 확인이 필요합니다."
    ),
    rdb_title: str = "생산계획 근거",
    rdb_url: str = "/plans/1?mode=read",
    document_title: str = "LINE-A01 병목 대응 기준",
    used_llm_generation: bool = False,
) -> dict:
    security_code = None
    if security_status == "BLOCKED_UNAUTHORIZED":
        security_code = "CHAT_SECURITY_004"

    sources = []
    urls = []
    if rdb_evidence_count > 0:
        urls.append({"label": "RDB 근거", "url": rdb_url, "type": "PLAN"})
        sources.append(
            {
                "sourceType": "PLAN",
                "title": rdb_title,
                "summary": "RDB View에서 조회한 근거입니다.",
                "url": rdb_url,
                "referenceId": 1,
                "source": "chat_production_plan_evidence_view",
                "basisTime": "2026-05-12T10:35:00+09:00",
                "sourceOrigin": "RDB",
            }
        )
    if document_source_count > 0:
        urls.append(
            {
                "label": "LINE-A01 병목 대응 기준",
                "url": "/company-info/line-a01-bottleneck-guide",
                "type": "COMPANY_INFO",
            }
        )
        sources.append(
            {
                "sourceType": "COMPANY_INFO",
                "title": document_title,
                "summary": "Qdrant에서 조회한 회사정보 근거입니다.",
                "url": "/company-info/line-a01-bottleneck-guide",
                "referenceId": None,
                "source": "company-line-a01-bottleneck-guide:chunk-0001",
                "basisTime": "2026-05-12T10:35:00+09:00",
                "sourceOrigin": "QDRANT",
                "relevanceScore": 0.92,
            }
        )

    resolved_answer = (
        answer
        if security_status == "PASSED"
        else "현재 역할 권한으로는 답변할 수 없는 요청입니다."
    )

    return {
        "sessionId": 10,
        "messageId": 24,
        "intent": intent.value,
        "answer": resolved_answer,
        "basisTime": "2026-05-12T10:35:00+09:00",
        "urls": urls,
        "sources": sources,
        "securityResult": {
            "status": security_status,
            "code": security_code,
            "reason": "RAG 시나리오 점검용 응답입니다.",
        },
        "modelResult": {
            "usedVectorSearch": used_vector_search,
            "usedRdbEvidence": rdb_evidence_count > 0,
            "usedLlmGeneration": used_llm_generation,
            "rdbEvidenceCount": rdb_evidence_count,
            "documentSourceCount": document_source_count,
            "evidenceCount": evidence_count,
            "vectorSearchSkippedReason": None if used_vector_search else "Qdrant 미사용",
            "llmGenerationSkippedReason": (
                None
                if used_llm_generation
                else "LLM 답변 생성 기능이 비활성화되어 있습니다."
            ),
        },
    }


def _rdb_title_for_scenario(scenario_id: str) -> str:
    if scenario_id == "material-shortage-with-company-guide":
        return "MAT-FOAM-ADD 발포 첨가제 SHORTAGE"
    if scenario_id == "line-bottleneck-with-company-guide":
        return "LINE-PE-01 MAINTENANCE"
    return "납기 위험 RDB 근거"


def _document_title_for_scenario(scenario_id: str) -> str:
    if scenario_id == "company-overview-document-allowed":
        return "S-Map 회사 개요"
    if scenario_id == "manager-revenue-company-info-allowed":
        return "S-Map 매출 구조"
    if scenario_id == "line-bottleneck-with-company-guide":
        return "LINE-PE-01 병목 대응 기준"
    return "S-Map 생산 리스크 문서"


def test_select_rag_scenarios_returns_core_by_default() -> None:
    scenarios = check_rag_chat_scenarios.select_scenarios(None)

    assert [scenario.scenario_id for scenario in scenarios] == [
        "material-shortage-with-company-guide",
        "line-bottleneck-with-company-guide",
        "delivery-risk-with-company-guide",
    ]
    assert all(scenario.require_rdb_evidence for scenario in scenarios)
    assert all(scenario.require_vector_search for scenario in scenarios)


def test_select_rag_scenarios_supports_access_group() -> None:
    scenarios = check_rag_chat_scenarios.select_scenarios(None, ["access"])

    assert [scenario.scenario_id for scenario in scenarios] == [
        "operator-report-document-allowed",
        "operator-financial-rag-blocked",
    ]
    assert scenarios[0].role == "OPERATOR"
    assert scenarios[0].require_rdb_evidence is False
    assert scenarios[1].expected_security_results == (
        ("BLOCKED_UNAUTHORIZED", "CHAT_SECURITY_004"),
    )


def test_select_rag_scenarios_supports_company_group() -> None:
    scenarios = check_rag_chat_scenarios.select_scenarios(None, ["company"])

    assert [scenario.scenario_id for scenario in scenarios] == [
        "company-overview-document-allowed",
        "manager-revenue-company-info-allowed",
        "operator-revenue-company-info-blocked",
    ]
    assert [scenario.role for scenario in scenarios] == [
        "OPERATOR",
        "MANUFACTURING_MANAGER",
        "OPERATOR",
    ]
    assert all(not scenario.require_rdb_evidence for scenario in scenarios)
    assert scenarios[2].expected_security_results == (
        ("BLOCKED_UNAUTHORIZED", "CHAT_SECURITY_004"),
    )


def test_check_rag_chat_scenarios_calls_fastapi_for_each_core_scenario() -> None:
    captured_questions: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        captured_questions.append(body)
        for scenario in check_rag_chat_scenarios.CORE_RAG_CHAT_SCENARIOS:
            if scenario.question in body:
                return httpx.Response(
                    200,
                    json=_answer_response(
                        scenario.intent,
                        rdb_title=_rdb_title_for_scenario(scenario.scenario_id),
                        document_title=_document_title_for_scenario(scenario.scenario_id),
                    ),
                    request=request,
                )
        return httpx.Response(500, json={}, request=request)

    async def run() -> dict[str, Any]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_rag_chat_scenarios.check_rag_chat_scenarios(
                _build_args(),
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["checkStatus"] == "PASS"
    assert result["scenarioCount"] == 3
    assert len(captured_questions) == 3
    assert all(scenario["usedRdbEvidence"] is True for scenario in result["scenarios"])
    assert all(scenario["usedVectorSearch"] is True for scenario in result["scenarios"])
    assert all(scenario["documentSourceCount"] == 1 for scenario in result["scenarios"])


def test_check_rag_chat_scenarios_verifies_access_group() -> None:
    captured_roles: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        payload = json.loads(body)
        for scenario in check_rag_chat_scenarios.ACCESS_RAG_CHAT_SCENARIOS:
            if scenario.question in body:
                captured_roles.append(payload["user"]["role"])
                if scenario.scenario_id == "operator-financial-rag-blocked":
                    return httpx.Response(
                        200,
                        json=_answer_response(
                            scenario.intent,
                            evidence_count=0,
                            rdb_evidence_count=0,
                            document_source_count=0,
                            used_vector_search=False,
                            security_status="BLOCKED_UNAUTHORIZED",
                        ),
                        request=request,
                    )
                return httpx.Response(
                    200,
                    json=_answer_response(
                        scenario.intent,
                        evidence_count=1,
                        rdb_evidence_count=0,
                        document_source_count=1,
                        document_title=_document_title_for_scenario(scenario.scenario_id),
                    ),
                    request=request,
                )
        return httpx.Response(500, json={}, request=request)

    async def run() -> dict[str, Any]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_rag_chat_scenarios.check_rag_chat_scenarios(
                _build_args(scenario_group=["access"]),
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["checkStatus"] == "PASS"
    assert result["scenarioCount"] == 2
    assert captured_roles == ["OPERATOR", "OPERATOR"]
    assert [scenario["securityStatus"] for scenario in result["scenarios"]] == [
        "PASSED",
        "BLOCKED_UNAUTHORIZED",
    ]


def test_check_rag_chat_scenarios_verifies_company_group() -> None:
    captured_roles: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        payload = json.loads(body)
        for scenario in check_rag_chat_scenarios.COMPANY_INFO_RAG_CHAT_SCENARIOS:
            if scenario.question in body and payload["user"]["role"] == scenario.role:
                captured_roles.append(payload["user"]["role"])
                if scenario.scenario_id == "operator-revenue-company-info-blocked":
                    return httpx.Response(
                        200,
                        json=_answer_response(
                            scenario.intent,
                            evidence_count=0,
                            rdb_evidence_count=0,
                            document_source_count=0,
                            used_vector_search=False,
                            security_status="BLOCKED_UNAUTHORIZED",
                        ),
                        request=request,
                    )
                return httpx.Response(
                    200,
                    json=_answer_response(
                        scenario.intent,
                        evidence_count=1,
                        rdb_evidence_count=0,
                        document_source_count=1,
                        document_title=_document_title_for_scenario(scenario.scenario_id),
                    ),
                    request=request,
                )
        return httpx.Response(500, json={}, request=request)

    async def run() -> dict[str, Any]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_rag_chat_scenarios.check_rag_chat_scenarios(
                _build_args(scenario_group=["company"]),
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["checkStatus"] == "PASS"
    assert result["scenarioCount"] == 3
    assert captured_roles == ["OPERATOR", "MANUFACTURING_MANAGER", "OPERATOR"]
    assert [scenario["securityStatus"] for scenario in result["scenarios"]] == [
        "PASSED",
        "PASSED",
        "BLOCKED_UNAUTHORIZED",
    ]
    assert [scenario["requireRdbEvidence"] for scenario in result["scenarios"]] == [
        False,
        False,
        False,
    ]


def test_check_rag_chat_scenarios_fails_when_rdb_count_is_below_minimum() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                ChatIntent.LINE_BOTTLENECK,
                evidence_count=2,
                rdb_evidence_count=1,
                document_source_count=1,
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_rag_chat_scenarios.check_rag_chat_scenarios(
                _build_args(
                    scenario=["line-bottleneck-with-company-guide"],
                    min_rdb_evidence_count=2,
                ),
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_001"
    assert "RDB Evidence 개수가 기준보다 적습니다" in exc_info.value.message


def test_check_rag_chat_scenarios_fails_when_vector_search_is_not_used() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                ChatIntent.LINE_BOTTLENECK,
                used_vector_search=False,
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_rag_chat_scenarios.check_rag_chat_scenarios(
                _build_args(scenario=["line-bottleneck-with-company-guide"]),
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_001"
    assert "Qdrant Vector Search가 사용되지 않았습니다" in exc_info.value.message


def test_check_rag_chat_scenarios_verifies_required_llm_generation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                ChatIntent.LINE_BOTTLENECK,
                rdb_title="LINE-PE-01 MAINTENANCE",
                document_title="LINE-PE-01 병목 대응 기준",
                used_llm_generation=True,
            ),
            request=request,
        )

    async def run() -> dict[str, Any]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await check_rag_chat_scenarios.check_rag_chat_scenarios(
                _build_args(
                    scenario=["line-bottleneck-with-company-guide"],
                    require_llm_generation=True,
                ),
                http_client=http_client,
            )

    result = anyio.run(run)

    assert result["checkStatus"] == "PASS"
    assert result["scenarios"][0]["requireLlmGeneration"] is True
    assert result["scenarios"][0]["usedLlmGeneration"] is True


def test_check_rag_chat_scenarios_fails_when_llm_generation_is_required() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                ChatIntent.LINE_BOTTLENECK,
                rdb_title="LINE-PE-01 MAINTENANCE",
                document_title="LINE-PE-01 병목 대응 기준",
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_rag_chat_scenarios.check_rag_chat_scenarios(
                _build_args(
                    scenario=["line-bottleneck-with-company-guide"],
                    require_llm_generation=True,
                ),
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_LLM_004"
    assert "LLM 답변 생성이 사용되지 않았습니다" in exc_info.value.message


def test_check_rag_chat_scenarios_fails_when_company_info_mixes_rdb_evidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                ChatIntent.REPORT_LOOKUP,
                evidence_count=2,
                rdb_evidence_count=1,
                document_source_count=1,
                document_title="S-Map 회사 개요",
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_rag_chat_scenarios.check_rag_chat_scenarios(
                _build_args(scenario=["company-overview-document-allowed"]),
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_001"
    assert "RDB Evidence 개수가 허용 기준보다 많습니다" in exc_info.value.message


def test_check_rag_chat_scenarios_fails_when_rdb_url_is_not_read_only() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                ChatIntent.LINE_BOTTLENECK,
                rdb_title="LINE-PE-01 MAINTENANCE",
                rdb_url="/production-lines/103",
                document_title="LINE-PE-01 병목 대응 기준",
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_rag_chat_scenarios.check_rag_chat_scenarios(
                _build_args(scenario=["line-bottleneck-with-company-guide"]),
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_001"
    assert "read-only 형식이 아닙니다" in exc_info.value.message


def test_check_rag_chat_scenarios_fails_when_required_source_title_is_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_answer_response(
                ChatIntent.LINE_BOTTLENECK,
                rdb_title="LINE-PP-01 RUNNING",
                document_title="LINE-PP-01 병목 대응 기준",
            ),
            request=request,
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            await check_rag_chat_scenarios.check_rag_chat_scenarios(
                _build_args(scenario=["line-bottleneck-with-company-guide"]),
                http_client=http_client,
            )

    with pytest.raises(ChatServiceError) as exc_info:
        anyio.run(run)

    assert exc_info.value.code.value == "CHAT_EVIDENCE_001"
    assert "출처 제목에 필요한 문구가 없습니다" in exc_info.value.message


def test_check_rag_chat_scenarios_formats_text_result() -> None:
    output = check_rag_chat_scenarios.format_text_result(
        {
            "checkStatus": "PASS",
            "scenarioCount": 1,
            "scenarios": [
                {
                    "scenarioId": "line-bottleneck-with-company-guide",
                    "role": "MANUFACTURING_MANAGER",
                    "intent": "LINE_BOTTLENECK",
                    "securityStatus": "PASSED",
                    "securityCode": None,
                    "requireRdbEvidence": True,
                    "requireVectorSearch": True,
                    "rdbEvidenceCount": 1,
                    "documentSourceCount": 1,
                    "usedVectorSearch": True,
                    "requireLlmGeneration": False,
                    "usedLlmGeneration": False,
                    "sourceCount": 2,
                    "urlCount": 2,
                }
            ],
        }
    )

    assert "status=PASS" in output
    assert "scenario=line-bottleneck-with-company-guide" in output
    assert "requireVectorSearch=True" in output
    assert "requireLlmGeneration=False" in output
    assert "usedLlmGeneration=False" in output
    assert "documentSourceCount=1" in output


def test_check_rag_chat_scenarios_formats_markdown_result() -> None:
    output = check_rag_chat_scenarios.format_markdown_result(
        {
            "checkStatus": "PASS",
            "scenarioCount": 1,
            "scenarios": [
                {
                    "scenarioId": "line-bottleneck-with-company-guide",
                    "role": "MANUFACTURING_MANAGER",
                    "question": "LINE-PE-01 병목 현황과 대응 기준을 같이 알려줘",
                    "intent": "LINE_BOTTLENECK",
                    "securityStatus": "PASSED",
                    "securityCode": None,
                    "evidenceCount": 2,
                    "rdbEvidenceCount": 1,
                    "documentSourceCount": 1,
                    "requireLlmGeneration": False,
                    "usedLlmGeneration": False,
                    "answer": "핵심 답변: LINE-PE-01 병목 근거를 확인했습니다.",
                    "sourceDetails": [
                        {
                            "sourceOrigin": "RDB",
                            "sourceType": "LINE",
                            "title": "LINE-PE-01 MAINTENANCE",
                            "url": "/production-lines/103?mode=read",
                        },
                        {
                            "sourceOrigin": "QDRANT",
                            "sourceType": "COMPANY_INFO",
                            "title": "LINE-PE-01 병목 대응 기준",
                            "url": "/company-info/line-pe-01-bottleneck-guide",
                        },
                    ],
                    "urlDetails": [
                        {
                            "type": "LINE",
                            "label": "LINE-PE-01 MAINTENANCE",
                            "url": "/production-lines/103?mode=read",
                        }
                    ],
                }
            ],
        }
    )

    assert "# RAG 챗봇 시나리오 점검 결과" in output
    assert "## line-bottleneck-with-company-guide" in output
    assert "LINE-PE-01 병목 현황과 대응 기준을 같이 알려줘" in output
    assert "- LLM 생성: 요구 `False`, 사용 `False`" in output
    assert "```text\n핵심 답변: LINE-PE-01 병목 근거를 확인했습니다.\n```" in output
    assert "| `RDB` / `LINE` | LINE-PE-01 MAINTENANCE |" in output
    assert "| `LINE` | LINE-PE-01 MAINTENANCE | `/production-lines/103?mode=read` |" in output


def test_check_rag_chat_scenarios_main_does_not_expose_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_rag_chat_scenarios(args) -> dict:
        return {
            "checkStatus": "PASS",
            "scenarioCount": 1,
            "scenarios": [
                {
                    "scenarioId": "line-bottleneck-with-company-guide",
                    "role": "MANUFACTURING_MANAGER",
                    "intent": "LINE_BOTTLENECK",
                    "securityStatus": "PASSED",
                    "securityCode": None,
                    "requireRdbEvidence": True,
                    "requireVectorSearch": True,
                    "rdbEvidenceCount": 1,
                    "documentSourceCount": 1,
                    "usedVectorSearch": True,
                    "requireLlmGeneration": False,
                    "usedLlmGeneration": False,
                    "sourceCount": 2,
                    "urlCount": 2,
                }
            ],
        }

    monkeypatch.setattr(
        check_rag_chat_scenarios,
        "check_rag_chat_scenarios",
        fake_check_rag_chat_scenarios,
    )
    stdout = StringIO()

    exit_code = check_rag_chat_scenarios.main(
        ["--token", "secret-answer-token"],
        stdout=stdout,
    )

    assert exit_code == 0
    assert "status=PASS" in stdout.getvalue()
    assert "secret-answer-token" not in stdout.getvalue()


def test_check_rag_chat_scenarios_main_formats_markdown_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_rag_chat_scenarios(args) -> dict:
        return {
            "checkStatus": "PASS",
            "scenarioCount": 1,
            "scenarios": [
                {
                    "scenarioId": "line-bottleneck-with-company-guide",
                    "role": "MANUFACTURING_MANAGER",
                    "question": "LINE-PE-01 병목 현황과 대응 기준을 같이 알려줘",
                    "intent": "LINE_BOTTLENECK",
                    "securityStatus": "PASSED",
                    "securityCode": None,
                    "requireRdbEvidence": True,
                    "requireVectorSearch": True,
                    "evidenceCount": 2,
                    "rdbEvidenceCount": 1,
                    "documentSourceCount": 1,
                    "usedVectorSearch": True,
                    "requireLlmGeneration": False,
                    "usedLlmGeneration": False,
                    "sourceCount": 2,
                    "urlCount": 2,
                    "answer": "핵심 답변: LINE-PE-01 병목 근거를 확인했습니다.",
                    "sourceDetails": [
                        {
                            "sourceOrigin": "RDB",
                            "sourceType": "LINE",
                            "title": "LINE-PE-01 MAINTENANCE",
                            "url": "/production-lines/103?mode=read",
                        }
                    ],
                    "urlDetails": [
                        {
                            "type": "LINE",
                            "label": "LINE-PE-01 MAINTENANCE",
                            "url": "/production-lines/103?mode=read",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(
        check_rag_chat_scenarios,
        "check_rag_chat_scenarios",
        fake_check_rag_chat_scenarios,
    )
    stdout = StringIO()

    exit_code = check_rag_chat_scenarios.main(
        ["--token", "secret-answer-token", "--scenario-group", "core", "--markdown"],
        stdout=stdout,
    )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "# RAG 챗봇 시나리오 점검 결과" in output
    assert "## line-bottleneck-with-company-guide" in output
    assert "핵심 답변: LINE-PE-01 병목 근거를 확인했습니다." in output
    assert "secret-answer-token" not in output


def test_check_rag_chat_scenarios_main_returns_one_without_token() -> None:
    stderr = StringIO()

    exit_code = check_rag_chat_scenarios.main(
        ["--base-url", "http://fastapi.local"],
        stderr=stderr,
    )

    assert exit_code == 1
    assert "RAG 챗봇 시나리오 점검 실패" in stderr.getvalue()
    assert "code=CHAT_SECURITY_003" in stderr.getvalue()
