from argparse import Namespace
from io import StringIO
from typing import Any

import anyio

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode
from scripts import check_chat_quality_scenarios


def _build_args(**overrides: Any) -> Namespace:
    values = {
        "base_url": "http://fastapi.local",
        "path": None,
        "token": "answer-token",
        "env_file": None,
        "timeout_seconds": 10.0,
        "profile": "minimal",
        "role": "MANUFACTURING_MANAGER",
        "user_id": 1,
        "company_name": "S-MAP",
        "require_llm_generation": False,
        "require_llm_cache_miss": False,
        "max_llm_total_tokens": None,
        "network": False,
        "markdown": False,
        "json": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _scenario_result(
    *,
    scenario_id: str,
    security_status: str = "PASSED",
    rdb_evidence_count: int = 1,
    document_source_count: int = 0,
    used_llm_generation: bool = False,
    llm_cache_hit: bool = False,
    total_tokens: int | None = None,
    url_count: int = 1,
) -> dict[str, Any]:
    llm_usage = None
    if total_tokens is not None:
        llm_usage = {
            "promptTokens": total_tokens - 20,
            "completionTokens": 20,
            "totalTokens": total_tokens,
        }
    return {
        "scenarioId": scenario_id,
        "securityStatus": security_status,
        "securityCode": (
            "CHAT_SECURITY_004"
            if security_status == "BLOCKED_UNAUTHORIZED"
            else None
        ),
        "rdbEvidenceCount": rdb_evidence_count,
        "documentSourceCount": document_source_count,
        "usedLlmGeneration": used_llm_generation,
        "llmCacheHit": llm_cache_hit,
        "llmUsage": llm_usage,
        "urlCount": url_count,
    }


def test_check_chat_quality_scenarios_validate_only_builds_execution_plan() -> None:
    args = _build_args(
        profile="standard",
        require_llm_generation=True,
        require_llm_cache_miss=True,
        max_llm_total_tokens=300,
    )
    result = check_chat_quality_scenarios.build_validate_only_result(
        args,
        Settings(api_v1_prefix="/api/v1"),
    )

    assert result["checkStatus"] == "VALIDATED"
    assert result["mode"] == "VALIDATE_ONLY"
    assert result["path"] == "/api/v1/chat/answer"
    assert result["tokenConfigured"] is True
    assert result["rdbScenarioGroups"] == ["core", "access"]
    assert result["ragScenarioGroups"] == ["core", "access", "company"]
    assert result["requireLlmGeneration"] is True
    assert result["requireLlmCacheMiss"] is True
    assert result["maxLlmTotalTokens"] == 300
    assert "Role 기반 금액성 정보 차단" in result["qualityCriteria"]
    assert "Role별 실제 업무 질문 매트릭스 확인" in result["qualityCriteria"]


def test_check_chat_quality_scenarios_business_profile_builds_role_execution_plan() -> None:
    result = check_chat_quality_scenarios.build_validate_only_result(
        _build_args(profile="business", max_llm_total_tokens=2500),
        Settings(api_v1_prefix="/api/v1"),
    )

    assert result["checkStatus"] == "VALIDATED"
    assert result["profile"] == "business"
    assert result["rdbScenarioGroups"] == ["core", "access", "filtered"]
    assert result["ragScenarioGroups"] == ["core", "company", "role"]
    assert result["maxLlmTotalTokens"] == 2500


def test_check_chat_quality_scenarios_network_runs_rdb_rag_and_summarizes_quality() -> None:
    captured: dict[str, Any] = {}

    async def fake_rdb_checker(args: Namespace) -> dict[str, Any]:
        captured["rdb_groups"] = args.scenario_group
        captured["rdb_require_llm_generation"] = args.require_llm_generation
        captured["rdb_require_llm_cache_miss"] = args.require_llm_cache_miss
        captured["rdb_max_llm_total_tokens"] = args.max_llm_total_tokens
        return {
            "checkStatus": "PASS",
            "scenarioCount": 2,
            "scenarios": [
                _scenario_result(
                    scenario_id="operator-financial-blocked",
                    security_status="BLOCKED_UNAUTHORIZED",
                    rdb_evidence_count=0,
                    url_count=0,
                ),
                _scenario_result(
                    scenario_id="operator-urgent-order-allowed",
                    rdb_evidence_count=2,
                    used_llm_generation=True,
                    total_tokens=80,
                ),
            ],
        }

    async def fake_rag_checker(args: Namespace) -> dict[str, Any]:
        captured["rag_groups"] = args.scenario_group
        captured["rag_require_llm_generation"] = args.require_llm_generation
        captured["rag_require_llm_cache_miss"] = args.require_llm_cache_miss
        captured["rag_max_llm_total_tokens"] = args.max_llm_total_tokens
        return {
            "checkStatus": "PASS",
            "scenarioCount": 1,
            "scenarios": [
                _scenario_result(
                    scenario_id="company-overview-document-allowed",
                    rdb_evidence_count=0,
                    document_source_count=2,
                    used_llm_generation=True,
                    llm_cache_hit=False,
                    total_tokens=90,
                )
            ],
        }

    result = anyio.run(
        check_chat_quality_scenarios.check_chat_quality_scenarios,
        _build_args(
            network=True,
            require_llm_generation=True,
            require_llm_cache_miss=True,
            max_llm_total_tokens=200,
        ),
        fake_rdb_checker,
        fake_rag_checker,
    )

    assert result["checkStatus"] == "PASS"
    assert captured == {
        "rdb_groups": ["access"],
        "rdb_require_llm_generation": True,
        "rdb_require_llm_cache_miss": True,
        "rdb_max_llm_total_tokens": 200,
        "rag_groups": ["company"],
        "rag_require_llm_generation": True,
        "rag_require_llm_cache_miss": True,
        "rag_max_llm_total_tokens": 200,
    }
    assert result["qualitySummary"] == {
        "scenarioCount": 3,
        "blockedUnauthorizedCount": 1,
        "rdbEvidenceScenarioCount": 1,
        "qdrantSourceScenarioCount": 1,
        "llmGenerationCount": 2,
        "llmCacheHitCount": 0,
        "totalLlmTokens": 170,
        "maxScenarioLlmTokens": 90,
        "urlScenarioCount": 2,
    }


def test_check_chat_quality_scenarios_formats_markdown_validate_only() -> None:
    result = check_chat_quality_scenarios.build_validate_only_result(
        _build_args(profile="minimal"),
        Settings(),
    )

    output = check_chat_quality_scenarios.format_markdown_result(result)

    assert "# 챗봇 품질 시나리오 점검 결과" in output
    assert "- 프로필: `minimal`" in output
    assert "## 품질 기준" in output
    assert "- RDB 시나리오 그룹: `access`" in output
    assert "- RAG 시나리오 그룹: `company`" in output
    assert "OPERATOR 비금액성 조회 허용" in output


def test_check_chat_quality_scenarios_main_does_not_expose_secret(
    monkeypatch,
) -> None:
    async def fake_check_chat_quality_scenarios(args) -> dict[str, Any]:
        return {
            "checkStatus": "VALIDATED",
            "mode": "VALIDATE_ONLY",
            "networkChecked": False,
            "profile": "minimal",
            "baseUrl": "http://fastapi.local",
            "path": "/api/v1/chat/answer",
            "tokenConfigured": True,
            "rdbScenarioGroups": ["access"],
            "ragScenarioGroups": ["company"],
            "qualityCriteria": list(check_chat_quality_scenarios.QUALITY_CRITERIA),
            "requireLlmGeneration": False,
            "requireLlmCacheMiss": False,
            "maxLlmTotalTokens": None,
        }

    monkeypatch.setattr(
        check_chat_quality_scenarios,
        "check_chat_quality_scenarios",
        fake_check_chat_quality_scenarios,
    )
    stdout = StringIO()

    exit_code = check_chat_quality_scenarios.main(
        ["--token", "secret-answer-token"],
        stdout=stdout,
    )

    assert exit_code == 0
    assert "status=VALIDATED" in stdout.getvalue()
    assert "secret-answer-token" not in stdout.getvalue()


def test_check_chat_quality_scenarios_main_returns_one_on_service_error(
    monkeypatch,
) -> None:
    async def fake_check_chat_quality_scenarios(args) -> dict[str, Any]:
        raise ChatServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_SECURITY_003,
            message="FastAPI chat answer internal token이 설정되지 않았습니다.",
        )

    monkeypatch.setattr(
        check_chat_quality_scenarios,
        "check_chat_quality_scenarios",
        fake_check_chat_quality_scenarios,
    )
    stderr = StringIO()

    exit_code = check_chat_quality_scenarios.main(["--network"], stderr=stderr)

    assert exit_code == 1
    assert "챗봇 품질 시나리오 점검 실패" in stderr.getvalue()
    assert "code=CHAT_SECURITY_003" in stderr.getvalue()
