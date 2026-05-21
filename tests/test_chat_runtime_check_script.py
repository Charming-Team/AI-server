from argparse import Namespace
from io import StringIO
from typing import Any

import anyio
import pytest

from app.core.config import Settings
from scripts import check_chat_runtime


def _build_args(**overrides: Any) -> Namespace:
    values = {
        "preset": "none",
        "env_file": None,
        "network": False,
        "require_rdb_evidence": False,
        "require_vector_search": False,
        "require_document_index": False,
        "include_vector_smoke": False,
        "include_document_api_smoke": False,
        "include_answer_api_smoke": False,
        "include_recommendation_api_smoke": False,
        "include_answer_output_policy_smoke": False,
        "include_rdb_chat_scenarios": False,
        "rdb_chat_scenario_group": None,
        "rdb_chat_scenario": None,
        "answer_api_base_url": "http://fastapi.local",
        "answer_api_question": "자재 부족 현황 알려줘",
        "answer_api_role": "MANUFACTURING_MANAGER",
        "answer_api_user_id": 1,
        "answer_api_timeout_seconds": 10.0,
        "answer_api_min_evidence_count": 0,
        "answer_api_min_document_source_count": 0,
        "recommendation_api_base_url": "http://fastapi.local",
        "recommendation_api_keyword": "라인",
        "recommendation_api_role": "MANUFACTURING_MANAGER",
        "recommendation_api_user_id": 1,
        "recommendation_api_timeout_seconds": 10.0,
        "recommendation_api_min_item_count": 1,
        "document_api_base_url": "http://fastapi.local",
        "document_api_smoke_document_id": "smoke-document-api-contract",
        "document_api_timeout_seconds": 10.0,
        "skip_rdb_privilege_check": False,
        "qdrant_min_points": 0,
        "json": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _base_ready_settings(**overrides: Any) -> Settings:
    values = {
        "chat_answer_internal_token": "answer-token",
        "chat_recommendation_internal_token": "recommendation-token",
        "document_index_internal_token": "document-token",
        "llm_enabled": True,
        "llm_base_url": "http://llm.local",
        "llm_model": "test-model",
    }
    values.update(overrides)
    return Settings(**values)


def test_check_chat_runtime_builds_required_components() -> None:
    args = _build_args(
        require_rdb_evidence=True,
        require_vector_search=True,
        require_document_index=True,
    )

    assert check_chat_runtime.build_required_components(args) == [
        "rdbEvidence",
        "qdrantSearch",
        "ragSearchPipeline",
        "documentIndexPipeline",
    ]


def test_check_chat_runtime_builds_required_components_from_full_preset() -> None:
    args = _build_args(preset="full")

    assert check_chat_runtime.build_required_components(args) == [
        "rdbEvidence",
        "qdrantSearch",
        "ragSearchPipeline",
        "documentIndexPipeline",
    ]
    assert args.include_answer_api_smoke is True
    assert args.include_recommendation_api_smoke is True
    assert args.include_document_api_smoke is True
    assert args.include_vector_smoke is True
    assert args.include_answer_output_policy_smoke is True
    assert args.include_rdb_chat_scenarios is True
    assert args.answer_api_min_evidence_count == 1
    assert args.answer_api_min_document_source_count == 1


def test_check_chat_runtime_validate_only_passes_with_rdb_evidence() -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
    )
    args = _build_args(require_rdb_evidence=True)

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["mode"] == "VALIDATE_ONLY"
    assert [step["name"] for step in result["steps"]] == [
        "readiness",
        "rdbEvidenceViews",
    ]
    assert result["steps"][1]["result"]["checkStatus"] == "VALIDATED"
    assert "secret" not in check_chat_runtime.format_json_result(result)


def test_check_chat_runtime_fails_when_required_rdb_is_disabled() -> None:
    settings = _base_ready_settings(evidence_lookup_enabled=True)
    args = _build_args(require_rdb_evidence=True)

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "FAIL"
    assert result["steps"][0]["status"] == "FAIL"
    assert result["steps"][1]["name"] == "rdbEvidenceViews"
    assert result["steps"][1]["status"] == "FAIL"
    assert result["steps"][1]["error"]["code"] == "CHAT_EVIDENCE_004"
    assert result["summary"]["failedStepCount"] == 2
    assert result["summary"]["failedSteps"][0]["name"] == "readiness"
    assert result["summary"]["failedSteps"][1]["name"] == "rdbEvidenceViews"
    assert "RDB DSN" in result["summary"]["nextActions"][1]


def test_check_chat_runtime_validate_only_checks_qdrant_when_vector_is_required() -> None:
    settings = _base_ready_settings(
        qdrant_search_enabled=True,
        embedding_enabled=True,
        qdrant_collection="smap_internal_documents",
    )
    args = _build_args(require_vector_search=True)

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert [step["name"] for step in result["steps"]] == [
        "readiness",
        "qdrantCollection",
        "qdrantDocumentPayloads",
    ]
    assert result["steps"][1]["result"]["checkStatus"] == "VALIDATED"
    assert result["steps"][2]["result"]["checkStatus"] == "VALIDATED"


def test_check_chat_runtime_network_runs_qdrant_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _base_ready_settings(
        qdrant_search_enabled=True,
        embedding_enabled=True,
        qdrant_collection="smap_internal_documents",
    )
    args = _build_args(network=True, require_vector_search=True)
    calls: list[str] = []

    async def fake_qdrant_collection_check(settings_arg, args_arg):
        calls.append("collection")
        return {"checkStatus": "PASS", "collectionName": settings_arg.qdrant_collection}

    async def fake_qdrant_payload_check(settings_arg, args_arg):
        calls.append("payload")
        return {"checkStatus": "PASS", "pointCount": 0}

    monkeypatch.setattr(
        check_chat_runtime,
        "run_qdrant_collection_check",
        fake_qdrant_collection_check,
    )
    monkeypatch.setattr(
        check_chat_runtime,
        "run_qdrant_payload_check",
        fake_qdrant_payload_check,
    )

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["mode"] == "NETWORK"
    assert calls == ["collection", "payload"]


def test_check_chat_runtime_network_runs_vector_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _base_ready_settings(
        qdrant_search_enabled=True,
        embedding_enabled=True,
        qdrant_collection="smap_internal_documents",
    )
    args = _build_args(
        network=True,
        require_vector_search=True,
        include_vector_smoke=True,
    )
    calls: list[str] = []

    async def fake_qdrant_collection_check(settings_arg, args_arg):
        return {"checkStatus": "PASS"}

    async def fake_qdrant_payload_check(settings_arg, args_arg):
        return {"checkStatus": "PASS"}

    async def fake_vector_smoke(settings_arg, args_arg):
        calls.append(settings_arg.qdrant_collection)
        return {"checkStatus": "PASS", "matchedSourceCount": 1}

    monkeypatch.setattr(
        check_chat_runtime,
        "run_qdrant_collection_check",
        fake_qdrant_collection_check,
    )
    monkeypatch.setattr(
        check_chat_runtime,
        "run_qdrant_payload_check",
        fake_qdrant_payload_check,
    )
    monkeypatch.setattr(check_chat_runtime, "run_qdrant_vector_smoke", fake_vector_smoke)

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["steps"][-1]["name"] == "qdrantVectorSmoke"
    assert calls == ["smap_internal_documents"]


def test_check_chat_runtime_validate_only_checks_vector_smoke_metadata() -> None:
    settings = _base_ready_settings(
        qdrant_search_enabled=True,
        embedding_enabled=True,
        qdrant_collection="smap_internal_documents",
    )
    args = _build_args(include_vector_smoke=True)

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["mode"] == "VALIDATE_ONLY"
    assert result["steps"][-1]["name"] == "qdrantVectorSmoke"
    assert result["steps"][-1]["result"] == {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "networkChecked": False,
        "collectionName": "smap_internal_documents",
        "documentId": "smoke-company-line-bottleneck",
        "pointId": "5700a63b-27df-5374-8281-b1e6cd67f3d8",
        "intent": "LINE_BOTTLENECK",
        "role": "MANUFACTURING_MANAGER",
        "embeddingDimension": 1024,
        "qdrantUrlConfigured": True,
        "apiKeyConfigured": False,
    }


def test_check_chat_runtime_vector_smoke_validate_only_validates_sample_point() -> None:
    settings = _base_ready_settings(
        qdrant_search_enabled=True,
        embedding_enabled=True,
        qdrant_collection="smap_internal_documents",
        embedding_dimension=0,
    )
    args = _build_args(include_vector_smoke=True)

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "FAIL"
    assert result["steps"][-1]["name"] == "qdrantVectorSmoke"
    assert result["steps"][-1]["status"] == "FAIL"
    assert result["steps"][-1]["error"] == {
        "code": "CHAT_EMBEDDING_003",
        "message": "Qdrant smoke test 벡터 차원은 1 이상이어야 합니다.",
    }


def test_check_chat_runtime_rdb_preset_enables_core_api_smokes() -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
    )
    args = _build_args(preset="rdb")

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert [step["name"] for step in result["steps"]] == [
        "readiness",
        "answerOutputPolicySmoke",
        "rdbEvidenceViews",
        "rdbChatScenarios",
        "answerApiSmoke",
        "recommendationApiSmoke",
    ]
    assert result["steps"][1]["result"]["checkStatus"] == "PASS"
    assert result["steps"][3]["result"]["scenarioGroups"] == ["core", "access"]
    assert result["steps"][3]["result"]["scenarioCount"] == 9
    assert result["steps"][4]["result"]["minEvidenceCount"] == 1
    assert result["steps"][4]["result"]["requireRdbEvidence"] is True


def test_check_chat_runtime_full_preset_validate_only_runs_all_core_checks() -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
        qdrant_search_enabled=True,
        embedding_enabled=True,
        qdrant_collection="smap_internal_documents",
    )
    args = _build_args(preset="full")

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["mode"] == "VALIDATE_ONLY"
    assert [step["name"] for step in result["steps"]] == [
        "readiness",
        "answerOutputPolicySmoke",
        "rdbEvidenceViews",
        "rdbChatScenarios",
        "qdrantCollection",
        "qdrantDocumentPayloads",
        "qdrantVectorSmoke",
        "documentApiSmoke",
        "answerApiSmoke",
        "recommendationApiSmoke",
    ]
    assert result["requiredComponents"] == [
        "rdbEvidence",
        "qdrantSearch",
        "ragSearchPipeline",
        "documentIndexPipeline",
    ]
    assert result["summary"] == {
        "totalStepCount": 10,
        "passedStepCount": 10,
        "failedStepCount": 0,
        "failedSteps": [],
        "nextActions": [],
    }


def test_check_chat_runtime_validate_only_checks_answer_output_policy_smoke() -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
    )
    args = _build_args(include_answer_output_policy_smoke=True)

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["mode"] == "VALIDATE_ONLY"
    assert [step["name"] for step in result["steps"]] == [
        "readiness",
        "answerOutputPolicySmoke",
        "rdbEvidenceViews",
    ]
    assert result["steps"][1]["result"]["checkStatus"] == "PASS"
    assert result["steps"][1]["result"]["caseCount"] == 4


def test_check_chat_runtime_validate_only_checks_rdb_chat_scenarios() -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
    )
    args = _build_args(
        include_rdb_chat_scenarios=True,
        rdb_chat_scenario_group=["access"],
        rdb_chat_scenario=["operator-financial-blocked"],
    )

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["mode"] == "VALIDATE_ONLY"
    assert [step["name"] for step in result["steps"]] == [
        "readiness",
        "rdbEvidenceViews",
        "rdbChatScenarios",
    ]
    assert result["steps"][2]["result"] == {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "networkChecked": False,
        "baseUrl": "http://fastapi.local",
        "path": "/api/v1/chat/answer",
        "tokenConfigured": True,
        "scenarioGroups": ["access"],
        "scenarioCount": 1,
        "scenarioIds": ["operator-financial-blocked"],
    }


def test_check_chat_runtime_rdb_chat_scenarios_require_token() -> None:
    settings = Settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
    )
    args = _build_args(include_rdb_chat_scenarios=True)

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "FAIL"
    assert result["steps"][-1]["name"] == "rdbChatScenarios"
    assert result["steps"][-1]["status"] == "FAIL"
    assert result["steps"][-1]["error"]["code"] == "CHAT_SECURITY_003"
    assert "Role 기반 접근 제어" in result["summary"]["nextActions"][-1]


def test_check_chat_runtime_network_runs_rdb_chat_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
    )
    args = _build_args(
        network=True,
        include_rdb_chat_scenarios=True,
        rdb_chat_scenario_group=["access"],
        rdb_chat_scenario=["operator-financial-blocked"],
    )
    captured: dict[str, Any] = {}

    async def fake_rdb_check(settings_arg, args_arg):
        return {"checkStatus": "PASS", "viewCount": 7}

    async def fake_rdb_chat_scenarios(scenario_args):
        captured["base_url"] = scenario_args.base_url
        captured["path"] = scenario_args.path
        captured["token"] = scenario_args.token
        captured["scenario_group"] = scenario_args.scenario_group
        captured["scenario"] = scenario_args.scenario
        return {
            "checkStatus": "PASS",
            "scenarioCount": 1,
            "scenarios": [
                {
                    "scenarioId": "operator-financial-blocked",
                    "securityStatus": "BLOCKED_UNAUTHORIZED",
                    "securityCode": "CHAT_SECURITY_004",
                }
            ],
        }

    monkeypatch.setattr(
        check_chat_runtime,
        "run_rdb_evidence_view_check",
        fake_rdb_check,
    )
    monkeypatch.setattr(
        check_chat_runtime.check_rdb_chat_scenarios,
        "check_rdb_chat_scenarios",
        fake_rdb_chat_scenarios,
    )

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["mode"] == "NETWORK"
    assert result["steps"][-1]["name"] == "rdbChatScenarios"
    assert captured == {
        "base_url": "http://fastapi.local",
        "path": "/api/v1/chat/answer",
        "token": "answer-token",
        "scenario_group": ["access"],
        "scenario": ["operator-financial-blocked"],
    }


def test_check_chat_runtime_fails_when_answer_output_policy_smoke_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
    )
    args = _build_args(include_answer_output_policy_smoke=True)

    def fake_answer_output_policy_smoke():
        return {
            "checkStatus": "FAIL",
            "failedCaseCount": 1,
        }

    monkeypatch.setattr(
        check_chat_runtime,
        "run_answer_output_policy_smoke",
        fake_answer_output_policy_smoke,
    )

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "FAIL"
    assert result["steps"][1]["name"] == "answerOutputPolicySmoke"
    assert result["steps"][1]["status"] == "FAIL"
    assert result["summary"]["failedSteps"][0]["name"] == "answerOutputPolicySmoke"
    assert "LLM 출력 보안 정책" in result["summary"]["nextActions"][0]


def test_check_chat_runtime_validate_only_checks_document_api_smoke() -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
    )
    args = _build_args(include_document_api_smoke=True)

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["mode"] == "VALIDATE_ONLY"
    assert result["steps"][-1]["name"] == "documentApiSmoke"
    assert result["steps"][-1]["result"] == {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "networkChecked": False,
        "baseUrl": "http://fastapi.local",
        "indexPath": "/api/v1/chat/internal/documents/index",
        "deletePath": "/api/v1/chat/internal/documents/delete",
        "documentId": "smoke-document-api-contract",
        "tokenConfigured": True,
        "minIndexedCount": 0,
        "allowSkipped": True,
        "requireDocumentIndex": False,
    }


def test_check_chat_runtime_document_api_smoke_requires_real_index_when_required() -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
        qdrant_search_enabled=True,
        embedding_enabled=True,
        qdrant_collection="smap_internal_documents",
    )
    args = _build_args(
        include_document_api_smoke=True,
        require_document_index=True,
    )

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["steps"][-1]["name"] == "documentApiSmoke"
    assert result["steps"][-1]["result"]["minIndexedCount"] == 1
    assert result["steps"][-1]["result"]["allowSkipped"] is False
    assert result["steps"][-1]["result"]["requireDocumentIndex"] is True


def test_check_chat_runtime_document_api_smoke_requires_token() -> None:
    settings = Settings(chat_answer_internal_token="answer-token")
    args = _build_args(include_document_api_smoke=True)

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "FAIL"
    assert result["steps"][-1]["name"] == "documentApiSmoke"
    assert result["steps"][-1]["status"] == "FAIL"
    assert result["steps"][-1]["error"]["code"] == "CHAT_SECURITY_003"


def test_check_chat_runtime_network_runs_document_api_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
    )
    args = _build_args(network=True, include_document_api_smoke=True)
    calls: list[str] = []

    async def fake_index_check(**kwargs):
        calls.append(f"index:{kwargs['document'].document_id}")
        return {
            "checkStatus": "PASS",
            "documentId": kwargs["document"].document_id,
            "indexedCount": 0,
            "skippedReason": "임베딩 기능이 비활성화되어 있습니다.",
        }

    async def fake_delete_check(**kwargs):
        calls.append(f"delete:{kwargs['request'].document_id}")
        return {
            "checkStatus": "PASS",
            "documentId": kwargs["request"].document_id,
            "operationStatus": "completed",
        }

    async def fake_rdb_check(settings_arg, args_arg):
        return {"checkStatus": "PASS", "viewCount": 7}

    monkeypatch.setattr(
        check_chat_runtime.check_document_index_api,
        "check_document_index_api",
        fake_index_check,
    )
    monkeypatch.setattr(
        check_chat_runtime.check_document_delete_api,
        "check_document_delete_api",
        fake_delete_check,
    )
    monkeypatch.setattr(
        check_chat_runtime,
        "run_rdb_evidence_view_check",
        fake_rdb_check,
    )

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["mode"] == "NETWORK"
    assert result["steps"][-1]["name"] == "documentApiSmoke"
    assert calls == [
        "index:smoke-document-api-contract",
        "delete:smoke-document-api-contract",
    ]


def test_check_chat_runtime_network_requires_document_index_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
        qdrant_search_enabled=True,
        embedding_enabled=True,
        qdrant_collection="smap_internal_documents",
    )
    args = _build_args(
        network=True,
        include_document_api_smoke=True,
        require_document_index=True,
    )
    captured: dict[str, Any] = {}

    async def fake_index_check(**kwargs):
        captured["min_indexed_count"] = kwargs["min_indexed_count"]
        captured["allow_skipped"] = kwargs["allow_skipped"]
        return {
            "checkStatus": "PASS",
            "documentId": kwargs["document"].document_id,
            "indexedCount": kwargs["min_indexed_count"],
            "skippedReason": None,
        }

    async def fake_delete_check(**kwargs):
        return {
            "checkStatus": "PASS",
            "documentId": kwargs["request"].document_id,
            "operationStatus": "completed",
        }

    async def fake_rdb_check(settings_arg, args_arg):
        return {"checkStatus": "PASS", "viewCount": 7}

    async def fake_qdrant_collection_check(settings_arg, args_arg):
        return {"checkStatus": "PASS", "collectionName": settings_arg.qdrant_collection}

    async def fake_qdrant_payload_check(settings_arg, args_arg):
        return {"checkStatus": "PASS", "pointCount": 0}

    monkeypatch.setattr(
        check_chat_runtime.check_document_index_api,
        "check_document_index_api",
        fake_index_check,
    )
    monkeypatch.setattr(
        check_chat_runtime.check_document_delete_api,
        "check_document_delete_api",
        fake_delete_check,
    )
    monkeypatch.setattr(
        check_chat_runtime,
        "run_rdb_evidence_view_check",
        fake_rdb_check,
    )
    monkeypatch.setattr(
        check_chat_runtime,
        "run_qdrant_collection_check",
        fake_qdrant_collection_check,
    )
    monkeypatch.setattr(
        check_chat_runtime,
        "run_qdrant_payload_check",
        fake_qdrant_payload_check,
    )

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["steps"][-1]["name"] == "documentApiSmoke"
    assert captured == {
        "min_indexed_count": 1,
        "allow_skipped": False,
    }


def test_check_chat_runtime_validate_only_checks_answer_api_smoke() -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
    )
    args = _build_args(
        include_answer_api_smoke=True,
        answer_api_min_evidence_count=1,
    )

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["mode"] == "VALIDATE_ONLY"
    assert result["steps"][-1]["name"] == "answerApiSmoke"
    assert result["steps"][-1]["result"] == {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "networkChecked": False,
        "baseUrl": "http://fastapi.local",
        "path": "/api/v1/chat/answer",
        "question": "자재 부족 현황 알려줘",
        "role": "MANUFACTURING_MANAGER",
        "tokenConfigured": True,
        "minEvidenceCount": 1,
        "requireRdbEvidence": False,
        "minDocumentSourceCount": 0,
        "requireVectorSearch": False,
    }


def test_check_chat_runtime_answer_api_smoke_requires_token() -> None:
    settings = Settings(document_index_internal_token="document-token")
    args = _build_args(include_answer_api_smoke=True)

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "FAIL"
    assert result["steps"][-1]["name"] == "answerApiSmoke"
    assert result["steps"][-1]["status"] == "FAIL"
    assert result["steps"][-1]["error"]["code"] == "CHAT_SECURITY_003"


def test_check_chat_runtime_network_runs_answer_api_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
    )
    args = _build_args(
        network=True,
        include_answer_api_smoke=True,
        require_rdb_evidence=True,
        answer_api_min_evidence_count=2,
    )
    calls: list[str] = []

    async def fake_rdb_check(settings_arg, args_arg):
        return {"checkStatus": "PASS", "viewCount": 7}

    async def fake_answer_check(**kwargs):
        calls.append(kwargs["request"].question)
        return {
            "checkStatus": "PASS",
            "url": f"{kwargs['base_url']}/{kwargs['path'].lstrip('/')}",
            "intent": "MATERIAL_SHORTAGE",
            "evidenceCount": kwargs["min_evidence_count"],
            "usedRdbEvidence": kwargs["require_rdb_evidence"],
            "usedVectorSearch": kwargs["require_vector_search"],
        }

    monkeypatch.setattr(
        check_chat_runtime,
        "run_rdb_evidence_view_check",
        fake_rdb_check,
    )
    monkeypatch.setattr(
        check_chat_runtime.check_chat_answer,
        "check_chat_answer",
        fake_answer_check,
    )

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["mode"] == "NETWORK"
    assert result["steps"][-1]["name"] == "answerApiSmoke"
    assert calls == ["자재 부족 현황 알려줘"]
    assert result["steps"][-1]["result"]["evidenceCount"] == 2
    assert result["steps"][-1]["result"]["usedRdbEvidence"] is True


def test_check_chat_runtime_validate_only_checks_recommendation_api_smoke() -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
    )
    args = _build_args(
        include_recommendation_api_smoke=True,
        recommendation_api_role="OPERATOR",
    )

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["mode"] == "VALIDATE_ONLY"
    assert result["steps"][-1]["name"] == "recommendationApiSmoke"
    assert result["steps"][-1]["result"] == {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "networkChecked": False,
        "baseUrl": "http://fastapi.local",
        "path": "/api/v1/chat/recommendations",
        "role": "OPERATOR",
        "keywordConfigured": True,
        "tokenConfigured": True,
        "minItemCount": 1,
    }


def test_check_chat_runtime_recommendation_api_smoke_requires_token() -> None:
    settings = Settings(chat_answer_internal_token="answer-token")
    args = _build_args(include_recommendation_api_smoke=True)

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "FAIL"
    assert result["steps"][-1]["name"] == "recommendationApiSmoke"
    assert result["steps"][-1]["status"] == "FAIL"
    assert result["steps"][-1]["error"]["code"] == "CHAT_SECURITY_003"


def test_check_chat_runtime_network_runs_recommendation_api_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _base_ready_settings(
        rdb_evidence_enabled=True,
        rdb_evidence_dsn="postgresql://reader:secret@postgres.local:5432/smap",
    )
    args = _build_args(
        network=True,
        include_recommendation_api_smoke=True,
        recommendation_api_min_item_count=2,
    )
    calls: list[str] = []

    async def fake_rdb_check(settings_arg, args_arg):
        return {"checkStatus": "PASS", "viewCount": 7}

    async def fake_recommendation_check(**kwargs):
        calls.append(kwargs["request"].user.role)
        return {
            "checkStatus": "PASS",
            "role": kwargs["request"].user.role,
            "itemCount": kwargs["min_item_count"],
            "fallbackUsed": kwargs["expect_fallback"],
        }

    monkeypatch.setattr(
        check_chat_runtime,
        "run_rdb_evidence_view_check",
        fake_rdb_check,
    )
    monkeypatch.setattr(
        check_chat_runtime.check_chat_recommendations,
        "check_chat_recommendations",
        fake_recommendation_check,
    )

    result = anyio.run(check_chat_runtime.check_chat_runtime, settings, args)

    assert result["checkStatus"] == "PASS"
    assert result["mode"] == "NETWORK"
    assert result["steps"][-1]["name"] == "recommendationApiSmoke"
    assert calls == ["MANUFACTURING_MANAGER"]
    assert result["steps"][-1]["result"]["itemCount"] == 2


def test_check_chat_runtime_text_output_includes_step_status() -> None:
    output = check_chat_runtime.format_text_result(
        {
            "checkStatus": "FAIL",
            "mode": "VALIDATE_ONLY",
            "networkChecked": False,
            "requiredComponents": ["rdbEvidence"],
            "summary": {
                "totalStepCount": 1,
                "passedStepCount": 0,
                "failedStepCount": 1,
                "failedSteps": [
                    {
                        "name": "readiness",
                        "code": "CHAT_EVIDENCE_004",
                        "message": "RDB Evidence DSN이 설정되지 않았습니다.",
                        "action": "RDB Evidence 설정을 확인하세요.",
                    }
                ],
                "nextActions": ["RDB Evidence 설정을 확인하세요."],
            },
            "steps": [
                {
                    "name": "readiness",
                    "status": "FAIL",
                    "error": {
                        "code": "CHAT_EVIDENCE_004",
                        "message": "RDB Evidence DSN이 설정되지 않았습니다.",
                    },
                }
            ],
        }
    )

    assert "status=FAIL" in output
    assert "requiredComponents=rdbEvidence" in output
    assert "summary=passed:0 failed:1 total:1" in output
    assert "readiness: status=FAIL code=CHAT_EVIDENCE_004" in output
    assert "failure=readiness code=CHAT_EVIDENCE_004" in output
    assert "nextAction=RDB Evidence 설정을 확인하세요." in output


def test_check_chat_runtime_builds_failure_summary_from_result_error() -> None:
    steps = [
        {
            "name": "qdrantCollection",
            "status": "FAIL",
            "result": {
                "checkStatus": "FAIL",
                "error": {
                    "code": "CHAT_QDRANT_002",
                    "message": "Qdrant 컬렉션 조회에 실패했습니다.",
                },
            },
        }
    ]

    summary = check_chat_runtime.build_runtime_summary(steps)

    assert summary["totalStepCount"] == 1
    assert summary["passedStepCount"] == 0
    assert summary["failedStepCount"] == 1
    assert summary["failedSteps"][0] == {
        "name": "qdrantCollection",
        "code": "CHAT_QDRANT_002",
        "message": "Qdrant 컬렉션 조회에 실패했습니다.",
        "action": (
            "Qdrant URL, collection 이름, embedding dimension 설정이 일치하는지 "
            "확인하세요."
        ),
    }


def test_check_chat_runtime_main_returns_two_on_failed_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_chat_runtime(settings, args):
        return {
            "checkStatus": "FAIL",
            "mode": "VALIDATE_ONLY",
            "networkChecked": False,
            "requiredComponents": ["rdbEvidence"],
            "steps": [],
        }

    monkeypatch.setattr(check_chat_runtime, "check_chat_runtime", fake_check_chat_runtime)
    stdout = StringIO()

    exit_code = check_chat_runtime.main(["--require-rdb-evidence"], stdout=stdout)

    assert exit_code == 2
    assert "status=FAIL" in stdout.getvalue()


def test_check_chat_runtime_main_returns_zero_on_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check_chat_runtime(settings, args):
        return {
            "checkStatus": "PASS",
            "mode": "VALIDATE_ONLY",
            "networkChecked": False,
            "requiredComponents": [],
            "steps": [],
        }

    monkeypatch.setattr(check_chat_runtime, "check_chat_runtime", fake_check_chat_runtime)
    stdout = StringIO()

    exit_code = check_chat_runtime.main(["--json"], stdout=stdout)

    assert exit_code == 0
    assert '"checkStatus": "PASS"' in stdout.getvalue()
