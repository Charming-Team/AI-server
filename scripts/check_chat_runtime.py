import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any, TextIO

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatIntent
from scripts import (
    chat_check_common,
    check_answer_output_policy,
    check_chat_answer,
    check_chat_readiness,
    check_chat_recommendations,
    check_document_delete_api,
    check_document_index_api,
    check_llm_completion,
    check_qdrant_collection,
    check_qdrant_document_payloads,
    check_qdrant_vector_search,
    check_rag_chat_scenarios,
    check_rag_end_to_end,
    check_rdb_chat_scenarios,
    check_rdb_evidence_views,
)

StepRunner = Callable[[], Awaitable[dict[str, Any]] | dict[str, Any]]

STEP_ACTION_GUIDE = {
    "readiness": (
        "readiness 구성값을 확인하고 누락된 내부 토큰, RDB, Qdrant, LLM 설정을 "
        "보완하세요."
    ),
    "answerOutputPolicySmoke": (
        "LLM 출력 보안 정책의 프롬프트 인젝션, 민감정보, OPERATOR 금액성 답변 "
        "차단 규칙을 확인하세요."
    ),
    "llmCompletionSmoke": (
        "LLM_ENABLED, LLM_BASE_URL, LLM_MODEL 설정과 OpenAI-compatible "
        "chat completions 응답 형식을 확인하세요."
    ),
    "rdbEvidenceViews": (
        "RDB DSN, chat_evidence view 생성 여부, smap_chat_reader read-only 권한을 "
        "확인하세요."
    ),
    "rdbChatScenarios": (
        "챗봇 답변 API, RDB Evidence, Role 기반 접근 제어 시나리오의 intent, "
        "securityStatus, securityCode 결과를 확인하세요."
    ),
    "ragChatScenarios": (
        "RDB Evidence, Qdrant 문서 출처, Vector Search 사용 여부가 함께 충족되는지 "
        "확인하세요."
    ),
    "ragEndToEndSmoke": (
        "FastAPI 문서 등록, 챗봇 답변, 문서 삭제 API가 같은 설정과 토큰으로 "
        "연결되는지 확인하세요."
    ),
    "qdrantCollection": (
        "Qdrant URL, collection 이름, embedding dimension 설정이 일치하는지 "
        "확인하세요."
    ),
    "qdrantDocumentPayloads": (
        "Qdrant payload의 documentId, allowedRoles, intentTags, url 메타데이터를 "
        "확인하세요."
    ),
    "qdrantVectorSmoke": (
        "Embedding/Qdrant 연결과 smoke 문서 저장, 검색, 삭제 경로를 확인하세요."
    ),
    "documentApiSmoke": (
        "문서 인덱싱 내부 토큰, FastAPI base URL, Qdrant/Embedding 설정을 "
        "확인하세요."
    ),
    "answerApiSmoke": (
        "챗봇 답변 내부 토큰, FastAPI base URL, RDB/Qdrant Evidence, LLM 생성 조건을 "
        "확인하세요."
    ),
    "recommendationApiSmoke": (
        "추천 질문 내부 토큰, Role별 추천 규칙, OPERATOR read-only URL 조건을 "
        "확인하세요."
    ),
}
DEFAULT_STEP_ACTION = "실패한 step의 설정과 네트워크 연결, 응답 형식을 확인하세요."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="챗봇 RDB/Qdrant/Vector 런타임 준비 상태를 한 번에 점검합니다."
    )
    parser.add_argument(
        "--preset",
        choices=("none", "rdb", "qdrant", "rag", "full"),
        default="none",
        help=(
            "자주 쓰는 점검 옵션 묶음. rdb는 RDB Evidence와 답변/추천 API, "
            "qdrant는 Qdrant 컬렉션/페이로드/벡터 smoke, "
            "rag는 Qdrant/문서/답변/추천 API, full은 전체 경로를 점검합니다."
        ),
    )
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. 생략하면 기본 .env 설정을 사용합니다.",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="RDB와 Qdrant에 실제 네트워크 연결을 수행합니다.",
    )
    parser.add_argument(
        "--require-rdb-evidence",
        action="store_true",
        help="RDB Evidence View 파이프라인이 준비되어 있어야 합니다.",
    )
    parser.add_argument(
        "--require-vector-search",
        action="store_true",
        help="Qdrant Vector Search 파이프라인이 준비되어 있어야 합니다.",
    )
    parser.add_argument(
        "--require-document-index",
        action="store_true",
        help="문서 인덱싱 파이프라인이 준비되어 있어야 합니다.",
    )
    parser.add_argument(
        "--require-llm-generation",
        action="store_true",
        help="챗봇 답변 API smoke check에서 LLM 답변 생성 사용을 요구합니다.",
    )
    parser.add_argument(
        "--include-vector-smoke",
        action="store_true",
        help="Qdrant에 임시 문서를 저장/검색/삭제하는 Vector smoke check를 수행합니다.",
    )
    parser.add_argument(
        "--include-document-api-smoke",
        action="store_true",
        help=(
            "FastAPI 문서 인덱싱/삭제 내부 API를 smoke 문서로 호출해 "
            "계약과 토큰 설정을 점검합니다."
        ),
    )
    parser.add_argument(
        "--include-answer-api-smoke",
        action="store_true",
        help=(
            "FastAPI 챗봇 답변 내부 API를 smoke 질문으로 호출해 "
            "응답 계약과 Evidence 조건을 점검합니다."
        ),
    )
    parser.add_argument(
        "--include-recommendation-api-smoke",
        action="store_true",
        help=(
            "FastAPI 추천 질문 내부 API를 smoke 요청으로 호출해 "
            "Role 기반 추천 계약을 점검합니다."
        ),
    )
    parser.add_argument(
        "--include-answer-output-policy-smoke",
        action="store_true",
        help=(
            "LLM 출력 보안 정책의 핵심 차단 케이스를 로컬에서 점검합니다. "
            "네트워크 연결은 수행하지 않습니다."
        ),
    )
    parser.add_argument(
        "--include-llm-smoke",
        action="store_true",
        help=(
            "LLM chat completions 연결을 점검합니다. 네트워크 모드에서는 "
            "실제 LLM 서버를 호출합니다."
        ),
    )
    parser.add_argument(
        "--include-rdb-chat-scenarios",
        action="store_true",
        help=(
            "RDB Evidence 기반 챗봇 질문 시나리오를 실행합니다. "
            "기본으로 core/access 시나리오 그룹을 점검합니다."
        ),
    )
    parser.add_argument(
        "--include-rag-chat-scenarios",
        action="store_true",
        help=(
            "RDB Evidence와 Qdrant 문서 출처가 함께 필요한 RAG 챗봇 질문 "
            "시나리오를 실행합니다. 기본으로 core 시나리오 그룹을 점검합니다."
        ),
    )
    parser.add_argument(
        "--include-rag-end-to-end-smoke",
        action="store_true",
        help=(
            "FastAPI 문서 등록, 챗봇 답변, 문서 삭제까지 하나의 RAG smoke "
            "흐름으로 점검합니다."
        ),
    )
    parser.add_argument(
        "--rdb-chat-scenario-group",
        action="append",
        choices=sorted(check_rdb_chat_scenarios.RDB_CHAT_SCENARIO_GROUPS),
        help=(
            "RDB 챗봇 시나리오 그룹입니다. core, access, filtered 중 선택하며 "
            "여러 번 지정할 수 있습니다. 생략하면 core/access를 실행합니다."
        ),
    )
    parser.add_argument(
        "--rdb-chat-scenario",
        action="append",
        choices=[
            scenario.scenario_id
            for scenario in check_rdb_chat_scenarios.ALL_RDB_CHAT_SCENARIOS
        ],
        help="실행할 RDB 챗봇 시나리오 ID입니다. 여러 번 지정할 수 있습니다.",
    )
    parser.add_argument(
        "--rag-chat-scenario-group",
        action="append",
        choices=sorted(check_rag_chat_scenarios.RAG_CHAT_SCENARIO_GROUPS),
        help=(
            "RAG 챗봇 시나리오 그룹입니다. core, access 중 선택하며 "
            "여러 번 지정할 수 있습니다. 생략하면 core를 실행합니다."
        ),
    )
    parser.add_argument(
        "--rag-chat-scenario",
        action="append",
        choices=[
            scenario.scenario_id
            for scenario in check_rag_chat_scenarios.ALL_RAG_CHAT_SCENARIOS
        ],
        help="실행할 RAG 챗봇 시나리오 ID입니다. 여러 번 지정할 수 있습니다.",
    )
    parser.add_argument(
        "--recommendation-api-base-url",
        default=check_chat_recommendations.DEFAULT_BASE_URL,
        help="추천 질문 API smoke check에 사용할 FastAPI base URL",
    )
    parser.add_argument(
        "--recommendation-api-keyword",
        default=check_chat_recommendations.DEFAULT_KEYWORD,
        help="추천 질문 API smoke check 키워드",
    )
    parser.add_argument(
        "--recommendation-api-role",
        default=check_chat_recommendations.DEFAULT_ROLE,
        help="추천 질문 API smoke check 사용자 Role",
    )
    parser.add_argument(
        "--recommendation-api-user-id",
        type=int,
        default=1,
        help="추천 질문 API smoke check 사용자 ID",
    )
    parser.add_argument(
        "--recommendation-api-timeout-seconds",
        type=float,
        default=10.0,
        help="추천 질문 API smoke check HTTP timeout seconds",
    )
    parser.add_argument(
        "--recommendation-api-min-item-count",
        type=int,
        default=1,
        help="추천 질문 API smoke check에서 요구하는 최소 추천 질문 개수",
    )
    parser.add_argument(
        "--answer-api-base-url",
        default=check_chat_answer.DEFAULT_BASE_URL,
        help="챗봇 답변 API smoke check에 사용할 FastAPI base URL",
    )
    parser.add_argument(
        "--answer-api-question",
        default=check_chat_answer.DEFAULT_QUESTION,
        help="챗봇 답변 API smoke check 질문",
    )
    parser.add_argument(
        "--answer-api-role",
        default="MANUFACTURING_MANAGER",
        help="챗봇 답변 API smoke check 사용자 Role",
    )
    parser.add_argument(
        "--answer-api-user-id",
        type=int,
        default=1,
        help="챗봇 답변 API smoke check 사용자 ID",
    )
    parser.add_argument(
        "--answer-api-timeout-seconds",
        type=float,
        default=10.0,
        help="챗봇 답변 API smoke check HTTP timeout seconds",
    )
    parser.add_argument(
        "--answer-api-min-evidence-count",
        type=int,
        default=0,
        help="챗봇 답변 API smoke check에서 요구하는 최소 Evidence 개수",
    )
    parser.add_argument(
        "--answer-api-min-document-source-count",
        type=int,
        default=0,
        help="챗봇 답변 API smoke check에서 요구하는 최소 Qdrant 문서 출처 개수",
    )
    parser.add_argument(
        "--document-api-base-url",
        default=check_document_index_api.DEFAULT_BASE_URL,
        help="문서 인덱싱/삭제 API smoke check에 사용할 FastAPI base URL",
    )
    parser.add_argument(
        "--document-api-smoke-document-id",
        default=check_document_delete_api.DEFAULT_DOCUMENT_ID,
        help="문서 API smoke check에 사용할 임시 documentId",
    )
    parser.add_argument(
        "--document-api-timeout-seconds",
        type=float,
        default=10.0,
        help="문서 API smoke check HTTP timeout seconds",
    )
    parser.add_argument(
        "--skip-rdb-privilege-check",
        action="store_true",
        help="RDB View SELECT 점검만 수행하고 read-only 권한 점검은 건너뜁니다.",
    )
    parser.add_argument(
        "--qdrant-min-points",
        type=int,
        default=0,
        help="Qdrant payload 점검에서 요구하는 최소 point 개수",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON",
    )
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    if args.env_file:
        return Settings(_env_file=args.env_file)
    return Settings()


def apply_runtime_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = getattr(args, "preset", "none")
    if preset == "none":
        return args

    if preset in {"rdb", "rag", "full"}:
        args.require_rdb_evidence = True

    if preset in {"qdrant", "rag", "full"}:
        args.include_vector_smoke = True

    if preset in {"rag", "full"}:
        args.require_vector_search = True
        args.require_document_index = True
        args.include_document_api_smoke = True
        args.include_rag_chat_scenarios = True
        args.include_rag_end_to_end_smoke = True
        args.answer_api_min_document_source_count = max(
            args.answer_api_min_document_source_count,
            1,
        )

    if preset in {"rdb", "rag", "full"}:
        args.include_answer_api_smoke = True
        args.include_recommendation_api_smoke = True
        args.include_answer_output_policy_smoke = True
    if preset in {"rdb", "full"}:
        args.include_rdb_chat_scenarios = True
    if preset == "full":
        args.require_llm_generation = True
        args.include_llm_smoke = True
    args.answer_api_min_evidence_count = max(args.answer_api_min_evidence_count, 1)
    return args


def build_required_components(args: argparse.Namespace) -> list[str]:
    args = apply_runtime_preset(args)
    return check_chat_readiness.build_required_components(args)


async def check_chat_runtime(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, Any]:
    args = apply_runtime_preset(args)
    required_components = build_required_components(args)
    readiness_result = check_chat_readiness.build_readiness_result(
        settings,
        required_components=required_components,
    )
    steps = [
        await run_step(
            "readiness",
            lambda: readiness_result,
            fail_when=lambda result: should_fail_readiness(result, args),
        )
    ]

    if args.include_answer_output_policy_smoke:
        steps.append(
            await run_step(
                "answerOutputPolicySmoke",
                run_answer_output_policy_smoke,
                fail_when=lambda result: result["checkStatus"] != "PASS",
            )
        )

    if args.include_llm_smoke:
        steps.append(
            await run_step(
                "llmCompletionSmoke",
                lambda: run_llm_completion_smoke(settings, args),
            )
        )

    if should_check_rdb(settings, args):
        steps.append(
            await run_step(
                "rdbEvidenceViews",
                lambda: run_rdb_evidence_view_check(settings, args),
            )
        )

    if args.include_rdb_chat_scenarios:
        steps.append(
            await run_step(
                "rdbChatScenarios",
                lambda: run_rdb_chat_scenarios(settings, args),
            )
        )

    if args.include_rag_chat_scenarios:
        steps.append(
            await run_step(
                "ragChatScenarios",
                lambda: run_rag_chat_scenarios(settings, args),
            )
        )

    if should_check_qdrant(settings, args):
        steps.append(
            await run_step(
                "qdrantCollection",
                lambda: run_qdrant_collection_check(settings, args),
            )
        )
        steps.append(
            await run_step(
                "qdrantDocumentPayloads",
                lambda: run_qdrant_payload_check(settings, args),
            )
        )

    if args.include_vector_smoke:
        steps.append(
            await run_step(
                "qdrantVectorSmoke",
                lambda: run_qdrant_vector_smoke(settings, args),
            )
        )

    if args.include_document_api_smoke:
        steps.append(
            await run_step(
                "documentApiSmoke",
                lambda: run_document_api_smoke(settings, args),
            )
        )

    if args.include_rag_end_to_end_smoke:
        steps.append(
            await run_step(
                "ragEndToEndSmoke",
                lambda: run_rag_end_to_end_smoke(settings, args),
            )
        )

    if args.include_answer_api_smoke:
        steps.append(
            await run_step(
                "answerApiSmoke",
                lambda: run_answer_api_smoke(settings, args),
            )
        )

    if args.include_recommendation_api_smoke:
        steps.append(
            await run_step(
                "recommendationApiSmoke",
                lambda: run_recommendation_api_smoke(settings, args),
            )
        )

    check_status = "PASS" if all(step["status"] == "PASS" for step in steps) else "FAIL"
    summary = build_runtime_summary(steps)
    return {
        "checkStatus": check_status,
        "mode": "NETWORK" if args.network else "VALIDATE_ONLY",
        "networkChecked": bool(args.network),
        "requiredComponents": required_components,
        "summary": summary,
        "steps": steps,
    }


async def run_step(
    name: str,
    runner: StepRunner,
    fail_when: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    try:
        result_or_awaitable = runner()
        if hasattr(result_or_awaitable, "__await__"):
            result = await result_or_awaitable
        else:
            result = result_or_awaitable
        status = "FAIL" if fail_when and fail_when(result) else "PASS"
        return {
            "name": name,
            "status": status,
            "result": result,
        }
    except ChatServiceError as exc:
        return {
            "name": name,
            "status": "FAIL",
            "error": {
                "code": exc.code.value,
                "message": exc.message,
            },
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "FAIL",
            "error": {
                "code": "UNKNOWN",
                "message": str(exc),
            },
        }


def should_check_rdb(settings: Settings, args: argparse.Namespace) -> bool:
    if getattr(args, "preset", "none") == "qdrant" and not args.require_rdb_evidence:
        return False
    return bool(settings.rdb_evidence_enabled or args.require_rdb_evidence)


def should_check_qdrant(settings: Settings, args: argparse.Namespace) -> bool:
    return bool(
        settings.qdrant_search_enabled
        or settings.embedding_enabled
        or args.require_vector_search
        or args.require_document_index
        or args.include_vector_smoke
    )


def should_fail_readiness(result: dict[str, Any], args: argparse.Namespace) -> bool:
    if getattr(args, "preset", "none") == "qdrant":
        return bool(result.get("requirementFailures"))
    return result["status"] != "ready"


def build_runtime_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    failed_steps = [step for step in steps if step["status"] != "PASS"]
    failure_items = [build_failure_item(step) for step in failed_steps]
    next_actions = list(
        dict.fromkeys(item["action"] for item in failure_items if item["action"])
    )
    return {
        "totalStepCount": len(steps),
        "passedStepCount": len(steps) - len(failed_steps),
        "failedStepCount": len(failed_steps),
        "failedSteps": failure_items,
        "nextActions": next_actions,
    }


def build_failure_item(step: dict[str, Any]) -> dict[str, Any]:
    code, message = extract_failure_reason(step)
    return {
        "name": step["name"],
        "code": code,
        "message": message,
        "action": STEP_ACTION_GUIDE.get(step["name"], DEFAULT_STEP_ACTION),
    }


def extract_failure_reason(step: dict[str, Any]) -> tuple[str | None, str]:
    if "error" in step:
        error = step["error"]
        return error.get("code"), error.get("message", "step 실행에 실패했습니다.")

    result = step.get("result")
    if isinstance(result, dict):
        requirement_failures = result.get("requirementFailures")
        if isinstance(requirement_failures, list) and requirement_failures:
            first_failure = requirement_failures[0]
            if isinstance(first_failure, dict):
                return (
                    _optional_text(first_failure.get("code")),
                    _optional_text(first_failure.get("reason"))
                    or "필수 readiness 조건을 만족하지 못했습니다.",
                )

        error = result.get("error")
        if isinstance(error, dict):
            return (
                _optional_text(error.get("code")),
                _optional_text(error.get("message")) or "step 결과가 실패했습니다.",
            )

        status = result.get("status") or result.get("checkStatus")
        if isinstance(status, str):
            return None, f"step 결과 상태가 {status}입니다."

    return None, "step 결과가 실패했습니다."


def _optional_text(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


async def run_rdb_evidence_view_check(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.network:
        return check_rdb_evidence_views.build_validate_only_result(settings)
    return await check_rdb_evidence_views.check_rdb_evidence_views(
        settings,
        check_privileges=not args.skip_rdb_privilege_check,
    )


async def run_rdb_chat_scenarios(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, Any]:
    token = check_chat_answer.resolve_answer_token(
        argparse.Namespace(token=None),
        settings,
    )
    path = f"{settings.api_v1_prefix}/chat/answer"
    scenario_groups = args.rdb_chat_scenario_group or ["core", "access"]
    scenarios = check_rdb_chat_scenarios.select_scenarios(
        args.rdb_chat_scenario,
        scenario_groups,
    )

    if not args.network:
        return {
            "checkStatus": "VALIDATED",
            "mode": "VALIDATE_ONLY",
            "networkChecked": False,
            "baseUrl": args.answer_api_base_url,
            "path": path,
            "tokenConfigured": bool(token),
            "scenarioGroups": scenario_groups,
            "scenarioCount": len(scenarios),
            "scenarioIds": [scenario.scenario_id for scenario in scenarios],
        }

    scenario_args = argparse.Namespace(
        base_url=args.answer_api_base_url,
        path=path,
        token=token,
        env_file=None,
        timeout_seconds=args.answer_api_timeout_seconds,
        role=args.answer_api_role,
        user_id=args.answer_api_user_id,
        company_name="S-MAP",
        session_id=1,
        message_id=1,
        requested_at=chat_check_common.DEFAULT_REQUESTED_AT,
        scenario=args.rdb_chat_scenario,
        scenario_group=scenario_groups,
        min_evidence_count=None,
        json=False,
    )
    return await check_rdb_chat_scenarios.check_rdb_chat_scenarios(scenario_args)


async def run_rag_chat_scenarios(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, Any]:
    token = check_chat_answer.resolve_answer_token(
        argparse.Namespace(token=None),
        settings,
    )
    path = f"{settings.api_v1_prefix}/chat/answer"
    scenario_groups = args.rag_chat_scenario_group or ["core"]
    scenarios = check_rag_chat_scenarios.select_scenarios(
        args.rag_chat_scenario,
        scenario_groups,
    )

    if not args.network:
        return {
            "checkStatus": "VALIDATED",
            "mode": "VALIDATE_ONLY",
            "networkChecked": False,
            "baseUrl": args.answer_api_base_url,
            "path": path,
            "tokenConfigured": bool(token),
            "scenarioGroups": scenario_groups,
            "scenarioCount": len(scenarios),
            "scenarioIds": [scenario.scenario_id for scenario in scenarios],
        }

    scenario_args = argparse.Namespace(
        base_url=args.answer_api_base_url,
        path=path,
        token=token,
        env_file=None,
        timeout_seconds=args.answer_api_timeout_seconds,
        role=args.answer_api_role,
        user_id=args.answer_api_user_id,
        company_name="S-MAP",
        session_id=1,
        message_id=1,
        requested_at=chat_check_common.DEFAULT_REQUESTED_AT,
        scenario=args.rag_chat_scenario,
        scenario_group=scenario_groups,
        min_evidence_count=None,
        min_rdb_evidence_count=None,
        min_document_source_count=None,
        json=False,
    )
    return await check_rag_chat_scenarios.check_rag_chat_scenarios(scenario_args)


async def run_qdrant_collection_check(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.network:
        return check_qdrant_collection.build_validate_only_result(settings)
    result = await check_qdrant_collection.check_collection(settings)
    error = check_qdrant_collection.build_dimension_mismatch_error(result)
    return {
        "checkStatus": "PASS" if result.is_dimension_matched else "FAIL",
        **result.__dict__,
        "error": error.model_dump(mode="json") if error is not None else None,
    }


async def run_qdrant_payload_check(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.network:
        return check_qdrant_document_payloads.build_validate_only_result(
            settings,
            limit=20,
            min_points=args.qdrant_min_points,
        )
    return await check_qdrant_document_payloads.check_qdrant_document_payloads(
        settings,
        limit=20,
        min_points=args.qdrant_min_points,
    )


async def run_qdrant_vector_smoke(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, Any]:
    smoke_args = argparse.Namespace(
        document_id=check_qdrant_vector_search.DEFAULT_DOCUMENT_ID,
        title=check_qdrant_vector_search.DEFAULT_TITLE,
        content=check_qdrant_vector_search.DEFAULT_CONTENT,
        url=check_qdrant_vector_search.DEFAULT_URL,
        intent=ChatIntent.LINE_BOTTLENECK.value,
        role="MANUFACTURING_MANAGER",
        company_name="S-MAP",
        question=check_qdrant_vector_search.DEFAULT_QUESTION,
        user_id=1,
        session_id=1,
        message_id=1,
        requested_at=chat_check_common.DEFAULT_REQUESTED_AT,
    )
    document = check_qdrant_vector_search.build_sample_document(smoke_args)
    if not args.network:
        vector = check_qdrant_vector_search.build_static_vector(
            settings.embedding_dimension
        )
        point = check_qdrant_vector_search.build_sample_point(
            settings,
            document,
            vector,
        )
        return check_qdrant_vector_search.build_validate_only_result(
            settings,
            document,
            point,
            smoke_args,
        )

    request = chat_check_common.build_chat_answer_request(smoke_args)
    return await check_qdrant_vector_search.check_qdrant_vector_search(
        settings,
        document,
        request,
        ChatIntent.LINE_BOTTLENECK,
    )


async def run_document_api_smoke(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, Any]:
    token = check_document_index_api.resolve_index_token(
        argparse.Namespace(token=None),
        settings,
    )
    index_path = f"{settings.api_v1_prefix}/chat/internal/documents/index"
    delete_path = f"{settings.api_v1_prefix}/chat/internal/documents/delete"
    document = check_document_index_api.build_sample_document(
        argparse.Namespace(
            document_id=args.document_api_smoke_document_id,
            document_type=check_document_index_api.DEFAULT_DOCUMENT_TYPE,
            title=check_document_index_api.DEFAULT_TITLE,
            content=check_document_index_api.DEFAULT_CONTENT,
            summary=None,
            url=check_document_index_api.DEFAULT_URL,
            reference_type="SYSTEM",
            reference_id=None,
            basis_time=None,
            roles=["MANUFACTURING_MANAGER"],
            intents=[ChatIntent.LINE_BOTTLENECK.value],
            requested_by_role="MANUFACTURING_MANAGER",
            company_name="S-MAP",
        )
    )
    delete_request = check_document_delete_api.build_delete_request(
        argparse.Namespace(document_id=args.document_api_smoke_document_id)
    )
    require_document_index = bool(args.require_document_index)
    min_indexed_count = 1 if require_document_index else 0
    allow_skipped = not require_document_index

    if not args.network:
        return {
            "checkStatus": "VALIDATED",
            "mode": "VALIDATE_ONLY",
            "networkChecked": False,
            "baseUrl": args.document_api_base_url,
            "indexPath": index_path,
            "deletePath": delete_path,
            "documentId": args.document_api_smoke_document_id,
            "tokenConfigured": bool(token),
            "minIndexedCount": min_indexed_count,
            "allowSkipped": allow_skipped,
            "requireDocumentIndex": require_document_index,
        }

    index_result = await check_document_index_api.check_document_index_api(
        base_url=args.document_api_base_url,
        path=index_path,
        token=token,
        document=document,
        timeout_seconds=args.document_api_timeout_seconds,
        min_indexed_count=min_indexed_count,
        allow_skipped=allow_skipped,
    )
    delete_result = await check_document_delete_api.check_document_delete_api(
        base_url=args.document_api_base_url,
        path=delete_path,
        token=token,
        request=delete_request,
        timeout_seconds=args.document_api_timeout_seconds,
    )
    return {
        "checkStatus": "PASS",
        "mode": "NETWORK",
        "networkChecked": True,
        "documentId": args.document_api_smoke_document_id,
        "index": index_result,
        "delete": delete_result,
    }


async def run_rag_end_to_end_smoke(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, Any]:
    answer_token = check_rag_end_to_end.resolve_answer_token(
        argparse.Namespace(answer_token=None),
        settings,
    )
    document_token = check_rag_end_to_end.resolve_document_token(
        argparse.Namespace(document_token=None),
        settings,
    )
    smoke_args = argparse.Namespace(
        base_url=args.answer_api_base_url,
        answer_token=None,
        document_token=None,
        env_file=None,
        timeout_seconds=args.answer_api_timeout_seconds,
        document_id=args.document_api_smoke_document_id,
        title=check_qdrant_vector_search.DEFAULT_TITLE,
        content=check_qdrant_vector_search.DEFAULT_CONTENT,
        url=check_qdrant_vector_search.DEFAULT_URL,
        question=check_qdrant_vector_search.DEFAULT_QUESTION,
        role=args.answer_api_role,
        user_id=args.answer_api_user_id,
        company_name="S-MAP",
        session_id=1,
        message_id=1,
        requested_at=chat_check_common.DEFAULT_REQUESTED_AT,
        min_indexed_count=1,
        min_document_source_count=max(args.answer_api_min_document_source_count, 1),
        min_evidence_count=max(args.answer_api_min_evidence_count, 1),
        require_rdb_evidence=bool(args.require_rdb_evidence),
        keep_document=False,
        validate_only=not args.network,
        json=False,
    )

    if not args.network:
        return check_rag_end_to_end.build_validate_only_result(
            smoke_args,
            settings,
            answer_token,
            document_token,
        )

    return await check_rag_end_to_end.check_rag_end_to_end(
        args=smoke_args,
        settings=settings,
        answer_token=answer_token,
        document_token=document_token,
    )


async def run_answer_api_smoke(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, Any]:
    token = check_chat_answer.resolve_answer_token(
        argparse.Namespace(token=None),
        settings,
    )
    path = f"{settings.api_v1_prefix}/chat/answer"
    request = chat_check_common.build_chat_answer_request(
        argparse.Namespace(
            question=args.answer_api_question,
            role=args.answer_api_role,
            user_id=args.answer_api_user_id,
            company_name="S-MAP",
            session_id=1,
            message_id=1,
            requested_at=chat_check_common.DEFAULT_REQUESTED_AT,
        )
    )
    require_rdb_evidence = bool(args.require_rdb_evidence)
    require_vector_search = bool(args.require_vector_search)

    if not args.network:
        return {
            "checkStatus": "VALIDATED",
            "mode": "VALIDATE_ONLY",
            "networkChecked": False,
            "baseUrl": args.answer_api_base_url,
            "path": path,
            "question": request.question,
            "role": request.user.role,
            "tokenConfigured": bool(token),
            "minEvidenceCount": args.answer_api_min_evidence_count,
            "requireRdbEvidence": require_rdb_evidence,
            "minDocumentSourceCount": args.answer_api_min_document_source_count,
            "requireVectorSearch": require_vector_search,
            "requireLlmGeneration": bool(args.require_llm_generation),
        }

    return await check_chat_answer.check_chat_answer(
        base_url=args.answer_api_base_url,
        path=path,
        token=token,
        request=request,
        timeout_seconds=args.answer_api_timeout_seconds,
        min_evidence_count=args.answer_api_min_evidence_count,
        require_rdb_evidence=require_rdb_evidence,
        min_document_source_count=args.answer_api_min_document_source_count,
        require_vector_search=require_vector_search,
        require_llm_generation=bool(args.require_llm_generation),
    )


def run_answer_output_policy_smoke() -> dict[str, Any]:
    return check_answer_output_policy.check_answer_output_policy()


async def run_llm_completion_smoke(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.network:
        return check_llm_completion.build_validate_only_result(settings)
    return await check_llm_completion.check_llm_completion(settings)


async def run_recommendation_api_smoke(
    settings: Settings,
    args: argparse.Namespace,
) -> dict[str, Any]:
    token = check_chat_recommendations.resolve_recommendation_token(
        argparse.Namespace(token=None),
        settings,
    )
    path = f"{settings.api_v1_prefix}/chat/recommendations"
    request = check_chat_recommendations.build_request(
        argparse.Namespace(
            user_id=args.recommendation_api_user_id,
            role=args.recommendation_api_role,
            company_name="S-MAP",
            status="ACTIVE",
            keyword=args.recommendation_api_keyword,
        )
    )

    if not args.network:
        return {
            "checkStatus": "VALIDATED",
            "mode": "VALIDATE_ONLY",
            "networkChecked": False,
            "baseUrl": args.recommendation_api_base_url,
            "path": path,
            "role": request.user.role,
            "keywordConfigured": bool(request.keyword),
            "tokenConfigured": bool(token),
            "minItemCount": args.recommendation_api_min_item_count,
        }

    return await check_chat_recommendations.check_chat_recommendations(
        base_url=args.recommendation_api_base_url,
        path=path,
        token=token,
        request=request,
        timeout_seconds=args.recommendation_api_timeout_seconds,
        min_item_count=args.recommendation_api_min_item_count,
        expect_fallback=False,
    )


def format_text_result(result: dict[str, Any]) -> str:
    summary = result.get("summary", {})
    lines = [
        f"status={result['checkStatus']}",
        f"mode={result['mode']}",
        f"networkChecked={result['networkChecked']}",
        f"requiredComponents={','.join(result['requiredComponents'])}",
        (
            "summary="
            f"passed:{summary.get('passedStepCount', 0)} "
            f"failed:{summary.get('failedStepCount', 0)} "
            f"total:{summary.get('totalStepCount', len(result['steps']))}"
        ),
    ]
    for step in result["steps"]:
        line = f"{step['name']}: status={step['status']}"
        if "error" in step:
            error = step["error"]
            line = f"{line} code={error['code']} message={error['message']}"
        lines.append(line)
    for item in summary.get("failedSteps", []):
        line = f"failure={item['name']}"
        if item.get("code"):
            line = f"{line} code={item['code']}"
        line = f"{line} message={item['message']}"
        lines.append(line)
    for action in summary.get("nextActions", []):
        lines.append(f"nextAction={action}")
    return "\n".join(lines)


def format_json_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


def main(
    argv: list[str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    args = build_parser().parse_args(argv)

    try:
        settings = build_settings(args)
        result = asyncio.run(check_chat_runtime(settings, args))
    except Exception as exc:
        print(f"챗봇 런타임 통합 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0 if result["checkStatus"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
