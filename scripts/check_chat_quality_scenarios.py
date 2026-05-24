import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from typing import Any, TextIO

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from scripts import (
    chat_check_common,
    check_answer_output_policy,
    check_chat_answer,
    check_rag_chat_scenarios,
    check_rdb_chat_scenarios,
)

ScenarioChecker = Callable[[argparse.Namespace], Awaitable[dict[str, Any]]]

QUALITY_CRITERIA = (
    "Role별 실제 업무 질문 매트릭스 확인",
    "OPERATOR 비금액성 조회 허용과 금액성 정보 차단 동시 확인",
    "Role 기반 금액성 정보 차단",
    "RDB Evidence 또는 Qdrant 문서 출처 확인",
    "화면 이동 URL 포함 여부 확인",
    "LLM 생성 여부와 캐시 사용 여부 확인",
    "LLM total token 상한 확인",
    "민감정보 출력 정책 확인",
)
QUALITY_PROFILES: dict[str, dict[str, list[str]]] = {
    "minimal": {
        "rdbGroups": ["access"],
        "ragGroups": ["company"],
    },
    "standard": {
        "rdbGroups": ["core", "access"],
        "ragGroups": ["core", "access", "company"],
    },
    "business": {
        "rdbGroups": ["core", "access", "filtered"],
        "ragGroups": ["core", "company", "role"],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="챗봇 품질 시나리오를 RDB, Qdrant, LLM 기준으로 한 번에 점검합니다."
    )
    parser.add_argument(
        "--base-url",
        default=check_chat_answer.DEFAULT_BASE_URL,
        help="FastAPI base URL",
    )
    parser.add_argument(
        "--path",
        help="Chat answer path. 생략하면 Settings.api_v1_prefix 기준으로 생성합니다.",
    )
    parser.add_argument("--token", help="FastAPI chat answer internal token")
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. CLI 인자가 있으면 해당 값이 우선합니다.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--profile",
        choices=sorted(QUALITY_PROFILES),
        default="minimal",
        help="실행할 품질 시나리오 범위입니다.",
    )
    parser.add_argument("--role", default="MANUFACTURING_MANAGER", help="기본 사용자 Role")
    parser.add_argument("--user-id", type=int, default=1, help="사용자 ID")
    parser.add_argument("--company-name", default="S-MAP", help="회사명 메타데이터")
    parser.add_argument(
        "--require-llm-generation",
        action="store_true",
        help="챗봇 응답이 fallback이 아니라 LLM 생성 답변을 사용했는지 검증합니다.",
    )
    parser.add_argument(
        "--require-llm-cache-miss",
        action="store_true",
        help=(
            "LLM 답변이 캐시가 아니라 실제 생성 경로에서 만들어졌는지 검증합니다. "
            "배포 직후 LLM 연결 확인용으로만 사용합니다."
        ),
    )
    parser.add_argument(
        "--max-llm-total-tokens",
        type=int,
        default=None,
        help="모든 품질 시나리오에 적용할 LLM total token 최대 허용값",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="실제 FastAPI 챗봇 API를 호출합니다. 생략하면 실행 계획만 검증합니다.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="점검 결과를 리뷰용 Markdown으로 출력합니다.",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    if args.env_file:
        return Settings(_env_file=args.env_file)
    return Settings()


def resolve_answer_path(args: argparse.Namespace, settings: Settings) -> str:
    return args.path or f"{settings.api_v1_prefix}/chat/answer"


def resolve_answer_token(args: argparse.Namespace, settings: Settings) -> str:
    return check_chat_answer.resolve_answer_token(
        argparse.Namespace(token=args.token),
        settings,
    )


def build_validate_only_result(args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    profile = QUALITY_PROFILES[args.profile]
    token = args.token or settings.chat_answer_internal_token
    return {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "networkChecked": False,
        "profile": args.profile,
        "baseUrl": args.base_url,
        "path": resolve_answer_path(args, settings),
        "tokenConfigured": bool(token),
        "rdbScenarioGroups": profile["rdbGroups"],
        "ragScenarioGroups": profile["ragGroups"],
        "qualityCriteria": list(QUALITY_CRITERIA),
        "requireLlmGeneration": bool(args.require_llm_generation),
        "requireLlmCacheMiss": bool(args.require_llm_cache_miss),
        "maxLlmTotalTokens": args.max_llm_total_tokens,
    }


async def check_chat_quality_scenarios(
    args: argparse.Namespace,
    rdb_checker: ScenarioChecker | None = None,
    rag_checker: ScenarioChecker | None = None,
) -> dict[str, Any]:
    settings = build_settings(args)
    if not args.network:
        return build_validate_only_result(args, settings)

    token = resolve_answer_token(args, settings)
    path = resolve_answer_path(args, settings)
    profile = QUALITY_PROFILES[args.profile]
    output_policy_result = check_answer_output_policy.check_answer_output_policy()
    rdb_result = await (rdb_checker or check_rdb_chat_scenarios.check_rdb_chat_scenarios)(
        build_rdb_args(args, path, token, profile["rdbGroups"])
    )
    rag_result = await (rag_checker or check_rag_chat_scenarios.check_rag_chat_scenarios)(
        build_rag_args(args, path, token, profile["ragGroups"])
    )
    return {
        "checkStatus": "PASS",
        "mode": "NETWORK",
        "networkChecked": True,
        "profile": args.profile,
        "baseUrl": args.base_url,
        "path": path,
        "qualityCriteria": list(QUALITY_CRITERIA),
        "requireLlmGeneration": bool(args.require_llm_generation),
        "requireLlmCacheMiss": bool(args.require_llm_cache_miss),
        "maxLlmTotalTokens": args.max_llm_total_tokens,
        "answerOutputPolicy": output_policy_result,
        "rdbScenarios": rdb_result,
        "ragScenarios": rag_result,
        "qualitySummary": build_quality_summary(rdb_result, rag_result),
    }


def build_rdb_args(
    args: argparse.Namespace,
    path: str,
    token: str,
    scenario_groups: list[str],
) -> argparse.Namespace:
    return argparse.Namespace(
        base_url=args.base_url,
        path=path,
        token=token,
        env_file=None,
        timeout_seconds=args.timeout_seconds,
        role=args.role,
        user_id=args.user_id,
        company_name=args.company_name,
        session_id=1,
        message_id=1,
        requested_at=chat_check_common.DEFAULT_REQUESTED_AT,
        scenario=None,
        scenario_group=scenario_groups,
        min_evidence_count=None,
        require_llm_generation=bool(args.require_llm_generation),
        require_llm_cache_miss=bool(args.require_llm_cache_miss),
        max_llm_total_tokens=args.max_llm_total_tokens,
        markdown=False,
        json=False,
    )


def build_rag_args(
    args: argparse.Namespace,
    path: str,
    token: str,
    scenario_groups: list[str],
) -> argparse.Namespace:
    return argparse.Namespace(
        base_url=args.base_url,
        path=path,
        token=token,
        env_file=None,
        timeout_seconds=args.timeout_seconds,
        role=args.role,
        user_id=args.user_id,
        company_name=args.company_name,
        session_id=101,
        message_id=101,
        requested_at=chat_check_common.DEFAULT_REQUESTED_AT,
        scenario=None,
        scenario_group=scenario_groups,
        min_evidence_count=None,
        min_rdb_evidence_count=None,
        min_document_source_count=None,
        require_llm_generation=bool(args.require_llm_generation),
        require_llm_cache_miss=bool(args.require_llm_cache_miss),
        max_llm_total_tokens=args.max_llm_total_tokens,
        markdown=False,
        json=False,
    )


def build_quality_summary(
    rdb_result: dict[str, Any],
    rag_result: dict[str, Any],
) -> dict[str, Any]:
    scenarios = [
        *rdb_result.get("scenarios", []),
        *rag_result.get("scenarios", []),
    ]
    llm_usage_items = [
        scenario["llmUsage"]
        for scenario in scenarios
        if isinstance(scenario.get("llmUsage"), dict)
    ]
    return {
        "scenarioCount": len(scenarios),
        "blockedUnauthorizedCount": _count_by_security_status(
            scenarios,
            "BLOCKED_UNAUTHORIZED",
        ),
        "rdbEvidenceScenarioCount": sum(
            1 for scenario in scenarios if scenario.get("rdbEvidenceCount", 0) > 0
        ),
        "qdrantSourceScenarioCount": sum(
            1 for scenario in scenarios if scenario.get("documentSourceCount", 0) > 0
        ),
        "llmGenerationCount": sum(
            1 for scenario in scenarios if scenario.get("usedLlmGeneration")
        ),
        "llmCacheHitCount": sum(1 for scenario in scenarios if scenario.get("llmCacheHit")),
        "totalLlmTokens": sum(usage["totalTokens"] for usage in llm_usage_items),
        "maxScenarioLlmTokens": max(
            (usage["totalTokens"] for usage in llm_usage_items),
            default=0,
        ),
        "urlScenarioCount": sum(1 for scenario in scenarios if scenario.get("urlCount", 0) > 0),
    }


def _count_by_security_status(
    scenarios: list[dict[str, Any]],
    security_status: str,
) -> int:
    return sum(1 for scenario in scenarios if scenario.get("securityStatus") == security_status)


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"mode={result['mode']}",
        f"networkChecked={result['networkChecked']}",
        f"profile={result['profile']}",
        f"baseUrl={result['baseUrl']}",
        f"path={result['path']}",
        f"requireLlmGeneration={result['requireLlmGeneration']}",
        f"requireLlmCacheMiss={result['requireLlmCacheMiss']}",
        f"maxLlmTotalTokens={result['maxLlmTotalTokens']}",
    ]
    if result["mode"] == "VALIDATE_ONLY":
        lines.extend(
            [
                f"tokenConfigured={result['tokenConfigured']}",
                f"rdbScenarioGroups={','.join(result['rdbScenarioGroups'])}",
                f"ragScenarioGroups={','.join(result['ragScenarioGroups'])}",
                f"qualityCriteriaCount={len(result['qualityCriteria'])}",
            ]
        )
        return "\n".join(lines)

    summary = result["qualitySummary"]
    lines.extend(
        [
            f"scenarioCount={summary['scenarioCount']}",
            f"blockedUnauthorizedCount={summary['blockedUnauthorizedCount']}",
            f"rdbEvidenceScenarioCount={summary['rdbEvidenceScenarioCount']}",
            f"qdrantSourceScenarioCount={summary['qdrantSourceScenarioCount']}",
            f"llmGenerationCount={summary['llmGenerationCount']}",
            f"llmCacheHitCount={summary['llmCacheHitCount']}",
            f"totalLlmTokens={summary['totalLlmTokens']}",
            f"maxScenarioLlmTokens={summary['maxScenarioLlmTokens']}",
            f"urlScenarioCount={summary['urlScenarioCount']}",
        ]
    )
    return "\n".join(lines)


def format_markdown_result(result: dict[str, Any]) -> str:
    lines = [
        "# 챗봇 품질 시나리오 점검 결과",
        "",
        f"- 점검 상태: `{result['checkStatus']}`",
        f"- 실행 모드: `{result['mode']}`",
        f"- 프로필: `{result['profile']}`",
        f"- LLM 생성 필수: `{result['requireLlmGeneration']}`",
        f"- LLM 캐시 미스 필수: `{result['requireLlmCacheMiss']}`",
        f"- LLM 토큰 상한: `{result['maxLlmTotalTokens'] or '-'}`",
        "",
        "## 품질 기준",
        "",
    ]
    lines.extend(f"- {criterion}" for criterion in result["qualityCriteria"])

    if result["mode"] == "VALIDATE_ONLY":
        lines.extend(
            [
                "",
                "## 실행 계획",
                "",
                f"- 토큰 설정 여부: `{result['tokenConfigured']}`",
                f"- RDB 시나리오 그룹: `{', '.join(result['rdbScenarioGroups'])}`",
                f"- RAG 시나리오 그룹: `{', '.join(result['ragScenarioGroups'])}`",
            ]
        )
        return "\n".join(lines)

    summary = result["qualitySummary"]
    lines.extend(
        [
            "",
            "## 요약",
            "",
            "| 항목 | 값 |",
            "| --- | --- |",
            f"| 전체 시나리오 | `{summary['scenarioCount']}` |",
            f"| 권한 차단 확인 | `{summary['blockedUnauthorizedCount']}` |",
            f"| RDB 근거 사용 | `{summary['rdbEvidenceScenarioCount']}` |",
            f"| Qdrant 출처 사용 | `{summary['qdrantSourceScenarioCount']}` |",
            f"| LLM 생성 사용 | `{summary['llmGenerationCount']}` |",
            f"| LLM 캐시 사용 | `{summary['llmCacheHitCount']}` |",
            f"| 총 LLM 토큰 | `{summary['totalLlmTokens']}` |",
            f"| 최대 단일 시나리오 토큰 | `{summary['maxScenarioLlmTokens']}` |",
            f"| URL 포함 시나리오 | `{summary['urlScenarioCount']}` |",
        ]
    )
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
        result = asyncio.run(check_chat_quality_scenarios(args))
    except ChatServiceError as exc:
        print(f"챗봇 품질 시나리오 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        return 1
    except Exception as exc:
        print(f"챗봇 품질 시나리오 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    elif args.markdown:
        print(format_markdown_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
