import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TextIO

import httpx

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatErrorCode,
    ChatIntent,
    ChatUserContext,
)
from scripts import check_chat_answer

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_REQUESTED_AT = "2026-05-12T10:30:00+09:00"


@dataclass(frozen=True)
class RagChatScenario:
    scenario_id: str
    intent: ChatIntent
    question: str
    role: str | None = None
    expected_security_results: tuple[tuple[str, str | None], ...] = (("PASSED", None),)
    require_rdb_evidence: bool = True
    require_vector_search: bool = True
    min_evidence_count: int = 2
    min_rdb_evidence_count: int = 1
    max_rdb_evidence_count: int | None = None
    min_document_source_count: int = 1
    required_answer_fragments: tuple[str, ...] = ()
    required_source_title_fragments: tuple[str, ...] = ()
    forbidden_source_origins: tuple[str, ...] = ()
    require_rdb_urls_read_only: bool = True

    @property
    def expected_security_statuses(self) -> tuple[str, ...]:
        return tuple(status for status, _ in self.expected_security_results)

    @property
    def expected_security_codes(self) -> tuple[str | None, ...]:
        return tuple(code for _, code in self.expected_security_results)


CORE_RAG_CHAT_SCENARIOS: tuple[RagChatScenario, ...] = (
    RagChatScenario(
        scenario_id="material-shortage-with-company-guide",
        intent=ChatIntent.MATERIAL_SHORTAGE,
        question="MAT-FOAM-ADD 자재 부족 현황과 대응 기준을 같이 알려줘",
        required_answer_fragments=("핵심 답변", "확인 필요"),
        required_source_title_fragments=("MAT-FOAM-ADD",),
    ),
    RagChatScenario(
        scenario_id="line-bottleneck-with-company-guide",
        intent=ChatIntent.LINE_BOTTLENECK,
        question="LINE-PE-01 병목 현황과 대응 기준을 같이 알려줘",
        required_answer_fragments=("핵심 답변", "확인 필요"),
        required_source_title_fragments=("LINE-PE-01",),
    ),
    RagChatScenario(
        scenario_id="delivery-risk-with-company-guide",
        intent=ChatIntent.DELIVERY_RISK,
        question="납기 위험이 있는 주문과 주요 원인, 대응 기준을 알려줘",
    ),
)

ACCESS_RAG_CHAT_SCENARIOS: tuple[RagChatScenario, ...] = (
    RagChatScenario(
        scenario_id="operator-report-document-allowed",
        intent=ChatIntent.REPORT_LOOKUP,
        question="이번 달 월간 리포트에서 생산 리스크 요약해줘",
        role="OPERATOR",
        require_rdb_evidence=False,
        min_evidence_count=1,
        min_rdb_evidence_count=0,
        min_document_source_count=1,
        require_rdb_urls_read_only=False,
    ),
    RagChatScenario(
        scenario_id="operator-financial-rag-blocked",
        intent=ChatIntent.DELIVERY_RISK,
        question="납기 지연 시 예상 패널티와 계약 금액 관련 보고서를 찾아줘",
        role="OPERATOR",
        expected_security_results=(("BLOCKED_UNAUTHORIZED", "CHAT_SECURITY_004"),),
        require_rdb_evidence=False,
        require_vector_search=False,
        min_evidence_count=0,
        min_rdb_evidence_count=0,
        min_document_source_count=0,
        require_rdb_urls_read_only=False,
    ),
)

COMPANY_INFO_RAG_CHAT_SCENARIOS: tuple[RagChatScenario, ...] = (
    RagChatScenario(
        scenario_id="company-overview-document-allowed",
        intent=ChatIntent.REPORT_LOOKUP,
        question="S-Map 회사 개요 알려줘",
        role="OPERATOR",
        require_rdb_evidence=False,
        min_evidence_count=1,
        min_rdb_evidence_count=0,
        max_rdb_evidence_count=0,
        min_document_source_count=1,
        required_source_title_fragments=("S-Map",),
        forbidden_source_origins=("RDB",),
        require_rdb_urls_read_only=False,
    ),
    RagChatScenario(
        scenario_id="manager-revenue-company-info-allowed",
        intent=ChatIntent.REPORT_LOOKUP,
        question="S-Map 매출 구조 알려줘",
        role="MANUFACTURING_MANAGER",
        require_rdb_evidence=False,
        min_evidence_count=1,
        min_rdb_evidence_count=0,
        max_rdb_evidence_count=0,
        min_document_source_count=1,
        required_source_title_fragments=("매출",),
        forbidden_source_origins=("RDB",),
        require_rdb_urls_read_only=False,
    ),
    RagChatScenario(
        scenario_id="operator-revenue-company-info-blocked",
        intent=ChatIntent.REPORT_LOOKUP,
        question="S-Map 매출 구조 알려줘",
        role="OPERATOR",
        expected_security_results=(("BLOCKED_UNAUTHORIZED", "CHAT_SECURITY_004"),),
        require_rdb_evidence=False,
        require_vector_search=False,
        min_evidence_count=0,
        min_rdb_evidence_count=0,
        min_document_source_count=0,
        require_rdb_urls_read_only=False,
    ),
)

RAG_CHAT_SCENARIO_GROUPS = {
    "core": CORE_RAG_CHAT_SCENARIOS,
    "access": ACCESS_RAG_CHAT_SCENARIOS,
    "company": COMPANY_INFO_RAG_CHAT_SCENARIOS,
}
ALL_RAG_CHAT_SCENARIOS = tuple(
    scenario
    for group in RAG_CHAT_SCENARIO_GROUPS.values()
    for scenario in group
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FastAPI 챗봇 RDB + Qdrant RAG 실사용 시나리오를 점검합니다."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="FastAPI base URL")
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
    parser.add_argument("--role", default="MANUFACTURING_MANAGER", help="사용자 Role")
    parser.add_argument("--user-id", type=int, default=1, help="사용자 ID")
    parser.add_argument("--company-name", default="S-MAP", help="회사명 메타데이터")
    parser.add_argument("--session-id", type=int, default=1, help="세션 ID 시작값")
    parser.add_argument("--message-id", type=int, default=1, help="메시지 ID 시작값")
    parser.add_argument(
        "--requested-at",
        default=DEFAULT_REQUESTED_AT,
        help="요청 기준 시각. ISO datetime 형식",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.scenario_id for scenario in ALL_RAG_CHAT_SCENARIOS],
        help="특정 RAG 시나리오만 실행합니다. 여러 번 지정할 수 있습니다.",
    )
    parser.add_argument(
        "--scenario-group",
        action="append",
        choices=sorted(RAG_CHAT_SCENARIO_GROUPS),
        help=(
            "실행할 시나리오 묶음입니다. core, access, company 중 선택하며 "
            "여러 번 지정할 수 있습니다. 생략하면 core만 실행합니다."
        ),
    )
    parser.add_argument(
        "--min-evidence-count",
        type=int,
        default=None,
        help="모든 시나리오에 적용할 최소 전체 Evidence 개수",
    )
    parser.add_argument(
        "--min-rdb-evidence-count",
        type=int,
        default=None,
        help="모든 시나리오에 적용할 최소 RDB Evidence 개수",
    )
    parser.add_argument(
        "--min-document-source-count",
        type=int,
        default=None,
        help="모든 시나리오에 적용할 최소 Qdrant 문서 출처 개수",
    )
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
        help="모든 시나리오에 적용할 LLM total token 최대 허용값",
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


def select_scenarios(
    scenario_ids: list[str] | None,
    scenario_groups: list[str] | None = None,
) -> tuple[RagChatScenario, ...]:
    selected_scenarios = _select_scenario_groups(scenario_groups)
    if not scenario_ids:
        return selected_scenarios

    requested_ids = set(scenario_ids)
    search_space = selected_scenarios if scenario_groups else ALL_RAG_CHAT_SCENARIOS
    return tuple(
        scenario
        for scenario in search_space
        if scenario.scenario_id in requested_ids
    )


def resolve_answer_token(args: argparse.Namespace, settings: Settings) -> str:
    token = args.token or settings.chat_answer_internal_token
    if not token:
        raise ChatServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_SECURITY_003,
            message="FastAPI chat answer internal token이 설정되지 않았습니다.",
        )
    return token


async def check_rag_chat_scenarios(
    args: argparse.Namespace,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    settings = build_settings(args)
    token = resolve_answer_token(args, settings)
    path = args.path or f"{settings.api_v1_prefix}/chat/answer"
    scenarios = select_scenarios(args.scenario, args.scenario_group)
    scenario_results = []

    for index, scenario in enumerate(scenarios):
        result = await check_chat_answer.check_chat_answer(
            base_url=args.base_url,
            path=path,
            token=token,
            request=build_request(args, scenario, index),
            timeout_seconds=args.timeout_seconds,
            min_evidence_count=_resolve_min_evidence_count(args, scenario),
            require_rdb_evidence=scenario.require_rdb_evidence,
            min_document_source_count=_resolve_min_document_source_count(
                args,
                scenario,
            ),
            require_vector_search=scenario.require_vector_search,
            require_llm_generation=args.require_llm_generation,
            require_llm_cache_miss=args.require_llm_cache_miss,
            max_llm_total_tokens=args.max_llm_total_tokens,
            expected_security_status=_single_expected_security_status(scenario),
            expected_security_code=_single_expected_security_code(scenario),
            expected_intent=scenario.intent.value,
            http_client=http_client,
        )
        validate_scenario_result(args, scenario, result)
        scenario_results.append(
            {
                "scenarioId": scenario.scenario_id,
                "role": scenario.role or args.role,
                "question": scenario.question,
                "expectedIntent": scenario.intent.value,
                "expectedSecurityResults": [
                    {"status": status, "code": code}
                    for status, code in scenario.expected_security_results
                ],
                "requireRdbEvidence": scenario.require_rdb_evidence,
                "requireVectorSearch": scenario.require_vector_search,
                "minRdbEvidenceCount": _resolve_min_rdb_evidence_count(args, scenario),
                "maxRdbEvidenceCount": scenario.max_rdb_evidence_count,
                "minDocumentSourceCount": _resolve_min_document_source_count(
                    args,
                    scenario,
                ),
                "maxLlmTotalTokens": args.max_llm_total_tokens,
                "requireLlmCacheMiss": args.require_llm_cache_miss,
                **result,
            }
        )

    return {
        "checkStatus": "PASS",
        "scenarioCount": len(scenario_results),
        "scenarios": scenario_results,
    }


def build_request(
    args: argparse.Namespace,
    scenario: RagChatScenario,
    index: int,
) -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=args.session_id + index,
        messageId=args.message_id + index,
        user=ChatUserContext(
            userId=args.user_id,
            role=scenario.role or args.role,
            companyName=args.company_name,
            status="ACTIVE",
        ),
        question=scenario.question,
        requestedAt=datetime.fromisoformat(args.requested_at),
    )


def validate_scenario_result(
    args: argparse.Namespace,
    scenario: RagChatScenario,
    result: dict[str, Any],
) -> None:
    if result["intent"] != scenario.intent.value:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_EVIDENCE_001,
            message=(
                "RAG 챗봇 시나리오 intent가 예상과 다릅니다. "
                f"scenario={scenario.scenario_id}, "
                f"expected={scenario.intent.value}, actual={result['intent']}"
            ),
        )

    actual_security_result = (result["securityStatus"], result["securityCode"])
    if actual_security_result not in scenario.expected_security_results:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_EVIDENCE_001,
            message=(
                "RAG 챗봇 시나리오 보안 결과가 예상과 다릅니다. "
                f"scenario={scenario.scenario_id}, "
                f"expected={_format_security_results(scenario.expected_security_results)}, "
                f"actual={_format_security_result(actual_security_result)}"
            ),
        )

    min_rdb_evidence_count = _resolve_min_rdb_evidence_count(args, scenario)
    if result["rdbEvidenceCount"] < min_rdb_evidence_count:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_EVIDENCE_001,
            message=(
                "RAG 챗봇 응답 RDB Evidence 개수가 기준보다 적습니다. "
                f"scenario={scenario.scenario_id}, "
                f"expected>={min_rdb_evidence_count}, "
                f"actual={result['rdbEvidenceCount']}"
            ),
        )

    if (
        scenario.max_rdb_evidence_count is not None
        and result["rdbEvidenceCount"] > scenario.max_rdb_evidence_count
    ):
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_EVIDENCE_001,
            message=(
                "RAG 챗봇 응답 RDB Evidence 개수가 허용 기준보다 많습니다. "
                f"scenario={scenario.scenario_id}, "
                f"expected<={scenario.max_rdb_evidence_count}, "
                f"actual={result['rdbEvidenceCount']}"
            ),
        )

    for fragment in scenario.required_answer_fragments:
        if fragment not in result["answer"]:
            raise ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_EVIDENCE_001,
                message=(
                    "RAG 챗봇 응답 답변에 필요한 문구가 없습니다. "
                    f"scenario={scenario.scenario_id}, fragment={fragment}"
                ),
            )

    for fragment in scenario.required_source_title_fragments:
        if not _has_source_title_fragment(result, fragment):
            raise ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_EVIDENCE_001,
                message=(
                    "RAG 챗봇 응답 출처 제목에 필요한 문구가 없습니다. "
                    f"scenario={scenario.scenario_id}, fragment={fragment}"
                ),
            )

    for forbidden_origin in scenario.forbidden_source_origins:
        if _has_source_origin(result, forbidden_origin):
            raise ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_EVIDENCE_001,
                message=(
                    "RAG 챗봇 응답에 허용되지 않은 출처 유형이 포함되었습니다. "
                    f"scenario={scenario.scenario_id}, origin={forbidden_origin}"
                ),
            )

    if scenario.require_rdb_urls_read_only:
        invalid_urls = _find_non_read_only_rdb_urls(result)
        if invalid_urls:
            raise ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_EVIDENCE_001,
                message=(
                    "RAG 챗봇 응답 RDB 화면 이동 URL이 read-only 형식이 아닙니다. "
                    f"scenario={scenario.scenario_id}, urls={', '.join(invalid_urls)}"
                ),
            )


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"scenarioCount={result['scenarioCount']}",
    ]
    for scenario in result["scenarios"]:
        lines.append(
            "scenario="
            f"{scenario['scenarioId']} "
            f"role={scenario['role']} "
            f"intent={scenario['intent']} "
            f"securityStatus={scenario['securityStatus']} "
            f"securityCode={scenario['securityCode']} "
            f"requireRdbEvidence={scenario['requireRdbEvidence']} "
            f"requireVectorSearch={scenario['requireVectorSearch']} "
            f"rdbEvidenceCount={scenario['rdbEvidenceCount']} "
            f"maxRdbEvidenceCount={scenario.get('maxRdbEvidenceCount')} "
            f"documentSourceCount={scenario['documentSourceCount']} "
            f"usedVectorSearch={scenario['usedVectorSearch']} "
            f"requireLlmGeneration={scenario['requireLlmGeneration']} "
            f"usedLlmGeneration={scenario['usedLlmGeneration']} "
            f"llmCacheHit={scenario.get('llmCacheHit', False)} "
            f"requireLlmCacheMiss={scenario.get('requireLlmCacheMiss', False)} "
            f"llmUsage={check_chat_answer.format_llm_usage(scenario.get('llmUsage'))} "
            f"maxLlmTotalTokens={scenario.get('maxLlmTotalTokens')} "
            f"sourceCount={scenario['sourceCount']} "
            f"urlCount={scenario['urlCount']}"
        )
    return "\n".join(lines)


def format_json_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


def format_markdown_result(result: dict[str, Any]) -> str:
    lines = [
        "# RAG 챗봇 시나리오 점검 결과",
        "",
        f"- 점검 상태: `{result['checkStatus']}`",
        f"- 시나리오 수: `{result['scenarioCount']}`",
    ]

    for scenario in result["scenarios"]:
        lines.extend(
            [
                "",
                f"## {scenario['scenarioId']}",
                "",
                f"- Role: `{scenario['role']}`",
                f"- 질문: {scenario['question']}",
                f"- Intent: `{scenario['intent']}`",
                (
                    "- 보안 결과: "
                    f"`{scenario['securityStatus']}`"
                    f"{_format_optional_code(scenario['securityCode'])}"
                ),
                (
                    "- 근거 수: "
                    f"전체 `{scenario['evidenceCount']}`, "
                    f"RDB `{scenario['rdbEvidenceCount']}`, "
                    f"Qdrant `{scenario['documentSourceCount']}`"
                ),
                (
                    "- LLM 생성: "
                    f"요구 `{scenario['requireLlmGeneration']}`, "
                    f"사용 `{scenario['usedLlmGeneration']}`, "
                    f"캐시 `{scenario.get('llmCacheHit', False)}`, "
                    f"캐시 미스 요구 `{scenario.get('requireLlmCacheMiss', False)}`, "
                    "토큰 "
                    f"`{check_chat_answer.format_llm_usage(scenario.get('llmUsage'))}`, "
                    f"최대 `{scenario.get('maxLlmTotalTokens') or '-'}`"
                ),
            ]
        )
        answer = scenario.get("answer")
        if answer:
            lines.extend(["", "### 답변", "", "```text", answer, "```"])

        source_details = scenario.get("sourceDetails") or []
        if source_details:
            lines.extend(["", "### 출처", "", "| 유형 | 제목 | URL |", "| --- | --- | --- |"])
            lines.extend(
                (
                    f"| `{source['sourceOrigin'] or '-'}` / `{source['sourceType']}` "
                    f"| {_escape_markdown_cell(source['title'])} "
                    f"| `{source['url'] or '-'}` |"
                )
                for source in source_details
            )

        url_details = scenario.get("urlDetails") or []
        if url_details:
            lines.extend(
                ["", "### 화면 이동 URL", "", "| 유형 | 라벨 | URL |", "| --- | --- | --- |"]
            )
            lines.extend(
                (
                    f"| `{url['type']}` "
                    f"| {_escape_markdown_cell(url['label'])} "
                    f"| `{url['url']}` |"
                )
                for url in url_details
            )

    return "\n".join(lines)


def _select_scenario_groups(
    scenario_groups: list[str] | None,
) -> tuple[RagChatScenario, ...]:
    requested_groups = scenario_groups or ["core"]
    scenarios: list[RagChatScenario] = []
    seen_scenario_ids: set[str] = set()
    for group in requested_groups:
        for scenario in RAG_CHAT_SCENARIO_GROUPS[group]:
            if scenario.scenario_id in seen_scenario_ids:
                continue
            seen_scenario_ids.add(scenario.scenario_id)
            scenarios.append(scenario)
    return tuple(scenarios)


def _resolve_min_evidence_count(
    args: argparse.Namespace,
    scenario: RagChatScenario,
) -> int:
    if args.min_evidence_count is not None:
        return args.min_evidence_count
    return scenario.min_evidence_count


def _resolve_min_rdb_evidence_count(
    args: argparse.Namespace,
    scenario: RagChatScenario,
) -> int:
    if args.min_rdb_evidence_count is not None:
        return args.min_rdb_evidence_count
    return scenario.min_rdb_evidence_count


def _resolve_min_document_source_count(
    args: argparse.Namespace,
    scenario: RagChatScenario,
) -> int:
    if args.min_document_source_count is not None:
        return args.min_document_source_count
    return scenario.min_document_source_count


def _has_source_title_fragment(result: dict[str, Any], fragment: str) -> bool:
    normalized_fragment = fragment.casefold()
    return any(
        normalized_fragment in source["title"].casefold()
        for source in result["sourceDetails"]
    )


def _has_source_origin(result: dict[str, Any], source_origin: str) -> bool:
    normalized_origin = source_origin.casefold()
    return any(
        (source["sourceOrigin"] or "").casefold() == normalized_origin
        for source in result["sourceDetails"]
    )


def _find_non_read_only_rdb_urls(result: dict[str, Any]) -> list[str]:
    return [
        source["url"]
        for source in result["sourceDetails"]
        if source["sourceOrigin"] == "RDB"
        and source["url"]
        and "mode=read" not in source["url"]
    ]


def _single_expected_security_status(scenario: RagChatScenario) -> str | None:
    if len(scenario.expected_security_results) != 1:
        return None
    return scenario.expected_security_results[0][0]


def _single_expected_security_code(scenario: RagChatScenario) -> str | None:
    if len(scenario.expected_security_results) != 1:
        return None

    expected_code = scenario.expected_security_results[0][1]
    if expected_code is None:
        return "NONE"
    return expected_code


def _format_security_results(
    security_results: tuple[tuple[str, str | None], ...],
) -> str:
    return ",".join(
        _format_security_result(security_result)
        for security_result in security_results
    )


def _format_security_result(security_result: tuple[str, str | None]) -> str:
    status, code = security_result
    return f"{status}:{code or 'NONE'}"


def _format_optional_code(code: str | None) -> str:
    if code is None:
        return ""
    return f" / `{code}`"


def _escape_markdown_cell(value: str | None) -> str:
    if value is None:
        return "-"
    return value.replace("|", "\\|").replace("\n", "<br>")


def main(
    argv: list[str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    args = build_parser().parse_args(argv)

    try:
        result = asyncio.run(check_rag_chat_scenarios(args))
    except ChatServiceError as exc:
        print(f"RAG 챗봇 시나리오 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        return 1
    except Exception as exc:
        print(f"RAG 챗봇 시나리오 점검 실패: {exc}", file=error_output)
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
