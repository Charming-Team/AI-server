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
    min_document_source_count: int = 1

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
        question="RM-AL-001 자재 부족 현황과 대응 기준을 같이 알려줘",
    ),
    RagChatScenario(
        scenario_id="line-bottleneck-with-company-guide",
        intent=ChatIntent.LINE_BOTTLENECK,
        question="LINE-A01 병목 현황과 대응 기준을 같이 알려줘",
    ),
    RagChatScenario(
        scenario_id="delivery-risk-with-report",
        intent=ChatIntent.DELIVERY_RISK,
        question="납기 위험이 있는 주문과 관련 보고서 근거를 알려줘",
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
        min_document_source_count=1,
    ),
    RagChatScenario(
        scenario_id="manager-revenue-company-info-allowed",
        intent=ChatIntent.REPORT_LOOKUP,
        question="S-Map 매출 구조 알려줘",
        role="MANUFACTURING_MANAGER",
        require_rdb_evidence=False,
        min_evidence_count=1,
        min_rdb_evidence_count=0,
        min_document_source_count=1,
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
            "실행할 시나리오 묶음입니다. core, access 중 선택하며 "
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
            expected_security_status=_single_expected_security_status(scenario),
            expected_security_code=_single_expected_security_code(scenario),
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
                "minDocumentSourceCount": _resolve_min_document_source_count(
                    args,
                    scenario,
                ),
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
            f"documentSourceCount={scenario['documentSourceCount']} "
            f"usedVectorSearch={scenario['usedVectorSearch']} "
            f"sourceCount={scenario['sourceCount']} "
            f"urlCount={scenario['urlCount']}"
        )
    return "\n".join(lines)


def format_json_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


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
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
