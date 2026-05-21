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
class RdbChatScenario:
    scenario_id: str
    intent: ChatIntent
    question: str
    role: str | None = None
    expected_security_status: str = "PASSED"
    require_rdb_evidence: bool = True
    min_evidence_count: int = 1


DEFAULT_RDB_CHAT_SCENARIOS: tuple[RdbChatScenario, ...] = (
    RdbChatScenario(
        scenario_id="material-shortage",
        intent=ChatIntent.MATERIAL_SHORTAGE,
        question="자재 부족 현황 알려줘",
    ),
    RdbChatScenario(
        scenario_id="delivery-risk",
        intent=ChatIntent.DELIVERY_RISK,
        question="납기 위험이 있는 주문 알려줘",
    ),
    RdbChatScenario(
        scenario_id="production-plan",
        intent=ChatIntent.PRODUCTION_PLAN,
        question="생산계획 현황 알려줘",
    ),
    RdbChatScenario(
        scenario_id="line-bottleneck",
        intent=ChatIntent.LINE_BOTTLENECK,
        question="라인 병목 현황 알려줘",
    ),
    RdbChatScenario(
        scenario_id="work-priority",
        intent=ChatIntent.WORK_PRIORITY,
        question="작업 우선순위 알려줘",
    ),
)

ACCESS_CONTROL_RDB_CHAT_SCENARIOS: tuple[RdbChatScenario, ...] = (
    RdbChatScenario(
        scenario_id="operator-report-blocked",
        intent=ChatIntent.REPORT_LOOKUP,
        question="이번 달 월간 리포트 요약해줘",
        role="OPERATOR",
        expected_security_status="BLOCKED_UNAUTHORIZED",
        require_rdb_evidence=False,
        min_evidence_count=0,
    ),
    RdbChatScenario(
        scenario_id="operator-urgent-order-blocked",
        intent=ChatIntent.URGENT_ORDER_IMPACT,
        question="긴급 주문이 생산계획에 미치는 영향 알려줘",
        role="OPERATOR",
        expected_security_status="BLOCKED_UNAUTHORIZED",
        require_rdb_evidence=False,
        min_evidence_count=0,
    ),
    RdbChatScenario(
        scenario_id="operator-financial-blocked",
        intent=ChatIntent.DELIVERY_RISK,
        question="납기 지연 시 예상 패널티와 계약 금액 영향을 알려줘",
        role="OPERATOR",
        expected_security_status="BLOCKED_UNAUTHORIZED",
        require_rdb_evidence=False,
        min_evidence_count=0,
    ),
    RdbChatScenario(
        scenario_id="admin-chat-blocked",
        intent=ChatIntent.DELIVERY_RISK,
        question="납기 위험이 있는 주문 알려줘",
        role="ADMIN",
        expected_security_status="BLOCKED_UNAUTHORIZED",
        require_rdb_evidence=False,
        min_evidence_count=0,
    ),
)

FILTERED_RDB_CHAT_SCENARIOS: tuple[RdbChatScenario, ...] = (
    RdbChatScenario(
        scenario_id="material-shortage-this-week-target",
        intent=ChatIntent.MATERIAL_SHORTAGE,
        question="이번 주 RM-AL-001 자재 부족 현황 알려줘",
    ),
    RdbChatScenario(
        scenario_id="line-bottleneck-today-target",
        intent=ChatIntent.LINE_BOTTLENECK,
        question="오늘 LINE-A01 라인 병목 현황 알려줘",
    ),
    RdbChatScenario(
        scenario_id="production-plan-date-range",
        intent=ChatIntent.PRODUCTION_PLAN,
        question="2026-05-12부터 2026-05-18까지 생산계획 현황 알려줘",
    ),
)

RDB_CHAT_SCENARIO_GROUPS = {
    "core": DEFAULT_RDB_CHAT_SCENARIOS,
    "access": ACCESS_CONTROL_RDB_CHAT_SCENARIOS,
    "filtered": FILTERED_RDB_CHAT_SCENARIOS,
}
ALL_RDB_CHAT_SCENARIOS = tuple(
    scenario
    for group in RDB_CHAT_SCENARIO_GROUPS.values()
    for scenario in group
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FastAPI 챗봇 RDB Evidence 실사용 시나리오를 점검합니다."
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
        choices=[scenario.scenario_id for scenario in ALL_RDB_CHAT_SCENARIOS],
        help="특정 시나리오만 실행합니다. 여러 번 지정할 수 있습니다.",
    )
    parser.add_argument(
        "--scenario-group",
        action="append",
        choices=sorted(RDB_CHAT_SCENARIO_GROUPS),
        help=(
            "실행할 시나리오 묶음입니다. core, access, filtered 중 선택하며 "
            "여러 번 지정할 수 있습니다. 생략하면 core만 실행합니다."
        ),
    )
    parser.add_argument(
        "--min-evidence-count",
        type=int,
        default=None,
        help=(
            "모든 시나리오에 적용할 최소 RDB Evidence 개수. "
            "생략하면 시나리오 기본값을 사용합니다."
        ),
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
) -> tuple[RdbChatScenario, ...]:
    selected_scenarios = _select_scenario_groups(scenario_groups)
    if not scenario_ids:
        return selected_scenarios

    requested_ids = set(scenario_ids)
    search_space = selected_scenarios if scenario_groups else ALL_RDB_CHAT_SCENARIOS
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


async def check_rdb_chat_scenarios(
    args: argparse.Namespace,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    settings = build_settings(args)
    token = resolve_answer_token(args, settings)
    path = args.path or f"{settings.api_v1_prefix}/chat/answer"
    scenarios = select_scenarios(args.scenario, args.scenario_group)
    scenario_results = []

    for index, scenario in enumerate(scenarios):
        min_evidence_count = (
            args.min_evidence_count
            if args.min_evidence_count is not None
            else scenario.min_evidence_count
        )
        result = await check_chat_answer.check_chat_answer(
            base_url=args.base_url,
            path=path,
            token=token,
            request=build_request(args, scenario, index),
            timeout_seconds=args.timeout_seconds,
            min_evidence_count=min_evidence_count,
            require_rdb_evidence=scenario.require_rdb_evidence,
            http_client=http_client,
        )
        if result["intent"] != scenario.intent.value:
            raise ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_EVIDENCE_001,
                message=(
                    "챗봇 시나리오 intent가 예상과 다릅니다. "
                    f"scenario={scenario.scenario_id}, "
                    f"expected={scenario.intent.value}, actual={result['intent']}"
                ),
            )
        if result["securityStatus"] != scenario.expected_security_status:
            raise ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_EVIDENCE_001,
                message=(
                    "챗봇 시나리오 보안 상태가 예상과 다릅니다. "
                    f"scenario={scenario.scenario_id}, "
                    f"expected={scenario.expected_security_status}, "
                    f"actual={result['securityStatus']}"
                ),
            )

        scenario_results.append(
            {
                "scenarioId": scenario.scenario_id,
                "role": scenario.role or args.role,
                "question": scenario.question,
                "expectedIntent": scenario.intent.value,
                "expectedSecurityStatus": scenario.expected_security_status,
                "requireRdbEvidence": scenario.require_rdb_evidence,
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
    scenario: RdbChatScenario,
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
            f"requireRdbEvidence={scenario['requireRdbEvidence']} "
            f"rdbEvidenceCount={scenario['rdbEvidenceCount']} "
            f"sourceCount={scenario['sourceCount']} "
            f"urlCount={scenario['urlCount']}"
        )
    return "\n".join(lines)


def _select_scenario_groups(
    scenario_groups: list[str] | None,
) -> tuple[RdbChatScenario, ...]:
    requested_groups = scenario_groups or ["core"]
    scenarios: list[RdbChatScenario] = []
    seen_scenario_ids: set[str] = set()
    for group in requested_groups:
        for scenario in RDB_CHAT_SCENARIO_GROUPS[group]:
            if scenario.scenario_id in seen_scenario_ids:
                continue
            seen_scenario_ids.add(scenario.scenario_id)
            scenarios.append(scenario)
    return tuple(scenarios)


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
        result = asyncio.run(check_rdb_chat_scenarios(args))
    except ChatServiceError as exc:
        print(f"RDB 챗봇 시나리오 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        return 1
    except Exception as exc:
        print(f"RDB 챗봇 시나리오 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
