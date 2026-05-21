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
        choices=[scenario.scenario_id for scenario in DEFAULT_RDB_CHAT_SCENARIOS],
        help="특정 시나리오만 실행합니다. 여러 번 지정할 수 있습니다.",
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


def select_scenarios(scenario_ids: list[str] | None) -> tuple[RdbChatScenario, ...]:
    if not scenario_ids:
        return DEFAULT_RDB_CHAT_SCENARIOS

    requested_ids = set(scenario_ids)
    return tuple(
        scenario
        for scenario in DEFAULT_RDB_CHAT_SCENARIOS
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
    scenarios = select_scenarios(args.scenario)
    scenario_results = []

    for index, scenario in enumerate(scenarios):
        min_evidence_count = args.min_evidence_count or scenario.min_evidence_count
        result = await check_chat_answer.check_chat_answer(
            base_url=args.base_url,
            path=path,
            token=token,
            request=build_request(args, scenario, index),
            timeout_seconds=args.timeout_seconds,
            min_evidence_count=min_evidence_count,
            require_rdb_evidence=True,
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

        scenario_results.append(
            {
                "scenarioId": scenario.scenario_id,
                "question": scenario.question,
                "expectedIntent": scenario.intent.value,
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
            role=args.role,
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
            f"intent={scenario['intent']} "
            f"securityStatus={scenario['securityStatus']} "
            f"rdbEvidenceCount={scenario['rdbEvidenceCount']} "
            f"sourceCount={scenario['sourceCount']} "
            f"urlCount={scenario['urlCount']}"
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
