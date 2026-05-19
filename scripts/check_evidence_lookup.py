import argparse
import asyncio
import json
import sys
from typing import Any, TextIO

import httpx

from app.core.config import Settings
from app.features.chat.evidence_service import (
    EvidenceService,
    validate_evidence_lookup_settings,
)
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatErrorCode,
    ChatIntent,
)
from scripts import chat_check_common

DEFAULT_QUESTION = "자재 부족으로 영향받는 생산계획 알려줘"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spring RDB Evidence 내부 API 연결과 응답 계약을 점검합니다."
    )
    parser.add_argument("--base-url", help="Spring server base URL")
    parser.add_argument("--path", help="Evidence lookup path")
    parser.add_argument("--token", help="Spring internal token")
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. CLI 인자가 있으면 해당 값이 우선합니다.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        help="Evidence lookup request timeout seconds",
    )
    parser.add_argument(
        "--intent",
        choices=[intent.value for intent in ChatIntent if intent != ChatIntent.UNKNOWN],
        default=ChatIntent.MATERIAL_SHORTAGE.value,
        help="점검에 사용할 질문 의도",
    )
    chat_check_common.add_chat_request_arguments(parser, DEFAULT_QUESTION)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Spring 네트워크 호출 없이 설정과 요청 payload만 검증합니다.",
    )
    parser.add_argument(
        "--min-items",
        type=int,
        default=0,
        help="네트워크 점검에서 요구하는 최소 Evidence item 개수",
    )
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    values: dict[str, Any] = {"evidence_lookup_enabled": True}
    if args.base_url:
        values["evidence_lookup_base_url"] = args.base_url
    if args.path:
        values["evidence_lookup_path"] = args.path
    if args.token:
        values["evidence_lookup_internal_token"] = args.token
    if args.timeout_seconds is not None:
        values["evidence_lookup_timeout_seconds"] = args.timeout_seconds

    if args.env_file:
        return Settings(_env_file=args.env_file, **values)
    return Settings(**values)


def build_request(args: argparse.Namespace) -> ChatAnswerRequest:
    return chat_check_common.build_chat_answer_request(args)


def build_validate_only_result(
    settings: Settings,
    request: ChatAnswerRequest,
    intent: ChatIntent,
) -> dict[str, Any]:
    validate_evidence_lookup_settings(settings)
    service = EvidenceService(settings)
    return {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "url": service._evidence_lookup_url,
        "tokenConfigured": bool(settings.evidence_lookup_internal_token),
        "networkChecked": False,
        "payload": service._build_payload(request, intent),
    }


async def check_evidence_lookup(
    settings: Settings,
    request: ChatAnswerRequest,
    intent: ChatIntent,
    min_items: int = 0,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    service = EvidenceService(settings, http_client=http_client)
    result = await service.get_evidence(request, intent)
    item_count = len(result.items)
    if item_count < min_items:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_EVIDENCE_001,
            message=(
                "Spring Evidence 응답 item 개수가 기준보다 적습니다. "
                f"expected>={min_items}, actual={item_count}"
            ),
        )

    return {
        "checkStatus": "PASS",
        "mode": "NETWORK",
        "url": service._evidence_lookup_url,
        "intent": result.intent.value,
        "basisTime": result.basis_time.isoformat(),
        "itemCount": item_count,
        "minItems": min_items,
        "sourceTypes": sorted({item.type for item in result.items}),
        "networkChecked": True,
    }


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"mode={result['mode']}",
        f"url={result['url']}",
        f"tokenConfigured={result.get('tokenConfigured', True)}",
        f"networkChecked={result['networkChecked']}",
    ]
    if result["mode"] == "VALIDATE_ONLY":
        payload = result["payload"]
        lines.extend(
            [
                f"intent={payload['intent']}",
                f"role={payload['user']['role']}",
                f"targetType={payload['filters'].get('targetType')}",
                f"targetCode={payload['filters'].get('targetCode')}",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            f"intent={result['intent']}",
            f"basisTime={result['basisTime']}",
            f"itemCount={result['itemCount']}",
            f"minItems={result['minItems']}",
            f"sourceTypes={','.join(result['sourceTypes'])}",
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
        settings = build_settings(args)
        intent = ChatIntent(args.intent)
        request = build_request(args)
        if args.validate_only:
            result = build_validate_only_result(settings, request, intent)
        else:
            result = asyncio.run(
                check_evidence_lookup(
                    settings,
                    request,
                    intent,
                    min_items=args.min_items,
                )
            )
    except ChatServiceError as exc:
        print(f"Spring Evidence 연결 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        return 1
    except Exception as exc:
        print(f"Spring Evidence 연결 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
