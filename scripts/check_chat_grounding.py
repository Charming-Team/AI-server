import argparse
import asyncio
import json
import sys
from typing import Any, TextIO

import httpx

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode, ChatIntent
from scripts import chat_check_common, check_chat_answer, check_evidence_lookup

DEFAULT_FASTAPI_BASE_URL = "http://localhost:8000"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spring Evidence와 FastAPI 챗봇 답변 근거 반영 흐름을 한 번에 점검합니다."
    )
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. CLI 인자가 있으면 해당 값이 우선합니다.",
    )
    parser.add_argument("--spring-base-url", help="Spring server base URL")
    parser.add_argument("--spring-path", help="Spring Evidence lookup path")
    parser.add_argument("--spring-token", help="Spring Evidence internal token")
    parser.add_argument(
        "--fastapi-base-url",
        default=DEFAULT_FASTAPI_BASE_URL,
        help="FastAPI base URL",
    )
    parser.add_argument(
        "--fastapi-path",
        help="FastAPI chat answer path. 생략하면 Settings.api_v1_prefix 기준으로 생성합니다.",
    )
    parser.add_argument("--fastapi-token", help="FastAPI chat answer internal token")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--intent",
        choices=[intent.value for intent in ChatIntent if intent != ChatIntent.UNKNOWN],
        default=ChatIntent.MATERIAL_SHORTAGE.value,
        help="점검에 사용할 질문 의도",
    )
    chat_check_common.add_chat_request_arguments(
        parser,
        check_chat_answer.DEFAULT_QUESTION,
    )
    parser.add_argument(
        "--min-evidence-count",
        type=int,
        default=1,
        help="Spring Evidence와 FastAPI 답변에서 요구하는 최소 Evidence 개수",
    )
    parser.add_argument(
        "--allow-non-rdb-evidence",
        action="store_true",
        help="FastAPI 답변에서 RDB Evidence 사용 여부를 필수로 보지 않습니다.",
    )
    parser.add_argument(
        "--max-llm-total-tokens",
        type=int,
        default=None,
        help="FastAPI 챗봇 답변에서 허용할 LLM total token 최대값입니다.",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    values: dict[str, Any] = {}
    if args.spring_base_url:
        values["evidence_lookup_base_url"] = args.spring_base_url
    if args.spring_path:
        values["evidence_lookup_path"] = args.spring_path
    if args.spring_token:
        values["evidence_lookup_internal_token"] = args.spring_token
    if args.timeout_seconds is not None:
        values["evidence_lookup_timeout_seconds"] = args.timeout_seconds

    if args.env_file:
        return Settings(_env_file=args.env_file, **values)
    return Settings(**values)


def build_evidence_settings(args: argparse.Namespace) -> Settings:
    settings = build_settings(args)
    return settings.model_copy(update={"evidence_lookup_enabled": True})


def resolve_fastapi_token(args: argparse.Namespace, settings: Settings) -> str:
    token = args.fastapi_token or settings.chat_answer_internal_token
    if not token:
        raise ChatServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_SECURITY_003,
            message="FastAPI chat answer internal token이 설정되지 않았습니다.",
        )
    return token


async def check_chat_grounding(
    args: argparse.Namespace,
    spring_http_client: httpx.AsyncClient | None = None,
    fastapi_http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    settings = build_settings(args)
    evidence_settings = build_evidence_settings(args)
    request = chat_check_common.build_chat_answer_request(args)
    intent = ChatIntent(args.intent)
    fastapi_token = resolve_fastapi_token(args, settings)
    fastapi_path = args.fastapi_path or f"{settings.api_v1_prefix}/chat/answer"

    evidence_result = await check_evidence_lookup.check_evidence_lookup(
        evidence_settings,
        request,
        intent,
        min_items=args.min_evidence_count,
        http_client=spring_http_client,
    )
    answer_result = await check_chat_answer.check_chat_answer(
        base_url=args.fastapi_base_url,
        path=fastapi_path,
        token=fastapi_token,
        request=request,
        timeout_seconds=args.timeout_seconds,
        min_evidence_count=args.min_evidence_count,
        require_rdb_evidence=not args.allow_non_rdb_evidence,
        max_llm_total_tokens=args.max_llm_total_tokens,
        http_client=fastapi_http_client,
    )

    return {
        "checkStatus": "PASS",
        "intent": intent.value,
        "minEvidenceCount": args.min_evidence_count,
        "maxLlmTotalTokens": args.max_llm_total_tokens,
        "springEvidence": evidence_result,
        "fastapiAnswer": answer_result,
    }


def format_text_result(result: dict[str, Any]) -> str:
    spring_result = result["springEvidence"]
    answer_result = result["fastapiAnswer"]
    lines = [
        f"status={result['checkStatus']}",
        f"intent={result['intent']}",
        f"minEvidenceCount={result['minEvidenceCount']}",
        f"maxLlmTotalTokens={result.get('maxLlmTotalTokens')}",
        f"spring.url={spring_result['url']}",
        f"spring.itemCount={spring_result['itemCount']}",
        f"spring.sourceTypes={','.join(spring_result['sourceTypes'])}",
        f"fastapi.url={answer_result['url']}",
        f"fastapi.securityStatus={answer_result['securityStatus']}",
        f"fastapi.evidenceCount={answer_result['evidenceCount']}",
        f"fastapi.usedRdbEvidence={answer_result['usedRdbEvidence']}",
        f"fastapi.sourceCount={answer_result['sourceCount']}",
        f"fastapi.urlCount={answer_result['urlCount']}",
    ]
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
        result = asyncio.run(check_chat_grounding(args))
    except ChatServiceError as exc:
        print(f"챗봇 Grounding 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        return 1
    except Exception as exc:
        print(f"챗봇 Grounding 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
