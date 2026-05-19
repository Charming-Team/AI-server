import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import Any, TextIO

import httpx

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatAnswerResponse,
    ChatErrorCode,
    ChatUserContext,
)

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_QUESTION = "자재 부족 현황 알려줘"
DEFAULT_REQUESTED_AT = "2026-05-12T10:30:00+09:00"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FastAPI 챗봇 답변 API와 Evidence 반영 결과를 점검합니다."
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
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="점검 질문")
    parser.add_argument("--role", default="MANUFACTURING_MANAGER", help="사용자 Role")
    parser.add_argument("--user-id", type=int, default=1, help="사용자 ID")
    parser.add_argument("--company-name", default="S-MAP", help="회사명 메타데이터")
    parser.add_argument("--session-id", type=int, default=1, help="세션 ID")
    parser.add_argument("--message-id", type=int, default=1, help="메시지 ID")
    parser.add_argument(
        "--requested-at",
        default=DEFAULT_REQUESTED_AT,
        help="요청 기준 시각. ISO datetime 형식",
    )
    parser.add_argument(
        "--min-evidence-count",
        type=int,
        default=0,
        help="요구하는 최소 전체 Evidence 개수",
    )
    parser.add_argument(
        "--require-rdb-evidence",
        action="store_true",
        help="RDB Evidence가 실제로 사용됐는지 검증합니다.",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    if args.env_file:
        return Settings(_env_file=args.env_file)
    return Settings()


def build_request(args: argparse.Namespace) -> ChatAnswerRequest:
    return ChatAnswerRequest(
        sessionId=args.session_id,
        messageId=args.message_id,
        user=ChatUserContext(
            userId=args.user_id,
            role=args.role,
            companyName=args.company_name,
            status="ACTIVE",
        ),
        question=args.question,
        requestedAt=datetime.fromisoformat(args.requested_at),
    )


def build_answer_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def resolve_answer_path(args: argparse.Namespace, settings: Settings) -> str:
    if args.path:
        return args.path
    return f"{settings.api_v1_prefix}/chat/answer"


def resolve_answer_token(args: argparse.Namespace, settings: Settings) -> str:
    token = args.token or settings.chat_answer_internal_token
    if not token:
        raise ChatServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_SECURITY_003,
            message="FastAPI chat answer internal token이 설정되지 않았습니다.",
        )
    return token


async def check_chat_answer(
    base_url: str,
    path: str,
    token: str,
    request: ChatAnswerRequest,
    timeout_seconds: float,
    min_evidence_count: int = 0,
    require_rdb_evidence: bool = False,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    url = build_answer_url(base_url, path)
    response = await _post_chat_answer(
        url=url,
        token=token,
        request=request,
        timeout_seconds=timeout_seconds,
        http_client=http_client,
    )
    answer = ChatAnswerResponse.model_validate(response.json())
    evidence_count = answer.model_result.evidence_count
    if evidence_count < min_evidence_count:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_EVIDENCE_001,
            message=(
                "FastAPI 챗봇 응답 Evidence 개수가 기준보다 적습니다. "
                f"expected>={min_evidence_count}, actual={evidence_count}"
            ),
        )

    if require_rdb_evidence and not answer.model_result.used_rdb_evidence:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_EVIDENCE_001,
            message="FastAPI 챗봇 응답에 RDB Evidence가 사용되지 않았습니다.",
        )

    return {
        "checkStatus": "PASS",
        "url": url,
        "intent": answer.intent.value,
        "securityStatus": answer.security_result.status.value,
        "securityCode": (
            answer.security_result.code.value if answer.security_result.code else None
        ),
        "evidenceCount": evidence_count,
        "minEvidenceCount": min_evidence_count,
        "requireRdbEvidence": require_rdb_evidence,
        "rdbEvidenceCount": answer.model_result.rdb_evidence_count,
        "documentSourceCount": answer.model_result.document_source_count,
        "usedRdbEvidence": answer.model_result.used_rdb_evidence,
        "usedVectorSearch": answer.model_result.used_vector_search,
        "usedLlmGeneration": answer.model_result.used_llm_generation,
        "sourceCount": len(answer.sources),
        "urlCount": len(answer.urls),
    }


async def _post_chat_answer(
    url: str,
    token: str,
    request: ChatAnswerRequest,
    timeout_seconds: float,
    http_client: httpx.AsyncClient | None = None,
) -> httpx.Response:
    payload = request.model_dump(mode="json", by_alias=True)
    headers = {
        "Content-Type": "application/json",
        "X-Internal-Token": token,
    }

    try:
        if http_client is not None:
            response = await http_client.post(url, json=payload, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise ChatServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_SERVER_001,
            message=f"FastAPI 챗봇 답변 API 호출에 실패했습니다. {exc}",
        ) from exc

    if response.is_error:
        raise ChatServiceError(
            status_code=response.status_code,
            code=ChatErrorCode.CHAT_SERVER_001,
            message=(
                "FastAPI 챗봇 답변 API가 실패했습니다. "
                f"status={response.status_code}, body={response.text[:300]}"
            ),
        )

    return response


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"url={result['url']}",
        f"intent={result['intent']}",
        f"securityStatus={result['securityStatus']}",
        f"securityCode={result['securityCode']}",
        f"evidenceCount={result['evidenceCount']}",
        f"minEvidenceCount={result['minEvidenceCount']}",
        f"requireRdbEvidence={result['requireRdbEvidence']}",
        f"rdbEvidenceCount={result['rdbEvidenceCount']}",
        f"documentSourceCount={result['documentSourceCount']}",
        f"usedRdbEvidence={result['usedRdbEvidence']}",
        f"usedVectorSearch={result['usedVectorSearch']}",
        f"usedLlmGeneration={result['usedLlmGeneration']}",
        f"sourceCount={result['sourceCount']}",
        f"urlCount={result['urlCount']}",
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
        settings = build_settings(args)
        request = build_request(args)
        token = resolve_answer_token(args, settings)
        path = resolve_answer_path(args, settings)
        result = asyncio.run(
            check_chat_answer(
                base_url=args.base_url,
                path=path,
                token=token,
                request=request,
                timeout_seconds=args.timeout_seconds,
                min_evidence_count=args.min_evidence_count,
                require_rdb_evidence=args.require_rdb_evidence,
            )
        )
    except ChatServiceError as exc:
        print(f"FastAPI 챗봇 답변 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        return 1
    except Exception as exc:
        print(f"FastAPI 챗봇 답변 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
