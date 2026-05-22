import argparse
import asyncio
import json
import sys
from typing import Any, TextIO

import httpx
from pydantic import ValidationError

from app.core.config import Settings
from app.features.chat.document_index_service import DocumentIndexResult
from app.features.chat.document_payload import InternalDocumentInput
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode, ErrorResponse
from scripts.document_api_failure_actions import build_document_api_failure_actions

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_DOCUMENT_ID = "company-info-line-bottleneck"
DEFAULT_DOCUMENT_TYPE = "COMPANY_INFO"
DEFAULT_TITLE = "LINE-A01 병목 대응 기준"
DEFAULT_CONTENT = (
    "LINE-A01에서 대기 수량과 대기 시간이 함께 증가하면 라인 병목 가능성을 "
    "우선 확인한다. 처리량, 가동률, 설비 상태, 진행률을 함께 비교한다."
)
DEFAULT_URL = "/lines/LINE-A01?mode=read"
DEFAULT_ROLE = "MANUFACTURING_MANAGER"
DEFAULT_INTENT = "LINE_BOTTLENECK"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FastAPI 내부 문서 인덱싱 API 계약과 결과를 점검합니다."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="FastAPI base URL")
    parser.add_argument(
        "--path",
        help="Document index path. 생략하면 Settings.api_v1_prefix 기준으로 생성합니다.",
    )
    parser.add_argument("--token", help="FastAPI document index internal token")
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. CLI 인자가 있으면 해당 값이 우선합니다.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--document-id", default=DEFAULT_DOCUMENT_ID)
    parser.add_argument("--document-type", default=DEFAULT_DOCUMENT_TYPE)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--content", default=DEFAULT_CONTENT)
    parser.add_argument("--summary")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--reference-type")
    parser.add_argument("--reference-id", type=int)
    parser.add_argument("--basis-time")
    parser.add_argument(
        "--role",
        dest="roles",
        action="append",
        help=(
            "문서 allowedRoles 값. 여러 역할은 옵션을 반복해서 지정합니다. "
            "생략하면 MANUFACTURING_MANAGER를 사용합니다."
        ),
    )
    parser.add_argument(
        "--intent",
        dest="intents",
        action="append",
        help=(
            "문서 intentTags 값. 여러 의도는 옵션을 반복해서 지정합니다. "
            "생략하면 LINE_BOTTLENECK을 사용합니다."
        ),
    )
    parser.add_argument("--requested-by-role", default=DEFAULT_ROLE)
    parser.add_argument("--company-name", default="S-MAP")
    parser.add_argument(
        "--min-indexed-count",
        type=int,
        default=0,
        help="요구하는 최소 실제 Qdrant 인덱싱 청크 개수",
    )
    parser.add_argument(
        "--allow-skipped",
        action="store_true",
        help="임베딩 비활성화 등으로 문서 저장이 skip되어도 API 계약만 통과로 봅니다.",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    if args.env_file:
        return Settings(_env_file=args.env_file)
    return Settings()


def build_index_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def resolve_index_path(args: argparse.Namespace, settings: Settings) -> str:
    if args.path:
        return args.path
    return f"{settings.api_v1_prefix}/chat/internal/documents/index"


def resolve_index_token(args: argparse.Namespace, settings: Settings) -> str:
    token = args.token or settings.document_index_internal_token
    if not token:
        raise ChatServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_SECURITY_003,
            message="FastAPI document index internal token이 설정되지 않았습니다.",
        )
    return token


def build_sample_document(args: argparse.Namespace) -> InternalDocumentInput:
    payload = {
        "documentId": args.document_id,
        "documentType": args.document_type,
        "title": args.title,
        "content": args.content,
        "summary": args.summary,
        "url": args.url,
        "referenceType": args.reference_type,
        "referenceId": args.reference_id,
        "basisTime": args.basis_time,
        "allowedRoles": args.roles or [DEFAULT_ROLE],
        "companyName": args.company_name,
        "intentTags": args.intents or [DEFAULT_INTENT],
        "requestedByRole": args.requested_by_role,
    }
    try:
        return InternalDocumentInput.model_validate(payload)
    except ValidationError as exc:
        raise ChatServiceError(
            status_code=400,
            code=ChatErrorCode.CHAT_DOCUMENT_002,
            message="문서 인덱싱 API 점검 payload 필수 필드 또는 타입이 올바르지 않습니다.",
        ) from exc


async def check_document_index_api(
    base_url: str,
    path: str,
    token: str,
    document: InternalDocumentInput,
    timeout_seconds: float,
    min_indexed_count: int = 0,
    allow_skipped: bool = False,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    url = build_index_url(base_url, path)
    response = await _post_document_index(
        url=url,
        token=token,
        document=document,
        timeout_seconds=timeout_seconds,
        http_client=http_client,
    )
    result = _parse_index_result(response)
    _validate_index_result(
        result,
        min_indexed_count=min_indexed_count,
        allow_skipped=allow_skipped,
    )

    return {
        "checkStatus": "PASS",
        "url": url,
        "documentId": result.document_id,
        "documentType": document.document_type,
        "title": document.title,
        "contentCharCount": len(document.content),
        "requestedByRole": document.requested_by_role,
        "allowedRoleCount": len(document.allowed_roles),
        "intentTagCount": len(document.intent_tags),
        "chunkCount": result.chunk_count,
        "indexedCount": result.indexed_count,
        "minIndexedCount": min_indexed_count,
        "allowSkipped": allow_skipped,
        "skippedReason": result.skipped_reason,
        "operationId": result.operation.get("operation_id"),
        "operationStatus": result.operation.get("status"),
        "networkChecked": True,
    }


async def _post_document_index(
    url: str,
    token: str,
    document: InternalDocumentInput,
    timeout_seconds: float,
    http_client: httpx.AsyncClient | None = None,
) -> httpx.Response:
    payload = document.model_dump(mode="json", by_alias=True, exclude_none=True)
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
            message=f"FastAPI 문서 인덱싱 API 호출에 실패했습니다. {exc}",
        ) from exc

    if response.is_error:
        _raise_response_error(response)

    return response


def _parse_index_result(response: httpx.Response) -> DocumentIndexResult:
    try:
        return DocumentIndexResult.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        raise ChatServiceError(
            status_code=502,
            code=ChatErrorCode.CHAT_DOCUMENT_003,
            message="FastAPI 문서 인덱싱 API 응답 형식이 올바르지 않습니다.",
        ) from exc


def _validate_index_result(
    result: DocumentIndexResult,
    min_indexed_count: int,
    allow_skipped: bool,
) -> None:
    if result.skipped_reason and not allow_skipped:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_DOCUMENT_003,
            message=(
                "FastAPI 문서 인덱싱 API가 문서 저장을 생략했습니다. "
                f"skippedReason={result.skipped_reason}"
            ),
        )

    if result.indexed_count < min_indexed_count:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_DOCUMENT_003,
            message=(
                "FastAPI 문서 인덱싱 API indexedCount가 기준보다 적습니다. "
                f"expected>={min_indexed_count}, actual={result.indexed_count}"
            ),
        )


def _raise_response_error(response: httpx.Response) -> None:
    code = ChatErrorCode.CHAT_DOCUMENT_003
    message = (
        "FastAPI 문서 인덱싱 API가 실패했습니다. "
        f"status={response.status_code}, body={response.text[:300]}"
    )
    try:
        error = ErrorResponse.model_validate(response.json())
    except (ValueError, ValidationError):
        error = None

    if error is not None:
        code = _to_chat_error_code(error.code)
        message = f"FastAPI 문서 인덱싱 API가 실패했습니다. {error.message}"

    raise ChatServiceError(
        status_code=response.status_code,
        code=code,
        message=message,
    )


def _to_chat_error_code(code: ChatErrorCode | str) -> ChatErrorCode:
    if isinstance(code, ChatErrorCode):
        return code
    try:
        return ChatErrorCode(code)
    except ValueError:
        return ChatErrorCode.CHAT_DOCUMENT_003


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"url={result['url']}",
        f"documentId={result['documentId']}",
        f"documentType={result['documentType']}",
        f"title={result['title']}",
        f"contentCharCount={result['contentCharCount']}",
        f"requestedByRole={result['requestedByRole']}",
        f"allowedRoleCount={result['allowedRoleCount']}",
        f"intentTagCount={result['intentTagCount']}",
        f"chunkCount={result['chunkCount']}",
        f"indexedCount={result['indexedCount']}",
        f"minIndexedCount={result['minIndexedCount']}",
        f"allowSkipped={result['allowSkipped']}",
        f"skippedReason={result['skippedReason']}",
        f"operationId={result['operationId']}",
        f"operationStatus={result['operationStatus']}",
        f"networkChecked={result['networkChecked']}",
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
        token = resolve_index_token(args, settings)
        path = resolve_index_path(args, settings)
        document = build_sample_document(args)
        result = asyncio.run(
            check_document_index_api(
                base_url=args.base_url,
                path=path,
                token=token,
                document=document,
                timeout_seconds=args.timeout_seconds,
                min_indexed_count=args.min_indexed_count,
                allow_skipped=args.allow_skipped,
            )
        )
    except ChatServiceError as exc:
        print(f"FastAPI 문서 인덱싱 API 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        for next_action in build_document_api_failure_actions(exc):
            print(f"nextAction={next_action}", file=error_output)
        return 1
    except Exception as exc:
        print(f"FastAPI 문서 인덱싱 API 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
