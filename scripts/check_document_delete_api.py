import argparse
import asyncio
import json
import sys
from typing import Any, TextIO

import httpx
from pydantic import ValidationError

from app.core.config import Settings
from app.features.chat.document_index_service import DocumentDeleteResult
from app.features.chat.document_payload import InternalDocumentDeleteRequest
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode, ErrorResponse

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_DOCUMENT_ID = "smoke-document-api-contract"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FastAPI 내부 문서 삭제 API 계약과 결과를 점검합니다."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="FastAPI base URL")
    parser.add_argument(
        "--path",
        help="Document delete path. 생략하면 Settings.api_v1_prefix 기준으로 생성합니다.",
    )
    parser.add_argument("--token", help="FastAPI document index internal token")
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. CLI 인자가 있으면 해당 값이 우선합니다.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--document-id", default=DEFAULT_DOCUMENT_ID)
    parser.add_argument(
        "--expected-operation-status",
        help="기대하는 Qdrant operation status. 생략하면 형식만 검증합니다.",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    if args.env_file:
        return Settings(_env_file=args.env_file)
    return Settings()


def build_delete_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def resolve_delete_path(args: argparse.Namespace, settings: Settings) -> str:
    if args.path:
        return args.path
    return f"{settings.api_v1_prefix}/chat/internal/documents/delete"


def resolve_delete_token(args: argparse.Namespace, settings: Settings) -> str:
    token = args.token or settings.document_index_internal_token
    if not token:
        raise ChatServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_SECURITY_003,
            message="FastAPI document index internal token이 설정되지 않았습니다.",
        )
    return token


def build_delete_request(args: argparse.Namespace) -> InternalDocumentDeleteRequest:
    try:
        return InternalDocumentDeleteRequest.model_validate(
            {"documentId": args.document_id}
        )
    except ValidationError as exc:
        raise ChatServiceError(
            status_code=400,
            code=ChatErrorCode.CHAT_DOCUMENT_002,
            message="문서 삭제 API 점검 payload 필수 필드 또는 타입이 올바르지 않습니다.",
        ) from exc


async def check_document_delete_api(
    base_url: str,
    path: str,
    token: str,
    request: InternalDocumentDeleteRequest,
    timeout_seconds: float,
    expected_operation_status: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    url = build_delete_url(base_url, path)
    response = await _post_document_delete(
        url=url,
        token=token,
        request=request,
        timeout_seconds=timeout_seconds,
        http_client=http_client,
    )
    result = _parse_delete_result(response)
    _validate_delete_result(
        result,
        expected_operation_status=expected_operation_status,
    )

    return {
        "checkStatus": "PASS",
        "url": url,
        "documentId": result.document_id,
        "operationType": result.operation_type,
        "operationId": result.operation.get("operation_id"),
        "operationStatus": result.operation.get("status"),
        "expectedOperationStatus": expected_operation_status,
        "networkChecked": True,
    }


async def _post_document_delete(
    url: str,
    token: str,
    request: InternalDocumentDeleteRequest,
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
            message=f"FastAPI 문서 삭제 API 호출에 실패했습니다. {exc}",
        ) from exc

    if response.is_error:
        _raise_response_error(response)

    return response


def _parse_delete_result(response: httpx.Response) -> DocumentDeleteResult:
    try:
        return DocumentDeleteResult.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        raise ChatServiceError(
            status_code=502,
            code=ChatErrorCode.CHAT_DOCUMENT_003,
            message="FastAPI 문서 삭제 API 응답 형식이 올바르지 않습니다.",
        ) from exc


def _validate_delete_result(
    result: DocumentDeleteResult,
    expected_operation_status: str | None,
) -> None:
    operation_status = result.operation.get("status")
    if not isinstance(operation_status, str) or not operation_status:
        raise ChatServiceError(
            status_code=502,
            code=ChatErrorCode.CHAT_DOCUMENT_003,
            message="FastAPI 문서 삭제 API operation status가 올바르지 않습니다.",
        )

    if expected_operation_status is None:
        return

    if operation_status != expected_operation_status:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_DOCUMENT_003,
            message=(
                "FastAPI 문서 삭제 API operation status가 기준과 다릅니다. "
                f"expected={expected_operation_status}, actual={operation_status}"
            ),
        )


def _raise_response_error(response: httpx.Response) -> None:
    code = ChatErrorCode.CHAT_DOCUMENT_003
    message = (
        "FastAPI 문서 삭제 API가 실패했습니다. "
        f"status={response.status_code}, body={response.text[:300]}"
    )
    try:
        error = ErrorResponse.model_validate(response.json())
    except (ValueError, ValidationError):
        error = None

    if error is not None:
        code = _to_chat_error_code(error.code)
        message = f"FastAPI 문서 삭제 API가 실패했습니다. {error.message}"

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
        f"operationType={result['operationType']}",
        f"operationId={result['operationId']}",
        f"operationStatus={result['operationStatus']}",
        f"expectedOperationStatus={result['expectedOperationStatus']}",
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
        token = resolve_delete_token(args, settings)
        path = resolve_delete_path(args, settings)
        request = build_delete_request(args)
        result = asyncio.run(
            check_document_delete_api(
                base_url=args.base_url,
                path=path,
                token=token,
                request=request,
                timeout_seconds=args.timeout_seconds,
                expected_operation_status=args.expected_operation_status,
            )
        )
    except ChatServiceError as exc:
        print(f"FastAPI 문서 삭제 API 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        return 1
    except Exception as exc:
        print(f"FastAPI 문서 삭제 API 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
