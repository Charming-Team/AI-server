import argparse
import asyncio
import json
import sys
from typing import Any, TextIO

import httpx

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ErrorResponse
from scripts import check_document_index_api, validate_document_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "검증된 문서 payload JSON을 FastAPI 내부 문서 인덱싱 API로 일괄 등록합니다."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        help="인덱싱할 문서 payload JSON 파일 경로. 여러 파일은 옵션을 반복합니다.",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        help="인덱싱할 문서 payload JSON 파일이 들어 있는 디렉터리 경로",
    )
    parser.add_argument(
        "--base-url",
        default=check_document_index_api.DEFAULT_BASE_URL,
        help="FastAPI base URL",
    )
    parser.add_argument(
        "--path",
        help="문서 인덱싱 API path. 생략하면 Settings.api_v1_prefix 기준으로 생성합니다.",
    )
    parser.add_argument("--token", help="FastAPI document index internal token")
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. CLI 인자가 있으면 해당 값이 우선합니다.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--min-indexed-count",
        type=int,
        default=0,
        help="문서별 요구하는 최소 실제 Qdrant 인덱싱 청크 개수",
    )
    parser.add_argument(
        "--allow-skipped",
        action="store_true",
        help="임베딩 비활성화 등으로 문서 저장이 skip되어도 통과로 봅니다.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="payload 검증과 작업량 추정만 수행하고 API 호출은 하지 않습니다.",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    if args.env_file:
        return Settings(_env_file=args.env_file)
    return Settings()


def build_validate_only_result(
    input_paths: list[str],
    settings: Settings,
) -> dict[str, Any]:
    validation = validate_document_payload.validate_document_payload_files(
        input_paths,
        settings,
    )
    return {
        "checkStatus": "VALIDATED" if validation["invalidCount"] == 0 else "FAIL",
        "phase": "DRY_RUN",
        "networkChecked": False,
        "validation": validation,
        "documentCount": validation["validCount"],
        "indexedDocumentCount": 0,
        "failedDocumentCount": validation["invalidCount"],
        "totalChunkCount": validation["totalChunkCount"],
        "totalIndexedCount": 0,
    }


async def index_document_payloads(
    *,
    base_url: str,
    path: str,
    token: str,
    input_paths: list[str],
    settings: Settings,
    timeout_seconds: float,
    min_indexed_count: int = 0,
    allow_skipped: bool = False,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    validation = validate_document_payload.validate_document_payload_files(
        input_paths,
        settings,
    )
    if validation["invalidCount"]:
        return {
            "checkStatus": "FAIL",
            "phase": "VALIDATION",
            "networkChecked": False,
            "validation": validation,
            "documentCount": validation["validCount"],
            "indexedDocumentCount": 0,
            "failedDocumentCount": validation["invalidCount"],
            "totalChunkCount": validation["totalChunkCount"],
            "totalIndexedCount": 0,
        }

    items: list[dict[str, Any]] = []
    for input_path in input_paths:
        document = validate_document_payload.build_document(
            validate_document_payload.load_payload(input_path)
        )
        try:
            result = await check_document_index_api.check_document_index_api(
                base_url=base_url,
                path=path,
                token=token,
                document=document,
                timeout_seconds=timeout_seconds,
                min_indexed_count=min_indexed_count,
                allow_skipped=allow_skipped,
                http_client=http_client,
            )
            items.append(
                {
                    "inputPath": input_path,
                    "status": "PASS",
                    "documentId": result["documentId"],
                    "documentType": result["documentType"],
                    "chunkCount": result["chunkCount"],
                    "indexedCount": result["indexedCount"],
                    "skippedReason": result["skippedReason"],
                    "operationId": result["operationId"],
                    "operationStatus": result["operationStatus"],
                }
            )
        except ChatServiceError as exc:
            items.append(
                {
                    "inputPath": input_path,
                    "status": "FAIL",
                    "documentId": document.document_id,
                    "documentType": document.document_type,
                    "error": ErrorResponse(
                        code=exc.code,
                        message=exc.message,
                    ).model_dump(mode="json"),
                }
            )

    failed_count = sum(1 for item in items if item["status"] == "FAIL")
    return {
        "checkStatus": "FAIL" if failed_count else "PASS",
        "phase": "INDEX",
        "networkChecked": True,
        "url": check_document_index_api.build_index_url(base_url, path),
        "documentCount": len(input_paths),
        "indexedDocumentCount": len(input_paths) - failed_count,
        "failedDocumentCount": failed_count,
        "totalChunkCount": sum(
            item.get("chunkCount", 0) for item in items if item["status"] == "PASS"
        ),
        "totalIndexedCount": sum(
            item.get("indexedCount", 0) for item in items if item["status"] == "PASS"
        ),
        "minIndexedCount": min_indexed_count,
        "allowSkipped": allow_skipped,
        "results": items,
    }


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"phase={result['phase']}",
        f"networkChecked={result['networkChecked']}",
        f"documentCount={result['documentCount']}",
        f"indexedDocumentCount={result['indexedDocumentCount']}",
        f"failedDocumentCount={result['failedDocumentCount']}",
        f"totalChunkCount={result['totalChunkCount']}",
        f"totalIndexedCount={result['totalIndexedCount']}",
    ]
    if result["phase"] == "INDEX":
        lines.append(f"url={result['url']}")
        lines.append(f"minIndexedCount={result['minIndexedCount']}")
        lines.append(f"allowSkipped={result['allowSkipped']}")
        lines.extend(format_index_item(item) for item in result["results"])
        return "\n".join(lines)

    validation = result["validation"]
    lines.extend(
        [
            f"validationStatus={validation['status']}",
            f"validationInputCount={validation['inputCount']}",
            f"validationValidCount={validation['validCount']}",
            f"validationInvalidCount={validation['invalidCount']}",
            f"validationWarningCount={validation['warningCount']}",
            f"validationEstimatedQdrantUpsertPointCount="
            f"{validation['totalEstimatedQdrantUpsertPointCount']}",
        ]
    )
    return "\n".join(lines)


def format_index_item(item: dict[str, Any]) -> str:
    if item["status"] == "FAIL":
        error = item["error"]
        return (
            f"input={item['inputPath']} "
            f"status=FAIL "
            f"documentId={item['documentId']} "
            f"code={error['code']} "
            f"message={error['message']}"
        )

    return (
        f"input={item['inputPath']} "
        f"status=PASS "
        f"documentId={item['documentId']} "
        f"chunkCount={item['chunkCount']} "
        f"indexedCount={item['indexedCount']} "
        f"skippedReason={item['skippedReason']} "
        f"operationId={item['operationId']} "
        f"operationStatus={item['operationStatus']}"
    )


def format_json_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


def resolve_exit_code(result: dict[str, Any]) -> int:
    return 0 if result["checkStatus"] in {"PASS", "VALIDATED"} else 1


def format_error(error: ChatServiceError, as_json: bool) -> str:
    if as_json:
        return json.dumps(
            ErrorResponse(
                code=error.code,
                message=error.message,
            ).model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    return f"문서 payload 일괄 인덱싱 실패: {error.message}\ncode={error.code.value}"


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
        input_paths = validate_document_payload.normalize_input_paths(args)
        if args.dry_run:
            result = build_validate_only_result(input_paths, settings)
        else:
            token = check_document_index_api.resolve_index_token(args, settings)
            path = check_document_index_api.resolve_index_path(args, settings)
            result = asyncio.run(
                index_document_payloads(
                    base_url=args.base_url,
                    path=path,
                    token=token,
                    input_paths=input_paths,
                    settings=settings,
                    timeout_seconds=args.timeout_seconds,
                    min_indexed_count=args.min_indexed_count,
                    allow_skipped=args.allow_skipped,
                )
            )
    except ChatServiceError as exc:
        print(format_error(exc, args.json), file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return resolve_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
