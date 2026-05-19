import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from pydantic import ValidationError

from app.core.config import Settings
from app.features.chat.document_index_builder import DocumentIndexBuilder
from app.features.chat.document_index_policy import DocumentIndexPolicy
from app.features.chat.document_payload import InternalDocumentInput
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import ChatErrorCode, ErrorResponse
from app.features.chat.skip_reasons import DOCUMENT_CONTENT_EMPTY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qdrant 인덱싱용 내부 문서 payload를 로컬에서 검증합니다."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="검증할 문서 payload JSON 파일 경로",
    )
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. 생략하면 기본 .env 설정을 사용합니다.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON",
    )
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    env_file = getattr(args, "env_file", None)
    if env_file:
        return Settings(_env_file=env_file)
    return Settings()


def load_payload(input_path: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ChatServiceError(
            status_code=400,
            code=ChatErrorCode.CHAT_DOCUMENT_002,
            message="문서 payload 파일을 읽을 수 없습니다.",
        ) from exc
    except json.JSONDecodeError as exc:
        raise ChatServiceError(
            status_code=400,
            code=ChatErrorCode.CHAT_DOCUMENT_002,
            message="문서 payload JSON 형식이 올바르지 않습니다.",
        ) from exc

    if isinstance(payload, dict):
        return payload

    raise ChatServiceError(
        status_code=400,
        code=ChatErrorCode.CHAT_DOCUMENT_002,
        message="문서 payload는 JSON object 형식이어야 합니다.",
    )


def build_document(payload: dict[str, Any]) -> InternalDocumentInput:
    try:
        return InternalDocumentInput.model_validate(payload)
    except ValidationError as exc:
        raise ChatServiceError(
            status_code=400,
            code=ChatErrorCode.CHAT_DOCUMENT_002,
            message="문서 payload 필수 필드 또는 타입이 올바르지 않습니다.",
        ) from exc


def validate_document_payload(
    payload: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    document = build_document(payload)
    policy = DocumentIndexPolicy(max_content_chars=settings.document_content_max_chars)
    builder = DocumentIndexBuilder(settings)

    policy.validate(document)
    chunk_count = len(builder.build_payloads(document))
    if chunk_count > settings.document_max_chunks:
        raise ChatServiceError(
            status_code=400,
            code=ChatErrorCode.CHAT_DOCUMENT_002,
            message=(
                f"문서 청크는 최대 {settings.document_max_chunks}개까지 "
                "인덱싱할 수 있습니다."
            ),
        )

    return {
        "status": "VALID",
        "documentId": document.document_id,
        "documentType": document.document_type,
        "title": document.title,
        "chunkCount": chunk_count,
        "allowedRoles": document.allowed_roles,
        "intentTags": document.intent_tags,
        "requestedByRole": document.requested_by_role,
        "companyNameConfigured": bool(document.company_name),
        "urlConfigured": bool(document.url),
        "networkChecked": False,
        "skippedReason": DOCUMENT_CONTENT_EMPTY if chunk_count == 0 else None,
    }


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"status={result['status']}",
        f"documentId={result['documentId']}",
        f"documentType={result['documentType']}",
        f"title={result['title']}",
        f"chunkCount={result['chunkCount']}",
        f"allowedRoles={','.join(result['allowedRoles'])}",
        f"intentTags={','.join(result['intentTags'])}",
        f"companyNameConfigured={result['companyNameConfigured']}",
        f"urlConfigured={result['urlConfigured']}",
        f"networkChecked={result['networkChecked']}",
    ]
    if result["requestedByRole"]:
        lines.append(f"requestedByRole={result['requestedByRole']}")
    if result["skippedReason"]:
        lines.append(f"skippedReason={result['skippedReason']}")
    return "\n".join(lines)


def format_json_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


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
    return f"문서 payload 검증 실패: {error.message}\ncode={error.code.value}"


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
        payload = load_payload(args.input)
        result = validate_document_payload(payload, settings)
    except ChatServiceError as exc:
        print(format_error(exc, args.json), file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
