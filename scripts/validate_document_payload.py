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

WARNING_CONTENT_NEAR_LIMIT = "문서 본문 길이가 설정 한도에 근접했습니다."
WARNING_CHUNK_NEAR_LIMIT = "문서 청크 수가 설정 한도에 근접했습니다."
WARNING_DUPLICATE_CHUNKS = "중복 청크가 있어 임베딩 요청 입력은 중복 제거 후 계산됩니다."
WARNING_EMBEDDING_DISABLED = "임베딩 기능이 비활성화되어 실제 문서 저장은 생략됩니다."


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
    parser.add_argument(
        "--include-chunks",
        action="store_true",
        help="본문 원문 없이 청크 ID와 글자 수 등 안전한 청크 메타데이터를 출력합니다.",
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
    include_chunks: bool = False,
) -> dict[str, Any]:
    document = build_document(payload)
    policy = DocumentIndexPolicy(max_content_chars=settings.document_content_max_chars)
    builder = DocumentIndexBuilder(settings)

    policy.validate(document)
    chunk_payloads = builder.build_payloads(document)
    chunk_count = len(chunk_payloads)
    if chunk_count > settings.document_max_chunks:
        raise ChatServiceError(
            status_code=400,
            code=ChatErrorCode.CHAT_DOCUMENT_002,
            message=(
                f"문서 청크는 최대 {settings.document_max_chunks}개까지 "
                "인덱싱할 수 있습니다."
            ),
        )
    unique_embedding_input_count = len({chunk.chunk_text for chunk in chunk_payloads})

    result = {
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
        "contentCharCount": len(document.content),
        "embeddingEnabled": settings.embedding_enabled,
        "embeddingInputCount": chunk_count,
        "uniqueEmbeddingInputCount": unique_embedding_input_count,
        "estimatedEmbeddingRequestCount": (
            1 if settings.embedding_enabled and unique_embedding_input_count > 0 else 0
        ),
        "estimatedQdrantUpsertPointCount": (
            chunk_count if settings.embedding_enabled and chunk_count > 0 else 0
        ),
        "warnings": build_warnings(
            document=document,
            settings=settings,
            chunk_count=chunk_count,
            unique_embedding_input_count=unique_embedding_input_count,
        ),
    }
    if include_chunks:
        result["chunks"] = [
            {
                "chunkId": chunk.chunk_id,
                "charCount": len(chunk.chunk_text),
                "summaryConfigured": bool(chunk.summary),
                "urlConfigured": bool(chunk.url),
            }
            for chunk in chunk_payloads
        ]
    return result


def build_warnings(
    document: InternalDocumentInput,
    settings: Settings,
    chunk_count: int,
    unique_embedding_input_count: int,
) -> list[str]:
    warnings: list[str] = []
    if not settings.embedding_enabled:
        warnings.append(WARNING_EMBEDDING_DISABLED)
    if _is_near_limit(len(document.content), settings.document_content_max_chars):
        warnings.append(WARNING_CONTENT_NEAR_LIMIT)
    if _is_near_limit(chunk_count, settings.document_max_chunks):
        warnings.append(WARNING_CHUNK_NEAR_LIMIT)
    if unique_embedding_input_count < chunk_count:
        warnings.append(WARNING_DUPLICATE_CHUNKS)
    return warnings


def _is_near_limit(value: int, limit: int) -> bool:
    if limit <= 0 or value <= 0:
        return False
    return value / limit >= 0.8


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
        f"contentCharCount={result['contentCharCount']}",
        f"embeddingEnabled={result['embeddingEnabled']}",
        f"embeddingInputCount={result['embeddingInputCount']}",
        f"uniqueEmbeddingInputCount={result['uniqueEmbeddingInputCount']}",
        f"estimatedEmbeddingRequestCount={result['estimatedEmbeddingRequestCount']}",
        f"estimatedQdrantUpsertPointCount={result['estimatedQdrantUpsertPointCount']}",
    ]
    if result["requestedByRole"]:
        lines.append(f"requestedByRole={result['requestedByRole']}")
    if result["skippedReason"]:
        lines.append(f"skippedReason={result['skippedReason']}")
    for chunk in result.get("chunks", []):
        lines.append(
            "chunk="
            f"{chunk['chunkId']} "
            f"charCount={chunk['charCount']} "
            f"summaryConfigured={chunk['summaryConfigured']} "
            f"urlConfigured={chunk['urlConfigured']}"
        )
    for warning in result["warnings"]:
        lines.append(f"warning={warning}")
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
        result = validate_document_payload(
            payload,
            settings,
            include_chunks=args.include_chunks,
        )
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
