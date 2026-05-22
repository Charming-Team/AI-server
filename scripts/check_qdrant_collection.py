import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from typing import Any, TextIO

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.qdrant_client import (
    QdrantCollectionCheckResult,
    QdrantDocumentSearchClient,
)
from app.features.chat.schemas import ChatErrorCode, ErrorResponse

DIMENSION_MISMATCH_MESSAGE = (
    "Qdrant 컬렉션 vector dimension이 FastAPI 임베딩 설정과 일치하지 않습니다."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qdrant 컬렉션과 FastAPI 임베딩 설정이 맞는지 점검합니다."
    )
    parser.add_argument("--qdrant-url", help="Qdrant base URL")
    parser.add_argument("--collection", help="Qdrant collection name")
    parser.add_argument("--api-key", help="Qdrant API key")
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        help="Expected embedding vector dimension",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        help="Qdrant request timeout seconds",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON",
    )
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    values: dict[str, Any] = {}
    if args.qdrant_url:
        values["qdrant_url"] = args.qdrant_url
    if args.collection:
        values["qdrant_collection"] = args.collection
    if args.api_key:
        values["qdrant_api_key"] = args.api_key
    if args.embedding_dimension is not None:
        values["embedding_dimension"] = args.embedding_dimension
    if args.timeout_seconds is not None:
        values["qdrant_timeout_seconds"] = args.timeout_seconds
    return Settings(**values)


async def check_collection(settings: Settings) -> QdrantCollectionCheckResult:
    client = QdrantDocumentSearchClient(settings)
    return await client.check_collection()


def build_dimension_mismatch_error(
    result: QdrantCollectionCheckResult,
) -> ErrorResponse | None:
    if result.is_dimension_matched:
        return None

    return ErrorResponse(
        code=ChatErrorCode.CHAT_QDRANT_004,
        message=(
            f"{DIMENSION_MISMATCH_MESSAGE} "
            f"expected={result.expected_dimension}, "
            f"actual={result.actual_dimension or 'unknown'}"
        ),
    )


def format_text_result(result: QdrantCollectionCheckResult) -> str:
    check_status = "PASS" if result.is_dimension_matched else "FAIL"
    lines = [
        f"status={check_status}",
        f"collection={result.collection_name}",
        f"qdrantStatus={result.status or 'unknown'}",
        f"expectedDimension={result.expected_dimension}",
        f"actualDimension={result.actual_dimension or 'unknown'}",
        f"pointsCount={result.points_count if result.points_count is not None else 'unknown'}",
    ]
    error = build_dimension_mismatch_error(result)
    if error is not None:
        lines.extend(
            [
                f"code={error.code}",
                f"message={error.message}",
            ]
        )
    return "\n".join(lines)


def format_json_result(result: QdrantCollectionCheckResult) -> str:
    error = build_dimension_mismatch_error(result)
    return json.dumps(
        {
            "checkStatus": "PASS" if result.is_dimension_matched else "FAIL",
            **asdict(result),
            "error": error.model_dump(mode="json") if error is not None else None,
        },
        ensure_ascii=False,
        indent=2,
    )


def main(
    argv: list[str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    settings = build_settings(args)

    try:
        result = asyncio.run(check_collection(settings))
    except ChatServiceError as exc:
        print(f"Qdrant 컬렉션 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0 if result.is_dimension_matched else 2


if __name__ == "__main__":
    raise SystemExit(main())
