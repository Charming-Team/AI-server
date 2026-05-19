import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from typing import Any, TextIO

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError, ChatServiceError
from app.features.chat.qdrant_client import (
    QdrantCollectionCheckResult,
    QdrantDocumentIndexClient,
    QdrantDocumentSearchClient,
)
from app.features.chat.schemas import ChatErrorCode, ErrorResponse

DIMENSION_MISMATCH_MESSAGE = (
    "Qdrant 컬렉션 vector dimension이 FastAPI 임베딩 설정과 일치하지 않습니다."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qdrant 컬렉션이 없으면 생성하고, 있으면 dimension을 확인합니다."
    )
    parser.add_argument("--qdrant-url", help="Qdrant base URL")
    parser.add_argument("--collection", help="Qdrant collection name")
    parser.add_argument("--api-key", help="Qdrant API key")
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. CLI 인자가 있으면 해당 값이 우선합니다.",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        help="Expected embedding vector dimension",
    )
    parser.add_argument(
        "--distance",
        default="Cosine",
        choices=("Cosine", "Euclid", "Dot", "Manhattan"),
        help="Qdrant vector distance",
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
    env_file = getattr(args, "env_file", None)
    if env_file:
        return Settings(_env_file=env_file, **values)
    return Settings(**values)


async def check_collection(settings: Settings) -> QdrantCollectionCheckResult:
    client = QdrantDocumentSearchClient(settings)
    return await client.check_collection()


async def create_collection(settings: Settings, distance: str) -> dict:
    client = QdrantDocumentIndexClient(settings)
    return await client.create_collection(distance=distance)


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


async def ensure_collection(settings: Settings, distance: str) -> dict[str, Any]:
    try:
        check_result = await check_collection(settings)
    except ChatExternalServiceError as exc:
        if exc.status_code != 404:
            raise
        operation = await create_collection(settings, distance)
        check_result = await check_collection(settings)
        return {
            "action": "CREATED",
            "operation": operation,
            "collection": asdict(check_result),
            "error": None,
        }

    error = build_dimension_mismatch_error(check_result)
    return {
        "action": "EXISTS" if check_result.is_dimension_matched else "MISMATCH",
        "operation": None,
        "collection": asdict(check_result),
        "error": error.model_dump(mode="json") if error is not None else None,
    }


def format_text_result(result: dict[str, Any]) -> str:
    collection = result["collection"]
    points_count = (
        collection["points_count"] if collection["points_count"] is not None else "unknown"
    )
    lines = [
        f"action={result['action']}",
        f"collection={collection['collection_name']}",
        f"qdrantStatus={collection['status'] or 'unknown'}",
        f"expectedDimension={collection['expected_dimension']}",
        f"actualDimension={collection['actual_dimension'] or 'unknown'}",
        f"dimensionMatched={collection['is_dimension_matched']}",
        f"pointsCount={points_count}",
    ]
    error = result.get("error")
    if isinstance(error, dict):
        lines.extend(
            [
                f"code={error['code']}",
                f"message={error['message']}",
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
    settings = build_settings(args)

    try:
        result = asyncio.run(ensure_collection(settings, args.distance))
    except ChatServiceError as exc:
        print(f"Qdrant 컬렉션 생성/점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 2 if result["action"] == "MISMATCH" else 0


if __name__ == "__main__":
    raise SystemExit(main())
