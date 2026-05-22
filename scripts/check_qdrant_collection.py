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
    validate_qdrant_settings,
)
from app.features.chat.schemas import ChatErrorCode, ErrorResponse

DIMENSION_MISMATCH_MESSAGE = (
    "Qdrant 컬렉션 vector dimension이 FastAPI 임베딩 설정과 일치하지 않습니다."
)
QDRANT_NETWORK_FAILURE_ACTIONS = (
    "QDRANT_URL이 현재 실행 환경에서 접근 가능한 주소인지 확인하세요.",
    (
        "로컬 점검이면 kubectl port-forward로 Qdrant 6333 포트를 열고 "
        "http://localhost:6333을 사용하세요."
    ),
    "컬렉션이 없으면 scripts.create_qdrant_collection으로 먼저 생성하세요.",
)
QDRANT_COLLECTION_NOT_FOUND_ACTIONS = (
    "QDRANT_COLLECTION 값이 실제 Qdrant 컬렉션명과 일치하는지 확인하세요.",
    "컬렉션이 아직 없다면 scripts.create_qdrant_collection으로 생성하세요.",
)
QDRANT_SETTINGS_FAILURE_ACTIONS = (
    "QDRANT_URL과 QDRANT_COLLECTION 설정을 확인하세요.",
)
QDRANT_RESPONSE_FAILURE_ACTIONS = (
    "Qdrant 호환 API를 호출 중인지 확인하세요.",
    "프록시나 Ingress가 Qdrant JSON 응답을 HTML/오류 페이지로 바꾸지 않는지 확인하세요.",
)
QDRANT_DIMENSION_FAILURE_ACTIONS = (
    "EMBEDDING_DIMENSION과 Qdrant collection vector size를 같은 값으로 맞추세요.",
    "임베딩 모델을 바꿨다면 컬렉션 재생성 또는 별도 컬렉션 사용을 검토하세요.",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qdrant 컬렉션과 FastAPI 임베딩 설정이 맞는지 점검합니다."
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
        "--timeout-seconds",
        type=float,
        help="Qdrant request timeout seconds",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Qdrant 네트워크 호출 없이 로컬 설정만 검증합니다.",
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


def build_validate_only_result(settings: Settings) -> dict[str, Any]:
    validate_qdrant_settings(settings)
    return {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "collectionName": settings.qdrant_collection,
        "expectedDimension": settings.embedding_dimension,
        "qdrantUrlConfigured": bool(settings.qdrant_url.strip()),
        "apiKeyConfigured": bool(settings.qdrant_api_key),
        "networkChecked": False,
    }


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


def build_dimension_mismatch_actions(
    result: QdrantCollectionCheckResult,
) -> list[str]:
    if result.is_dimension_matched:
        return []
    return list(QDRANT_DIMENSION_FAILURE_ACTIONS)


def build_collection_failure_actions(exc: ChatServiceError) -> list[str]:
    if exc.code == ChatErrorCode.CHAT_QDRANT_001:
        return list(QDRANT_SETTINGS_FAILURE_ACTIONS)
    if exc.code == ChatErrorCode.CHAT_QDRANT_003:
        return list(QDRANT_RESPONSE_FAILURE_ACTIONS)
    if exc.status_code == 404:
        return list(QDRANT_COLLECTION_NOT_FOUND_ACTIONS)
    if exc.code == ChatErrorCode.CHAT_QDRANT_002:
        return list(QDRANT_NETWORK_FAILURE_ACTIONS)
    return ["Qdrant 설정, 네트워크 연결, 컬렉션 상태를 확인하세요."]


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
        lines.extend(
            f"nextAction={action}"
            for action in build_dimension_mismatch_actions(result)
        )
    return "\n".join(lines)


def format_json_result(result: QdrantCollectionCheckResult) -> str:
    error = build_dimension_mismatch_error(result)
    return json.dumps(
        {
            "checkStatus": "PASS" if result.is_dimension_matched else "FAIL",
            **asdict(result),
            "error": error.model_dump(mode="json") if error is not None else None,
            "nextActions": build_dimension_mismatch_actions(result),
        },
        ensure_ascii=False,
        indent=2,
    )


def format_validate_only_text_result(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "status=VALIDATED",
            "mode=validateOnly",
            f"collection={result['collectionName']}",
            f"expectedDimension={result['expectedDimension']}",
            f"qdrantUrlConfigured={result['qdrantUrlConfigured']}",
            f"apiKeyConfigured={result['apiKeyConfigured']}",
            f"networkChecked={result['networkChecked']}",
        ]
    )


def format_validate_only_json_result(result: dict[str, Any]) -> str:
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
        if args.validate_only:
            validate_only_result = build_validate_only_result(settings)
            if args.json:
                print(format_validate_only_json_result(validate_only_result), file=output)
            else:
                print(format_validate_only_text_result(validate_only_result), file=output)
            return 0

        result = asyncio.run(check_collection(settings))
    except ChatServiceError as exc:
        print(f"Qdrant 컬렉션 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        for action in build_collection_failure_actions(exc):
            print(f"nextAction={action}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0 if result.is_dimension_matched else 2


if __name__ == "__main__":
    raise SystemExit(main())
