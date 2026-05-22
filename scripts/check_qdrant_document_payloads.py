import argparse
import asyncio
import json
import sys
from typing import Any, TextIO

from pydantic import ValidationError

from app.core.config import Settings
from app.features.chat.access_control import BUSINESS_ROLES, QDRANT_DOCUMENT_TYPES
from app.features.chat.document_access_policy import DocumentAccessPolicy
from app.features.chat.document_payload import InternalDocumentPayload
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.qdrant_client import (
    QdrantDocumentSearchClient,
    validate_qdrant_settings,
)
from app.features.chat.schemas import ChatErrorCode, ChatIntent
from app.features.chat.source_url_policy import normalize_internal_url

QDRANT_PAYLOAD_SETTINGS_FAILURE_ACTIONS = (
    "QDRANT_URL과 QDRANT_COLLECTION 설정을 확인하세요.",
)
QDRANT_PAYLOAD_NETWORK_FAILURE_ACTIONS = (
    "Qdrant URL, collection 이름, port-forward 상태를 확인하세요.",
    "컬렉션이 없다면 scripts.create_qdrant_collection으로 먼저 생성하세요.",
)
QDRANT_PAYLOAD_MIN_POINTS_FAILURE_ACTIONS = (
    "보고서 또는 회사정보 문서가 Qdrant에 인덱싱되어 있는지 확인하세요.",
    "아직 문서를 넣지 않은 초기 상태라면 --min-points 0으로 점검하세요.",
)
QDRANT_PAYLOAD_CONTRACT_FAILURE_ACTIONS = (
    "Qdrant payload의 documentId, documentType, title, chunkText를 확인하세요.",
    "allowedRoles에는 OPERATOR, EXECUTIVE, MANUFACTURING_MANAGER 중 허용 역할만 넣으세요.",
    "intentTags에는 챗봇이 지원하는 intent 값을 넣고, url은 내부 relative path로 저장하세요.",
    "OPERATOR 허용 문서에는 계약 금액, 패널티, 비용 등 금액성 정보를 넣지 마세요.",
)
QDRANT_PAYLOAD_RESPONSE_FAILURE_ACTIONS = (
    "Qdrant scroll API 응답 형식이 예상 JSON 구조인지 확인하세요.",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qdrant에 저장된 챗봇 문서 payload 계약을 점검합니다."
    )
    parser.add_argument("--qdrant-url", help="Qdrant base URL")
    parser.add_argument("--collection", help="Qdrant collection name")
    parser.add_argument("--api-key", help="Qdrant API key")
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. CLI 인자가 있으면 해당 값이 우선합니다.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="점검할 Qdrant point 최대 개수",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=0,
        help="요구하는 최소 point 개수",
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
    if args.timeout_seconds is not None:
        values["qdrant_timeout_seconds"] = args.timeout_seconds

    if args.env_file:
        return Settings(_env_file=args.env_file, **values)
    return Settings(**values)


def build_validate_only_result(settings: Settings, limit: int, min_points: int) -> dict:
    validate_qdrant_settings(settings)
    return {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "collectionName": settings.qdrant_collection,
        "qdrantUrlConfigured": bool(settings.qdrant_url.strip()),
        "apiKeyConfigured": bool(settings.qdrant_api_key),
        "limit": limit,
        "minPoints": min_points,
        "networkChecked": False,
    }


async def check_qdrant_document_payloads(
    settings: Settings,
    *,
    limit: int = 20,
    min_points: int = 0,
    client: QdrantDocumentSearchClient | None = None,
) -> dict:
    validate_qdrant_settings(settings)
    qdrant_client = client or QdrantDocumentSearchClient(settings)
    points = await qdrant_client.scroll_points(limit=limit)
    invalid_points = validate_points(points)

    if len(points) < min_points:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_QDRANT_004,
            message=(
                "Qdrant 문서 point 개수가 기준보다 적습니다. "
                f"expected>={min_points}, actual={len(points)}"
            ),
        )

    if invalid_points:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_QDRANT_003,
            message=(
                "Qdrant 문서 payload 계약을 만족하지 않는 point가 있습니다. "
                f"invalidCount={len(invalid_points)}"
            ),
        )

    return {
        "checkStatus": "PASS",
        "mode": "NETWORK",
        "collectionName": settings.qdrant_collection,
        "networkChecked": True,
        "limit": limit,
        "minPoints": min_points,
        "pointCount": len(points),
        "invalidCount": 0,
        "documentTypes": sorted(_collect_values(points, "documentType")),
        "intentTags": sorted(_collect_list_values(points, "intentTags")),
        "allowedRoles": sorted(_collect_list_values(points, "allowedRoles")),
    }


def validate_points(points: list[dict]) -> list[dict]:
    invalid_points = []
    access_policy = DocumentAccessPolicy()
    for point in points:
        errors = validate_point(point, access_policy)
        if not errors:
            continue
        invalid_points.append(
            {
                "id": point.get("id"),
                "errors": errors,
            }
        )
    return invalid_points


def validate_point(point: dict, access_policy: DocumentAccessPolicy) -> list[str]:
    payload = point.get("payload")
    if not isinstance(payload, dict):
        return ["payload must be object"]

    errors: list[str] = []
    try:
        document = InternalDocumentPayload.model_validate(payload)
    except ValidationError as exc:
        return [f"payload validation failed: {exc.errors()[0]['loc']}"]

    if document.document_type not in QDRANT_DOCUMENT_TYPES:
        errors.append("documentType must be REPORT or COMPANY_INFO")

    if not document.allowed_roles:
        errors.append("allowedRoles is required")
    elif set(document.allowed_roles) - BUSINESS_ROLES:
        errors.append("allowedRoles contains unsupported role")

    if not document.intent_tags:
        errors.append("intentTags is required")
    elif set(document.intent_tags) - _allowed_intent_tags():
        errors.append("intentTags contains unsupported intent")

    if document.url and normalize_internal_url(document.url) is None:
        errors.append("url must be internal relative path")

    if not access_policy.allows_point(point, "OPERATOR"):
        errors.append("OPERATOR document contains restricted business terms")

    return errors


def build_payload_failure_actions(exc: ChatServiceError) -> list[str]:
    if exc.code == ChatErrorCode.CHAT_QDRANT_001:
        return list(QDRANT_PAYLOAD_SETTINGS_FAILURE_ACTIONS)
    if exc.code == ChatErrorCode.CHAT_QDRANT_002:
        return list(QDRANT_PAYLOAD_NETWORK_FAILURE_ACTIONS)
    if exc.code == ChatErrorCode.CHAT_QDRANT_003:
        if "payload 계약" in exc.message:
            return list(QDRANT_PAYLOAD_CONTRACT_FAILURE_ACTIONS)
        return list(QDRANT_PAYLOAD_RESPONSE_FAILURE_ACTIONS)
    if exc.code == ChatErrorCode.CHAT_QDRANT_004:
        return list(QDRANT_PAYLOAD_MIN_POINTS_FAILURE_ACTIONS)
    return ["Qdrant 문서 payload 저장 상태와 계약을 확인하세요."]


def _allowed_intent_tags() -> set[str]:
    return {intent.value for intent in ChatIntent if intent != ChatIntent.UNKNOWN}


def _collect_values(points: list[dict], key: str) -> set[str]:
    values = set()
    for point in points:
        payload = point.get("payload")
        if not isinstance(payload, dict):
            continue
        value = payload.get(key)
        if isinstance(value, str):
            values.add(value.strip().upper())
    return values


def _collect_list_values(points: list[dict], key: str) -> set[str]:
    values = set()
    for point in points:
        payload = point.get("payload")
        if not isinstance(payload, dict):
            continue
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        values.update(item.strip().upper() for item in value if isinstance(item, str))
    return values


def format_text_result(result: dict) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"mode={result['mode']}",
        f"collection={result['collectionName']}",
        f"networkChecked={result['networkChecked']}",
        f"limit={result['limit']}",
        f"minPoints={result['minPoints']}",
    ]
    if result["mode"] == "NETWORK":
        lines.extend(
            [
                f"pointCount={result['pointCount']}",
                f"invalidCount={result['invalidCount']}",
                f"documentTypes={','.join(result['documentTypes'])}",
                f"intentTags={','.join(result['intentTags'])}",
                f"allowedRoles={','.join(result['allowedRoles'])}",
            ]
        )
    else:
        lines.extend(
            [
                f"qdrantUrlConfigured={result['qdrantUrlConfigured']}",
                f"apiKeyConfigured={result['apiKeyConfigured']}",
            ]
        )
    return "\n".join(lines)


def format_json_result(result: dict) -> str:
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
        if args.validate_only:
            result = build_validate_only_result(
                settings,
                args.limit,
                args.min_points,
            )
        else:
            result = asyncio.run(
                check_qdrant_document_payloads(
                    settings,
                    limit=args.limit,
                    min_points=args.min_points,
                )
            )
    except ChatServiceError as exc:
        print(f"Qdrant 문서 payload 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        for action in build_payload_failure_actions(exc):
            print(f"nextAction={action}", file=error_output)
        return 1
    except Exception as exc:
        print(f"Qdrant 문서 payload 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
