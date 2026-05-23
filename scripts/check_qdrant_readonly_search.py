import argparse
import asyncio
import json
import sys
from typing import Any, TextIO
from urllib.parse import SplitResult, urlsplit, urlunsplit

from app.core.config import Settings
from app.features.chat.document_search_service import DocumentSearchService
from app.features.chat.embedding_client import validate_embedding_settings
from app.features.chat.embedding_service import EmbeddingService
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.qdrant_client import (
    QdrantDocumentSearchClient,
    validate_qdrant_settings,
)
from app.features.chat.runtime_mode import build_chat_runtime_mode
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatErrorCode,
    ChatIntent,
    DocumentSearchResult,
)
from scripts import chat_check_common

DEFAULT_QUESTION = "LINE-A01 병목 대응 기준 알려줘"
DEFAULT_INTENT = ChatIntent.LINE_BOTTLENECK.value
DEFAULT_ROLE = "MANUFACTURING_MANAGER"
DEFAULT_MIN_SOURCE_COUNT = 1

QDRANT_READONLY_SETTINGS_FAILURE_ACTIONS = (
    "QDRANT_URL과 QDRANT_COLLECTION 설정을 확인하세요.",
)
QDRANT_READONLY_EMBEDDING_FAILURE_ACTIONS = (
    "EMBEDDING_BASE_URL, EMBEDDING_PATH, EMBEDDING_MODEL, EMBEDDING_DIMENSION 설정을 확인하세요.",
    "임베딩 서비스가 실행 중인지 확인하세요.",
)
QDRANT_READONLY_QDRANT_NETWORK_FAILURE_ACTIONS = (
    "Qdrant URL, collection 이름, port-forward 상태를 확인하세요.",
    "컬렉션이 없다면 scripts.create_qdrant_collection으로 먼저 생성하세요.",
)
QDRANT_READONLY_RESPONSE_FAILURE_ACTIONS = (
    "Embedding/Qdrant 응답 형식과 Qdrant payload 계약을 확인하세요.",
)
QDRANT_READONLY_MATCH_FAILURE_ACTIONS = (
    "Qdrant에 질문과 관련된 보고서 또는 회사정보 문서가 들어있는지 확인하세요.",
    "allowedRoles, intentTags, url/referenceType/referenceId payload를 확인하세요.",
    "검색 질문과 intent가 실제 문서 메타데이터와 맞는지 확인하세요.",
)
QDRANT_READONLY_LOCAL_K8S_EMBEDDING_ACTION = (
    "로컬에서 실행 중이면 embedding-service는 Kubernetes 내부 DNS이므로 "
    "port-forward 또는 클러스터 내부 실행으로 점검하세요."
)
QDRANT_READONLY_LOCAL_K8S_QDRANT_ACTION = (
    "로컬에서 실행 중이면 Qdrant Kubernetes 내부 DNS 대신 port-forward URL을 사용하세요."
)

LOCALHOST_ENDPOINT_SCOPE = "LOCALHOST"
KUBERNETES_SERVICE_ENDPOINT_SCOPE = "KUBERNETES_SERVICE"
REMOTE_ENDPOINT_SCOPE = "REMOTE_OR_EXTERNAL"
UNKNOWN_ENDPOINT_SCOPE = "UNKNOWN"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qdrant에 이미 저장된 문서를 수정하지 않고 Embedding + Vector 검색 흐름을 점검합니다."
        )
    )
    parser.add_argument("--qdrant-url", help="Qdrant base URL")
    parser.add_argument("--collection", help="Qdrant collection name")
    parser.add_argument("--api-key", help="Qdrant API key")
    parser.add_argument("--embedding-base-url", help="Embedding service base URL")
    parser.add_argument("--embedding-path", help="Embedding service path")
    parser.add_argument("--embedding-api-key", help="Embedding API key")
    parser.add_argument("--embedding-model", help="Embedding model name")
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        help="Expected embedding vector dimension",
    )
    parser.add_argument(
        "--qdrant-top-k",
        type=int,
        help="Qdrant 검색 결과 최대 개수",
    )
    parser.add_argument(
        "--qdrant-score-threshold",
        type=float,
        help="Qdrant 검색 관련도 하한값",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        help="Qdrant request timeout seconds",
    )
    parser.add_argument(
        "--embedding-timeout-seconds",
        type=float,
        help="Embedding request timeout seconds",
    )
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. CLI 인자가 있으면 해당 값이 우선합니다.",
    )
    parser.add_argument(
        "--intent",
        choices=[intent.value for intent in ChatIntent if intent != ChatIntent.UNKNOWN],
        default=DEFAULT_INTENT,
        help="검색 질문에 사용할 intent",
    )
    parser.add_argument(
        "--min-source-count",
        type=int,
        default=DEFAULT_MIN_SOURCE_COUNT,
        help="요구하는 최소 Qdrant 문서 출처 개수",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Qdrant/Embedding 네트워크 호출 없이 로컬 설정만 검증합니다.",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    chat_check_common.add_chat_request_arguments(parser, DEFAULT_QUESTION)
    parser.set_defaults(role=DEFAULT_ROLE)
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    values: dict[str, Any] = {}
    if args.qdrant_url:
        values["qdrant_url"] = args.qdrant_url
    if args.collection:
        values["qdrant_collection"] = args.collection
    if args.api_key:
        values["qdrant_api_key"] = args.api_key
    if args.embedding_base_url:
        values["embedding_base_url"] = args.embedding_base_url
    if args.embedding_path:
        values["embedding_path"] = args.embedding_path
    if args.embedding_api_key:
        values["embedding_api_key"] = args.embedding_api_key
    if args.embedding_model:
        values["embedding_model"] = args.embedding_model
    if args.embedding_dimension is not None:
        values["embedding_dimension"] = args.embedding_dimension
    if args.qdrant_top_k is not None:
        values["qdrant_top_k"] = args.qdrant_top_k
    if args.qdrant_score_threshold is not None:
        values["qdrant_score_threshold"] = args.qdrant_score_threshold
    if args.timeout_seconds is not None:
        values["qdrant_timeout_seconds"] = args.timeout_seconds
    if args.embedding_timeout_seconds is not None:
        values["embedding_timeout_seconds"] = args.embedding_timeout_seconds

    if args.env_file:
        return Settings(_env_file=args.env_file, **values)
    return Settings(**values)


def build_search_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "qdrant_search_enabled": True,
            "embedding_enabled": True,
            "qdrant_top_k": max(1, settings.qdrant_top_k),
        }
    )


def build_validate_only_result(
    settings: Settings,
    request: ChatAnswerRequest,
    intent: ChatIntent,
    min_source_count: int,
) -> dict[str, Any]:
    search_settings = build_search_settings(settings)
    if min_source_count < 1:
        raise ChatServiceError(
            status_code=400,
            code=ChatErrorCode.CHAT_QDRANT_004,
            message="Qdrant read-only 검색 최소 출처 개수는 1 이상이어야 합니다.",
        )

    validate_qdrant_settings(search_settings)
    validate_embedding_settings(search_settings)
    return {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "networkChecked": False,
        "collectionName": search_settings.qdrant_collection,
        "runtimeMode": build_chat_runtime_mode(search_settings).model_dump(
            mode="json",
            by_alias=True,
        ),
        "endpointSummary": build_endpoint_summary(search_settings),
        "qdrantUrlConfigured": bool(search_settings.qdrant_url.strip()),
        "apiKeyConfigured": bool(search_settings.qdrant_api_key),
        "embeddingBaseUrlConfigured": bool(search_settings.embedding_base_url.strip()),
        "embeddingPathConfigured": bool(search_settings.embedding_path.strip()),
        "embeddingModel": search_settings.embedding_model,
        "embeddingDimension": search_settings.embedding_dimension,
        "question": request.question,
        "intent": intent.value,
        "role": request.user.role,
        "minSourceCount": min_source_count,
    }


async def check_qdrant_readonly_search(
    settings: Settings,
    request: ChatAnswerRequest,
    intent: ChatIntent,
    *,
    min_source_count: int = DEFAULT_MIN_SOURCE_COUNT,
    embedding_service: EmbeddingService | None = None,
    qdrant_client: QdrantDocumentSearchClient | None = None,
) -> dict[str, Any]:
    if min_source_count < 1:
        raise ChatServiceError(
            status_code=400,
            code=ChatErrorCode.CHAT_QDRANT_004,
            message="Qdrant read-only 검색 최소 출처 개수는 1 이상이어야 합니다.",
        )

    search_settings = build_search_settings(settings)
    search_service = DocumentSearchService(
        search_settings,
        embedding_service=embedding_service,
        qdrant_client=qdrant_client,
    )
    search_result = await search_service.search(request, intent)
    validate_search_result(search_result, min_source_count)

    return {
        "checkStatus": "PASS",
        "mode": "NETWORK",
        "networkChecked": True,
        "collectionName": search_settings.qdrant_collection,
        "runtimeMode": build_chat_runtime_mode(search_settings).model_dump(
            mode="json",
            by_alias=True,
        ),
        "endpointSummary": build_endpoint_summary(search_settings),
        "question": request.question,
        "intent": intent.value,
        "role": request.user.role,
        "sourceCount": len(search_result.sources),
        "minSourceCount": min_source_count,
        "sourceTitles": [source.title for source in search_result.sources],
        "sourceTypes": sorted({source.source_type for source in search_result.sources}),
        "usedReadOnlySearch": True,
    }


def validate_search_result(
    search_result: DocumentSearchResult,
    min_source_count: int,
) -> None:
    if not search_result.was_searched:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_QDRANT_004,
            message=(
                "Qdrant read-only 검색이 수행되지 않았습니다. "
                f"reason={search_result.skipped_reason or 'unknown'}"
            ),
        )

    if len(search_result.sources) < min_source_count:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_QDRANT_004,
            message=(
                "Qdrant read-only 검색 결과가 기준보다 적습니다. "
                f"expected>={min_source_count}, actual={len(search_result.sources)}, "
                f"reason={search_result.skipped_reason or 'unknown'}"
            ),
        )


def build_endpoint_summary(settings: Settings) -> dict[str, str]:
    return {
        "qdrantBaseUrl": redact_url(settings.qdrant_url),
        "qdrantEndpointScope": classify_endpoint_scope(settings.qdrant_url),
        "embeddingBaseUrl": redact_url(settings.embedding_base_url),
        "embeddingEndpointScope": classify_endpoint_scope(settings.embedding_base_url),
    }


def classify_endpoint_scope(url: str) -> str:
    parsed_url = urlsplit(url.strip())
    host = (parsed_url.hostname or "").lower()
    if not host:
        return UNKNOWN_ENDPOINT_SCOPE
    if host in {"localhost", "127.0.0.1", "::1"}:
        return LOCALHOST_ENDPOINT_SCOPE
    if _looks_like_kubernetes_service_host(host):
        return KUBERNETES_SERVICE_ENDPOINT_SCOPE
    return REMOTE_ENDPOINT_SCOPE


def redact_url(url: str) -> str:
    stripped_url = url.strip()
    parsed_url = urlsplit(stripped_url)
    if not parsed_url.netloc or ("@" not in parsed_url.netloc):
        return stripped_url

    hostname = parsed_url.hostname or ""
    port = f":{parsed_url.port}" if parsed_url.port else ""
    return urlunsplit(
        SplitResult(
            scheme=parsed_url.scheme,
            netloc=f"***:***@{hostname}{port}",
            path=parsed_url.path,
            query=parsed_url.query,
            fragment=parsed_url.fragment,
        )
    )


def _looks_like_kubernetes_service_host(host: str) -> bool:
    return (
        host.endswith(".svc")
        or ".svc." in host
        or host.endswith(".svc.cluster.local")
        or host in {"embedding-service", "qdrant", "qdrant-service"}
    )


def build_readonly_failure_actions(
    exc: ChatServiceError,
    settings: Settings | None = None,
) -> list[str]:
    if exc.code == ChatErrorCode.CHAT_QDRANT_001:
        return list(QDRANT_READONLY_SETTINGS_FAILURE_ACTIONS)
    if exc.code in {
        ChatErrorCode.CHAT_EMBEDDING_001,
        ChatErrorCode.CHAT_EMBEDDING_002,
        ChatErrorCode.CHAT_EMBEDDING_003,
        ChatErrorCode.CHAT_EMBEDDING_004,
    }:
        actions = list(QDRANT_READONLY_EMBEDDING_FAILURE_ACTIONS)
        if (
            settings is not None
            and classify_endpoint_scope(settings.embedding_base_url)
            == KUBERNETES_SERVICE_ENDPOINT_SCOPE
        ):
            actions.insert(0, QDRANT_READONLY_LOCAL_K8S_EMBEDDING_ACTION)
        return actions
    if exc.code == ChatErrorCode.CHAT_QDRANT_002:
        actions = list(QDRANT_READONLY_QDRANT_NETWORK_FAILURE_ACTIONS)
        if (
            settings is not None
            and classify_endpoint_scope(settings.qdrant_url)
            == KUBERNETES_SERVICE_ENDPOINT_SCOPE
        ):
            actions.insert(0, QDRANT_READONLY_LOCAL_K8S_QDRANT_ACTION)
        return actions
    if exc.code == ChatErrorCode.CHAT_QDRANT_003:
        return list(QDRANT_READONLY_RESPONSE_FAILURE_ACTIONS)
    if exc.code == ChatErrorCode.CHAT_QDRANT_004:
        return list(QDRANT_READONLY_MATCH_FAILURE_ACTIONS)
    return ["Qdrant read-only 검색 설정, 네트워크, payload 계약을 확인하세요."]


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"mode={result['mode']}",
        f"collection={result['collectionName']}",
        f"qdrantEndpointScope={result['endpointSummary']['qdrantEndpointScope']}",
        f"embeddingEndpointScope={result['endpointSummary']['embeddingEndpointScope']}",
        f"question={result['question']}",
        f"intent={result['intent']}",
        f"role={result['role']}",
        f"networkChecked={result['networkChecked']}",
    ]
    runtime_mode = result.get("runtimeMode")
    if isinstance(runtime_mode, dict):
        lines.extend(
            [
                f"groundingMode={runtime_mode['groundingMode']}",
                f"answerMode={runtime_mode['answerMode']}",
                f"ragSearchMode={runtime_mode['ragSearchMode']}",
            ]
        )
    if result["mode"] == "NETWORK":
        lines.extend(
            [
                f"sourceCount={result['sourceCount']}",
                f"minSourceCount={result['minSourceCount']}",
                f"sourceTitles={','.join(result['sourceTitles'])}",
                f"sourceTypes={','.join(result['sourceTypes'])}",
                f"usedReadOnlySearch={result['usedReadOnlySearch']}",
            ]
        )
    else:
        lines.extend(
            [
                f"qdrantUrlConfigured={result['qdrantUrlConfigured']}",
                f"apiKeyConfigured={result['apiKeyConfigured']}",
                f"embeddingBaseUrlConfigured={result['embeddingBaseUrlConfigured']}",
                f"embeddingPathConfigured={result['embeddingPathConfigured']}",
                f"embeddingModel={result['embeddingModel']}",
                f"embeddingDimension={result['embeddingDimension']}",
                f"minSourceCount={result['minSourceCount']}",
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
    settings: Settings | None = None

    try:
        settings = build_settings(args)
        request = chat_check_common.build_chat_answer_request(args)
        intent = ChatIntent(args.intent)
        if args.validate_only:
            result = build_validate_only_result(
                settings,
                request,
                intent,
                args.min_source_count,
            )
        else:
            result = asyncio.run(
                check_qdrant_readonly_search(
                    settings,
                    request,
                    intent,
                    min_source_count=args.min_source_count,
                )
            )
    except ChatServiceError as exc:
        print(f"Qdrant read-only 검색 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        for action in build_readonly_failure_actions(exc, settings):
            print(f"nextAction={action}", file=error_output)
        return 1
    except Exception as exc:
        print(f"Qdrant read-only 검색 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
