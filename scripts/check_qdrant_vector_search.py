import argparse
import asyncio
import json
import sys
from typing import Any, TextIO

from app.core.config import Settings
from app.features.chat.document_index_builder import DocumentIndexBuilder
from app.features.chat.document_payload import InternalDocumentInput, QdrantUpsertPoint
from app.features.chat.document_search_service import DocumentSearchService
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.qdrant_client import (
    QdrantDocumentIndexClient,
    QdrantDocumentSearchClient,
    validate_qdrant_settings,
)
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatErrorCode,
    ChatIntent,
    EmbeddingResult,
)
from scripts import chat_check_common

DEFAULT_DOCUMENT_ID = "smoke-company-line-bottleneck"
DEFAULT_TITLE = "LINE-A01 병목 대응 기준"
DEFAULT_CONTENT = (
    "LINE-A01에서 대기 시간이 증가하면 작업자는 대기 수량과 처리량을 먼저 확인합니다. "
    "제조관리직은 라인 상태, 설비 상태, 생산계획 순서를 함께 검토해 병목 원인을 판단합니다."
)
DEFAULT_URL = "/lines/LINE-A01"
DEFAULT_QUESTION = "LINE-A01 병목 대응 기준 알려줘"


class StaticEmbeddingService:
    def __init__(self, vector: list[float], model: str) -> None:
        self.vector = vector
        self.model = model

    async def embed_query(self, request: ChatAnswerRequest) -> EmbeddingResult:
        return EmbeddingResult(
            vector=self.vector,
            was_embedded=True,
            model=self.model,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qdrant 문서 저장부터 Vector 검색까지 챗봇 검색 흐름을 점검합니다."
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
        help="Smoke test에 사용할 벡터 차원",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        help="Qdrant request timeout seconds",
    )
    parser.add_argument(
        "--document-id",
        default=DEFAULT_DOCUMENT_ID,
        help="테스트로 저장할 문서 ID",
    )
    parser.add_argument("--title", default=DEFAULT_TITLE, help="테스트 문서 제목")
    parser.add_argument("--content", default=DEFAULT_CONTENT, help="테스트 문서 본문")
    parser.add_argument("--url", default=DEFAULT_URL, help="테스트 문서 내부 URL")
    parser.add_argument(
        "--intent",
        choices=[intent.value for intent in ChatIntent if intent != ChatIntent.UNKNOWN],
        default=ChatIntent.LINE_BOTTLENECK.value,
        help="테스트 문서와 검색에 사용할 intent",
    )
    parser.add_argument(
        "--keep-sample",
        action="store_true",
        help="점검 후 샘플 문서를 Qdrant에 남깁니다.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Qdrant 네트워크 호출 없이 로컬 설정과 샘플 문서만 검증합니다.",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    chat_check_common.add_chat_request_arguments(parser, DEFAULT_QUESTION)
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

    if args.env_file:
        return Settings(_env_file=args.env_file, **values)
    return Settings(**values)


def build_sample_document(args: argparse.Namespace) -> InternalDocumentInput:
    return InternalDocumentInput(
        documentId=args.document_id,
        documentType="COMPANY_INFO",
        title=args.title,
        content=args.content,
        summary=args.content,
        url=args.url,
        referenceType="LINE",
        referenceId=None,
        allowedRoles=[args.role],
        companyName=args.company_name,
        intentTags=[args.intent],
        requestedByRole="MANUFACTURING_MANAGER",
    )


def build_static_vector(dimension: int) -> list[float]:
    if dimension < 1:
        raise ChatServiceError(
            status_code=400,
            code=ChatErrorCode.CHAT_EMBEDDING_003,
            message="Qdrant smoke test 벡터 차원은 1 이상이어야 합니다.",
        )
    return [1.0, *([0.0] * (dimension - 1))]


def build_sample_point(
    settings: Settings,
    document: InternalDocumentInput,
    vector: list[float],
) -> QdrantUpsertPoint:
    builder = DocumentIndexBuilder(settings)
    payloads = builder.build_payloads(document)
    if not payloads:
        raise ChatServiceError(
            status_code=400,
            code=ChatErrorCode.CHAT_DOCUMENT_002,
            message="Qdrant smoke test 문서 본문이 비어 있습니다.",
        )
    return builder.build_point(payloads[0], vector)


def build_validate_only_result(
    settings: Settings,
    document: InternalDocumentInput,
    point: QdrantUpsertPoint,
    args: argparse.Namespace,
) -> dict[str, Any]:
    validate_qdrant_settings(settings)
    return {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "collectionName": settings.qdrant_collection,
        "documentId": document.document_id,
        "pointId": point.id,
        "intent": args.intent,
        "role": args.role,
        "embeddingDimension": settings.embedding_dimension,
        "qdrantUrlConfigured": bool(settings.qdrant_url.strip()),
        "apiKeyConfigured": bool(settings.qdrant_api_key),
        "networkChecked": False,
    }


async def check_qdrant_vector_search(
    settings: Settings,
    document: InternalDocumentInput,
    request: ChatAnswerRequest,
    intent: ChatIntent,
    *,
    keep_sample: bool = False,
    qdrant_index_client: QdrantDocumentIndexClient | None = None,
    qdrant_search_client: QdrantDocumentSearchClient | None = None,
) -> dict[str, Any]:
    validate_qdrant_settings(settings)
    vector = build_static_vector(settings.embedding_dimension)
    point = build_sample_point(settings, document, vector)
    index_client = qdrant_index_client or QdrantDocumentIndexClient(settings)
    search_client = qdrant_search_client or QdrantDocumentSearchClient(settings)

    await index_client.delete_by_document_id(document.document_id)
    upsert_operation = await index_client.upsert([point])

    try:
        search_settings = settings.model_copy(
            update={
                "qdrant_search_enabled": True,
                "qdrant_top_k": max(1, settings.qdrant_top_k),
                "qdrant_score_threshold": 0.0,
            }
        )
        search_service = DocumentSearchService(
            search_settings,
            embedding_service=StaticEmbeddingService(
                vector=vector,
                model=settings.embedding_model,
            ),
            qdrant_client=search_client,
        )
        search_result = await search_service.search(request, intent)
        matched_sources = [
            source
            for source in search_result.sources
            if source.source and source.source.startswith(document.document_id)
        ]
        if not matched_sources:
            raise ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_QDRANT_004,
                message="Qdrant에 저장한 샘플 문서를 Vector 검색 결과에서 찾지 못했습니다.",
            )

        return {
            "checkStatus": "PASS",
            "mode": "NETWORK",
            "collectionName": settings.qdrant_collection,
            "documentId": document.document_id,
            "pointId": point.id,
            "intent": intent.value,
            "role": request.user.role,
            "indexedCount": 1,
            "sourceCount": len(search_result.sources),
            "matchedSourceCount": len(matched_sources),
            "matchedTitles": [source.title for source in matched_sources],
            "usedCleanup": not keep_sample,
            "networkChecked": True,
            "upsertStatus": upsert_operation.get("status"),
        }
    finally:
        if not keep_sample:
            await index_client.delete_by_document_id(document.document_id)


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"mode={result['mode']}",
        f"collection={result['collectionName']}",
        f"documentId={result['documentId']}",
        f"pointId={result['pointId']}",
        f"intent={result['intent']}",
        f"role={result['role']}",
        f"networkChecked={result['networkChecked']}",
    ]
    if result["mode"] == "NETWORK":
        lines.extend(
            [
                f"indexedCount={result['indexedCount']}",
                f"sourceCount={result['sourceCount']}",
                f"matchedSourceCount={result['matchedSourceCount']}",
                f"matchedTitles={','.join(result['matchedTitles'])}",
                f"usedCleanup={result['usedCleanup']}",
                f"upsertStatus={result['upsertStatus']}",
            ]
        )
    else:
        lines.extend(
            [
                f"embeddingDimension={result['embeddingDimension']}",
                f"qdrantUrlConfigured={result['qdrantUrlConfigured']}",
                f"apiKeyConfigured={result['apiKeyConfigured']}",
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

    try:
        settings = build_settings(args)
        document = build_sample_document(args)
        vector = build_static_vector(settings.embedding_dimension)
        point = build_sample_point(settings, document, vector)
        if args.validate_only:
            result = build_validate_only_result(settings, document, point, args)
        else:
            request = chat_check_common.build_chat_answer_request(args)
            result = asyncio.run(
                check_qdrant_vector_search(
                    settings,
                    document,
                    request,
                    ChatIntent(args.intent),
                    keep_sample=args.keep_sample,
                )
            )
    except ChatServiceError as exc:
        print(f"Qdrant Vector 검색 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        return 1
    except Exception as exc:
        print(f"Qdrant Vector 검색 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
