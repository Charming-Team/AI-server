import argparse
import asyncio
import json
import sys
from typing import Any, TextIO

import httpx

from app.core.config import Settings
from app.features.chat.document_payload import InternalDocumentDeleteRequest
from app.features.chat.exceptions import ChatServiceError
from scripts import (
    chat_check_common,
    check_chat_answer,
    check_document_delete_api,
    check_document_index_api,
    check_qdrant_vector_search,
    rag_end_to_end_failure_actions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "FastAPI 문서 등록, 챗봇 답변, 문서 삭제까지 RAG end-to-end 흐름을 "
            "점검합니다."
        )
    )
    parser.add_argument(
        "--base-url",
        default=check_chat_answer.DEFAULT_BASE_URL,
        help="FastAPI base URL",
    )
    parser.add_argument("--answer-token", help="FastAPI chat answer internal token")
    parser.add_argument("--document-token", help="FastAPI document index internal token")
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. CLI 인자가 있으면 해당 값이 우선합니다.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--document-id",
        default=check_qdrant_vector_search.DEFAULT_DOCUMENT_ID,
        help="임시로 인덱싱할 문서 ID",
    )
    parser.add_argument(
        "--title",
        default=check_qdrant_vector_search.DEFAULT_TITLE,
        help="임시 문서 제목",
    )
    parser.add_argument(
        "--content",
        default=check_qdrant_vector_search.DEFAULT_CONTENT,
        help="임시 문서 본문",
    )
    parser.add_argument(
        "--url",
        default=check_qdrant_vector_search.DEFAULT_URL,
        help="임시 문서 내부 URL",
    )
    parser.add_argument(
        "--question",
        default=check_qdrant_vector_search.DEFAULT_QUESTION,
        help="문서 등록 후 챗봇에 물어볼 질문",
    )
    parser.add_argument("--role", default="MANUFACTURING_MANAGER", help="사용자 Role")
    parser.add_argument("--user-id", type=int, default=1, help="사용자 ID")
    parser.add_argument("--company-name", default="S-MAP", help="회사명 메타데이터")
    parser.add_argument("--session-id", type=int, default=1, help="세션 ID")
    parser.add_argument("--message-id", type=int, default=1, help="메시지 ID")
    parser.add_argument(
        "--requested-at",
        default=chat_check_common.DEFAULT_REQUESTED_AT,
        help="요청 기준 시각. ISO datetime 형식",
    )
    parser.add_argument(
        "--min-indexed-count",
        type=int,
        default=1,
        help="문서 등록 API에서 요구하는 최소 indexedCount",
    )
    parser.add_argument(
        "--min-document-source-count",
        type=int,
        default=1,
        help="챗봇 답변에서 요구하는 최소 Qdrant 문서 출처 개수",
    )
    parser.add_argument(
        "--min-evidence-count",
        type=int,
        default=1,
        help="챗봇 답변에서 요구하는 최소 전체 Evidence 개수",
    )
    parser.add_argument(
        "--require-rdb-evidence",
        action="store_true",
        help="챗봇 답변에 RDB Evidence가 함께 사용되어야 합니다.",
    )
    parser.add_argument(
        "--max-llm-total-tokens",
        type=int,
        default=None,
        help="챗봇 답변에서 허용할 LLM total token 최대값입니다.",
    )
    parser.add_argument(
        "--keep-document",
        action="store_true",
        help="점검 후 임시 문서를 삭제하지 않고 Qdrant에 남깁니다.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="네트워크 호출 없이 토큰, 경로, 샘플 문서 계약만 검증합니다.",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    if args.env_file:
        return Settings(_env_file=args.env_file)
    return Settings()


def build_sample_document(args: argparse.Namespace):
    return check_document_index_api.build_sample_document(
        argparse.Namespace(
            document_id=args.document_id,
            document_type="COMPANY_INFO",
            title=args.title,
            content=args.content,
            summary=args.content,
            url=args.url,
            reference_type="LINE",
            reference_id=None,
            basis_time=None,
            roles=[args.role],
            intents=["LINE_BOTTLENECK"],
            requested_by_role="MANUFACTURING_MANAGER",
            company_name=args.company_name,
        )
    )


def build_answer_request(args: argparse.Namespace):
    return chat_check_common.build_chat_answer_request(args)


def resolve_answer_token(args: argparse.Namespace, settings: Settings) -> str:
    return check_chat_answer.resolve_answer_token(
        argparse.Namespace(token=args.answer_token),
        settings,
    )


def resolve_document_token(args: argparse.Namespace, settings: Settings) -> str:
    return check_document_index_api.resolve_index_token(
        argparse.Namespace(token=args.document_token),
        settings,
    )


def build_paths(settings: Settings) -> dict[str, str]:
    return {
        "answerPath": f"{settings.api_v1_prefix}/chat/answer",
        "indexPath": f"{settings.api_v1_prefix}/chat/internal/documents/index",
        "deletePath": f"{settings.api_v1_prefix}/chat/internal/documents/delete",
    }


def build_validate_only_result(
    args: argparse.Namespace,
    settings: Settings,
    answer_token: str,
    document_token: str,
) -> dict[str, Any]:
    document = build_sample_document(args)
    request = build_answer_request(args)
    paths = build_paths(settings)
    return {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "networkChecked": False,
        "baseUrl": args.base_url,
        **paths,
        "documentId": document.document_id,
        "documentType": document.document_type,
        "question": request.question,
        "role": request.user.role,
        "answerTokenConfigured": bool(answer_token),
        "documentTokenConfigured": bool(document_token),
        "minIndexedCount": args.min_indexed_count,
        "minEvidenceCount": args.min_evidence_count,
        "minDocumentSourceCount": args.min_document_source_count,
        "requireRdbEvidence": args.require_rdb_evidence,
        "maxLlmTotalTokens": args.max_llm_total_tokens,
        "keepDocument": args.keep_document,
    }


async def check_rag_end_to_end(
    *,
    args: argparse.Namespace,
    settings: Settings,
    answer_token: str,
    document_token: str,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    paths = build_paths(settings)
    document = build_sample_document(args)
    request = build_answer_request(args)
    index_result: dict[str, Any] | None = None
    answer_result: dict[str, Any] | None = None
    cleanup_result: dict[str, Any] | None = None
    primary_error: ChatServiceError | None = None

    try:
        index_result = await check_document_index_api.check_document_index_api(
            base_url=args.base_url,
            path=paths["indexPath"],
            token=document_token,
            document=document,
            timeout_seconds=args.timeout_seconds,
            min_indexed_count=args.min_indexed_count,
            http_client=http_client,
        )
        answer_result = await check_chat_answer.check_chat_answer(
            base_url=args.base_url,
            path=paths["answerPath"],
            token=answer_token,
            request=request,
            timeout_seconds=args.timeout_seconds,
            min_evidence_count=args.min_evidence_count,
            require_rdb_evidence=args.require_rdb_evidence,
            min_document_source_count=args.min_document_source_count,
            require_vector_search=True,
            max_llm_total_tokens=args.max_llm_total_tokens,
            expected_security_status="PASSED",
            expected_security_code="NONE",
            http_client=http_client,
        )
    except ChatServiceError as exc:
        primary_error = exc
    finally:
        if index_result is not None and not args.keep_document:
            cleanup_result = await check_document_delete_api.check_document_delete_api(
                base_url=args.base_url,
                path=paths["deletePath"],
                token=document_token,
                request=InternalDocumentDeleteRequest(documentId=document.document_id),
                timeout_seconds=args.timeout_seconds,
                http_client=http_client,
            )

    if primary_error is not None:
        raise primary_error

    return {
        "checkStatus": "PASS",
        "mode": "NETWORK",
        "networkChecked": True,
        "baseUrl": args.base_url,
        **paths,
        "documentId": document.document_id,
        "question": request.question,
        "role": request.user.role,
        "index": index_result,
        "answer": answer_result,
        "cleanup": cleanup_result,
        "maxLlmTotalTokens": args.max_llm_total_tokens,
        "usedCleanup": cleanup_result is not None,
        "keepDocument": args.keep_document,
    }


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"mode={result['mode']}",
        f"networkChecked={result['networkChecked']}",
        f"baseUrl={result['baseUrl']}",
        f"answerPath={result['answerPath']}",
        f"indexPath={result['indexPath']}",
        f"deletePath={result['deletePath']}",
        f"documentId={result['documentId']}",
        f"question={result['question']}",
        f"role={result['role']}",
    ]
    if result["mode"] == "VALIDATE_ONLY":
        lines.extend(
            [
                f"answerTokenConfigured={result['answerTokenConfigured']}",
                f"documentTokenConfigured={result['documentTokenConfigured']}",
                f"minIndexedCount={result['minIndexedCount']}",
                f"minEvidenceCount={result['minEvidenceCount']}",
                f"minDocumentSourceCount={result['minDocumentSourceCount']}",
                f"requireRdbEvidence={result['requireRdbEvidence']}",
                f"maxLlmTotalTokens={result['maxLlmTotalTokens']}",
                f"keepDocument={result['keepDocument']}",
            ]
        )
        return "\n".join(lines)

    index = result["index"]
    answer = result["answer"]
    cleanup = result["cleanup"]
    lines.extend(
        [
            f"indexedCount={index['indexedCount']}",
            f"answerIntent={answer['intent']}",
            f"answerEvidenceCount={answer['evidenceCount']}",
            f"answerRdbEvidenceCount={answer['rdbEvidenceCount']}",
            f"answerDocumentSourceCount={answer['documentSourceCount']}",
            f"answerUsedVectorSearch={answer['usedVectorSearch']}",
            f"maxLlmTotalTokens={result.get('maxLlmTotalTokens')}",
            f"usedCleanup={result['usedCleanup']}",
            f"keepDocument={result['keepDocument']}",
        ]
    )
    if cleanup:
        lines.append(f"cleanupOperationStatus={cleanup['operationStatus']}")
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
        answer_token = resolve_answer_token(args, settings)
        document_token = resolve_document_token(args, settings)
        if args.validate_only:
            result = build_validate_only_result(
                args,
                settings,
                answer_token,
                document_token,
            )
        else:
            result = asyncio.run(
                check_rag_end_to_end(
                    args=args,
                    settings=settings,
                    answer_token=answer_token,
                    document_token=document_token,
                )
            )
    except ChatServiceError as exc:
        print(f"RAG end-to-end 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        for next_action in rag_end_to_end_failure_actions.build_rag_end_to_end_failure_actions(
            exc
        ):
            print(f"nextAction={next_action}", file=error_output)
        return 1
    except Exception as exc:
        print(f"RAG end-to-end 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
