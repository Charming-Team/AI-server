import argparse
import asyncio
import json
import sys
from typing import Any, TextIO

import httpx

from app.core.config import Settings
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatAnswerResponse,
    ChatErrorCode,
    ChatIntent,
    SecurityStatus,
)
from scripts import chat_check_common
from scripts.chat_api_failure_actions import build_answer_api_failure_actions

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_QUESTION = "자재 부족 현황 알려줘"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FastAPI 챗봇 답변 API와 Evidence 반영 결과를 점검합니다."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="FastAPI base URL")
    parser.add_argument(
        "--path",
        help="Chat answer path. 생략하면 Settings.api_v1_prefix 기준으로 생성합니다.",
    )
    parser.add_argument("--token", help="FastAPI chat answer internal token")
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. CLI 인자가 있으면 해당 값이 우선합니다.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    chat_check_common.add_chat_request_arguments(parser, DEFAULT_QUESTION)
    parser.add_argument(
        "--min-evidence-count",
        type=int,
        default=0,
        help="요구하는 최소 전체 Evidence 개수",
    )
    parser.add_argument(
        "--require-rdb-evidence",
        action="store_true",
        help="RDB Evidence가 실제로 사용됐는지 검증합니다.",
    )
    parser.add_argument(
        "--min-document-source-count",
        type=int,
        default=0,
        help="요구하는 최소 Qdrant 문서 출처 개수",
    )
    parser.add_argument(
        "--require-vector-search",
        action="store_true",
        help="Qdrant Vector Search가 실제로 수행됐는지 검증합니다.",
    )
    parser.add_argument(
        "--require-llm-generation",
        action="store_true",
        help="LLM 답변 생성이 실제로 수행됐는지 검증합니다.",
    )
    parser.add_argument(
        "--expected-llm-skipped-reason",
        help=(
            "기대하는 LLM 생성 스킵 사유입니다. 스킵 사유가 없어야 하면 NONE을 "
            "사용합니다."
        ),
    )
    parser.add_argument(
        "--expected-security-status",
        choices=[status.value for status in SecurityStatus],
        help="기대하는 응답 보안 상태. 예: PASSED, BLOCKED_UNAUTHORIZED",
    )
    parser.add_argument(
        "--expected-security-code",
        choices=[code.value for code in ChatErrorCode] + ["NONE"],
        help="기대하는 응답 보안 에러 코드. 코드가 없어야 하면 NONE을 사용합니다.",
    )
    parser.add_argument(
        "--expected-intent",
        choices=[intent.value for intent in ChatIntent],
        help="기대하는 질문 intent. 예: MATERIAL_SHORTAGE, LINE_BOTTLENECK",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="점검 결과를 리뷰용 Markdown으로 출력합니다.",
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    if args.env_file:
        return Settings(_env_file=args.env_file)
    return Settings()


def build_request(args: argparse.Namespace) -> ChatAnswerRequest:
    return chat_check_common.build_chat_answer_request(args)


def build_answer_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def resolve_answer_path(args: argparse.Namespace, settings: Settings) -> str:
    if args.path:
        return args.path
    return f"{settings.api_v1_prefix}/chat/answer"


def resolve_answer_token(args: argparse.Namespace, settings: Settings) -> str:
    token = args.token or settings.chat_answer_internal_token
    if not token:
        raise ChatServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_SECURITY_003,
            message="FastAPI chat answer internal token이 설정되지 않았습니다.",
        )
    return token


async def check_chat_answer(
    base_url: str,
    path: str,
    token: str,
    request: ChatAnswerRequest,
    timeout_seconds: float,
    min_evidence_count: int = 0,
    require_rdb_evidence: bool = False,
    min_document_source_count: int = 0,
    require_vector_search: bool = False,
    require_llm_generation: bool = False,
    expected_llm_skipped_reason: str | None = None,
    expected_security_status: str | None = None,
    expected_security_code: str | None = None,
    expected_intent: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    url = build_answer_url(base_url, path)
    response = await _post_chat_answer(
        url=url,
        token=token,
        request=request,
        timeout_seconds=timeout_seconds,
        http_client=http_client,
    )
    answer = ChatAnswerResponse.model_validate(response.json())
    actual_intent = answer.intent.value
    if expected_intent is not None and actual_intent != expected_intent:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_EVIDENCE_001,
            message=(
                "FastAPI 챗봇 응답 intent가 기대값과 다릅니다. "
                f"expected={expected_intent}, actual={actual_intent}"
            ),
        )

    security_status = answer.security_result.status.value
    security_code = (
        answer.security_result.code.value if answer.security_result.code else None
    )
    if expected_security_status and security_status != expected_security_status:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_SECURITY_001,
            message=(
                "FastAPI 챗봇 응답 보안 상태가 기대값과 다릅니다. "
                f"expected={expected_security_status}, actual={security_status}"
            ),
        )

    expected_security_code_value = _resolve_expected_security_code(
        expected_security_code
    )
    if (
        expected_security_code is not None
        and security_code != expected_security_code_value
    ):
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_SECURITY_001,
            message=(
                "FastAPI 챗봇 응답 보안 코드가 기대값과 다릅니다. "
                f"expected={expected_security_code_value}, actual={security_code}"
            ),
        )

    evidence_count = answer.model_result.evidence_count
    if evidence_count < min_evidence_count:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_EVIDENCE_001,
            message=(
                "FastAPI 챗봇 응답 Evidence 개수가 기준보다 적습니다. "
                f"expected>={min_evidence_count}, actual={evidence_count}"
            ),
        )

    if require_rdb_evidence and not answer.model_result.used_rdb_evidence:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_EVIDENCE_001,
            message="FastAPI 챗봇 응답에 RDB Evidence가 사용되지 않았습니다.",
        )

    document_source_count = answer.model_result.document_source_count
    if document_source_count < min_document_source_count:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_EVIDENCE_001,
            message=(
                "FastAPI 챗봇 응답 Qdrant 문서 출처 개수가 기준보다 적습니다. "
                f"expected>={min_document_source_count}, actual={document_source_count}"
            ),
        )

    if require_vector_search and not answer.model_result.used_vector_search:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_EVIDENCE_001,
            message="FastAPI 챗봇 응답에 Qdrant Vector Search가 사용되지 않았습니다.",
        )

    if require_llm_generation and not answer.model_result.used_llm_generation:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_LLM_004,
            message="FastAPI 챗봇 응답에 LLM 답변 생성이 사용되지 않았습니다.",
        )

    llm_skipped_reason = answer.model_result.llm_generation_skipped_reason
    expected_llm_skipped_reason_value = _resolve_expected_optional_text(
        expected_llm_skipped_reason
    )
    if (
        expected_llm_skipped_reason is not None
        and llm_skipped_reason != expected_llm_skipped_reason_value
    ):
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_LLM_004,
            message=(
                "FastAPI 챗봇 응답의 LLM 생성 스킵 사유가 기대값과 다릅니다. "
                f"expected={expected_llm_skipped_reason_value}, "
                f"actual={llm_skipped_reason}"
            ),
        )

    return {
        "checkStatus": "PASS",
        "url": url,
        "intent": actual_intent,
        "answer": answer.answer,
        "expectedIntent": expected_intent,
        "securityStatus": security_status,
        "expectedSecurityStatus": expected_security_status,
        "securityCode": security_code,
        "expectedSecurityCode": expected_security_code,
        "evidenceCount": evidence_count,
        "minEvidenceCount": min_evidence_count,
        "requireRdbEvidence": require_rdb_evidence,
        "rdbEvidenceCount": answer.model_result.rdb_evidence_count,
        "documentSourceCount": document_source_count,
        "minDocumentSourceCount": min_document_source_count,
        "usedRdbEvidence": answer.model_result.used_rdb_evidence,
        "usedVectorSearch": answer.model_result.used_vector_search,
        "requireVectorSearch": require_vector_search,
        "vectorSearchSkippedReason": answer.model_result.vector_search_skipped_reason,
        "usedLlmGeneration": answer.model_result.used_llm_generation,
        "llmCacheHit": answer.model_result.llm_cache_hit,
        "requireLlmGeneration": require_llm_generation,
        "llmGenerationSkippedReason": llm_skipped_reason,
        "expectedLlmGenerationSkippedReason": expected_llm_skipped_reason,
        "sourceCount": len(answer.sources),
        "sourceDetails": [
            {
                "sourceType": source.source_type,
                "title": source.title,
                "url": source.url,
                "sourceOrigin": source.source_origin,
            }
            for source in answer.sources
        ],
        "urlCount": len(answer.urls),
        "urlDetails": [
            {
                "label": url_item.label,
                "url": url_item.url,
                "type": url_item.type,
            }
            for url_item in answer.urls
        ],
    }


def _resolve_expected_security_code(expected_security_code: str | None) -> str | None:
    return _resolve_expected_optional_text(expected_security_code)


def _resolve_expected_optional_text(expected_text: str | None) -> str | None:
    if expected_text == "NONE":
        return None
    return expected_text


async def _post_chat_answer(
    url: str,
    token: str,
    request: ChatAnswerRequest,
    timeout_seconds: float,
    http_client: httpx.AsyncClient | None = None,
) -> httpx.Response:
    payload = request.model_dump(mode="json", by_alias=True)
    headers = {
        "Content-Type": "application/json",
        "X-Internal-Token": token,
    }

    try:
        if http_client is not None:
            response = await http_client.post(url, json=payload, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise ChatServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_SERVER_001,
            message=f"FastAPI 챗봇 답변 API 호출에 실패했습니다. {exc}",
        ) from exc

    if response.is_error:
        raise ChatServiceError(
            status_code=response.status_code,
            code=ChatErrorCode.CHAT_SERVER_001,
            message=(
                "FastAPI 챗봇 답변 API가 실패했습니다. "
                f"status={response.status_code}, body={response.text[:300]}"
            ),
        )

    return response


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"url={result['url']}",
        f"intent={result['intent']}",
        f"expectedIntent={result['expectedIntent']}",
        f"securityStatus={result['securityStatus']}",
        f"expectedSecurityStatus={result['expectedSecurityStatus']}",
        f"securityCode={result['securityCode']}",
        f"expectedSecurityCode={result['expectedSecurityCode']}",
        f"evidenceCount={result['evidenceCount']}",
        f"minEvidenceCount={result['minEvidenceCount']}",
        f"requireRdbEvidence={result['requireRdbEvidence']}",
        f"rdbEvidenceCount={result['rdbEvidenceCount']}",
        f"documentSourceCount={result['documentSourceCount']}",
        f"minDocumentSourceCount={result['minDocumentSourceCount']}",
        f"usedRdbEvidence={result['usedRdbEvidence']}",
        f"usedVectorSearch={result['usedVectorSearch']}",
        f"requireVectorSearch={result['requireVectorSearch']}",
        f"vectorSearchSkippedReason={result['vectorSearchSkippedReason']}",
        f"usedLlmGeneration={result['usedLlmGeneration']}",
        f"llmCacheHit={result['llmCacheHit']}",
        f"requireLlmGeneration={result['requireLlmGeneration']}",
        f"llmGenerationSkippedReason={result['llmGenerationSkippedReason']}",
        (
            "expectedLlmGenerationSkippedReason="
            f"{result['expectedLlmGenerationSkippedReason']}"
        ),
        f"sourceCount={result['sourceCount']}",
        f"urlCount={result['urlCount']}",
    ]
    return "\n".join(lines)


def format_json_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


def format_markdown_result(result: dict[str, Any]) -> str:
    lines = [
        "# 챗봇 답변 점검 결과",
        "",
        f"- 점검 상태: `{result['checkStatus']}`",
        f"- API URL: `{result['url']}`",
        f"- Intent: `{result['intent']}`",
        (
            "- 보안 결과: "
            f"`{result['securityStatus']}`"
            f"{_format_optional_code(result['securityCode'])}"
        ),
        (
            "- 근거 수: "
            f"전체 `{result['evidenceCount']}`, "
            f"RDB `{result['rdbEvidenceCount']}`, "
            f"Qdrant `{result['documentSourceCount']}`"
        ),
        (
            "- 생성 상태: "
            f"LLM `{result['usedLlmGeneration']}`, "
            f"LLM Cache `{result['llmCacheHit']}`, "
            f"Vector Search `{result['usedVectorSearch']}`"
        ),
    ]

    answer = result.get("answer")
    if answer:
        lines.extend(["", "## 답변", "", "```text", answer, "```"])

    source_details = result.get("sourceDetails") or []
    if source_details:
        lines.extend(["", "## 출처", "", "| 유형 | 제목 | URL |", "| --- | --- | --- |"])
        lines.extend(
            (
                f"| `{source['sourceOrigin'] or '-'}` / `{source['sourceType']}` "
                f"| {_escape_markdown_cell(source['title'])} "
                f"| `{source['url'] or '-'}` |"
            )
            for source in source_details
        )

    url_details = result.get("urlDetails") or []
    if url_details:
        lines.extend(
            ["", "## 화면 이동 URL", "", "| 유형 | 라벨 | URL |", "| --- | --- | --- |"]
        )
        lines.extend(
            (
                f"| `{url['type']}` "
                f"| {_escape_markdown_cell(url['label'])} "
                f"| `{url['url']}` |"
            )
            for url in url_details
        )

    return "\n".join(lines)


def _format_optional_code(code: str | None) -> str:
    if code is None:
        return ""
    return f" / `{code}`"


def _escape_markdown_cell(value: str | None) -> str:
    if value is None:
        return "-"
    return value.replace("|", "\\|").replace("\n", "<br>")


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
        request = build_request(args)
        token = resolve_answer_token(args, settings)
        path = resolve_answer_path(args, settings)
        result = asyncio.run(
            check_chat_answer(
                base_url=args.base_url,
                path=path,
                token=token,
                request=request,
                timeout_seconds=args.timeout_seconds,
                min_evidence_count=args.min_evidence_count,
                require_rdb_evidence=args.require_rdb_evidence,
                min_document_source_count=args.min_document_source_count,
                require_vector_search=args.require_vector_search,
                require_llm_generation=args.require_llm_generation,
                expected_llm_skipped_reason=args.expected_llm_skipped_reason,
                expected_security_status=args.expected_security_status,
                expected_security_code=args.expected_security_code,
                expected_intent=args.expected_intent,
            )
        )
    except ChatServiceError as exc:
        print(f"FastAPI 챗봇 답변 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        for next_action in build_answer_api_failure_actions(exc):
            print(f"nextAction={next_action}", file=error_output)
        return 1
    except Exception as exc:
        print(f"FastAPI 챗봇 답변 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    elif args.markdown:
        print(format_markdown_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
