import argparse
import asyncio
import json
import sys
from typing import Any, TextIO

import httpx

from app.core.config import Settings
from app.features.chat.access_control import (
    OPERATOR_RESTRICTED_TERMS,
    OPERATOR_ROLE,
    ROLE_INTENT_MATRIX,
)
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.schemas import (
    ChatErrorCode,
    ChatRecommendationRequest,
    ChatRecommendationResponse,
    ChatUserContext,
    ErrorResponse,
)
from scripts.chat_api_failure_actions import build_recommendation_api_failure_actions

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_ROLE = "MANUFACTURING_MANAGER"
DEFAULT_KEYWORD = "라인"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FastAPI 추천 질문 API와 Role 기반 추천 계약을 점검합니다."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="FastAPI base URL")
    parser.add_argument(
        "--path",
        help="Recommendation path. 생략하면 Settings.api_v1_prefix 기준으로 생성합니다.",
    )
    parser.add_argument("--token", help="FastAPI chat recommendation internal token")
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. CLI 인자가 있으면 해당 값이 우선합니다.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--keyword", default=DEFAULT_KEYWORD, help="추천 질문 검색 키워드")
    parser.add_argument("--role", default=DEFAULT_ROLE, help="사용자 Role")
    parser.add_argument("--user-id", type=int, default=1, help="사용자 ID")
    parser.add_argument("--company-name", default="S-MAP", help="회사명 메타데이터")
    parser.add_argument("--status", default="ACTIVE", help="사용자 상태")
    parser.add_argument(
        "--min-item-count",
        type=int,
        default=1,
        help="요구하는 최소 추천 질문 개수",
    )
    parser.add_argument(
        "--expect-fallback",
        action="store_true",
        help="fallback 추천 목록이 사용되어야 한다고 검증합니다.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON",
    )
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    if args.env_file:
        return Settings(_env_file=args.env_file)
    return Settings()


def build_recommendation_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def resolve_recommendation_path(args: argparse.Namespace, settings: Settings) -> str:
    if args.path:
        return args.path
    return f"{settings.api_v1_prefix}/chat/recommendations"


def resolve_recommendation_token(args: argparse.Namespace, settings: Settings) -> str:
    token = args.token or settings.chat_recommendation_internal_token
    if not token:
        raise ChatServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_SECURITY_003,
            message="FastAPI chat recommendation internal token이 설정되지 않았습니다.",
        )
    return token


def build_request(args: argparse.Namespace) -> ChatRecommendationRequest:
    return ChatRecommendationRequest(
        user=ChatUserContext(
            userId=args.user_id,
            role=args.role,
            companyName=args.company_name,
            status=args.status,
        ),
        keyword=args.keyword,
    )


async def check_chat_recommendations(
    base_url: str,
    path: str,
    token: str,
    request: ChatRecommendationRequest,
    timeout_seconds: float,
    min_item_count: int = 1,
    expect_fallback: bool = False,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    url = build_recommendation_url(base_url, path)
    response = await _post_chat_recommendations(
        url=url,
        token=token,
        request=request,
        timeout_seconds=timeout_seconds,
        http_client=http_client,
    )
    recommendations = ChatRecommendationResponse.model_validate(response.json())
    _validate_recommendations(
        recommendations,
        role=request.user.role,
        min_item_count=min_item_count,
        expect_fallback=expect_fallback,
    )

    return {
        "checkStatus": "PASS",
        "url": url,
        "role": request.user.role,
        "keywordConfigured": bool(request.keyword),
        "itemCount": len(recommendations.items),
        "minItemCount": min_item_count,
        "fallbackUsed": recommendations.fallback_used,
        "expectFallback": expect_fallback,
        "questionIds": [item.question_id for item in recommendations.items],
        "intents": [item.intent.value for item in recommendations.items],
        "urlCount": len([item for item in recommendations.items if item.url]),
        "networkChecked": True,
    }


async def _post_chat_recommendations(
    url: str,
    token: str,
    request: ChatRecommendationRequest,
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
            message=f"FastAPI 추천 질문 API 호출에 실패했습니다. {exc}",
        ) from exc

    if response.is_error:
        _raise_response_error(response)

    return response


def _validate_recommendations(
    recommendations: ChatRecommendationResponse,
    role: str,
    min_item_count: int,
    expect_fallback: bool,
) -> None:
    item_count = len(recommendations.items)
    if item_count < min_item_count:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_RECOMMEND_001,
            message=(
                "FastAPI 추천 질문 개수가 기준보다 적습니다. "
                f"expected>={min_item_count}, actual={item_count}"
            ),
        )

    if expect_fallback and not recommendations.fallback_used:
        raise ChatServiceError(
            status_code=500,
            code=ChatErrorCode.CHAT_RECOMMEND_001,
            message="FastAPI 추천 질문 응답에 fallback이 사용되지 않았습니다.",
        )

    normalized_role = role.strip().upper()
    allowed_intents = ROLE_INTENT_MATRIX.get(normalized_role, frozenset())
    for item in recommendations.items:
        if item.intent not in allowed_intents:
            raise ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_RECOMMEND_002,
                message=(
                    "FastAPI 추천 질문에 Role 권한 밖 intent가 포함되어 있습니다. "
                    f"role={normalized_role}, intent={item.intent.value}"
                ),
            )

        if normalized_role == OPERATOR_ROLE and "mode=read" not in item.url.casefold():
            raise ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_RECOMMEND_002,
                message="OPERATOR 추천 질문 URL은 read-only mode를 포함해야 합니다.",
            )

        if normalized_role == OPERATOR_ROLE and _has_operator_restricted_content(item):
            raise ChatServiceError(
                status_code=500,
                code=ChatErrorCode.CHAT_RECOMMEND_002,
                message="OPERATOR 추천 질문에 금액성 내용이 포함되어 있습니다.",
            )


def _has_operator_restricted_content(item: Any) -> bool:
    target = _compact(
        " ".join(
            (
                item.question,
                item.category,
                item.intent.value,
                item.url,
            )
        )
    )
    return any(_compact(term) in target for term in OPERATOR_RESTRICTED_TERMS)


def _compact(value: str) -> str:
    return "".join(value.casefold().split()).replace("_", "").replace("-", "")


def _raise_response_error(response: httpx.Response) -> None:
    code = ChatErrorCode.CHAT_RECOMMEND_001
    message = (
        "FastAPI 추천 질문 API가 실패했습니다. "
        f"status={response.status_code}, body={response.text[:300]}"
    )
    try:
        error = ErrorResponse.model_validate(response.json())
    except ValueError:
        error = None

    if error is not None:
        code = _to_chat_error_code(error.code)
        message = f"FastAPI 추천 질문 API가 실패했습니다. {error.message}"

    raise ChatServiceError(
        status_code=response.status_code,
        code=code,
        message=message,
    )


def _to_chat_error_code(code: ChatErrorCode | str) -> ChatErrorCode:
    if isinstance(code, ChatErrorCode):
        return code
    try:
        return ChatErrorCode(code)
    except ValueError:
        return ChatErrorCode.CHAT_RECOMMEND_001


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"url={result['url']}",
        f"role={result['role']}",
        f"keywordConfigured={result['keywordConfigured']}",
        f"itemCount={result['itemCount']}",
        f"minItemCount={result['minItemCount']}",
        f"fallbackUsed={result['fallbackUsed']}",
        f"expectFallback={result['expectFallback']}",
        f"questionIds={','.join(result['questionIds'])}",
        f"intents={','.join(result['intents'])}",
        f"urlCount={result['urlCount']}",
        f"networkChecked={result['networkChecked']}",
    ]
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
        request = build_request(args)
        token = resolve_recommendation_token(args, settings)
        path = resolve_recommendation_path(args, settings)
        result = asyncio.run(
            check_chat_recommendations(
                base_url=args.base_url,
                path=path,
                token=token,
                request=request,
                timeout_seconds=args.timeout_seconds,
                min_item_count=args.min_item_count,
                expect_fallback=args.expect_fallback,
            )
        )
    except ChatServiceError as exc:
        print(f"FastAPI 추천 질문 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        for next_action in build_recommendation_api_failure_actions(exc):
            print(f"nextAction={next_action}", file=error_output)
        return 1
    except Exception as exc:
        print(f"FastAPI 추천 질문 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
