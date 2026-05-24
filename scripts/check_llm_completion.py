import argparse
import asyncio
import json
import sys
from typing import Any, TextIO

import httpx

from app.core.config import Settings
from app.features.chat.answer_output_policy import AnswerOutputPolicy
from app.features.chat.exceptions import ChatServiceError
from app.features.chat.grounded_prompt_builder import GroundedPrompt
from app.features.chat.llm_client import (
    LlmClient,
    normalize_llm_provider,
    resolve_llm_base_url,
    validate_llm_settings,
)
from app.features.chat.schemas import ChatErrorCode

DEFAULT_SYSTEM_PROMPT = (
    "너는 사내 챗봇 LLM 연결 점검용 assistant다. "
    "제공된 요청에 한 문장으로만 답한다."
)
DEFAULT_USER_PROMPT = "LLM 연결 점검입니다. 짧게 정상 응답을 반환하세요."
OPENAI_NETWORK_CONFIRM_FLAG = "--allow-openai-network"
SMOKE_MAX_TOKENS = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenAI 또는 OpenAI-compatible LLM chat completions 연결을 점검합니다."
    )
    parser.add_argument(
        "--env-file",
        help="Settings를 로드할 env 파일 경로. 생략하면 기본 .env 설정을 사용합니다.",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="LLM 서버에 실제 chat completions 요청을 보냅니다.",
    )
    parser.add_argument(
        OPENAI_NETWORK_CONFIRM_FLAG,
        action="store_true",
        help=(
            "LLM_PROVIDER=openai 상태에서 실제 OpenAI API 호출을 허용합니다. "
            "Credit이 차감될 수 있으므로 수동 점검 때만 사용합니다."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print result as JSON")
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    if args.env_file:
        return Settings(_env_file=args.env_file)
    return Settings()


def build_smoke_prompt() -> GroundedPrompt:
    return GroundedPrompt(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_prompt=DEFAULT_USER_PROMPT,
    )


def validate_llm_smoke_settings(settings: Settings) -> None:
    if not settings.llm_enabled:
        raise ChatServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_LLM_001,
            message="LLM smoke check에는 LLM_ENABLED=true 설정이 필요합니다.",
        )
    validate_llm_settings(settings)


def validate_openai_network_allowed(settings: Settings, allow_openai_network: bool) -> None:
    if normalize_llm_provider(settings.llm_provider) != "openai":
        return
    if allow_openai_network:
        return

    raise ChatServiceError(
        status_code=400,
        code=ChatErrorCode.CHAT_LLM_001,
        message=(
            "OpenAI 네트워크 점검은 Credit이 차감될 수 있어 "
            f"{OPENAI_NETWORK_CONFIRM_FLAG} 옵션이 필요합니다."
        ),
    )


def build_smoke_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "llm_max_tokens": min(settings.llm_max_tokens, SMOKE_MAX_TOKENS),
        }
    )


def build_validate_only_result(settings: Settings) -> dict[str, Any]:
    validate_llm_smoke_settings(settings)
    return {
        "checkStatus": "VALIDATED",
        "mode": "VALIDATE_ONLY",
        "networkChecked": False,
        "llmEnabled": settings.llm_enabled,
        "provider": normalize_llm_provider(settings.llm_provider),
        "baseUrlConfigured": bool(resolve_llm_base_url(settings)),
        "modelConfigured": bool(settings.llm_model.strip()),
        "apiKeyConfigured": bool(settings.llm_api_key),
    }


def validate_llm_smoke_answer(
    answer: str,
    output_policy: AnswerOutputPolicy | None = None,
) -> None:
    security_result = (output_policy or AnswerOutputPolicy()).evaluate(
        answer,
        role="EXECUTIVE",
    )
    if security_result is None:
        return

    raise ChatServiceError(
        status_code=502,
        code=security_result.code or ChatErrorCode.CHAT_SECURITY_002,
        message=(
            "LLM smoke check 응답이 출력 보안 정책에 의해 차단되었습니다. "
            f"reason={security_result.reason}"
        ),
    )


async def check_llm_completion(
    settings: Settings,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    validate_llm_smoke_settings(settings)
    smoke_settings = build_smoke_settings(settings)
    answer = await LlmClient(smoke_settings, http_client=http_client).generate(
        build_smoke_prompt()
    )
    if not answer:
        raise ChatServiceError(
            status_code=502,
            code=ChatErrorCode.CHAT_LLM_004,
            message="LLM smoke check 응답이 비어 있습니다.",
        )
    validate_llm_smoke_answer(answer)

    return {
        "checkStatus": "PASS",
        "mode": "NETWORK",
        "networkChecked": True,
        "llmEnabled": settings.llm_enabled,
        "provider": normalize_llm_provider(settings.llm_provider),
        "baseUrlConfigured": True,
        "modelConfigured": True,
        "apiKeyConfigured": bool(settings.llm_api_key),
        "maxTokensUsed": smoke_settings.llm_max_tokens,
        "answerReceived": True,
        "answerLength": len(answer),
        "outputPolicyPassed": True,
    }


def format_text_result(result: dict[str, Any]) -> str:
    lines = [
        f"status={result['checkStatus']}",
        f"mode={result['mode']}",
        f"networkChecked={result['networkChecked']}",
        f"llmEnabled={result['llmEnabled']}",
        f"provider={result['provider']}",
        f"baseUrlConfigured={result['baseUrlConfigured']}",
        f"modelConfigured={result['modelConfigured']}",
        f"apiKeyConfigured={result['apiKeyConfigured']}",
    ]
    if result.get("networkChecked"):
        lines.append(f"maxTokensUsed={result['maxTokensUsed']}")
        lines.append(f"answerReceived={result['answerReceived']}")
        lines.append(f"answerLength={result['answerLength']}")
        lines.append(f"outputPolicyPassed={result['outputPolicyPassed']}")
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
        if args.network:
            validate_openai_network_allowed(
                settings,
                allow_openai_network=args.allow_openai_network,
            )
            result = asyncio.run(check_llm_completion(settings))
        else:
            result = build_validate_only_result(settings)
    except ChatServiceError as exc:
        print(f"LLM completion 점검 실패: {exc.message}", file=error_output)
        print(f"code={exc.code.value}", file=error_output)
        return 1
    except Exception as exc:
        print(f"LLM completion 점검 실패: {exc}", file=error_output)
        return 1

    if args.json:
        print(format_json_result(result), file=output)
    else:
        print(format_text_result(result), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
