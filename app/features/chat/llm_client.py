import httpx

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.grounded_prompt_builder import GroundedPrompt
from app.features.chat.schemas import ChatErrorCode

OPENAI_PROVIDER = "openai"
OPENAI_COMPATIBLE_PROVIDER = "openai_compatible"
OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_LOCAL_LLM_BASE_URL = "http://localhost:8001/v1"
SUPPORTED_LLM_PROVIDERS = (OPENAI_PROVIDER, OPENAI_COMPATIBLE_PROVIDER)
OPENAI_COST_GUARDRAIL_MAX_TOKENS = 1024
OPENAI_COST_GUARDRAIL_MAX_PROMPT_CHARS = 8_000
OPENAI_COST_GUARDRAIL_MIN_CACHE_TTL_SECONDS = 1.0
OPENAI_COST_GUARDRAIL_MIN_CACHE_ENTRIES = 1


def normalize_llm_provider(provider: str) -> str:
    return provider.strip().casefold().replace("-", "_")


def validate_llm_settings(settings: Settings) -> None:
    provider = normalize_llm_provider(settings.llm_provider)
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise ChatExternalServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_LLM_001,
            message=f"지원하지 않는 LLM provider입니다: {settings.llm_provider}",
        )

    missing_fields: list[str] = []
    if provider != OPENAI_PROVIDER and not settings.llm_base_url.strip():
        missing_fields.append("llm_base_url")
    if not settings.llm_model.strip():
        missing_fields.append("llm_model")
    if provider == OPENAI_PROVIDER and not settings.llm_api_key:
        missing_fields.append("llm_api_key")

    if not missing_fields:
        validate_openai_model_allowlist(settings)
        validate_openai_cost_guardrails(settings)
        return

    raise ChatExternalServiceError(
        status_code=503,
        code=ChatErrorCode.CHAT_LLM_001,
        message=f"LLM 필수 설정이 누락되었습니다: {', '.join(missing_fields)}",
    )


def validate_openai_model_allowlist(settings: Settings) -> None:
    if normalize_llm_provider(settings.llm_provider) != OPENAI_PROVIDER:
        return

    allowed_models = {
        model.strip()
        for model in settings.llm_allowed_models
        if model.strip()
    }
    if not allowed_models:
        raise ChatExternalServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_LLM_001,
            message=(
                "OpenAI 모델 allowlist가 설정되지 않았습니다: "
                "llm_allowed_models"
            ),
        )

    if settings.llm_model.strip() in allowed_models:
        return

    raise ChatExternalServiceError(
        status_code=503,
        code=ChatErrorCode.CHAT_LLM_001,
        message="OpenAI 허용 모델이 아닙니다: llm_model",
    )


def validate_openai_cost_guardrails(settings: Settings) -> None:
    if normalize_llm_provider(settings.llm_provider) != OPENAI_PROVIDER:
        return

    violations: list[str] = []
    if settings.llm_max_tokens > OPENAI_COST_GUARDRAIL_MAX_TOKENS:
        violations.append(
            f"llm_max_tokens<={OPENAI_COST_GUARDRAIL_MAX_TOKENS}"
        )
    if settings.prompt_max_total_chars > OPENAI_COST_GUARDRAIL_MAX_PROMPT_CHARS:
        violations.append(
            f"prompt_max_total_chars<={OPENAI_COST_GUARDRAIL_MAX_PROMPT_CHARS}"
        )
    if not settings.llm_response_cache_enabled:
        violations.append("llm_response_cache_enabled=true")
    if (
        settings.llm_response_cache_ttl_seconds
        < OPENAI_COST_GUARDRAIL_MIN_CACHE_TTL_SECONDS
    ):
        violations.append(
            "llm_response_cache_ttl_seconds>="
            f"{OPENAI_COST_GUARDRAIL_MIN_CACHE_TTL_SECONDS:g}"
        )
    if settings.llm_response_cache_max_entries < OPENAI_COST_GUARDRAIL_MIN_CACHE_ENTRIES:
        violations.append(
            "llm_response_cache_max_entries>="
            f"{OPENAI_COST_GUARDRAIL_MIN_CACHE_ENTRIES}"
        )

    if not violations:
        return

    raise ChatExternalServiceError(
        status_code=503,
        code=ChatErrorCode.CHAT_LLM_001,
        message=(
            "OpenAI 비용 가드레일을 만족하지 않습니다: "
            + ", ".join(violations)
        ),
    )


def resolve_llm_base_url(settings: Settings) -> str:
    provider = normalize_llm_provider(settings.llm_provider)
    base_url = settings.llm_base_url.strip()
    if provider == OPENAI_PROVIDER and (
        not base_url or base_url == _DEFAULT_LOCAL_LLM_BASE_URL
    ):
        return OPENAI_BASE_URL
    return base_url


class LlmClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.http_client = http_client

    async def generate(self, prompt: GroundedPrompt) -> str:
        validate_llm_settings(self.settings)
        payload = self._build_payload(prompt)
        try:
            if self.http_client is not None:
                response = await self.http_client.post(
                    self._chat_completions_url,
                    json=payload,
                    headers=self._headers,
                )
                return self._parse_response(response)

            async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
                response = await client.post(
                    self._chat_completions_url,
                    json=payload,
                    headers=self._headers,
                )
                return self._parse_response(response)
        except httpx.HTTPError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_LLM_003,
                message="LLM 서버 호출에 실패했습니다.",
            ) from exc

    def _build_payload(self, prompt: GroundedPrompt) -> dict:
        return {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
        }

    def _parse_response(self, response: httpx.Response) -> str:
        try:
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_LLM_003,
                message="LLM 서버 호출에 실패했습니다.",
            ) from exc
        except ValueError as exc:
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_LLM_002,
                message="LLM 응답 형식이 올바르지 않습니다.",
            ) from exc

        if not isinstance(body, dict):
            self._raise_invalid_response_shape()

        choices = body.get("choices", [])
        if not choices:
            return ""
        if not isinstance(choices, list):
            self._raise_invalid_response_shape()

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            self._raise_invalid_response_shape()

        message = first_choice.get("message", {})
        if not isinstance(message, dict):
            self._raise_invalid_response_shape()

        content = message.get("content", "")
        if content is None:
            return ""
        if not isinstance(content, str):
            self._raise_invalid_response_shape()
        return content.strip()

    def _raise_invalid_response_shape(self) -> None:
        raise ChatExternalServiceError(
            status_code=502,
            code=ChatErrorCode.CHAT_LLM_002,
            message="LLM 응답 형식이 올바르지 않습니다.",
        )

    @property
    def _chat_completions_url(self) -> str:
        return f"{resolve_llm_base_url(self.settings).rstrip('/')}/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        return headers
