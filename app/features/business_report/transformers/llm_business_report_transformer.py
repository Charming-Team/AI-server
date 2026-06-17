import os
from typing import Any

from dotenv import load_dotenv
from openai import APITimeoutError, OpenAI

from app.core.config import Settings, get_settings
from app.core.langsmith_tracing import configure_langsmith_tracing_from_settings
from app.features.business_report.prompts.business_report_writing_prompt import (
    BUSINESS_REPORT_WRITING_SYSTEM_PROMPT,
    build_business_report_writing_user_prompt,
)
from app.features.business_report.schemas.source import BusinessReportSource

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - langgraph installs langsmith in normal runtime.

    def traceable(*args, **kwargs):
        def decorator(func):
            return func

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

load_dotenv()


def _process_business_report_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    source = inputs.get("source")
    settings = get_settings()
    safe_inputs: dict[str, Any] = {
        "report_id": getattr(source, "report_id", None),
        "report_type": getattr(source, "report_type", None),
        "report_title": getattr(source, "report_title", None),
        "target_start_date": str(getattr(source, "target_start_date", "")),
        "target_end_date": str(getattr(source, "target_end_date", "")),
    }
    if settings.langsmith_trace_payloads:
        safe_inputs["report_content"] = getattr(source, "report_content", None)
        safe_inputs["report_evidence"] = getattr(source, "report_evidence", None)
    return safe_inputs


def _process_business_report_trace_outputs(output: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.langsmith_trace_payloads:
        return {"answer_length": len(output or "")}
    return {"answer": output}


class LlmBusinessReportTransformer:
    def __init__(self) -> None:
        self.settings: Settings = get_settings()
        configure_langsmith_tracing_from_settings(self.settings)
        self.enabled = os.getenv("BUSINESS_REPORT_LLM_ENABLED", "").lower() == "true"
        self.model = os.getenv("LLM_MODEL", "")
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.timeout_seconds = self._resolve_timeout_seconds()
        self.client: OpenAI | None = None

        if self.enabled and self.api_key:
            self.client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)

    def run(
        self,
        source: BusinessReportSource,
    ) -> str:
        if not self.enabled:
            raise RuntimeError("BUSINESS_REPORT_LLM_ENABLED 설정이 비활성화되어 있습니다.")

        if not self.model:
            raise RuntimeError("LLM_MODEL 설정이 누락되었습니다.")

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY 또는 LLM_API_KEY 설정이 누락되었습니다.")

        if self.client is None:
            raise RuntimeError("Business report OpenAI client를 초기화할 수 없습니다.")

        user_prompt = build_business_report_writing_user_prompt(source)
        return self._generate_business_report_content(source, user_prompt)

    @traceable(
        name="business_report.llm_generation",
        run_type="llm",
        process_inputs=_process_business_report_trace_inputs,
        process_outputs=_process_business_report_trace_outputs,
    )
    def _generate_business_report_content(
        self,
        source: BusinessReportSource,
        user_prompt: str,
    ) -> str:
        del source

        try:
            request_options: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": BUSINESS_REPORT_WRITING_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "response_format": {"type": "json_object"},
            }
            if self._supports_custom_temperature():
                request_options["temperature"] = 0.1

            response = self._create_completion_with_timeout_retry(request_options)
        except Exception as error:
            raise RuntimeError(f"Business report LLM 호출에 실패했습니다: {error}") from error

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("Business report LLM 응답이 비어 있습니다.")

        return content.strip()

    def _create_completion_with_timeout_retry(self, request_options: dict[str, Any]):
        max_attempts = 2
        last_timeout: APITimeoutError | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return self.client.chat.completions.create(**request_options)
            except APITimeoutError as error:
                last_timeout = error
                if attempt >= max_attempts:
                    break

        raise RuntimeError(
            f"Business report LLM 응답 시간이 초과되었습니다. "
            f"timeoutSeconds={self.timeout_seconds} attempts={max_attempts}"
        ) from last_timeout

    def _supports_custom_temperature(self) -> bool:
        normalized_model = self.model.strip().lower()
        return not normalized_model.startswith("gpt-5")

    def _resolve_timeout_seconds(self) -> float:
        raw_timeout = (
            os.getenv("BUSINESS_REPORT_LLM_TIMEOUT_SECONDS")
            or os.getenv("LLM_TIMEOUT_SECONDS")
            or "180"
        )
        try:
            timeout_seconds = float(raw_timeout)
        except ValueError:
            return 180.0

        return timeout_seconds if timeout_seconds > 0 else 180.0
