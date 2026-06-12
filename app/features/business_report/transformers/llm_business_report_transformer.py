import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.features.business_report.prompts.business_report_writing_prompt import (
    BUSINESS_REPORT_WRITING_SYSTEM_PROMPT,
    build_business_report_writing_user_prompt,
)
from app.features.business_report.schemas.source import BusinessReportSource

load_dotenv()


class LlmBusinessReportTransformer:
    def __init__(self) -> None:
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

            response = self.client.chat.completions.create(**request_options)
        except Exception as error:
            raise RuntimeError(f"Business report LLM 호출에 실패했습니다: {error}") from error

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("Business report LLM 응답이 비어 있습니다.")

        return content.strip()

    def _supports_custom_temperature(self) -> bool:
        normalized_model = self.model.strip().lower()
        return not normalized_model.startswith("gpt-5")

    def _resolve_timeout_seconds(self) -> float:
        raw_timeout = (
            os.getenv("BUSINESS_REPORT_LLM_TIMEOUT_SECONDS")
            or os.getenv("LLM_TIMEOUT_SECONDS")
            or "60"
        )
        try:
            timeout_seconds = float(raw_timeout)
        except ValueError:
            return 60.0

        return timeout_seconds if timeout_seconds > 0 else 60.0
