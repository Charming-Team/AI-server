import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.features.report.prompts.report_writing_prompt import (
    REPORT_WRITING_SYSTEM_PROMPT,
    build_report_writing_user_prompt,
)

load_dotenv()
logger = logging.getLogger(__name__)
REPORT_LLM_TIMEOUT_SECONDS = 60.0


def resolve_report_llm_timeout_seconds() -> float:
    """Resolve the report LLM timeout with a safe default fallback."""
    raw_timeout = os.getenv("REPORT_LLM_TIMEOUT_SECONDS", str(REPORT_LLM_TIMEOUT_SECONDS))
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError:
        return REPORT_LLM_TIMEOUT_SECONDS
    return timeout_seconds if timeout_seconds > 0 else REPORT_LLM_TIMEOUT_SECONDS


class LlmReportWritingAgent:
    """Improve report markdown with an LLM while keeping the base report as fallback."""

    def __init__(self) -> None:
        self.enabled = os.getenv("REPORT_LLM_ENABLED", "false").lower() == "true"
        self.model = os.getenv("REPORT_LLM_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.timeout_seconds = resolve_report_llm_timeout_seconds()

        self.client: OpenAI | None = None

        if self.enabled and self.api_key:
            self.client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)

    def run(
        self,
        *,
        title: str,
        period_text: str,
        sections: dict[str, Any],
        base_markdown: str,
    ) -> str:
        """Generate polished markdown with timeout-bound LLM fallback behavior.

        The full prompt is never logged. Only coarse routing and fallback status are
        emitted so operational logs remain safe.
        """
        logger.info(
            "report_llm.status enabled=%s model_configured=%s has_api_key=%s",
            self.enabled,
            bool(self.model),
            bool(self.api_key),
        )

        if not self.enabled:
            logger.info("report_llm.fallback reason=disabled")
            return base_markdown

        if self.client is None:
            logger.warning("report_llm.fallback reason=client_unavailable")
            return base_markdown

        user_prompt = build_report_writing_user_prompt(
            title=title,
            period_text=period_text,
            sections=sections,
            base_markdown=base_markdown,
        )

        try:
            logger.info("report_llm.call.started")

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": REPORT_WRITING_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            logger.info("report_llm.call.completed")

            content = response.choices[0].message.content

            if not content or not content.strip():
                logger.warning("report_llm.fallback reason=empty_response")
                return base_markdown

            return content.strip()

        except Exception as error:
            logger.warning("report_llm.fallback reason=llm_error error=%s", type(error).__name__)
            return base_markdown
