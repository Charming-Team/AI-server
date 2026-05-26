# 기본 markdown과 sections를 받아 문장 개선
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.features.report.prompts.report_writing_prompt import (
    REPORT_WRITING_SYSTEM_PROMPT,
    build_report_writing_user_prompt,
)

load_dotenv()


class LlmReportWritingAgent:
    def __init__(self) -> None:
        self.enabled = os.getenv("REPORT_LLM_ENABLED", "false").lower() == "true"
        self.model = os.getenv("REPORT_LLM_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")

        self.client: OpenAI | None = None

        if self.enabled and self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def run(
        self,
        *,
        title: str,
        period_text: str,
        sections: dict[str, Any],
        base_markdown: str,
    ) -> str:
        print(
            "[LlmReportWritingAgent] "
            f"enabled={self.enabled}, "
            f"model={self.model}, "
            f"has_api_key={bool(self.api_key)}"
        )

        if not self.enabled:
            print("[LlmReportWritingAgent] LLM disabled. fallback to base_markdown")
            return base_markdown

        if self.client is None:
            print("[LlmReportWritingAgent] OpenAI client is None. fallback to base_markdown")
            return base_markdown

        user_prompt = build_report_writing_user_prompt(
            title=title,
            period_text=period_text,
            sections=sections,
            base_markdown=base_markdown,
        )

        try:
            print("[LlmReportWritingAgent] Calling OpenAI API...")

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

            print("[LlmReportWritingAgent] OpenAI API call success")

            content = response.choices[0].message.content

            if not content or not content.strip():
                print("[LlmReportWritingAgent] Empty response. fallback to base_markdown")
                return base_markdown

            return content.strip()

        except Exception as error:
            print(f"[LlmReportWritingAgent] OpenAI API call failed: {error}")
            return base_markdown