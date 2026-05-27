import os

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
        self.client: OpenAI | None = None

        if self.enabled and self.api_key:
            self.client = OpenAI(api_key=self.api_key)

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
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": BUSINESS_REPORT_WRITING_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format={"type": "json_object"},
            )
        except Exception as error:
            raise RuntimeError(f"Business report LLM 호출에 실패했습니다: {error}") from error

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("Business report LLM 응답이 비어 있습니다.")

        return content.strip()
