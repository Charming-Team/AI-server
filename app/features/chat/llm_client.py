import httpx

from app.core.config import Settings
from app.features.chat.grounded_prompt_builder import GroundedPrompt


class LlmClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.http_client = http_client

    async def generate(self, prompt: GroundedPrompt) -> str:
        payload = self._build_payload(prompt)
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
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if not isinstance(content, str):
            return ""
        return content.strip()

    @property
    def _chat_completions_url(self) -> str:
        return f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        return headers
