import hashlib
import time
from dataclasses import dataclass

from app.features.chat.grounded_prompt_builder import GroundedPrompt


@dataclass(frozen=True)
class LlmResponseCacheEntry:
    answer: str
    expires_at: float


class LlmResponseCache:
    def __init__(
        self,
        *,
        enabled: bool = True,
        ttl_seconds: float = 60.0,
        max_entries: int = 128,
    ) -> None:
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: dict[str, LlmResponseCacheEntry] = {}

    def get(self, key: str) -> str | None:
        if not self._is_enabled():
            return None

        now = time.monotonic()
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            self._entries.pop(key, None)
            return None
        return entry.answer

    def put(self, key: str, answer: str) -> None:
        if not self._is_enabled() or not answer:
            return

        self._remove_expired_entries(time.monotonic())
        self._entries[key] = LlmResponseCacheEntry(
            answer=answer,
            expires_at=time.monotonic() + self.ttl_seconds,
        )
        self._trim_entries()

    def _is_enabled(self) -> bool:
        return self.enabled and self.ttl_seconds > 0 and self.max_entries > 0

    def _remove_expired_entries(self, now: float) -> None:
        expired_keys = [
            key for key, entry in self._entries.items() if entry.expires_at <= now
        ]
        for key in expired_keys:
            self._entries.pop(key, None)

    def _trim_entries(self) -> None:
        while len(self._entries) > self.max_entries:
            oldest_key = min(
                self._entries,
                key=lambda key: self._entries[key].expires_at,
            )
            self._entries.pop(oldest_key, None)


def build_llm_response_cache_key(
    prompt: GroundedPrompt,
    *,
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    raw_key = "\n".join(
        [
            provider.strip().casefold(),
            model.strip(),
            str(max_tokens),
            str(temperature),
            prompt.system_prompt,
            prompt.user_prompt,
        ]
    )
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
