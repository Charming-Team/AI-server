from typing import Any

from app.features.chat.schemas import EvidenceItem
from app.features.chat.security_policy import SecurityPolicy


class GroundingSecurityPolicy:
    def __init__(self, security_policy: SecurityPolicy | None = None) -> None:
        self.security_policy = security_policy or SecurityPolicy()

    def allows_evidence_item(self, item: EvidenceItem) -> bool:
        return not self._contains_blocked_content(
            item.title,
            item.summary,
            item.data,
        )

    def allows_qdrant_point(self, point: dict) -> bool:
        payload = point.get("payload")
        if not isinstance(payload, dict):
            return False

        return not self._contains_blocked_content(
            payload.get("title"),
            payload.get("summary"),
            payload.get("chunkText"),
        )

    def _contains_blocked_content(self, *values: Any) -> bool:
        return any(self._contains_blocked_value(value) for value in values)

    def _contains_blocked_value(self, value: Any) -> bool:
        if isinstance(value, str):
            return self.security_policy.evaluate(value) is not None
        if isinstance(value, dict):
            return any(
                self._contains_blocked_value(key)
                or self._contains_blocked_value(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(self._contains_blocked_value(item) for item in value)
        return False
