from typing import Any

from app.features.chat.grounding_security_policy import GroundingSecurityPolicy
from app.features.chat.role_access_policy import RoleAccessPolicy
from app.features.chat.schemas import ChatUserContext, EvidenceItem, EvidenceResult


class EvidenceAccessPolicy:
    _operator_financial_key_terms = (
        "contractamount",
        "latepenaltyamount",
        "penalty",
        "revenue",
        "profit",
        "cost",
        "price",
        "margin",
        "sales",
        "financial",
        "payment",
        "invoice",
        "계약금액",
        "계약",
        "패널티",
        "지체상금",
        "매출",
        "수익",
        "손익",
        "비용",
        "원가",
        "금액",
    )
    _redacted_value = "[권한 제한]"

    def __init__(
        self,
        grounding_security_policy: GroundingSecurityPolicy | None = None,
    ) -> None:
        self.grounding_security_policy = (
            grounding_security_policy or GroundingSecurityPolicy()
        )

    def sanitize(
        self,
        role: str | ChatUserContext,
        evidence_result: EvidenceResult,
    ) -> EvidenceResult:
        role_name = self._role_name(role)

        return evidence_result.model_copy(
            update={
                "items": [
                    sanitized_item
                    for item in evidence_result.items
                    if (
                        sanitized_item := self._sanitize_item(
                            item,
                            role_name,
                        )
                    )
                    is not None
                ]
            }
        )

    def _sanitize_item(
        self,
        item: EvidenceItem,
        role: str,
    ) -> EvidenceItem | None:
        if not self._is_role_allowed(item, role):
            return None

        if not self.grounding_security_policy.allows_evidence_item(item):
            return None

        if role != "OPERATOR":
            return item

        if self._contains_restricted_term(item.title) or self._contains_restricted_term(
            item.summary
        ):
            return None
        return item.model_copy(update={"data": self._sanitize_data(item.data)})

    def _is_role_allowed(self, item: EvidenceItem, role: str) -> bool:
        return not item.allowed_roles or role in item.allowed_roles

    def _sanitize_data(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._sanitize_data(item)
                for key, item in value.items()
                if not self._is_restricted_key(key)
            }
        if isinstance(value, list):
            return [self._sanitize_data(item) for item in value]
        if isinstance(value, str) and self._contains_restricted_term(value):
            return self._redacted_value
        return value

    def _is_restricted_key(self, key: object) -> bool:
        if not isinstance(key, str):
            return False

        normalized_key = self._compact(self._normalize(key))
        return any(term in normalized_key for term in self._operator_financial_key_terms)

    def _contains_restricted_term(self, value: str) -> bool:
        normalized_value = self._normalize(value)
        compact_value = self._compact(normalized_value)
        return any(
            self._contains_term(term, normalized_value, compact_value)
            for term in RoleAccessPolicy.operator_restricted_terms
        )

    def _contains_term(
        self,
        term: str,
        normalized_value: str,
        compact_value: str,
    ) -> bool:
        normalized_term = self._normalize(term)
        compact_term = self._compact(normalized_term)
        return normalized_term in normalized_value or compact_term in compact_value

    def _normalize(self, value: str) -> str:
        return value.casefold()

    def _compact(self, value: str) -> str:
        return "".join(value.split()).replace("_", "").replace("-", "")

    def _role_name(self, role: str | ChatUserContext) -> str:
        if isinstance(role, ChatUserContext):
            return role.role.strip().upper()
        return role.strip().upper()
