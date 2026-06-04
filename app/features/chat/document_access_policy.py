from typing import Any

from app.features.chat.access_control import OPERATOR_RESTRICTED_TERMS, OPERATOR_ROLE
from app.features.chat.document_payload import InternalDocumentInput
from app.features.chat.schemas import ChatSource, DocumentSearchResult
from app.features.chat.skip_reasons import QDRANT_OPERATOR_RESTRICTED_CONTENT


class DocumentAccessPolicy:
    _operator_role = OPERATOR_ROLE

    def allows_document(self, document: InternalDocumentInput) -> bool:
        if not self._includes_operator(document.allowed_roles):
            return True

        return not self._contains_restricted_content(
            document.title,
            document.summary,
            document.content,
        )

    def allows_point(self, point: dict, role: str) -> bool:
        if role.strip().upper() != self._operator_role:
            return True

        payload = point.get("payload")
        if not isinstance(payload, dict):
            return False

        return not self._contains_restricted_content(
            payload.get("title"),
            payload.get("summary"),
            payload.get("chunkText"),
        )

    def allows_source(self, source: ChatSource, role: str) -> bool:
        if role.strip().upper() != self._operator_role:
            return True

        return not self._contains_restricted_content(
            source.title,
            source.summary,
        )

    def sanitize_search_result(
        self,
        document_result: DocumentSearchResult,
        role: str,
    ) -> DocumentSearchResult:
        filtered_sources = [
            source
            for source in document_result.sources
            if self.allows_source(source, role)
        ]
        if len(filtered_sources) == len(document_result.sources):
            return document_result

        skipped_reason = document_result.skipped_reason
        if not filtered_sources and document_result.sources and skipped_reason is None:
            skipped_reason = QDRANT_OPERATOR_RESTRICTED_CONTENT

        return document_result.model_copy(
            update={
                "sources": filtered_sources,
                "skipped_reason": skipped_reason,
            }
        )

    def _contains_restricted_content(self, *values: Any) -> bool:
        return any(
            isinstance(value, str) and self._contains_restricted_term(value)
            for value in values
        )

    def _includes_operator(self, allowed_roles: list[str]) -> bool:
        return any(role.strip().upper() == self._operator_role for role in allowed_roles)

    def _contains_restricted_term(self, value: str) -> bool:
        normalized_value = self._normalize(value)
        compact_value = self._compact(normalized_value)
        return any(
            self._contains_term(term, normalized_value, compact_value)
            for term in OPERATOR_RESTRICTED_TERMS
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
