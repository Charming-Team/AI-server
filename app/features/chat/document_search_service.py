from pydantic import ValidationError

from app.core.config import Settings
from app.features.chat.access_control import (
    BUSINESS_ROLES,
    QDRANT_DOCUMENT_TYPES,
)
from app.features.chat.document_access_policy import DocumentAccessPolicy
from app.features.chat.document_payload import QdrantSearchPoint
from app.features.chat.embedding_service import EmbeddingService
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.grounding_security_policy import GroundingSecurityPolicy
from app.features.chat.navigation_target_policy import (
    has_invalid_navigation_url,
    has_navigation_target,
)
from app.features.chat.qdrant_client import QdrantDocumentSearchClient
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatErrorCode,
    ChatIntent,
    ChatSource,
    DocumentSearchResult,
)
from app.features.chat.skip_reasons import (
    QDRANT_DOCUMENT_TYPE_NOT_ALLOWED,
    QDRANT_GROUNDING_SECURITY_BLOCKED,
    QDRANT_INTENT_NOT_ALLOWED,
    QDRANT_NAVIGATION_TARGET_MISSING,
    QDRANT_NO_RESULTS,
    QDRANT_OPERATOR_RESTRICTED_CONTENT,
    QDRANT_ROLE_NOT_ALLOWED,
    QDRANT_SCORE_THRESHOLD_NOT_MET,
    QDRANT_UNKNOWN_INTENT,
)


class DocumentSearchService:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService | None = None,
        qdrant_client: QdrantDocumentSearchClient | None = None,
        document_access_policy: DocumentAccessPolicy | None = None,
        grounding_security_policy: GroundingSecurityPolicy | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service or EmbeddingService(settings)
        self.qdrant_client = qdrant_client or QdrantDocumentSearchClient(settings)
        self.document_access_policy = document_access_policy or DocumentAccessPolicy()
        self.grounding_security_policy = (
            grounding_security_policy or GroundingSecurityPolicy()
        )

    async def search(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> DocumentSearchResult:
        if not self.settings.qdrant_search_enabled:
            return DocumentSearchResult(was_searched=False)

        if intent == ChatIntent.UNKNOWN:
            return DocumentSearchResult(
                was_searched=False,
                skipped_reason=QDRANT_UNKNOWN_INTENT,
            )

        embedding_result = await self.embedding_service.embed_query(request)
        if not embedding_result.was_embedded:
            return DocumentSearchResult(
                was_searched=False,
                skipped_reason=embedding_result.skipped_reason,
            )

        search_payload = self._build_search_payload(embedding_result.vector, request, intent)
        points = await self.qdrant_client.search(search_payload)
        role_filtered_points = self._filter_points_by_role(points, request.user.role)
        intent_filtered_points = self._filter_points_by_intent(
            role_filtered_points,
            intent,
        )
        document_type_filtered_points = self._filter_points_by_document_type(
            intent_filtered_points
        )
        access_filtered_points = self._filter_points_by_content_access(
            document_type_filtered_points,
            request.user.role,
        )
        security_filtered_points = self._filter_points_by_grounding_security(
            access_filtered_points
        )
        score_filtered_points = self._filter_points_by_score(security_filtered_points)
        navigation_filtered_points = self._filter_points_by_navigation_target(
            score_filtered_points
        )
        if not navigation_filtered_points:
            return DocumentSearchResult(
                was_searched=True,
                skipped_reason=self._build_no_result_reason(
                    points,
                    role_filtered_points,
                    intent_filtered_points,
                    document_type_filtered_points,
                    access_filtered_points,
                    security_filtered_points,
                    score_filtered_points,
                    navigation_filtered_points,
                ),
            )

        return DocumentSearchResult(
            was_searched=True,
            sources=self._build_sources(navigation_filtered_points),
        )

    def _build_search_payload(
        self,
        vector: list[float],
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> dict:
        payload = {
            "vector": vector,
            "limit": self.settings.qdrant_top_k,
            "with_payload": True,
            "filter": self._build_search_filter(request, intent),
        }
        if self.settings.qdrant_score_threshold > 0:
            payload["score_threshold"] = self.settings.qdrant_score_threshold
        return payload

    def _build_search_filter(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> dict:
        must_conditions: list[dict] = [
            {
                "key": "allowedRoles",
                "match": {"any": self._allowed_roles_for_search(request.user.role)},
            },
        ]
        if intent != ChatIntent.UNKNOWN:
            must_conditions.append(
                {"key": "intentTags", "match": {"any": [intent.value]}}
            )
        return {"must": must_conditions}

    def _allowed_roles_for_search(self, role: str) -> list[str]:
        normalized_role = role.strip().upper()
        if normalized_role in BUSINESS_ROLES:
            return sorted(BUSINESS_ROLES)
        return [normalized_role]

    def _filter_points_by_role(self, points: list[dict], role: str) -> list[dict]:
        return [
            point
            for point in points
            if self._is_role_allowed(point, role)
        ]

    def _is_role_allowed(self, point: dict, role: str) -> bool:
        payload = point.get("payload")
        if not isinstance(payload, dict):
            return False

        normalized_role = role.strip().upper()
        if normalized_role in BUSINESS_ROLES:
            return True

        allowed_roles = payload.get("allowedRoles")
        if not isinstance(allowed_roles, list):
            return False

        return any(
            isinstance(allowed_role, str)
            and allowed_role.strip().upper() == normalized_role
            for allowed_role in allowed_roles
        )

    def _filter_points_by_intent(
        self,
        points: list[dict],
        intent: ChatIntent,
    ) -> list[dict]:
        return [
            point
            for point in points
            if self._is_intent_allowed(point, intent)
        ]

    def _is_intent_allowed(self, point: dict, intent: ChatIntent) -> bool:
        payload = point.get("payload")
        if not isinstance(payload, dict):
            return False

        intent_tags = payload.get("intentTags")
        if not isinstance(intent_tags, list):
            return False

        normalized_intent = intent.value.strip().upper()
        return any(
            isinstance(intent_tag, str)
            and intent_tag.strip().upper() == normalized_intent
            for intent_tag in intent_tags
        )

    def _filter_points_by_score(self, points: list[dict]) -> list[dict]:
        threshold = self.settings.qdrant_score_threshold
        if threshold <= 0:
            return points

        return [
            point
            for point in points
            if self._is_score_above_threshold(point, threshold)
        ]

    def _filter_points_by_navigation_target(self, points: list[dict]) -> list[dict]:
        return [
            point
            for point in points
            if self._has_navigation_target(point)
        ]

    def _has_navigation_target(self, point: dict) -> bool:
        payload = point.get("payload")
        if not isinstance(payload, dict):
            return False

        if has_invalid_navigation_url(payload.get("url")):
            return False

        return has_navigation_target(
            payload.get("url"),
            payload.get("referenceType"),
            payload.get("referenceId"),
        )

    def _filter_points_by_document_type(self, points: list[dict]) -> list[dict]:
        return [
            point
            for point in points
            if self._is_document_type_allowed(point)
        ]

    def _is_document_type_allowed(self, point: dict) -> bool:
        payload = point.get("payload")
        if not isinstance(payload, dict):
            return False

        document_type = payload.get("documentType")
        if not isinstance(document_type, str):
            return False

        return document_type.strip().upper() in QDRANT_DOCUMENT_TYPES

    def _filter_points_by_content_access(self, points: list[dict], role: str) -> list[dict]:
        return [
            point
            for point in points
            if self.document_access_policy.allows_point(point, role)
        ]

    def _filter_points_by_grounding_security(self, points: list[dict]) -> list[dict]:
        return [
            point
            for point in points
            if self.grounding_security_policy.allows_qdrant_point(point)
        ]

    def _is_score_above_threshold(self, point: dict, threshold: float) -> bool:
        score = point.get("score")
        return isinstance(score, (int, float)) and score >= threshold

    def _build_no_result_reason(
        self,
        points: list[dict],
        role_filtered_points: list[dict],
        intent_filtered_points: list[dict],
        document_type_filtered_points: list[dict],
        access_filtered_points: list[dict],
        security_filtered_points: list[dict],
        score_filtered_points: list[dict],
        navigation_filtered_points: list[dict],
    ) -> str:
        if points and not role_filtered_points:
            return QDRANT_ROLE_NOT_ALLOWED
        if role_filtered_points and not intent_filtered_points:
            return QDRANT_INTENT_NOT_ALLOWED
        if intent_filtered_points and not document_type_filtered_points:
            return QDRANT_DOCUMENT_TYPE_NOT_ALLOWED
        if intent_filtered_points and not access_filtered_points:
            return QDRANT_OPERATOR_RESTRICTED_CONTENT
        if access_filtered_points and not security_filtered_points:
            return QDRANT_GROUNDING_SECURITY_BLOCKED
        if (
            security_filtered_points
            and not score_filtered_points
            and self.settings.qdrant_score_threshold > 0
        ):
            return QDRANT_SCORE_THRESHOLD_NOT_MET
        if score_filtered_points and not navigation_filtered_points:
            return QDRANT_NAVIGATION_TARGET_MISSING
        return QDRANT_NO_RESULTS

    def _build_sources(self, points: list[dict]) -> list[ChatSource]:
        try:
            return [
                QdrantSearchPoint.model_validate(point).to_chat_source()
                for point in points
            ]
        except ValidationError as exc:
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_QDRANT_003,
                message="Qdrant 문서 payload 형식이 올바르지 않습니다.",
            ) from exc
