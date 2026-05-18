import httpx
from pydantic import ValidationError

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.query_filter_extractor import QueryFilterExtractor
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatErrorCode,
    ChatIntent,
    EvidenceLookupFilters,
    EvidenceLookupRequest,
    EvidenceLookupUser,
    EvidenceResult,
)


class EvidenceService:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
        query_filter_extractor: QueryFilterExtractor | None = None,
    ) -> None:
        self.settings = settings
        self.http_client = http_client
        self.query_filter_extractor = query_filter_extractor or QueryFilterExtractor()

    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        if not self.settings.evidence_lookup_enabled:
            return self._empty_result(request, intent)

        self._validate_lookup_settings()
        payload = self._build_payload(request, intent)
        try:
            if self.http_client is not None:
                response = await self.http_client.post(
                    self._evidence_lookup_url,
                    json=payload,
                    headers=self._headers,
                )
                return self._parse_response(response, expected_intent=intent)

            async with httpx.AsyncClient(
                timeout=self.settings.evidence_lookup_timeout_seconds
            ) as client:
                response = await client.post(
                    self._evidence_lookup_url,
                    json=payload,
                    headers=self._headers,
                )
                return self._parse_response(response, expected_intent=intent)
        except httpx.HTTPError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_EVIDENCE_002,
                message="RDB Evidence 조회에 실패했습니다.",
            ) from exc

    def _empty_result(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        return EvidenceResult(
            intent=intent,
            basis_time=request.requested_at,
            items=[],
        )

    @property
    def _evidence_lookup_url(self) -> str:
        return f"{self.settings.evidence_lookup_base_url}{self.settings.evidence_lookup_path}"

    @property
    def _headers(self) -> dict[str, str]:
        if not self.settings.evidence_lookup_internal_token:
            return {}
        return {"X-Internal-Token": self.settings.evidence_lookup_internal_token}

    def _validate_lookup_settings(self) -> None:
        if self.settings.evidence_lookup_internal_token:
            return

        raise ChatExternalServiceError(
            status_code=503,
            code=ChatErrorCode.CHAT_SECURITY_003,
            message="RDB Evidence 내부 토큰이 설정되지 않았습니다.",
        )

    def _build_payload(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> dict:
        lookup_request = EvidenceLookupRequest(
            session_id=request.session_id,
            message_id=request.message_id,
            intent=intent,
            question=request.question,
            user=EvidenceLookupUser(
                user_id=request.user.user_id,
                role=request.user.role,
                company_name=request.user.company_name,
            ),
            filters=EvidenceLookupFilters.model_validate(
                self.query_filter_extractor.extract_filters(request.question)
            ),
        )
        return lookup_request.model_dump(mode="json", by_alias=True)

    def _parse_response(
        self,
        response: httpx.Response,
        expected_intent: ChatIntent,
    ) -> EvidenceResult:
        try:
            response.raise_for_status()
            result = EvidenceResult.model_validate(response.json())
        except httpx.HTTPStatusError as exc:
            raise ChatExternalServiceError(
                status_code=503,
                code=ChatErrorCode.CHAT_EVIDENCE_002,
                message="RDB Evidence 조회에 실패했습니다.",
            ) from exc
        except (ValueError, ValidationError) as exc:
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_EVIDENCE_003,
                message="RDB Evidence 응답 형식이 올바르지 않습니다.",
            ) from exc

        if result.intent != expected_intent:
            raise ChatExternalServiceError(
                status_code=502,
                code=ChatErrorCode.CHAT_EVIDENCE_003,
                message="RDB Evidence 응답 의도가 요청 의도와 일치하지 않습니다.",
            )
        return result
