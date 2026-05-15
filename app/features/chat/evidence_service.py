import httpx
from pydantic import ValidationError

from app.core.config import Settings
from app.features.chat.exceptions import ChatExternalServiceError
from app.features.chat.schemas import (
    ChatAnswerRequest,
    ChatErrorCode,
    ChatIntent,
    EvidenceResult,
)


class EvidenceService:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.http_client = http_client

    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        if not self.settings.evidence_lookup_enabled:
            return self._empty_result(request, intent)

        payload = self._build_payload(request, intent)
        try:
            if self.http_client is not None:
                response = await self.http_client.post(
                    self._evidence_lookup_url,
                    json=payload,
                    headers=self._headers,
                )
                return self._parse_response(response)

            async with httpx.AsyncClient(
                timeout=self.settings.evidence_lookup_timeout_seconds
            ) as client:
                response = await client.post(
                    self._evidence_lookup_url,
                    json=payload,
                    headers=self._headers,
                )
                return self._parse_response(response)
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

    def _build_payload(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> dict:
        return {
            "sessionId": request.session_id,
            "messageId": request.message_id,
            "intent": intent,
            "question": request.question,
            "user": {
                "userId": request.user.user_id,
                "role": request.user.role,
                "department": request.user.department,
                "companyName": request.user.company_name,
            },
            "filters": {
                "limit": 5,
                "fromDate": None,
                "toDate": None,
                "targetCode": None,
            },
        }

    def _parse_response(self, response: httpx.Response) -> EvidenceResult:
        try:
            response.raise_for_status()
            return EvidenceResult.model_validate(response.json())
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
