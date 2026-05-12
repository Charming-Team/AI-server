import httpx

from app.core.config import Settings
from app.schemas.chat import ChatAnswerRequest, ChatIntent, EvidenceResult


class EvidenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        if not self.settings.evidence_lookup_enabled:
            return self._empty_result(request, intent)

        payload = self._build_payload(request, intent)
        async with httpx.AsyncClient(
            timeout=self.settings.evidence_lookup_timeout_seconds
        ) as client:
            response = await client.post(
                self._evidence_lookup_url,
                json=payload,
                headers=self._headers,
            )
            response.raise_for_status()

        return EvidenceResult.model_validate(response.json())

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
