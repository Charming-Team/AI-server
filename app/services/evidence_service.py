from app.schemas.chat import ChatAnswerRequest, ChatIntent, EvidenceResult


class EvidenceService:
    async def get_evidence(
        self,
        request: ChatAnswerRequest,
        intent: ChatIntent,
    ) -> EvidenceResult:
        return EvidenceResult(
            intent=intent,
            basis_time=request.requested_at,
            items=[],
        )
